import argparse
import io
import json
import logging
import re
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests

client_logger = logging.getLogger("YuE2Client")

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_AR_LORA = "loras/ar_lora_inst_v3abc.safetensors"
DEFAULT_NAR_LORA = "loras/nar_lora_joint_v4.safetensors"

# Local/ephemeral state keys that must be stripped before sending yue_kwargs to YuE2Pipeline
EPHEMERAL_KEYS = (
    "id",
    "_source_style",
    "_source_abc",
    "pause",
    "volume",
    "active_style",
    "ar_scale",
    "nar_scale",
    "ar_lora",
    "nar_lora",
    "abc_base_dir",
    "abc_order",
    "strip_abc_tempo",
    "enable_bpm_modification",
    "bpm_source",
    "bpm_override",
    "_target_bpm",
    "_enable_bpm",
    "enable_normalization",
    "enable_reverb",
    "reverb_room_size",
    "reverb_damping",
    "reverb_wet_level",
    "reverb_dry_level",
    "target_lufs",
)


def build_api_payload(
    raw_metadata: Dict[str, Any],
    default_ar_lora: Optional[str] = DEFAULT_AR_LORA,
    default_nar_lora: Optional[str] = DEFAULT_NAR_LORA,
    default_ar_scale: float = 0.5,
    default_nar_scale: float = 1.0,
    abc_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extracts LoRA parameters, strips local filesystem/controller keys, and formats
    the payload expected by the YuE2 server's /generate endpoint.
    """
    yue_kwargs = raw_metadata.copy()

    if abc_override is not None:
        yue_kwargs["abc"] = abc_override

    # --- CONDITIONAL ABC TEMPO TAG REPLACEMENT ---
    enable_bpm = yue_kwargs.get("_enable_bpm", True)
    target_bpm = yue_kwargs.get("_target_bpm")
    strip_tempo = str(yue_kwargs.get("strip_abc_tempo", False)).lower() == "true"

    if "abc" in yue_kwargs and isinstance(yue_kwargs["abc"], str):
        original_abc = yue_kwargs["abc"]
        abc_str = original_abc

        if enable_bpm and target_bpm:
            # Replace full-line Q: tags (e.g., "Q: 1/4=120" -> "Q: 1/4={target_bpm}")
            abc_str = re.sub(r'(^Q:[^=\n]*=?\s*)\d+', rf'\g<1>{target_bpm}', original_abc, flags=re.MULTILINE)
            # Replace inline Q: tags (e.g., "[Q: 120]" -> "[Q: {target_bpm}]")
            abc_str = re.sub(r'(\[Q:[^=\]]*=?\s*)\d+', rf'\g<1>{target_bpm}', abc_str)
            
            if abc_str != original_abc:
                client_logger.info(f"Tempo tags (Q:) forced to {target_bpm} BPM.")
            else:
                client_logger.info("No tempo tags (Q:) found in the ABC file to overwrite.")
        elif strip_tempo:
            # Fallback to stripping if enabled and no enforcement is applied
            abc_str = re.sub(r'^Q:.*$\n?', '', original_abc, flags=re.MULTILINE)
            abc_str = re.sub(r'\[Q:[^\]]*\]', '', abc_str)
            client_logger.info("Tempo tags (Q:) stripped from ABC file.")
            
        yue_kwargs["abc"] = abc_str
    # -------------------------------------------

    ar_lora = yue_kwargs.pop("ar_lora", None) or default_ar_lora
    nar_lora = yue_kwargs.pop("nar_lora", None) or default_nar_lora
    ar_scale = float(yue_kwargs.pop("ar_scale", default_ar_scale))
    nar_scale = float(yue_kwargs.pop("nar_scale", default_nar_scale))

    for key in EPHEMERAL_KEYS:
        yue_kwargs.pop(key, None)

    return {
        "ar_lora": str(ar_lora) if ar_lora else None,
        "nar_lora": str(nar_lora) if nar_lora else None,
        "ar_scale": ar_scale,
        "nar_scale": nar_scale,
        "yue_kwargs": yue_kwargs,
    }


def generate_and_download(
    payload: Dict[str, Any],
    api_url: str = DEFAULT_API_URL,
    poll_interval: float = 3.0,
    request_timeout: float = 15.0,
    download_timeout: float = 60.0,
    shutdown_event: Optional[threading.Event] = None,
) -> Optional[bytes]:
    """
    Submits a generation payload to the YuE2 server, polls until completion,
    and returns the resulting ZIP archive bytes (or None if interrupted by shutdown_event).
    """
    base_url = api_url.rstrip("/")

    # 1. Submit task
    submit_resp = requests.post(f"{base_url}/generate", json=payload, timeout=request_timeout)
    submit_resp.raise_for_status()
    task_id = submit_resp.json()["task_id"]
    client_logger.info(f"Task submitted to {base_url}. Task ID: {task_id}")

    # 2. Poll for status
    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            return None

        status_resp = requests.get(f"{base_url}/status/{task_id}", timeout=request_timeout)
        status_resp.raise_for_status()
        status_data = status_resp.json()
        status = status_data.get("status")

        if status == "completed":
            break
        elif status == "failed":
            error_msg = status_data.get("error", "Unknown server error")
            raise RuntimeError(f"Server generation failed: {error_msg}")

        if shutdown_event is not None:
            shutdown_event.wait(poll_interval)
        else:
            time.sleep(poll_interval)

    if shutdown_event is not None and shutdown_event.is_set():
        return None

    # 3. Download result
    download_resp = requests.get(f"{base_url}/download/{task_id}", timeout=download_timeout)
    download_resp.raise_for_status()
    return download_resp.content


def extract_zip_bytes(zip_content: bytes, output_dir: Path) -> Path:
    """Extracts an in-memory ZIP byte stream to the target directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
        zf.extractall(output_dir)
    return output_dir.resolve()


def run_recreate_cli(
    metadata_file: Path,
    output_dir: Path,
    abc_file: Optional[Path] = None,
    api_url: str = DEFAULT_API_URL,
) -> int:
    """Shared execution helper for recreating a track from metadata (with optional custom ABC)."""
    if not metadata_file.exists():
        print(f"Error: Metadata file not found - {metadata_file}")
        return 1

    abc_override = None
    if abc_file is not None:
        if not abc_file.exists():
            print(f"Error: ABC file not found - {abc_file}")
            return 1
        abc_override = abc_file.read_text(encoding="utf-8")

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    payload = build_api_payload(metadata, abc_override=abc_override)

    print(f"Submitting task to {api_url}...")
    try:
        zip_content = generate_and_download(payload, api_url=api_url)
    except Exception as e:
        print(f"Generation failed: {e}")
        return 1

    if not zip_content:
        print("Generation cancelled.")
        return 1

    print("Done! Extracting artifacts...")
    saved_path = extract_zip_bytes(zip_content, output_dir)
    print(f"Success! Track saved to {saved_path}")
    return 0


def recreate_main():
    """Unified CLI entrypoint for `infinite-radio-recreate`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Recreate a track from a metadata.json file, optionally overriding the .abc score."
    )
    parser.add_argument("metadata_file", type=Path, help="Path to the metadata.json file")
    parser.add_argument(
        "-a", "--abc", dest="abc_file", type=Path, default=None, help="Optional path to an edited .abc file"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Folder to save the output (defaults to recreated_track or custom_track)"
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the YuE2 API Server")
    args = parser.parse_args()

    default_out = Path("custom_track") if args.abc_file else Path("recreated_track")
    output_dir = args.output or default_out

    sys.exit(
        run_recreate_cli(
            metadata_file=args.metadata_file,
            output_dir=output_dir,
            abc_file=args.abc_file,
            api_url=args.api_url,
        )
    )


if __name__ == "__main__":
    recreate_main()
