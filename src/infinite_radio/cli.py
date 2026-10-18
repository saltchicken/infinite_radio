import warnings
import io
import zipfile
import argparse
import json
import yaml
import random
import threading
import queue
import time
import logging
import sqlite3
from pathlib import Path

import requests
import numpy as np
import soundfile as sf
import uvicorn
import pyloudnorm as pyln

from .utils.shuffle_utils import smart_choice
from .player import audio_player
from .remote_api import create_remote_api
from .client import (
    DEFAULT_API_URL,
    DEFAULT_AR_LORA,
    DEFAULT_NAR_LORA,
    build_api_payload,
    generate_and_download,
)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
main_logger = logging.getLogger("Main")

DEFAULT_STATE = {
    "volume": 0.4,
    "pause": False,
    "active_style": ["styles/upvoted/"],
    "ar_lora": DEFAULT_AR_LORA,
    "nar_lora": DEFAULT_NAR_LORA,
    "ar_scale": 0.5,
    "nar_scale": 1.0,
    "abc": "",
    "abc_order": "random",  # Options: "random" or "sequential"
    "cot": "melody",
    "abc_base_dir": "sheetsage/output",
    "strip_abc_tempo": False,
    # DSP Settings
    "enable_normalization": True,
    "enable_reverb": True,
    "reverb_room_size": 0.3,
    "reverb_damping": 0.5,
    "reverb_wet_level": 0.15,
    "reverb_dry_level": 0.85,
    "target_lufs": -14.0,
}

# Refresh cached file lists every 5 minutes in case new files are added on disk
CACHE_TTL_SECONDS = 300.0


def load_config(config_path: Path) -> dict:
    """Reads the static YAML configuration file to set base defaults."""
    state = DEFAULT_STATE.copy()
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
            if user_config:
                state.update(user_config)
            main_logger.info(f"Loaded configuration from {config_path.name}")
        except Exception as e:
            main_logger.warning(f"Could not parse {config_path.name} ({e}). Using hardcoded defaults.")
    else:
        main_logger.info(f"No {config_path.name} found. Using hardcoded defaults.")
    
    # Ensure ephemeral state starts clean regardless of config file
    state["pause"] = False 
    return state


def _resolve_style_files(active_styles: list[str], base_dir: Path) -> list[Path]:
    """Scans the filesystem for matching style .json files."""
    style_files = []
    for style_req in active_styles:
        style_path = (base_dir / style_req) if not Path(style_req).is_absolute() else Path(style_req)
        if style_path.is_dir():
            style_files.extend(style_path.rglob("*.json"))
        elif style_path.is_file():
            style_files.append(style_path)
        else:
            main_logger.warning(f"Style path not found: {style_path}")
    return sorted(set(style_files))


def _resolve_abc_files(
    abc_reqs: list[str],
    abc_base_dir: str | None,
    cot_mode: str,
    base_dir: Path,
) -> list[Path]:
    """Scans the filesystem (or SQLite DB) for matching .abc notation files while preserving request order."""
    abc_files = []
    seen = set()
    db_path = base_dir / "votes.db"

    for abc_req in abc_reqs:
        # Intercept the virtual favorites directory flag
        if abc_req == "⭐ Favorites (SQLite)":
            if db_path.exists():
                try:
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT file_path FROM favorite_abcs")
                        for row in cursor.fetchall():
                            f_path = Path(row[0])
                            if f_path.exists() and f_path not in seen:
                                seen.add(f_path)
                                abc_files.append(f_path)
                except Exception as e:
                    main_logger.warning(f"Failed to load favorites from database: {e}")
            else:
                main_logger.warning("Favorites DB does not exist yet. Vote up some ABCs first!")
            continue

        # Standard file path resolution
        req_path = Path(abc_req)
        if req_path.is_absolute():
            abc_path = req_path
        elif abc_base_dir:
            base_abc = Path(abc_base_dir)
            abc_path = (base_abc / req_path) if base_abc.is_absolute() else (base_dir / base_abc / req_path)
        else:
            abc_path = base_dir / req_path

        matched = []
        if abc_path.is_dir():
            if cot_mode in ["full", "melody"]:
                matched = sorted(abc_path.rglob(f"*-{cot_mode}.abc"))
            else:
                matched = sorted(abc_path.rglob("*.abc"))
        elif abc_path.is_file():
            matched = [abc_path]
        else:
            main_logger.warning(f"ABC path not found: {abc_path}")

        for f in matched:
            if f not in seen:
                seen.add(f)
                abc_files.append(f)

    return abc_files


def main():
    parser = argparse.ArgumentParser(description="Infinite Radio Client")
    parser.add_argument(
        "-c", "--config",
        dest="config_file",
        type=Path,
        default=Path.cwd() / "config.yaml",
        help="Path to the static YAML configuration file",
    )
    parser.add_argument("--save-dir", type=Path, default=Path("saved_tracks"), help="Directory to save explicitly saved tracks")
    parser.add_argument("--crossfade", type=float, default=6.0)
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the YuE2 API Server")
    parser.add_argument("--ar-lora", type=str, help="Fallback path to the AR LoRA safetensors file on server")
    parser.add_argument("--nar-lora", type=str, help="Fallback path to the NAR LoRA safetensors file on server")
    
    # New secure UI routing arguments
    parser.add_argument("--ui-host", default="127.0.0.1", help="Host address for the local Web UI")
    parser.add_argument("--ui-port", type=int, default=8001, help="Port for the local Web UI")
    
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    config_file = args.config_file.resolve()
    base_dir = config_file.parent

    shared_state = load_config(config_file)

    pause_event = threading.Event()
    if shared_state.get("pause", False):
        pause_event.set()

    skip_event = threading.Event()
    shutdown_event = threading.Event()
    config_lock = threading.Lock()
    play_queue = queue.Queue()

    playback_info = {}

    player_thread = threading.Thread(
        target=audio_player,
        args=(play_queue, shared_state, config_lock, pause_event, skip_event, shutdown_event, playback_info, args.crossfade),
        daemon=True
    )

    remote_app = create_remote_api(
        shared_state=shared_state,
        play_queue=play_queue,
        pause_event=pause_event,
        skip_event=skip_event,
        config_lock=config_lock,
        playback_info=playback_info,
        base_path=base_dir,
        save_dir=args.save_dir,
    )
    
    api_thread = threading.Thread(
        target=uvicorn.run,
        args=(remote_app,),
        kwargs={"host": args.ui_host, "port": args.ui_port, "log_config": None, "access_log": False},
        daemon=True
    )

    api_thread.start()
    player_thread.start()

    main_logger.info("==============================================")
    main_logger.info("📻 Infinite Radio Local Playback Started")
    main_logger.info(f"🌐 Connected to YuE2 API at {api_url}")
    main_logger.info(f"📱 Web Control & Config UI: http://{args.ui_host}:{args.ui_port}")
    main_logger.info("==============================================")

    track_number = 1
    style_history = []
    abc_history = []
    abc_index = 0

    # Filesystem I/O cache state
    style_cache_key = None
    style_cache_files: list[Path] = []
    style_cache_time = 0.0

    abc_cache_key = None
    abc_cache_files: list[Path] = []
    abc_cache_time = 0.0

    try:
        while not shutdown_event.is_set():
            while pause_event.is_set() and not shutdown_event.is_set():
                shutdown_event.wait(1.0)

            while play_queue.qsize() >= 3 and not shutdown_event.is_set():
                shutdown_event.wait(1.0)

            if shutdown_event.is_set():
                break
            if pause_event.is_set():
                continue

            with config_lock:
                controller_req = shared_state.copy()

            chosen_style_path = None
            chosen_abc_path = None

            style_data = {}
            if "active_style" in controller_req and controller_req["active_style"]:
                active_styles = controller_req["active_style"]
                if isinstance(active_styles, str):
                    active_styles = [s.strip() for s in active_styles.split(",") if s.strip()]

                current_style_key = tuple(active_styles)
                now = time.monotonic()
                if current_style_key != style_cache_key or (now - style_cache_time) > CACHE_TTL_SECONDS:
                    style_cache_files = _resolve_style_files(active_styles, base_dir)
                    style_cache_key = current_style_key
                    style_cache_time = now

                if style_cache_files:
                    chosen_style = smart_choice(style_cache_files, style_history, max_memory=4)
                    chosen_style_path = str(chosen_style)
                    try:
                        style_data = json.loads(chosen_style.read_text(encoding="utf-8"))
                    except OSError as e:
                        main_logger.warning(f"Cached style file inaccessible ({e}). Invalidating style cache.")
                        style_cache_key = None
                        continue
                    except json.JSONDecodeError as e:
                        main_logger.error(f"Invalid JSON in style file {chosen_style.name} ({e}). Skipping.")
                        # Remove the corrupted file from the active rotation so it isn't repeatedly selected
                        if chosen_style in style_cache_files:
                            style_cache_files.remove(chosen_style)
                        continue
                else:
                    main_logger.warning(f"No .json files found for active_style(s): {active_styles}")

            active_request = style_data.copy()
            active_request.update(controller_req)

            active_request["seed"] = active_request.get("seed", random.randint(0, 2**32 - 1))
            cot_mode = active_request.get("cot", "full")

            # Extract the base directory and playback order for ABC files
            abc_base_dir = active_request.pop("abc_base_dir", None)
            abc_order = active_request.pop("abc_order", "random")

            if isinstance(active_request.get("lyrics"), list):
                chosen = random.choice(active_request["lyrics"])
                active_request["lyrics"] = "\n".join(chosen) if isinstance(chosen, list) else chosen

            if "choices" in active_request:
                choices = active_request.pop("choices")
                if isinstance(active_request.get("style"), str):
                    for key, options in choices.items():
                        chosen_val = str(random.choice(options))
                        active_request["style"] = active_request["style"].replace(f"{{{key}}}", chosen_val)

            if "abc" in active_request and active_request["abc"]:
                abc_reqs = active_request["abc"]
                if isinstance(abc_reqs, str):
                    abc_reqs = [s.strip() for s in abc_reqs.split(",") if s.strip()]

                raw_abc = None
                current_abc_key = (tuple(abc_reqs), str(abc_base_dir), str(cot_mode), str(abc_order))
                now = time.monotonic()
                if current_abc_key != abc_cache_key or (now - abc_cache_time) > CACHE_TTL_SECONDS:
                    abc_cache_files = _resolve_abc_files(abc_reqs, abc_base_dir, cot_mode, base_dir)
                    if current_abc_key != abc_cache_key:
                        abc_index = 0
                    abc_cache_key = current_abc_key
                    abc_cache_time = now

                if abc_cache_files:
                    if abc_order == "sequential":
                        chosen_abc = abc_cache_files[abc_index % len(abc_cache_files)]
                    else:
                        chosen_abc = smart_choice(abc_cache_files, abc_history, max_memory=15)

                    chosen_abc_path = str(chosen_abc)
                    try:
                        raw_abc = chosen_abc.read_text(encoding="utf-8")
                    except OSError as e:
                        main_logger.warning(f"Cached ABC file inaccessible ({e}). Invalidating ABC cache.")
                        abc_cache_key = None
                        continue
                else:
                    main_logger.warning(f"No valid .abc files found for cot '{cot_mode}'. Reverting to standard generation.")
                    active_request.pop("abc", None)

                if raw_abc:
                    active_request["abc"] = raw_abc

            else:
                active_request.pop("abc", None)

            # Build API payload and stripped yue_kwargs via shared client utility
            api_payload = build_api_payload(
                active_request,
                default_ar_lora=args.ar_lora,
                default_nar_lora=args.nar_lora,
                default_ar_scale=1.0,
                default_nar_scale=1.0,
            )

            # Build metadata internally for saving if requested via Web UI
            metadata_to_save = api_payload["yue_kwargs"].copy()
            metadata_to_save["_source_style"] = chosen_style_path
            metadata_to_save["_source_abc"] = chosen_abc_path
            metadata_to_save["ar_lora"] = api_payload["ar_lora"]
            metadata_to_save["nar_lora"] = api_payload["nar_lora"]
            metadata_to_save["ar_scale"] = api_payload["ar_scale"]
            metadata_to_save["nar_scale"] = api_payload["nar_scale"]

            start_time = time.time()
            try:
                zip_content = generate_and_download(
                    api_payload,
                    api_url=api_url,
                    shutdown_event=shutdown_event,
                )
            except (requests.RequestException, RuntimeError) as e:
                main_logger.error(f"API request failed: {e}. Retrying in 5 seconds...")
                shutdown_event.wait(5.0)
                continue

            if not zip_content or shutdown_event.is_set():
                continue

            elapsed = time.time() - start_time

            # Decode audio directly from memory
            try:
                with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                    if "audio.flac" not in zf.namelist():
                        main_logger.error("Server response did not contain audio.flac")
                        continue
                    audio_bytes = zf.read("audio.flac")
                    
                data, fs = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            except Exception as e:
                main_logger.error(f"Failed to decode audio stream: {e}")
                continue

            if data.ndim == 1:
                data = np.column_stack([data, data])

            if controller_req.get("enable_normalization", True):
                target_lufs = float(controller_req.get("target_lufs", -14.0))

                # Measure and normalize the perceived loudness
                meter = pyln.Meter(fs)
                current_loudness = meter.integrated_loudness(data)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    data = pyln.normalize.loudness(data, current_loudness, target_lufs)

                # Prevent hard clipping from normalization gain
                data = np.clip(data, -0.99, 0.99)
            
            # Make sure memory is strictly C-contiguous and float32 for sounddevice
            data = np.ascontiguousarray(data, dtype=np.float32)

            play_queue.put((data, fs, metadata_to_save, zip_content, track_number))
            main_logger.info(f"Track {track_number} finished generating in {elapsed:.2f} seconds. Queued for next crossfade.")

            track_number += 1
            if abc_order == "sequential" and abc_cache_files:
                abc_index = (abc_index + 1) % len(abc_cache_files)

    except KeyboardInterrupt:
        main_logger.info("Shutdown signal received (Ctrl+C). Terminating gracefully...")
    except Exception as e:
        main_logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        shutdown_event.set()
        pause_event.clear()

        main_logger.info("Waiting for background processes to exit...")
        player_thread.join(timeout=3)
        main_logger.info("Infinite Radio has shut down safely.")


if __name__ == "__main__":
    main()
