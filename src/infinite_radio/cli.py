import io
import zipfile
import argparse
import json
import random
import threading
import queue
import time
import shutil
import logging
from pathlib import Path

import requests
import numpy as np
import soundfile as sf
import uvicorn

from .utils.shuffle_utils import smart_choice
from .player import audio_player
from .remote_api import create_remote_api

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
    "ar_lora": "loras/ar_lora_inst_v3abc.safetensors",
    "nar_lora": "loras/nar_lora_joint_v4.safetensors",
    "ar_scale": 0.5,
    "nar_scale": 1.0,
    "abc": "",
    "cot": "melody",
    "abc_base_dir": "sheetsage/output",
}


def load_initial_state(state_path: Path) -> dict:
    state = DEFAULT_STATE.copy()
    if state_path.exists():
        try:
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            state.update(saved)
            main_logger.info(f"Loaded saved state from {state_path.name}")
        except Exception as e:
            main_logger.warning(f"Could not parse {state_path.name} ({e}). Using defaults.")
    else:
        try:
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            main_logger.info(f"Created initial state file at {state_path.name}")
        except Exception as e:
            main_logger.warning(f"Could not write initial state file: {e}")
    return state


def main():
    parser = argparse.ArgumentParser(description="Infinite Radio Client")
    parser.add_argument(
        "--state-file",
        dest="state_file",
        type=Path,
        default=Path.cwd() / "controller.json",
        help="Path to persist Web UI settings across restarts",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crossfade", type=float, default=6.0)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Base URL of the YuE2 API Server")
    parser.add_argument("--ar-lora", type=str, help="Fallback path to the AR LoRA safetensors file on server")
    parser.add_argument("--nar-lora", type=str, help="Fallback path to the NAR LoRA safetensors file on server")
    
    # New secure UI routing arguments
    parser.add_argument("--ui-host", default="127.0.0.1", help="Host address for the local Web UI")
    parser.add_argument("--ui-port", type=int, default=8001, help="Port for the local Web UI")
    
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    args.output.mkdir(parents=True, exist_ok=True)
    state_file = args.state_file.resolve()
    base_dir = state_file.parent

    shared_state = load_initial_state(state_file)

    pause_event = threading.Event()
    if shared_state.get("pause", False):
        pause_event.set()

    skip_event = threading.Event()
    shutdown_event = threading.Event()
    config_lock = threading.Lock()
    play_queue = queue.Queue()

    playback_info = {"current_dir": None}

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
        state_file=state_file,
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
    track_history = []
    style_history = []
    abc_history = []

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

            track_log_details = []
            style_data = {}
            if "active_style" in controller_req and controller_req["active_style"]:
                active_styles = controller_req["active_style"]
                if isinstance(active_styles, str):
                    active_styles = [s.strip() for s in active_styles.split(",") if s.strip()]

                style_files = []
                for style_req in active_styles:
                    style_path = (base_dir / style_req) if not Path(style_req).is_absolute() else Path(style_req)
                    if style_path.is_dir():
                        style_files.extend(style_path.rglob("*.json"))
                    elif style_path.is_file():
                        style_files.append(style_path)
                    else:
                        main_logger.warning(f"Style path not found: {style_path}")

                if style_files:
                    style_files = list(set(style_files))
                    chosen_style = smart_choice(style_files, style_history, max_memory=4)
                    chosen_style_path = str(chosen_style)
                    track_log_details.append(f"Style: {chosen_style.parent.name}/{chosen_style.name}")
                    style_data = json.loads(chosen_style.read_text(encoding="utf-8"))
                else:
                    main_logger.warning(f"No .json files found for active_style(s): {active_styles}")

            active_request = style_data.copy()
            active_request.update(controller_req)

            # Extract LoRA settings for the API wrapper
            requested_ar_scale = float(active_request.get("ar_scale", 1.0))
            requested_nar_scale = float(active_request.get("nar_scale", 1.0))
            requested_ar_lora = active_request.get("ar_lora") or args.ar_lora
            requested_nar_lora = active_request.get("nar_lora") or args.nar_lora

            active_request["seed"] = active_request.get("seed", random.randint(0, 2**32 - 1))
            cot_mode = active_request.get("cot", "full")

            # Extract the base directory for ABC files
            abc_base_dir = active_request.pop("abc_base_dir", None)

            # Cleaned up ephemeral keys from pop target list
            for key in ["pause", "volume", "active_style", "ar_scale", "nar_scale", "ar_lora", "nar_lora", "id", "_source_style", "_source_abc"]:
                active_request.pop(key, None)

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

                abc_files = []
                raw_abc = None

                for abc_req in abc_reqs:
                    req_path = Path(abc_req)
                    if req_path.is_absolute():
                        abc_path = req_path
                    elif abc_base_dir:
                        base_abc = Path(abc_base_dir)
                        abc_path = (base_abc / req_path) if base_abc.is_absolute() else (base_dir / base_abc / req_path)
                    else:
                        abc_path = base_dir / req_path

                    if abc_path.is_dir():
                        if cot_mode in ["full", "melody"]:
                            abc_files.extend(abc_path.rglob(f"*-{cot_mode}.abc"))
                        else:
                            abc_files.extend(abc_path.rglob("*.abc"))
                    elif abc_path.is_file():
                        abc_files.append(abc_path)
                    else:
                        main_logger.warning(f"ABC path not found: {abc_path}")

                if abc_files:
                    abc_files = list(set(abc_files))
                    chosen_abc = smart_choice(abc_files, abc_history, max_memory=15)
                    chosen_abc_path = str(chosen_abc)
                    track_log_details.append(f"Mode: '{cot_mode}' | Score: {chosen_abc.parent.name}/{chosen_abc.name}")
                    raw_abc = chosen_abc.read_text(encoding="utf-8")
                else:
                    main_logger.warning(f"No valid .abc files found for cot '{cot_mode}'. Reverting to standard generation.")
                    active_request.pop("abc", None)

                if raw_abc:
                    active_request["abc"] = raw_abc

            else:
                active_request.pop("abc", None)

            current_output_dir = args.output / f"track_{track_number:03d}"
            current_output_dir.mkdir(exist_ok=True)

            # Save metadata locally
            metadata_to_save = active_request.copy()
            metadata_to_save["_source_style"] = chosen_style_path
            metadata_to_save["_source_abc"] = chosen_abc_path
            
            # --- NEW: Re-inject LoRA bindings into metadata for recreate_track.py ---
            metadata_to_save["ar_lora"] = str(requested_ar_lora) if requested_ar_lora else None
            metadata_to_save["nar_lora"] = str(requested_nar_lora) if requested_nar_lora else None
            metadata_to_save["ar_scale"] = requested_ar_scale
            metadata_to_save["nar_scale"] = requested_nar_scale

            metadata_path = current_output_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_to_save, f, indent=2)

            combined_log = f"✅ Requesting Track {track_number} | " + " ".join(track_log_details)
            main_logger.info(combined_log)

            api_payload = {
                "ar_lora": str(requested_ar_lora) if requested_ar_lora else None,
                "nar_lora": str(requested_nar_lora) if requested_nar_lora else None,
                "ar_scale": requested_ar_scale,
                "nar_scale": requested_nar_scale,
                "yue_kwargs": active_request,
            }

            start_time = time.time()
            zip_content = None
            try:
                # 1. Submit Request
                submit_resp = requests.post(f"{api_url}/generate", json=api_payload, timeout=15)
                submit_resp.raise_for_status()
                task_id = submit_resp.json()["task_id"]
                main_logger.info(f"Task submitted to server. Task ID: {task_id}")

                # 2. Poll Status
                completed = False
                while not completed and not shutdown_event.is_set():
                    status_resp = requests.get(f"{api_url}/status/{task_id}", timeout=15)
                    status_resp.raise_for_status()
                    status_data = status_resp.json()
                    
                    if status_data["status"] == "completed":
                        completed = True
                    elif status_data["status"] == "failed":
                        error_msg = status_data.get("error", "Unknown API error")
                        raise RuntimeError(f"Server generation failed: {error_msg}")
                    else:
                        shutdown_event.wait(3.0)

                if shutdown_event.is_set():
                    continue

                # 3. Download Result
                download_resp = requests.get(f"{api_url}/download/{task_id}", timeout=60)
                download_resp.raise_for_status()
                zip_content = download_resp.content

            except (requests.RequestException, RuntimeError) as e:
                main_logger.error(f"API request failed: {e}. Retrying in 5 seconds...")
                shutdown_event.wait(5.0)
                continue

            elapsed = time.time() - start_time

            # Extract returned artifacts into current_output_dir
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                zf.extractall(current_output_dir)

            audio_path = current_output_dir / "audio.flac"
            if not audio_path.exists():
                main_logger.error(f"Server response did not contain audio.flac in {current_output_dir}")
                continue

            data, fs = sf.read(audio_path, dtype="float32")
            if data.ndim == 1:
                data = np.column_stack([data, data])

            play_queue.put((data, fs, current_output_dir))
            main_logger.info(f"Track {track_number} finished generating in {elapsed:.2f} seconds. Queued for next crossfade.")

            track_history.append(current_output_dir)
            if len(track_history) > 10:
                old_track_dir = track_history.pop(0)
                if old_track_dir.exists() and old_track_dir != playback_info.get("current_dir"):
                    try:
                        shutil.rmtree(old_track_dir)
                        main_logger.info(f"Ring buffer: Deleted old track {old_track_dir.name} from disk.")
                    except OSError as e:
                        main_logger.error(f"Ring buffer: Failed to delete {old_track_dir.name}: {e}")

            track_number += 1

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
