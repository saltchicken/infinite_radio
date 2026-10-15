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

from .utils.abc_utils import force_instrumental_abc
from .utils.shuffle_utils import smart_choice
from .watcher import load_controller_config, config_watcher
from .player import audio_player
from .remote_api import create_remote_api

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
main_logger = logging.getLogger("Main")


def main():
    parser = argparse.ArgumentParser(description="Infinite Radio Client")
    parser.add_argument("--request", type=Path, default=Path.cwd() / "controller.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crossfade", type=float, default=6.0)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Base URL of the YuE2 API Server")
    parser.add_argument("--ar-lora", type=str, help="Fallback path to the AR LoRA safetensors file on server")
    parser.add_argument("--nar-lora", type=str, help="Fallback path to the NAR LoRA safetensors file on server")
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    args.output.mkdir(parents=True, exist_ok=True)

    shared_state = {"volume": 0.4, "pause": False}
    try:
        initial_data = load_controller_config(args.request)
        shared_state.update(initial_data)
    except Exception as e:
        main_logger.error(f"Error loading initial config: {e}. Ensure controller.json exists.")
        return

    pause_event = threading.Event()
    skip_event = threading.Event()
    shutdown_event = threading.Event()
    config_lock = threading.Lock()
    play_queue = queue.Queue()

    playback_info = {"current_dir": None}

    watcher_thread = threading.Thread(
        target=config_watcher,
        args=(args.request, shared_state, shutdown_event, config_lock),
        daemon=True
    )
    player_thread = threading.Thread(
        target=audio_player,
        args=(play_queue, shared_state, config_lock, pause_event, skip_event, shutdown_event, playback_info, args.crossfade),
        daemon=True
    )

    remote_app = create_remote_api(shared_state, pause_event, skip_event, config_lock, playback_info, args.request.parent)
    api_thread = threading.Thread(
        target=uvicorn.run,
        args=(remote_app,),
        kwargs={"host": "0.0.0.0", "port": 8001, "log_level": "warning", "use_colors": True},
        daemon=True
    )

    api_thread.start()
    watcher_thread.start()
    player_thread.start()

    main_logger.info("==============================================")
    main_logger.info("📻 Infinite Radio Local Playback Started")
    main_logger.info(f"🌐 Connected to YuE2 API at {api_url}")
    main_logger.info(f"📱 Remote Control API listening on http://0.0.0.0:8001")
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
                    active_styles = [active_styles]

                style_files = []
                for style_req in active_styles:
                    style_path = args.request.parent / style_req
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

            active_request["seed"] = random.randint(0, 2**32 - 1)
            cot_mode = active_request.get("cot", "full")

            inst_name = active_request.pop("abc_instrument", None)
            midi_prog = active_request.pop("abc_midi_program", None)

            # Cleaned up ephemeral keys from pop target list
            for key in ["pause", "volume", "active_style", "ar_scale", "nar_scale", "ar_lora", "nar_lora"]:
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
                    abc_reqs = [abc_reqs]

                abc_files = []
                raw_abc = None

                for abc_req in abc_reqs:
                    abc_path = Path(abc_req)
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
                    if inst_name is not None and midi_prog is not None:
                        active_request["abc"] = force_instrumental_abc(raw_abc, inst_name, midi_prog)
                        track_log_details.append("Live Edited ABC String applied (forced instrumental).")
                    else:
                        active_request["abc"] = raw_abc

            current_output_dir = args.output / f"track_{track_number:03d}"
            current_output_dir.mkdir(exist_ok=True)

            # Save metadata locally
            metadata_to_save = active_request.copy()
            metadata_to_save["_source_style"] = chosen_style_path
            metadata_to_save["_source_abc"] = chosen_abc_path

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
                        # Continue polling
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
        watcher_thread.join(timeout=3)
        player_thread.join(timeout=3)
        main_logger.info("Infinite Radio has shut down safely.")


if __name__ == "__main__":
    main()
