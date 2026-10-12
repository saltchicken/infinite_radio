import sys
import torch
import argparse
import json
import random
import threading
import queue
import time
import shutil
import numpy as np
import soundfile as sf
import logging
from pathlib import Path
from yue2 import YuE2Pipeline

from .utils.abc_utils import force_instrumental_abc
from .utils.lora_utils import load_ckpt, merge_lora
from .utils.shuffle_utils import smart_choice
from .watcher import load_controller_config, config_watcher
from .player import audio_player

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
main_logger = logging.getLogger("Main")
sys_logger = logging.getLogger("System")


class StreamFilter:
    """Filter to intercept and drop logs containing [YuE2]"""
    def __init__(self, stream, target="[YuE2]"):
        self.stream = stream
        self.target = target
        self.buf = ""

    def write(self, data):
        self.buf += data
        
        # Process both newlines and carriage returns
        while True:
            if "\n" in self.buf:
                splitter = "\n"
            elif "\r" in self.buf:
                splitter = "\r"
            else:
                break
                
            line, self.buf = self.buf.split(splitter, 1)
            
            if self.target not in line:
                self.stream.write(line + splitter)
                self.stream.flush() # Force terminal to render the update

    def flush(self):
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, default=Path.cwd() / "controller.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crossfade", type=float, default=6.0)
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    parser.add_argument("--ar-lora", type=Path, help="Path to the AR LoRA safetensors file")
    parser.add_argument("--nar-lora", type=Path, help="Path to the NAR LoRA safetensors file")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    shared_state = {}
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
        args=(args.request, shared_state, pause_event, skip_event, shutdown_event, config_lock, playback_info),
        daemon=True)
    player_thread = threading.Thread(
        target=audio_player,
        args=(play_queue, shared_state, config_lock, pause_event, skip_event, shutdown_event, playback_info, args.crossfade),
        daemon=True)

    watcher_thread.start()
    player_thread.start()

    main_logger.info("==============================================")
    main_logger.info("📻 Infinite Radio Local Playback Started")
    main_logger.info("==============================================")

    pipe = None
    current_ar_scale = None
    current_nar_scale = None
    current_ar_lora = None
    current_nar_lora = None
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

            track_log_details = []
            style_data = {}
            if "active_style" in controller_req and controller_req["active_style"]:
                style_path = args.request.parent / controller_req["active_style"]
                if style_path.is_dir():
                    style_files = list(style_path.glob("*.json"))
                    if style_files:
                        chosen_style = smart_choice(style_files, style_history, max_memory=4)
                        track_log_details.append(f"Style: {chosen_style.name}")
                        style_data = json.loads(chosen_style.read_text(encoding="utf-8"))
                    else:
                        main_logger.warning(f"No .json files found in style directory {style_path}.")
                elif style_path.is_file():
                    track_log_details.append(f"Style: {style_path.name}")
                    style_data = json.loads(style_path.read_text(encoding="utf-8"))
                else:
                    main_logger.warning(f"Style path not found: {style_path}")

            active_request = style_data.copy()
            active_request.update(controller_req)

            requested_ar_scale = float(active_request.get("ar_scale", 1.0))
            requested_nar_scale = float(active_request.get("nar_scale", 1.0))

            req_ar_lora_str = active_request.get("ar_lora", "")
            req_nar_lora_str = active_request.get("nar_lora", "")

            requested_ar_lora = Path(req_ar_lora_str) if req_ar_lora_str else args.ar_lora
            requested_nar_lora = Path(req_nar_lora_str) if req_nar_lora_str else args.nar_lora

            if (pipe is None or current_ar_scale != requested_ar_scale or
                    current_nar_scale != requested_nar_scale or
                    current_ar_lora != requested_ar_lora or
                    current_nar_lora != requested_nar_lora):

                if pipe is not None:
                    sys_logger.info("🔄 LoRA configuration change detected. Reloading model...")
                    del pipe
                    torch.cuda.empty_cache()
                else:
                    main_logger.info("Loading YuE2 Model into VRAM...")

                pipe = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda")
                top_model = pipe._load_model()
                bb = top_model.model

                if requested_ar_lora and requested_ar_lora.exists():
                    main_logger.info(f"Loading AR LoRA from {requested_ar_lora.name}...")
                    ar_data = load_ckpt(str(requested_ar_lora), "cuda")
                    n_merged = merge_lora(bb, "self_attn", "mlp", ar_data["lora"], scale=requested_ar_scale, dev="cuda")
                    main_logger.info(f"Merged {n_merged} AR LoRA linear layers (scale: {requested_ar_scale}).")

                if requested_nar_lora and requested_nar_lora.exists():
                    main_logger.info(f"Loading NAR LoRA from {requested_nar_lora.name}...")
                    nar_data = load_ckpt(str(requested_nar_lora), "cuda")
                    n_merged = merge_lora(bb, "nar_self_attn", "nar_mlp", nar_data["lora"], scale=requested_nar_scale, dev="cuda")
                    top_model.vae2llm.load_state_dict({k: v.to(torch.bfloat16) for k, v in nar_data["io"]["vae2llm"].items()})
                    top_model.llm2vae.load_state_dict({k: v.to(torch.bfloat16) for k, v in nar_data["io"]["llm2vae"].items()})
                    main_logger.info(f"Merged {n_merged} NAR LoRA linear layers (scale: {requested_nar_scale}) & injected custom VAE/LLM crossovers.")

                current_ar_scale, current_nar_scale = requested_ar_scale, requested_nar_scale
                current_ar_lora, current_nar_lora = requested_ar_lora, requested_nar_lora

            active_request["seed"] = random.randint(0, 2**32 - 1)
            cot_mode = active_request.get("cot", "full")

            inst_name = active_request.pop("abc_instrument", None)
            midi_prog = active_request.pop("abc_midi_program", None)

            for key in ["pause", "skip", "volume", "save", "active_style", "ar_scale", "nar_scale", "ar_lora", "nar_lora"]:
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
                abc_path = Path(active_request["abc"])
                raw_abc = None

                if abc_path.is_dir():
                    abc_files = list(abc_path.rglob(f"*-{cot_mode}.abc")) if cot_mode in ["full", "melody"] else list(abc_path.rglob("*.abc"))
                    if abc_files:
                        chosen_abc = smart_choice(abc_files, abc_history, max_memory=15)
                        track_log_details.append(f"Mode: '{cot_mode}' | Score: {chosen_abc.name}")
                        raw_abc = chosen_abc.read_text(encoding="utf-8")
                    else:
                        main_logger.warning(f"No .abc files found for cot '{cot_mode}' in directory {abc_path}. Reverting to standard generation.")
                        active_request.pop("abc")
                elif abc_path.is_file():
                    track_log_details.append(f"Loading ABC sheet music from: {abc_path.name}")
                    raw_abc = abc_path.read_text(encoding="utf-8")
                else:
                    main_logger.warning(f"ABC path not found at {abc_path}. Reverting to standard generation.")
                    active_request.pop("abc")

                if raw_abc:
                    if inst_name is not None and midi_prog is not None:
                        active_request["abc"] = force_instrumental_abc(raw_abc, inst_name, midi_prog)
                        track_log_details.append("Live Edited ABC String applied (forced instrumental).")
                    else:
                        active_request["abc"] = raw_abc

            current_output_dir = args.output / f"track_{track_number:03d}"
            current_output_dir.mkdir(exist_ok=True)

            metadata_path = current_output_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(active_request, f, indent=2)

            combined_log = f"Generating Track {track_number} | " + " ".join(track_log_details)
            main_logger.info(combined_log)

            start_time = time.time()
            original_stdout, original_stderr = sys.stdout, sys.stderr
            sys.stdout = StreamFilter(sys.stdout)
            sys.stderr = StreamFilter(sys.stderr)

            try:
                song = pipe(**active_request)
            finally:
                if getattr(sys.stdout, 'buf', None) and "[YuE2]" not in sys.stdout.buf:
                    original_stdout.write(sys.stdout.buf)
                if getattr(sys.stderr, 'buf', None) and "[YuE2]" not in sys.stderr.buf:
                    original_stderr.write(sys.stderr.buf)
                sys.stdout, sys.stderr = original_stdout, original_stderr

            elapsed = time.time() - start_time
            song.save_artifacts(current_output_dir)
            audio_path = current_output_dir / "audio.flac"

            data, fs = sf.read(audio_path, dtype="float32")
            if data.ndim == 1:
                data = np.column_stack([data, data])

            play_queue.put((data, fs, current_output_dir))
            main_logger.info(f"✅ Track {track_number} finished generating in {elapsed:.2f} seconds. Queued for next crossfade.")

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
        if 'pipe' in locals() and pipe is not None:
            del pipe
        torch.cuda.empty_cache()

        main_logger.info("Waiting for background processes to exit...")
        watcher_thread.join(timeout=3)
        player_thread.join(timeout=3)
        main_logger.info("Infinite Radio has shut down safely.")


if __name__ == "__main__":
    main()
