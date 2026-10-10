import torch
import argparse
import json
import random
import threading
import queue
import time
import shutil
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from yue2 import YuE2Pipeline
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- LoRA Loading Helpers ---
def _mods(prefix): 
    return [(f"{prefix}self_attn", n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")] + \
           [(f"{prefix}mlp", n) for n in ("gate_proj", "up_proj", "down_proj")]

def load_ckpt(path, map_location="cuda"):
    from safetensors.torch import load_file
    from safetensors import safe_open
    
    t = load_file(path, device=str(map_location))
    with safe_open(path, "pt") as f: 
        meta = f.metadata() or {}
        
    prefix = ""
    layers = sorted({int(k.split(".")[1]) for k in t if k.startswith("layers.")})
    lora = []
    
    for L in layers:
        for blk, proj in _mods(prefix): 
            lora += [t[f"layers.{L}.{blk}.{proj}.lora_A"], t[f"layers.{L}.{blk}.{proj}.lora_B"]]
            
    return {"lora": lora, "rank": int(meta.get("rank", lora[0].shape[0]))}

@torch.no_grad()
def merge_lora(bb, attn_name, mlp_name, tensors, scale=1.0, dev="cuda"):
    it = iter(tensors)
    n_merged = 0
    for layer in bb.layers:
        for mod, names in ((getattr(layer, attn_name), ("q_proj", "k_proj", "v_proj", "o_proj")),
                           (getattr(layer, mlp_name), ("gate_proj", "up_proj", "down_proj"))):
            for n in names:
                A = next(it).to(dev).float()
                B = next(it).to(dev).float()
                lin = getattr(mod, n)
                # Fold LoRA deltas (W += scale * B @ A) into the base weights
                lin.weight.add_((scale * (B @ A)).to(lin.weight.dtype))
                n_merged += 1
    return n_merged
# ----------------------------

def load_merged_config(request_path):
    """Reads controller.json and merges it with the active style file."""
    controller_data = json.loads(request_path.read_text(encoding="utf-8"))
    
    playback_defaults = {
        "volume": 0.3,
        "pause": False,
        "skip": False,
    }
    for k, v in playback_defaults.items():
        controller_data.setdefault(k, v)

    style_data = {}
    if "active_style" in controller_data:
        style_path = request_path.parent / controller_data["active_style"]
        if style_path.exists():
            style_data = json.loads(style_path.read_text(encoding="utf-8"))
        else:
            print(f"\n[Warning] Style file not found: {style_path}")

    merged = style_data.copy()
    merged.update(controller_data)
    return merged, controller_data

class ConfigHandler(FileSystemEventHandler):
    def __init__(self, request_path, shared_state, pause_event, skip_event, config_lock):
        self.request_path = request_path.resolve()
        self.shared_state = shared_state
        self.pause_event = pause_event
        self.skip_event = skip_event
        self.config_lock = config_lock

    def on_any_event(self, event):
        if not event.is_directory and Path(event.src_path).resolve() == self.request_path:
            time.sleep(0.05)
            self.update_config()

    def update_config(self):
        try:
            merged_data, controller_data = load_merged_config(self.request_path)
            
            with self.config_lock:
                self.shared_state.clear()
                self.shared_state.update(merged_data)
            
            if controller_data.get("skip", False):
                self.skip_event.set()
                controller_data["skip"] = False
                with open(self.request_path, "w", encoding="utf-8") as f:
                    json.dump(controller_data, f, indent=2)

            if self.shared_state.get("pause", False):
                self.pause_event.set()
            else:
                self.pause_event.clear()
        except Exception:
            pass

def config_watcher(request_path, shared_state, pause_event, skip_event, shutdown_event, config_lock):
    event_handler = ConfigHandler(request_path, shared_state, pause_event, skip_event, config_lock)
    observer = Observer()
    observer.schedule(event_handler, path=str(request_path.parent.resolve()), recursive=False)
    observer.start()
    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(1.0)
    finally:
        observer.stop()
        observer.join()

def audio_player(play_queue, shared_state, config_lock, pause_event, skip_event, shutdown_event, crossfade_sec=6.0):
    stream = None
    current_tail = None

    def stream_write_interruptible(audio_data, chunk_frames, current_fs):
        for i in range(0, len(audio_data), chunk_frames):
            if shutdown_event.is_set():
                return "SHUTDOWN"
            
            if skip_event.is_set():
                skip_event.clear()
                print("\n[Player] ⏭️ Skipping current track...")
                return "SKIP"

            while pause_event.is_set() and not shutdown_event.is_set():
                if stream and stream.active:
                    stream.stop()
                    print("\n[Player] Paused. Set 'pause': false in controller.json to resume.")
                shutdown_event.wait(0.1)

            if shutdown_event.is_set():
                return "SHUTDOWN"

            with config_lock:
                volume = float(shared_state.get("volume", 1.0))

            chunk = audio_data[i:i + chunk_frames] * volume

            if stream is not None:
                if not stream.active:
                    stream.start()
                stream.write(chunk) 

        return "OK"

    try:
        while not shutdown_event.is_set():
            try:
                data, fs, track_name = play_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            track_length = len(data)
            max_fade = track_length // 2
            fade_samples = min(int(crossfade_sec * fs), max_fade)
            chunk_size = int(fs * 0.25)

            if stream is None:
                stream = sd.OutputStream(samplerate=fs, channels=data.shape[1], dtype="float32")
                stream.start()
            elif stream.samplerate != fs:
                stream.stop()
                stream.close()
                stream = sd.OutputStream(samplerate=fs, channels=data.shape[1], dtype="float32")
                stream.start()

            print(f"\n[Player] Transitioning into Track {track_name}...")

            t = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)[:, None]
            fade_out = 1.0 - t
            fade_in = t

            if current_tail is not None:
                tail = current_tail[-fade_samples:] if len(current_tail) > fade_samples else current_tail
                head = data[:fade_samples]
                transition = (tail * fade_out) + (head * fade_in)
                
                status = stream_write_interruptible(transition, chunk_size, fs)
                if status == "SHUTDOWN": break
                if status == "SKIP":
                    current_tail = None 
                    play_queue.task_done()
                    continue

                body = data[fade_samples:-fade_samples]
            else:
                body = data[:-fade_samples]

            current_tail = data[-fade_samples:]
            
            status = stream_write_interruptible(body, chunk_size, fs)
            if status == "SHUTDOWN": break
            if status == "SKIP":
                current_tail = None
                play_queue.task_done()
                continue

            play_queue.task_done()

    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        print("\n[Player] Audio stream closed and resources released.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, default=Path.cwd() / "controller.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crossfade", type=float, default=6.0)
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    parser.add_argument("--ar-lora", type=Path, help="Path to the AR LoRA safetensors file")
    parser.add_argument("--ar-scale", type=float, default=1.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    shared_state = {}
    try:
        initial_data, _ = load_merged_config(args.request)
        shared_state.update(initial_data)
    except Exception as e:
        print(f"Error loading initial config: {e}. Ensure controller.json exists.")
        return

    pause_event = threading.Event()
    skip_event = threading.Event()
    shutdown_event = threading.Event()
    config_lock = threading.Lock()
    play_queue = queue.Queue()

    watcher_thread = threading.Thread(target=config_watcher,
                                      args=(args.request, shared_state, pause_event, skip_event, shutdown_event, config_lock),
                                      daemon=True)
    player_thread = threading.Thread(target=audio_player,
                                     args=(play_queue, shared_state, config_lock, pause_event, skip_event, shutdown_event, args.crossfade),
                                     daemon=True)

    watcher_thread.start()
    player_thread.start()

    print(f"\n==============================================")
    print(f"📻 Infinite Radio Local Playback Started")
    print(f"==============================================\n")
    print("Loading YuE2 Model into VRAM... (This only happens once)")

    try:
        with YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda") as pipe:
            
            # --- Inject LoRA Weights ---
            if args.ar_lora and args.ar_lora.exists():
                print(f"Loading AR LoRA from {args.ar_lora.name}...")
                lora_data = load_ckpt(str(args.ar_lora), "cuda")
                # Get the underlying base model inside the pipeline
                base_model = pipe._load_model().model
                
                n_merged = merge_lora(base_model, "self_attn", "mlp", lora_data["lora"], scale=args.ar_scale, dev="cuda")
                print(f"Merged {n_merged} AR LoRA linear layers (scale: {args.ar_scale}).")
            # ---------------------------
            
            track_number = 1
            track_history = []  

            while not shutdown_event.is_set():
                while pause_event.is_set() and not shutdown_event.is_set():
                    shutdown_event.wait(1.0)

                while play_queue.qsize() >= 3 and not shutdown_event.is_set():
                    shutdown_event.wait(1.0)

                if shutdown_event.is_set():
                    break

                if pause_event.is_set():
                    continue

                print(f"\n--- Generating Track {track_number} ---")

                with config_lock:
                    active_request = shared_state.copy()

                active_request["seed"] = random.randint(0, 2**32 - 1)
                active_request.pop("pause", None)
                active_request.pop("skip", None)
                active_request.pop("volume", None)
                active_request.pop("active_style", None) 

                if isinstance(active_request.get("lyrics"), list):
                    chosen = random.choice(active_request["lyrics"])
                    if isinstance(chosen, list):
                        active_request["lyrics"] = "\n".join(chosen)
                    else:
                        active_request["lyrics"] = chosen

                if "choices" in active_request:
                    choices = active_request.pop("choices") 
                    if isinstance(active_request.get("style"), str):
                        for key, options in choices.items():
                            chosen_val = str(random.choice(options))
                            active_request["style"] = active_request["style"].replace(f"{{{key}}}", chosen_val)

                current_output_dir = args.output / f"track_{track_number:03d}"
                current_output_dir.mkdir(exist_ok=True)

                metadata_path = current_output_dir / "metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(active_request, f, indent=2)

                song = pipe(**active_request)
                song.save_artifacts(current_output_dir)
                audio_path = current_output_dir / "audio.flac"

                print(f"Track {track_number} rendered! Decoding audio into memory...")

                data, fs = sf.read(audio_path, dtype="float32")
                if data.ndim == 1:
                    data = np.column_stack([data, data])

                track_name = audio_path.parent.name.split('_')[-1]
                play_queue.put((data, fs, track_name))
                print(f"Track {track_number} queued for next crossfade.")

                track_history.append(current_output_dir)
                if len(track_history) > 10:
                    old_track_dir = track_history.pop(0)
                    try:
                        shutil.rmtree(old_track_dir)
                        print(f"Ring buffer: Deleted old track {old_track_dir.name} from disk.")
                    except OSError as e:
                        print(f"Ring buffer: Failed to delete {old_track_dir.name}: {e}")

                track_number += 1

    except KeyboardInterrupt:
        print("\n\n[Main] Shutdown signal received (Ctrl+C). Terminating gracefully...")
    except Exception as e:
        print(f"\n\n[Main] An unexpected error occurred: {e}")
    finally:
        shutdown_event.set()
        pause_event.clear()

        print("[Main] Waiting for background processes to exit...")
        watcher_thread.join(timeout=3)
        player_thread.join(timeout=3)
        print("[Main] Infinite Radio has shut down safely.")

if __name__ == "__main__":
    main()
