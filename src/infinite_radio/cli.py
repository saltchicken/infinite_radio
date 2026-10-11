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
from datetime import datetime
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
        
    prefix = "nar_" if any(".nar_self_attn." in k for k in t) else ""
    layers = sorted({int(k.split(".")[1]) for k in t if k.startswith("layers.")})
    lora = []
    
    for L in layers:
        for blk, proj in _mods(prefix): 
            lora += [t[f"layers.{L}.{blk}.{proj}.lora_A"], t[f"layers.{L}.{blk}.{proj}.lora_B"]]
            
    out = {"lora": lora, "rank": int(meta.get("rank", lora[0].shape[0]))}
    if prefix: 
        out["io"] = {m: {k.split(".", 1)[1]: v for k, v in t.items() if k.startswith(m + ".")} for m in ("vae2llm", "llm2vae")}
    return out

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
                lin.weight.add_((scale * (B @ A)).to(lin.weight.dtype))
                n_merged += 1
    return n_merged
# ----------------------------

def load_merged_config(request_path):
    controller_data = json.loads(request_path.read_text(encoding="utf-8"))
    
    playback_defaults = {
        "volume": 0.3,
        "pause": False,
        "skip": False,
        "save": False,
        "ar_scale": 1.0,
        "nar_scale": 1.0 
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
    def __init__(self, request_path, shared_state, pause_event, skip_event, config_lock, playback_info):
        self.request_path = request_path.resolve()
        self.shared_state = shared_state
        self.pause_event = pause_event
        self.skip_event = skip_event
        self.config_lock = config_lock
        self.playback_info = playback_info

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
            
            config_changed = False
            
            if controller_data.get("skip", False):
                self.skip_event.set()
                controller_data["skip"] = False
                config_changed = True
                
            if controller_data.get("save", False):
                controller_data["save"] = False
                config_changed = True
                
                curr_dir = self.playback_info.get("current_dir")
                if curr_dir and curr_dir.exists():
                    save_base = curr_dir.parent / "saved"
                    save_base.mkdir(exist_ok=True)
                    
                    # Generate timestamp and append it to the original folder name
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_dest = save_base / f"{curr_dir.name}_{timestamp}"
                    
                    counter = 1
                    while save_dest.exists():
                        save_dest = save_base / f"{curr_dir.name}_{timestamp}_{counter}"
                        counter += 1
                        
                    try:
                        shutil.copytree(curr_dir, save_dest)
                        print(f"\n[System] 💾 Saved currently playing track to: {save_dest}")
                    except Exception as e:
                        print(f"\n[System] ❌ Failed to save track: {e}")
                else:
                    print("\n[System] ⚠️ No track currently playing to save.")

            if config_changed:
                with open(self.request_path, "w", encoding="utf-8") as f:
                    json.dump(controller_data, f, indent=2)

            if self.shared_state.get("pause", False):
                self.pause_event.set()
            else:
                self.pause_event.clear()
        except Exception:
            pass

def config_watcher(request_path, shared_state, pause_event, skip_event, shutdown_event, config_lock, playback_info):
    event_handler = ConfigHandler(request_path, shared_state, pause_event, skip_event, config_lock, playback_info)
    observer = Observer()
    observer.schedule(event_handler, path=str(request_path.parent.resolve()), recursive=False)
    observer.start()
    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(1.0)
    finally:
        observer.stop()
        observer.join()

def audio_player(play_queue, shared_state, config_lock, pause_event, skip_event, shutdown_event, playback_info, crossfade_sec=6.0):
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
                data, fs, current_track_dir = play_queue.get(timeout=1.0)
                playback_info["current_dir"] = current_track_dir
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

            print(f"\n[Player] Transitioning into {current_track_dir.name}...")

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
    parser.add_argument("--nar-lora", type=Path, help="Path to the NAR LoRA safetensors file")
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
    
    playback_info = {"current_dir": None}

    watcher_thread = threading.Thread(target=config_watcher,
                                      args=(args.request, shared_state, pause_event, skip_event, shutdown_event, config_lock, playback_info),
                                      daemon=True)
    player_thread = threading.Thread(target=audio_player,
                                     args=(play_queue, shared_state, config_lock, pause_event, skip_event, shutdown_event, playback_info, args.crossfade),
                                     daemon=True)

    watcher_thread.start()
    player_thread.start()

    print(f"\n==============================================")
    print(f"📻 Infinite Radio Local Playback Started")
    print(f"==============================================\n")

    pipe = None
    current_ar_scale = None
    current_nar_scale = None
    track_number = 1
    track_history = []  

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
                active_request = shared_state.copy()

            requested_ar_scale = float(active_request.get("ar_scale", 1.0))
            requested_nar_scale = float(active_request.get("nar_scale", 1.0))

            if pipe is None or current_ar_scale != requested_ar_scale or current_nar_scale != requested_nar_scale:
                if pipe is not None:
                    print(f"\n🔄 [System] LoRA scale change detected (AR: {current_ar_scale} -> {requested_ar_scale}, NAR: {current_nar_scale} -> {requested_nar_scale}). Reloading model...")
                    del pipe
                    torch.cuda.empty_cache()
                else:
                    print("Loading YuE2 Model into VRAM...")

                pipe = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda")
                top_model = pipe._load_model()
                bb = top_model.model
                
                if args.ar_lora and args.ar_lora.exists():
                    print(f"Loading AR LoRA from {args.ar_lora.name}...")
                    ar_data = load_ckpt(str(args.ar_lora), "cuda")
                    n_merged = merge_lora(bb, "self_attn", "mlp", ar_data["lora"], scale=requested_ar_scale, dev="cuda")
                    print(f"Merged {n_merged} AR LoRA linear layers (scale: {requested_ar_scale}).")
                
                if args.nar_lora and args.nar_lora.exists():
                    print(f"Loading NAR LoRA from {args.nar_lora.name}...")
                    nar_data = load_ckpt(str(args.nar_lora), "cuda")
                    n_merged = merge_lora(bb, "nar_self_attn", "nar_mlp", nar_data["lora"], scale=requested_nar_scale, dev="cuda")
                    top_model.vae2llm.load_state_dict({k: v.to(torch.bfloat16) for k, v in nar_data["io"]["vae2llm"].items()})
                    top_model.llm2vae.load_state_dict({k: v.to(torch.bfloat16) for k, v in nar_data["io"]["llm2vae"].items()})
                    print(f"Merged {n_merged} NAR LoRA linear layers (scale: {requested_nar_scale}) & injected custom VAE/LLM crossovers.")

                current_ar_scale = requested_ar_scale
                current_nar_scale = requested_nar_scale

            print(f"\n--- Generating Track {track_number} ---")

            active_request["seed"] = random.randint(0, 2**32 - 1)
            active_request.pop("pause", None)
            active_request.pop("skip", None)
            active_request.pop("volume", None)
            active_request.pop("save", None)
            active_request.pop("active_style", None) 
            active_request.pop("ar_scale", None)
            active_request.pop("nar_scale", None)

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

            if "abc" in active_request and active_request["abc"]:
                abc_file = Path(active_request["abc"])
                if abc_file.is_file():
                    print(f"🎵 [Cover Mode] Loading ABC sheet music from: {abc_file}")
                    active_request["abc"] = abc_file.read_text(encoding="utf-8")
                else:
                    print(f"⚠️ [Warning] ABC file not found at {abc_file}. Reverting to standard generation.")
                    active_request.pop("abc")

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

            play_queue.put((data, fs, current_output_dir))
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
        if 'pipe' in locals() and pipe is not None:
            del pipe
        torch.cuda.empty_cache()

        print("[Main] Waiting for background processes to exit...")
        watcher_thread.join(timeout=3)
        player_thread.join(timeout=3)
        print("[Main] Infinite Radio has shut down safely.")

if __name__ == "__main__":
    main()
