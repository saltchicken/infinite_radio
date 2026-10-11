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
import sounddevice as sd
import soundfile as sf
import logging
import re
from pathlib import Path
from datetime import datetime
from yue2 import YuE2Pipeline
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
sys_logger = logging.getLogger("System")
player_logger = logging.getLogger("Player")
main_logger = logging.getLogger("Main")
# -----------------------------

# --- ABC Automation Helper ---
def force_instrumental_abc(abc_text, instrument_name="Lead Synth", midi_program=81):
    """Strips lyrics and forces a specific instrument/MIDI program in the ABC text."""
    lines = abc_text.splitlines()
    out_lines = []
    
    for line in lines:
        stripped = line.lstrip()
        # 1. Drop all lyric lines
        if stripped.startswith('w:'):
            continue
            
        # 2. Modify Voice lines to force instrument names
        if stripped.startswith('V:'):
            # Overwrite full name
            if 'name=' in line:
                line = re.sub(r'name="[^"]*"', f'name="{instrument_name}"', line)
            else:
                line = line.rstrip() + f' name="{instrument_name}"'
                
            # Overwrite short name (snm) to remove lingering "Vocal" triggers
            if 'snm=' in line:
                line = re.sub(r'snm="[^"]*"', f'snm="{instrument_name}"', line)
                
            out_lines.append(line)
            # Inject standard MIDI directive immediately after the voice declaration
            out_lines.append(f'%%MIDI program {midi_program}')
            continue
            
        out_lines.append(line)
        
    return '\n'.join(out_lines)
# -----------------------------

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

# --- Smart Shuffle Helper ---
def smart_choice(options, history, max_memory):
    if not options:
        return None
    
    available = [opt for opt in options if opt not in history]
    
    if not available:
        if len(options) > 1 and history:
            available = [opt for opt in options if opt != history[-1]]
        else:
            available = options
            
    chosen = random.choice(available)
    history.append(chosen)
    
    while len(history) > max_memory:
        history.pop(0)
        
    return chosen
# ----------------------------

def load_controller_config(request_path):
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

    return controller_data

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
            controller_data = load_controller_config(self.request_path)
            
            with self.config_lock:
                self.shared_state.clear()
                self.shared_state.update(controller_data)
            
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
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_dest = save_base / f"{curr_dir.name}_{timestamp}"
                    
                    counter = 1
                    while save_dest.exists():
                        save_dest = save_base / f"{curr_dir.name}_{timestamp}_{counter}"
                        counter += 1
                        
                    try:
                        shutil.copytree(curr_dir, save_dest)
                        sys_logger.info(f"💾 Saved currently playing track to: {save_dest}")
                    except Exception as e:
                        sys_logger.error(f"❌ Failed to save track: {e}")
                else:
                    sys_logger.warning("⚠️ No track currently playing to save.")

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
                player_logger.info("⏭️ Skipping current track...")
                return "SKIP"

            while pause_event.is_set() and not shutdown_event.is_set():
                if stream and stream.active:
                    stream.stop()
                    player_logger.info("Paused. Set 'pause': false in controller.json to resume.")
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

            player_logger.info(f"Transitioning into {current_track_dir.name}...")

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
        player_logger.info("Audio stream closed and resources released.")

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

    watcher_thread = threading.Thread(target=config_watcher,
                                      args=(args.request, shared_state, pause_event, skip_event, shutdown_event, config_lock, playback_info),
                                      daemon=True)
    player_thread = threading.Thread(target=audio_player,
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

            # Array to collect log statements for this track
            track_log_details = []

            style_data = {}
            if "active_style" in controller_req and controller_req["active_style"]:
                style_path = args.request.parent / controller_req["active_style"]
                if style_path.is_dir():
                    style_files = list(style_path.glob("*.json"))
                    if style_files:
                        chosen_style = smart_choice(style_files, style_history, max_memory=4)
                        track_log_details.append(f"🎲 [Style Mode] Selected style: {chosen_style.name}")
                        style_data = json.loads(chosen_style.read_text(encoding="utf-8"))
                    else:
                        main_logger.warning(f"No .json files found in style directory {style_path}.")
                elif style_path.is_file():
                    track_log_details.append(f"🎲 [Style Mode] Selected style: {style_path.name}")
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

            if (pipe is None or 
                current_ar_scale != requested_ar_scale or 
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

                current_ar_scale = requested_ar_scale
                current_nar_scale = requested_nar_scale
                current_ar_lora = requested_ar_lora
                current_nar_lora = requested_nar_lora

            active_request["seed"] = random.randint(0, 2**32 - 1)
            cot_mode = active_request.get("cot", "full")
            
            # --- Safely pop custom variables BEFORE pipeline execution ---
            inst_name = active_request.pop("abc_instrument", None)
            midi_prog = active_request.pop("abc_midi_program", None)

            for key in ["pause", "skip", "volume", "save", "active_style", "ar_scale", "nar_scale", "ar_lora", "nar_lora"]:
                active_request.pop(key, None)

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

            # --- DYNAMIC ABC MODIFICATION BLOCK ---
            if "abc" in active_request and active_request["abc"]:
                abc_path = Path(active_request["abc"])
                raw_abc = None
                
                if abc_path.is_dir():
                    if cot_mode in ["full", "melody"]:
                        abc_files = list(abc_path.rglob(f"*-{cot_mode}.abc"))
                    else:
                        abc_files = list(abc_path.rglob("*.abc"))
                        
                    if abc_files:
                        chosen_abc = smart_choice(abc_files, abc_history, max_memory=15)
                        track_log_details.append(f"🎵 [Cover Mode] Selected '{cot_mode}' ABC sheet music: {chosen_abc.name}")
                        raw_abc = chosen_abc.read_text(encoding="utf-8")
                    else:
                        main_logger.warning(f"No .abc files found for cot '{cot_mode}' in directory {abc_path}. Reverting to standard generation.")
                        active_request.pop("abc")
                        
                elif abc_path.is_file():
                    track_log_details.append(f"🎵 [Cover Mode] Loading ABC sheet music from: {abc_path.name}")
                    raw_abc = abc_path.read_text(encoding="utf-8")
                    
                else:
                    main_logger.warning(f"ABC path not found at {abc_path}. Reverting to standard generation.")
                    active_request.pop("abc")
                    
                if raw_abc:
                    if inst_name is not None and midi_prog is not None:
                        edited_abc = force_instrumental_abc(raw_abc, inst_name, midi_prog)
                        active_request["abc"] = edited_abc
                        track_log_details.append("📝 Live Edited ABC String applied (forced instrumental).")
                    else:
                        active_request["abc"] = raw_abc
                        track_log_details.append("📄 Passing raw ABC string untouched (no instrumental override parameters set).")
            # ----------------------------------------

            current_output_dir = args.output / f"track_{track_number:03d}"
            current_output_dir.mkdir(exist_ok=True)

            metadata_path = current_output_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(active_request, f, indent=2)

            # Append the final generating line and print out the grouped log block
            track_log_details.append(f"⚙️ Track {track_number} is generating...")
            combined_log = f"--- Generating Track {track_number} ---\n" + "\n".join(track_log_details)
            main_logger.info(combined_log)

            start_time = time.time()

            # Filter to intercept and drop logs containing [YuE2]
            class StreamFilter:
                def __init__(self, stream, target="[YuE2]"):
                    self.stream = stream
                    self.target = target
                    self.buf = ""
                def write(self, data):
                    self.buf += data
                    while "\n" in self.buf:
                        line, self.buf = self.buf.split("\n", 1)
                        # Check if the target is anywhere in the line (bypasses color codes)
                        if self.target not in line:
                            self.stream.write(line + "\n")
                def flush(self):
                    self.stream.flush()
                def __getattr__(self, attr):
                    # Passthrough for things like .isatty() needed by tqdm/progress bars
                    return getattr(self.stream, attr)

            original_stdout, original_stderr = sys.stdout, sys.stderr
            sys.stdout = StreamFilter(sys.stdout)
            sys.stderr = StreamFilter(sys.stderr)

            try:
                song = pipe(**active_request)
            finally:
                # Flush any remaining non-YuE2 text in the buffer, then restore streams
                if getattr(sys.stdout, 'buf', None) and "[YuE2]" not in sys.stdout.buf:
                    original_stdout.write(sys.stdout.buf)
                if getattr(sys.stderr, 'buf', None) and "[YuE2]" not in sys.stderr.buf:
                    original_stderr.write(sys.stderr.buf)
                sys.stdout, sys.stderr = original_stdout, original_stderr

            elapsed = time.time() - start_time
            main_logger.info(f"✅ Track {track_number} finished generating in {elapsed:.2f} seconds.")

            song.save_artifacts(current_output_dir)
            audio_path = current_output_dir / "audio.flac"

            data, fs = sf.read(audio_path, dtype="float32")
            if data.ndim == 1:
                data = np.column_stack([data, data])

            play_queue.put((data, fs, current_output_dir))
            main_logger.info(f"Track {track_number} queued for next crossfade.")

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
