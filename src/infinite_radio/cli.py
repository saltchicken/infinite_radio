import argparse
import json
import random
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from yue2 import YuE2Pipeline
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigHandler(FileSystemEventHandler):
    def __init__(self, request_path, shared_state, pause_event, config_lock):
        self.request_path = request_path.resolve()
        self.shared_state = shared_state
        self.pause_event = pause_event
        self.config_lock = config_lock

    def on_any_event(self, event):
        # Trigger on modifications or atomic replace events from text editors
        if not event.is_directory and Path(event.src_path).resolve() == self.request_path:
            # Small delay to ensure text editors finish writing before we read
            time.sleep(0.05)
            self.update_config()

    def update_config(self):
        try:
            data = json.loads(self.request_path.read_text(encoding="utf-8"))
            # Safely write to shared_state
            with self.config_lock:
                self.shared_state.update(data)
                
            if self.shared_state.get("pause", False):
                self.pause_event.set()
            else:
                self.pause_event.clear()
        except Exception:
            # Ignore errors if the file is caught mid-write or has invalid JSON
            pass

def config_watcher(request_path, shared_state, pause_event, shutdown_event, config_lock):
    event_handler = ConfigHandler(request_path, shared_state, pause_event, config_lock)
    observer = Observer()
    
    # Watch the directory containing song.json rather than the file itself 
    # to catch atomic saves (where editors delete and recreate the file)
    observer.schedule(event_handler, path=str(request_path.parent.resolve()), recursive=False)
    observer.start()
    
    try:
        # Keep the thread alive until shutdown is signaled
        while not shutdown_event.is_set():
            shutdown_event.wait(1.0)
    finally:
        observer.stop()
        observer.join()

def audio_player(play_queue, pause_event, shutdown_event, crossfade_sec=6.0):
    stream = None
    current_tail = None
    
    def stream_write_interruptible(audio_data, chunk_frames):
        for i in range(0, len(audio_data), chunk_frames):
            if shutdown_event.is_set():
                return False
                
            while pause_event.is_set() and not shutdown_event.is_set():
                if stream.active:
                    stream.stop()
                    print("\n[Player] Paused. Set 'pause': false in song.json to resume.")
                shutdown_event.wait(0.1)
                
            if shutdown_event.is_set():
                return False
                
            if not stream.active:
                stream.start()
            stream.write(audio_data[i:i + chunk_frames])
        return True

    try:
        while not shutdown_event.is_set():
            try:
                # Get the pre-decoded numpy array to prevent buffer underruns
                data, fs, track_name = play_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            # Safely cap fade_samples so short tracks don't cause array indexing errors
            track_length = len(data)
            max_fade = track_length // 2 
            fade_samples = min(int(crossfade_sec * fs), max_fade)
            
            chunk_size = int(fs * 0.25) 
            
            if stream is None:
                stream = sd.OutputStream(
                    samplerate=fs, channels=data.shape[1], dtype="float32"
                )
                stream.start()
            elif stream.samplerate != fs:
                # Handle sample rate changes safely if the AI ever outputs one
                stream.stop()
                stream.close()
                stream = sd.OutputStream(
                    samplerate=fs, channels=data.shape[1], dtype="float32"
                )
                stream.start()

            print(f"\n[Player] Transitioning into Track {track_name}...")

            t = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)[:, None]
            
            # Linear crossfade prevents digital clipping
            fade_out = 1.0 - t
            fade_in = t

            if current_tail is not None:
                # Handle cases where the previous tail was longer than the current capped fade_samples
                tail = current_tail[-fade_samples:] if len(current_tail) > fade_samples else current_tail
                
                head = data[:fade_samples]
                transition = (tail * fade_out) + (head * fade_in)
                if not stream_write_interruptible(transition, chunk_size):
                    break
                body = data[fade_samples:-fade_samples]
            else:
                body = data[:-fade_samples]

            current_tail = data[-fade_samples:]
            if not stream_write_interruptible(body, chunk_size):
                break
                
            play_queue.task_done()
            
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
            print("\n[Player] Audio stream closed and resources released.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, default=Path.cwd() / "song.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crossfade", type=float, default=6.0)
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    shared_state = json.loads(args.request.read_text(encoding="utf-8"))
    pause_event = threading.Event()
    shutdown_event = threading.Event()
    config_lock = threading.Lock()
    play_queue = queue.Queue()

    watcher_thread = threading.Thread(target=config_watcher, args=(args.request, shared_state, pause_event, shutdown_event, config_lock), daemon=True)
    player_thread = threading.Thread(target=audio_player, args=(play_queue, pause_event, shutdown_event, args.crossfade), daemon=True)
    
    watcher_thread.start()
    player_thread.start()

    print("Loading YuE2 Model into VRAM... (This only happens once)")

    try:
        with YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda") as pipe:
            track_number = 1
            while not shutdown_event.is_set():
                while pause_event.is_set() and not shutdown_event.is_set():
                    shutdown_event.wait(1.0)

                while play_queue.qsize() >= 1 and not shutdown_event.is_set():
                    shutdown_event.wait(1.0)
                    
                if shutdown_event.is_set():
                    break
                    
                if pause_event.is_set():
                    continue 
                    
                print(f"\n--- Generating Track {track_number} ---")
                
                # Safely copy the state
                with config_lock:
                    active_request = shared_state.copy()
                    
                active_request["seed"] = random.randint(0, 2**32 - 1)
                active_request.pop("pause", None)
                
                current_output_dir = args.output / f"track_{track_number:03d}"
                current_output_dir.mkdir(exist_ok=True)
                
                song = pipe(**active_request)
                # song = pipe(**active_request, best_of=3)
                song.save_artifacts(current_output_dir)
                audio_path = current_output_dir / "audio.flac"
                
                print(f"Track {track_number} rendered! Decoding audio into memory...")
                
                # Pre-read the file into RAM in the main thread to prevent underruns
                data, fs = sf.read(audio_path, dtype="float32")
                if data.ndim == 1:
                    data = np.column_stack([data, data])
                    
                track_name = audio_path.parent.name.split('_')[-1]
                play_queue.put((data, fs, track_name))
                print(f"Track {track_number} queued for next crossfade.")
                
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
