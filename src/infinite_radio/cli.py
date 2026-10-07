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

def config_watcher(request_path, shared_state, pause_event, shutdown_event):
    while not shutdown_event.is_set():
        try:
            data = json.loads(request_path.read_text(encoding="utf-8"))
            shared_state.update(data)
            if shared_state.get("pause", False):
                pause_event.set()
            else:
                pause_event.clear()
        except Exception:
            pass 
        # Use wait instead of sleep so it can break immediately on shutdown
        shutdown_event.wait(0.5)

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
                # Use a timeout so we can periodically check shutdown_event
                audio_path = play_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            data, fs = sf.read(audio_path, dtype="float32")
            
            if data.ndim == 1:
                data = np.column_stack([data, data])
                
            fade_samples = int(crossfade_sec * fs)
            chunk_size = int(fs * 0.25) 
            
            if stream is None:
                stream = sd.OutputStream(
                    samplerate=fs, channels=data.shape[1], dtype="float32"
                )
                stream.start()

            track_name = audio_path.parent.name.split('_')[-1]
            print(f"\n[Player] Transitioning into Track {track_name}...")

            t = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)[:, None]
            fade_out = np.cos(0.5 * np.pi * t)
            fade_in = np.sin(0.5 * np.pi * t)

            if current_tail is not None:
                head = data[:fade_samples]
                transition = (current_tail * fade_out) + (head * fade_in)
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
    play_queue = queue.Queue()

    watcher_thread = threading.Thread(target=config_watcher, args=(args.request, shared_state, pause_event, shutdown_event), daemon=True)
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
                
                active_request = shared_state.copy()
                active_request["seed"] = random.randint(0, 2**32 - 1)
                active_request.pop("pause", None)
                
                current_output_dir = args.output / f"track_{track_number:03d}"
                current_output_dir.mkdir(exist_ok=True)
                
                # If pipeline throws an error or is interrupted, the outer try/except handles it
                song = pipe(**active_request)
                song.save_artifacts(current_output_dir)
                audio_path = current_output_dir / "audio.flac"
                
                print(f"Track {track_number} rendered! Queued for next crossfade.")
                play_queue.put(audio_path)
                track_number += 1
                
    except KeyboardInterrupt:
        print("\n\n[Main] Shutdown signal received (Ctrl+C). Terminating gracefully...")
    except Exception as e:
        print(f"\n\n[Main] An unexpected error occurred: {e}")
    finally:
        # Trigger shutdown for background threads
        shutdown_event.set()
        
        # Unpause just in case the player thread is stuck in a paused loop
        pause_event.clear()
        
        print("[Main] Waiting for background processes to exit...")
        watcher_thread.join(timeout=3)
        player_thread.join(timeout=3)
        print("[Main] Infinite Radio has shut down safely.")

if __name__ == "__main__":
    main()
