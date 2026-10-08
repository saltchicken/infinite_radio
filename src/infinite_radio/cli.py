import argparse
import json
import random
import threading
import queue
import time
import struct
import shutil
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from yue2 import YuE2Pipeline
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Streaming Globals ---
STREAM_CLIENTS = []
STREAM_AUDIO_FORMAT = {"fs": None, "channels": None}

def create_wav_header(sample_rate, channels, bits_per_sample=16):
    """Generates an endless WAV header (0xFFFFFFFF size fields) for continuous streaming."""
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
                         b'RIFF', 0xFFFFFFFF, b'WAVE', b'fmt ', 16,
                         1, channels, sample_rate, byte_rate, block_align,
                         bits_per_sample, b'data', 0xFFFFFFFF)
    return header

class AudioStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream.wav':
            # Wait until the first track has decoded so we know the sample rate
            while STREAM_AUDIO_FORMAT['fs'] is None:
                time.sleep(0.5)
                
            self.send_response(200)
            self.send_header('Content-Type', 'audio/wav')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()

            # Create a queue for this specific client
            q = queue.Queue(maxsize=100)
            STREAM_CLIENTS.append(q)

            try:
                # Send the initial WAV header
                self.wfile.write(create_wav_header(STREAM_AUDIO_FORMAT['fs'], STREAM_AUDIO_FORMAT['channels']))
                
                # Continuously stream chunks as they are generated
                while True:
                    chunk = q.get()
                    if chunk is None:
                        break
                    self.wfile.write(chunk)
            except Exception:
                pass  # Client disconnected normally
            finally:
                if q in STREAM_CLIENTS:
                    STREAM_CLIENTS.remove(q)
        else:
            # Serve the mobile-friendly web player
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Infinite Radio</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
            </head>
            <body>
              <h1>📻 Infinite Radio</h1>
              <audio controls autoplay>
                  <source src="/stream.wav" type="audio/wav">
              </audio>
              <p>Live stream active</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs to keep the console clean

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
            data = json.loads(self.request_path.read_text(encoding="utf-8"))
            with self.config_lock:
                self.shared_state.update(data)
            
            # Trigger skip and auto-reset the JSON to false to prevent endless skipping
            if data.get("skip", False):
                self.skip_event.set()
                data["skip"] = False
                with open(self.request_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

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

def audio_player(play_queue, pause_event, skip_event, shutdown_event, crossfade_sec=6.0, disable_local=False):
    stream = None
    current_tail = None
    global STREAM_AUDIO_FORMAT

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
                    print("\n[Player] Paused. Set 'pause': false in song.json to resume.")
                shutdown_event.wait(0.1)

            if shutdown_event.is_set():
                return "SHUTDOWN"

            chunk = audio_data[i:i + chunk_frames]

            # 1. Broadcast to HTTP clients
            if STREAM_CLIENTS:
                # Convert float32 [-1.0, 1.0] to int16 PCM bytes
                pcm_16 = (chunk * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
                for q in list(STREAM_CLIENTS):
                    try:
                        q.put_nowait(pcm_16)
                    except queue.Full:
                        pass # Drop chunk if a client is lagging behind

            # 2. Local Audio Playback & Pacing
            if not disable_local and stream is not None:
                if not stream.active:
                    stream.start()
                stream.write(chunk) # stream.write automatically blocks, pacing the loop
            else:
                # If local hardware audio is disabled, manually pace the loop based on sample rate
                time.sleep(len(chunk) / current_fs)

        return "OK"

    try:
        while not shutdown_event.is_set():
            try:
                data, fs, track_name = play_queue.get(timeout=1.0)
                
                # Update global format for incoming stream clients
                STREAM_AUDIO_FORMAT['fs'] = fs
                STREAM_AUDIO_FORMAT['channels'] = data.shape[1]
                
            except queue.Empty:
                continue

            track_length = len(data)
            max_fade = track_length // 2
            fade_samples = min(int(crossfade_sec * fs), max_fade)
            chunk_size = int(fs * 0.25)

            if not disable_local:
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
                    current_tail = None # Clear tail so the next song starts clean
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

def start_http_server(port):
    # Bind to 0.0.0.0 so it's accessible over the local network
    server = ThreadingHTTPServer(('0.0.0.0', port), AudioStreamHandler)
    server.serve_forever()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, default=Path.cwd() / "song.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crossfade", type=float, default=6.0)
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    parser.add_argument("--stream-port", type=int, default=8000, help="Port to host the web stream")
    parser.add_argument("--disable-local", action="store_true", help="Only stream over network; disable local computer speakers")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    shared_state = json.loads(args.request.read_text(encoding="utf-8"))
    pause_event = threading.Event()
    skip_event = threading.Event()
    shutdown_event = threading.Event()
    config_lock = threading.Lock()
    play_queue = queue.Queue()

    watcher_thread = threading.Thread(target=config_watcher,
                                      args=(args.request, shared_state, pause_event, skip_event, shutdown_event, config_lock),
                                      daemon=True)
    player_thread = threading.Thread(target=audio_player,
                                     args=(play_queue, pause_event, skip_event, shutdown_event, args.crossfade, args.disable_local),
                                     daemon=True)
    http_thread = threading.Thread(target=start_http_server, 
                                   args=(args.stream_port,), 
                                   daemon=True)

    watcher_thread.start()
    player_thread.start()
    http_thread.start()

    print(f"\n==============================================")
    print(f"📡 Web Stream hosted on port {args.stream_port}")
    print(f"   From this PC: http://localhost:{args.stream_port}")
    print(f"   From Phone:   http://<YOUR_LOCAL_IP>:{args.stream_port}")
    print(f"==============================================\n")
    print("Loading YuE2 Model into VRAM... (This only happens once)")

    try:
        with YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda") as pipe:
            track_number = 1
            track_history = []  # Tracks paths for ring buffer logic

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

                with config_lock:
                    active_request = shared_state.copy()

                active_request["seed"] = random.randint(0, 2**32 - 1)
                active_request.pop("pause", None)
                active_request.pop("skip", None)  # Prevent kwargs error in YuE pipeline

                current_output_dir = args.output / f"track_{track_number:03d}"
                current_output_dir.mkdir(exist_ok=True)

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

                # Keep only the latest 10 tracks by deleting the oldest
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
