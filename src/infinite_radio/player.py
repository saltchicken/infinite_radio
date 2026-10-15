import queue
import numpy as np
import sounddevice as sd
import logging

player_logger = logging.getLogger("Player")

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
                
                # Apply a quick 0.5-second linear fade-out before skipping
                with config_lock:
                    volume = float(shared_state.get("volume", 0.4))
                fade_samples = int(0.5 * current_fs)
                remaining = audio_data[i:i + fade_samples] * volume
                
                if len(remaining) > 0 and stream is not None:
                    t = np.linspace(1.0, 0.0, len(remaining), dtype=np.float32)
                    if remaining.ndim == 2:
                        t = t[:, None]
                    
                    if not stream.active:
                        stream.start()
                    stream.write(remaining * t)
                    
                return "SKIP"

            while pause_event.is_set() and not shutdown_event.is_set():
                if stream and stream.active:
                    stream.stop()
                    player_logger.info("Paused. Press Play in the Remote Web UI to resume.")
                shutdown_event.wait(0.1)

            if shutdown_event.is_set():
                return "SHUTDOWN"

            with config_lock:
                volume = float(shared_state.get("volume", 0.4))

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
                if status == "SHUTDOWN":
                    break
                if status == "SKIP":
                    current_tail = None
                    play_queue.task_done()
                    continue

                body = data[fade_samples:-fade_samples]
            else:
                body = data[:-fade_samples]

            current_tail = data[-fade_samples:]

            status = stream_write_interruptible(body, chunk_size, fs)
            if status == "SHUTDOWN":
                break
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
