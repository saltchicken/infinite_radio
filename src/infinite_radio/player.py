import queue
import numpy as np
import sounddevice as sd
import logging

player_logger = logging.getLogger("Player")


def audio_player(
    play_queue,
    shared_state,
    config_lock,
    pause_event,
    skip_event,
    shutdown_event,
    playback_info,
    crossfade_sec=6.0,
):
    stream = None
    current_tail = None
    tail_fs = None

    def stream_write_interruptible(audio_data, chunk_frames, current_fs):
        for i in range(0, len(audio_data), chunk_frames):
            if shutdown_event.is_set():
                return "SHUTDOWN"

            if skip_event.is_set():
                skip_event.clear()
                player_logger.info("⏭️ Skipping current track...")

                # Apply a quick 0.5-second equal-power fade-out before skipping
                with config_lock:
                    volume = float(shared_state.get("volume", 0.4))
                fade_samples = int(0.5 * current_fs)
                remaining = audio_data[i : i + fade_samples] * volume

                if len(remaining) > 0 and stream is not None:
                    t = np.linspace(0.0, 1.0, len(remaining), dtype=np.float32)
                    fade_curve = np.cos(t * (np.pi / 2.0)).astype(np.float32)
                    if remaining.ndim == 2:
                        fade_curve = fade_curve[:, None]

                    if not stream.active:
                        stream.start()
                    stream.write(remaining * fade_curve)

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

            chunk = audio_data[i : i + chunk_frames] * volume

            if stream is not None:
                if not stream.active:
                    stream.start()
                stream.write(chunk)

        return "OK"

    try:
        while not shutdown_event.is_set():
            try:
                data, fs, metadata, zip_content, track_num = play_queue.get(timeout=1.0)
                playback_info["metadata"] = metadata
                playback_info["zip_content"] = zip_content
                playback_info["track_num"] = track_num
            except queue.Empty:
                continue

            track_length = len(data)
            max_fade = track_length // 2
            fade_samples = min(int(crossfade_sec * fs), max_fade)
            # 50ms chunks for responsive volume slider, pause, and skip controls
            chunk_size = max(1, int(fs * 0.05))

            if stream is None:
                stream = sd.OutputStream(samplerate=fs, channels=data.shape[1], dtype="float32")
                stream.start()
            elif stream.samplerate != fs or stream.channels != data.shape[1]:
                player_logger.info(f"Audio format changed to {fs}Hz ({data.shape[1]}ch). Reopening stream...")
                stream.stop()
                stream.close()
                stream = sd.OutputStream(samplerate=fs, channels=data.shape[1], dtype="float32")
                stream.start()
                # Discard incompatible tail from previous sample rate/channel count
                current_tail = None
                tail_fs = None

            player_logger.info(f"Transitioning into Track {track_num}...")

            if current_tail is not None and tail_fs == fs and fade_samples > 0:
                overlap_samples = min(len(current_tail), fade_samples)
                tail = current_tail[-overlap_samples:]
                head = data[:overlap_samples]

                # Equal-power crossfade curves (constant perceived loudness)
                t = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)[:, None]
                fade_out = np.cos(t * (np.pi / 2.0)).astype(np.float32)
                fade_in = np.sin(t * (np.pi / 2.0)).astype(np.float32)

                transition = (tail * fade_out) + (head * fade_in)

                status = stream_write_interruptible(transition, chunk_size, fs)
                if status == "SHUTDOWN":
                    break
                if status == "SKIP":
                    current_tail = None
                    tail_fs = None
                    play_queue.task_done()
                    continue

                body = data[overlap_samples:-fade_samples] if fade_samples > 0 else data[overlap_samples:]
            else:
                body = data[:-fade_samples] if fade_samples > 0 else data

            if fade_samples > 0:
                current_tail = data[-fade_samples:]
                tail_fs = fs
            else:
                current_tail = None
                tail_fs = None

            status = stream_write_interruptible(body, chunk_size, fs)
            if status == "SHUTDOWN":
                break
            if status == "SKIP":
                current_tail = None
                tail_fs = None
                play_queue.task_done()
                continue

            play_queue.task_done()

    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        player_logger.info("Audio stream closed and resources released.")
