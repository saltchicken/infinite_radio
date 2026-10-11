import json
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys_logger = logging.getLogger("System")

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
