import json
import queue
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

api_logger = logging.getLogger("RemoteAPI")

class ConfigUpdatePayload(BaseModel):
    settings: Dict[str, Any]


def _format_short_path(raw_path: str | None) -> str | None:
    """Formats a full path into 'parent_folder/filename' for clean mobile display."""
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.parent and p.parent.name:
        return f"{p.parent.name}/{p.name}"
    return p.name


def _persist_state(state_file: Path, shared_state: dict):
    """Atomically writes the current shared_state to disk so Web UI changes survive restarts."""
    try:
        tmp_path = state_file.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(shared_state, indent=2), encoding="utf-8")
        tmp_path.replace(state_file)
    except Exception as e:
        api_logger.warning(f"Failed to persist state to {state_file.name}: {e}")


def create_remote_api(
    shared_state: dict,
    play_queue: queue.Queue,
    pause_event,
    skip_event,
    config_lock,
    playback_info: dict,
    base_path: Path,
    state_file: Path,
):
    app = FastAPI(title="Infinite Radio Remote Control")

    @app.get("/", response_class=HTMLResponse)
    def serve_ui():
        # Serve the HTML cleanly from the static directory
        html_path = Path(__file__).parent / "static" / "index.html"
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="Web UI template not found")
        return FileResponse(html_path)

    @app.get("/api/options")
    def get_options():
        """Auto-discovers available LoRA files in loras/ to populate Web UI dropdowns."""
        lora_dir = base_path / "loras"
        loras = []
        if lora_dir.is_dir():
            loras = sorted(
                str(p.relative_to(base_path)) for p in lora_dir.rglob("*.safetensors")
            )
        return {"loras": loras}

    @app.get("/api/config")
    def get_config():
        with config_lock:
            return shared_state.copy()

    @app.post("/api/config")
    def update_config(payload: ConfigUpdatePayload):
        with config_lock:
            vol = shared_state.get("volume", 0.4)
            paused = shared_state.get("pause", False)
            shared_state.clear()
            shared_state.update(payload.settings)
            shared_state["volume"] = vol
            shared_state["pause"] = paused
            snapshot = shared_state.copy()

        _persist_state(state_file, snapshot)
        api_logger.info("⚙️ Generation settings updated via Web UI.")
        return {"status": "success", "config": snapshot}

    @app.post("/api/flush")
    def flush_queue():
        """Drains all buffered tracks from play_queue and cleans up their folders on disk."""
        flushed_count = 0
        while True:
            try:
                _, _, track_dir = play_queue.get_nowait()
                play_queue.task_done()
                flushed_count += 1
                if track_dir and track_dir.exists() and track_dir != playback_info.get("current_dir"):
                    try:
                        shutil.rmtree(track_dir)
                    except OSError:
                        pass
            except queue.Empty:
                break

        api_logger.info(f"🧹 Flushed {flushed_count} queued track(s) from buffer.")
        return {"status": "success", "flushed": flushed_count}

    @app.get("/api/state")
    def get_state():
        with config_lock:
            vol = float(shared_state.get("volume", 0.4))
            paused = bool(shared_state.get("pause", False))

        curr_dir = playback_info.get("current_dir")
        track_name = None
        style_name = None
        abc_name = None

        if curr_dir and curr_dir.exists():
            track_name = curr_dir.name
            meta_path = curr_dir / "metadata.json"
            if meta_path.exists():
                try:
                    meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
                    style_name = _format_short_path(meta_data.get("_source_style"))
                    abc_name = _format_short_path(meta_data.get("_source_abc"))
                except Exception:
                    pass

        return {
            "volume": vol,
            "paused": paused,
            "track": track_name,
            "style": style_name,
            "abc": abc_name,
            "queued": play_queue.qsize(),
        }

    @app.post("/api/volume")
    def update_volume(level: float):
        clamped = max(0.0, min(1.0, level))
        with config_lock:
            shared_state["volume"] = clamped
            snapshot = shared_state.copy()
        _persist_state(state_file, snapshot)
        return {"status": "success", "volume": clamped}

    @app.post("/api/skip")
    def skip_track():
        skip_event.set()
        return {"status": "success", "message": "Track skipped"}

    @app.post("/api/pause")
    def toggle_pause(state: bool):
        with config_lock:
            shared_state["pause"] = state
            snapshot = shared_state.copy()
        if state:
            pause_event.set()
        else:
            pause_event.clear()
        _persist_state(state_file, snapshot)
        return {"status": "success", "paused": state}

    @app.post("/api/save")
    def save_track():
        curr_dir = playback_info.get("current_dir")
        if not curr_dir or not curr_dir.exists():
            raise HTTPException(status_code=400, detail="No track playing")

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
            api_logger.info(f"💾 Saved currently playing track to: {save_dest}")
            return {"status": "success", "saved_to": str(save_dest)}
        except Exception as e:
            api_logger.error(f"❌ Failed to save track: {e}")
            raise HTTPException(status_code=500, detail="Save failed")

    @app.post("/api/vote/{feature}/{vote_type}")
    def vote(feature: str, vote_type: str):
        if feature not in ["style", "abc"] or vote_type not in ["up", "down"]:
            raise HTTPException(status_code=400, detail="Invalid feature")

        curr_dir = playback_info.get("current_dir")
        if not curr_dir or not curr_dir.exists():
            raise HTTPException(status_code=400, detail="No track playing")

        meta_path = curr_dir / "metadata.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="No metadata found")

        try:
            meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
            votes_file = base_path / "votes.jsonl"
            
            feature_val = meta_data.get("_source_style") if feature == "style" else meta_data.get("_source_abc")
            
            vote_record = {
                "timestamp": datetime.now().isoformat(),
                "vote": vote_type,
                "feature": feature,
                "value": feature_val,
                "track": curr_dir.name
            }

            if feature == "style" and meta_data.get("lyrics"):
                vote_record["lyrics"] = meta_data.get("lyrics")

            with open(votes_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(vote_record) + "\n")
            
            icon = "⭐" if vote_type == "up" else "👎"
            api_logger.info(f"{icon} {vote_type.title()}voted {feature}! Saved to {votes_file.name}")
            return {"status": "success", "record": vote_record}
            
        except Exception as e:
            api_logger.error(f"❌ Failed to process vote: {e}")
            raise HTTPException(status_code=500, detail="Vote failed")

    return app
