import io
import json
import queue
import zipfile
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


def _discover_styles(base_path: Path) -> Dict[str, list[str]]:
    """Scans styles/ for both directories containing JSONs and individual .json files."""
    styles_dir = base_path / "styles"
    folders = set()
    files = []

    if styles_dir.is_dir():
        for json_file in styles_dir.rglob("*.json"):
            rel_file = json_file.relative_to(base_path).as_posix()
            files.append(rel_file)

            # Add parent directories up to 'styles/' so users can select whole categories
            parent = json_file.parent
            while parent != base_path and parent.is_relative_to(styles_dir):
                folders.add(parent.relative_to(base_path).as_posix() + "/")
                parent = parent.parent

    return {
        "folders": sorted(folders),
        "files": sorted(files),
    }


def _discover_abc(base_path: Path, abc_base_dir_str: str | None) -> Dict[str, list[str]]:
    """
    Scans abc_base_dir (e.g., sheetsage/output) for ABC directories and files.
    Collapses SheetSage's '<song>-score/melody' and '<song>-score/full' folders
    so the user sees clean soundtrack/game folders and song score folders.
    """
    if abc_base_dir_str:
        abc_root = Path(abc_base_dir_str)
        if not abc_root.is_absolute():
            abc_root = base_path / abc_root
    else:
        abc_root = base_path / "sheetsage" / "output"

    folders = set()
    song_folders = set()

    if abc_root.is_dir():
        for abc_file in abc_root.rglob("*.abc"):
            parent = abc_file.parent
            # SheetSage structure: <collection>/<game>/<song>-score/{melody,full}/<song>.abc
            if parent.name in ("melody", "full") and parent.parent != abc_root:
                score_dir = parent.parent
                song_folders.add(score_dir.relative_to(abc_root).as_posix() + "/")
                curr = score_dir.parent
            else:
                curr = parent

            while curr != abc_root and curr.is_relative_to(abc_root):
                folders.add(curr.relative_to(abc_root).as_posix() + "/")
                curr = curr.parent

    return {
        "collections": sorted(folders),
        "songs": sorted(song_folders),
    }


def create_remote_api(
    shared_state: dict,
    play_queue: queue.Queue,
    pause_event,
    skip_event,
    config_lock,
    playback_info: dict,
    base_path: Path,
    state_file: Path,
    save_dir: Path,
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
        """Auto-discovers available LoRAs, style folders/files, and ABC directories for the Web UI."""
        lora_dir = base_path / "loras"
        loras = []
        if lora_dir.is_dir():
            loras = sorted(
                p.relative_to(base_path).as_posix() for p in lora_dir.rglob("*.safetensors")
            )

        with config_lock:
            abc_base_dir_str = shared_state.get("abc_base_dir", "sheetsage/output")

        styles_info = _discover_styles(base_path)
        abc_info = _discover_abc(base_path, abc_base_dir_str)

        return {
            "loras": loras,
            "styles": styles_info,
            "abc": abc_info,
        }

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
        """Drains all buffered tracks from play_queue in memory."""
        flushed_count = 0
        while True:
            try:
                _ = play_queue.get_nowait()
                play_queue.task_done()
                flushed_count += 1
            except queue.Empty:
                break
        api_logger.info(f"🧹 Flushed {flushed_count} queued track(s) from buffer.")
        return {"status": "success", "flushed": flushed_count}

    @app.get("/api/state")
    def get_state():
        with config_lock:
            vol = float(shared_state.get("volume", 0.4))
            paused = bool(shared_state.get("pause", False))

        metadata = playback_info.get("metadata", {})
        track_num = playback_info.get("track_num")

        track_name = f"Track {track_num}" if track_num else None
        style_name = _format_short_path(metadata.get("_source_style"))
        abc_name = _format_short_path(metadata.get("_source_abc"))

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
        zip_content = playback_info.get("zip_content")
        metadata = playback_info.get("metadata")
        track_num = playback_info.get("track_num")

        if not zip_content or not metadata:
            raise HTTPException(status_code=400, detail="No track playing")

        save_base = save_dir
        save_base.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dest = save_base / f"track_{track_num}_{timestamp}"

        counter = 1
        while save_dest.exists():
            save_dest = save_base / f"track_{track_num}_{timestamp}_{counter}"
            counter += 1

        try:
            save_dest.mkdir()
            # Unpack the in-memory zip to disk
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                zf.extractall(save_dest)
            
            # Save the metadata config along with it
            with open(save_dest / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            api_logger.info(f"💾 Saved currently playing track to: {save_dest}")
            return {"status": "success", "saved_to": str(save_dest)}
        except Exception as e:
            api_logger.error(f"❌ Failed to save track: {e}")
            raise HTTPException(status_code=500, detail="Save failed")

    @app.post("/api/vote/{feature}/{vote_type}")
    def vote(feature: str, vote_type: str):
        if feature not in ["style", "abc"] or vote_type not in ["up", "down"]:
            raise HTTPException(status_code=400, detail="Invalid feature")

        metadata = playback_info.get("metadata")
        track_num = playback_info.get("track_num")

        if not metadata:
            raise HTTPException(status_code=400, detail="No track playing")

        try:
            votes_file = base_path / "votes.jsonl"
            feature_val = metadata.get("_source_style") if feature == "style" else metadata.get("_source_abc")

            vote_record = {
                "timestamp": datetime.now().isoformat(),
                "vote": vote_type,
                "feature": feature,
                "value": feature_val,
                "track": f"Track {track_num}",
            }

            if feature == "style" and metadata.get("lyrics"):
                vote_record["lyrics"] = metadata.get("lyrics")

            with open(votes_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(vote_record) + "\n")

            icon = "⭐" if vote_type == "up" else "👎"
            api_logger.info(f"{icon} {vote_type.title()}voted {feature}! Saved to {votes_file.name}")
            return {"status": "success", "record": vote_record}

        except Exception as e:
            api_logger.error(f"❌ Failed to process vote: {e}")
            raise HTTPException(status_code=500, detail="Vote failed")

    return app
