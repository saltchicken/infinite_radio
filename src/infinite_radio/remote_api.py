import io
import json
import time
import queue
import zipfile
import logging
import threading
import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

api_logger = logging.getLogger("RemoteAPI")

# Cache directory scans for the Web UI dropdowns for 5 minutes
OPTIONS_CACHE_TTL = 300.0


class ConfigUpdatePayload(BaseModel):
    settings: Dict[str, Any]


class DSPUpdatePayload(BaseModel):
    enable_normalization: bool
    enable_reverb: bool
    reverb_room_size: float
    reverb_damping: float
    reverb_wet_level: float
    reverb_dry_level: float


def _format_short_path(raw_path: str | None, is_abc: bool = False) -> str | None:
    """Formats a full path into 'parent_folder/filename' without extensions for clean mobile display."""
    if not raw_path:
        return None
    p = Path(raw_path)
    name = p.stem  # drops the extension (.json, .abc)

    # For SheetSage ABCs, navigate up to the game folder instead of showing "melody" or "full"
    if is_abc and p.parent.name in ("melody", "full"):
        # Structure: <game>/<song>-score/{melody,full}/<song>.abc
        game_folder = p.parents[2].name if len(p.parents) > 2 else ""
        if game_folder:
            return f"{game_folder}/{name}"

    if p.parent and p.parent.name:
        return f"{p.parent.name}/{name}"
    return name


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
    save_dir: Path,
):
    app = FastAPI(title="Infinite Radio Remote Control")

    db_path = base_path / "votes.db"
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS votes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    vote TEXT,
                    feature TEXT,
                    value TEXT,
                    track TEXT,
                    lyrics TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS favorite_abcs (
                    file_path TEXT PRIMARY KEY
                )
            ''')
            conn.commit()
    except Exception as e:
        api_logger.error(f"Failed to initialize SQLite database: {e}")

    options_cache_lock = threading.Lock()
    options_cache: Dict[str, Any] = {
        "key": None,
        "time": 0.0,
        "data": None,
    }

    @app.get("/", response_class=HTMLResponse)
    def serve_ui():
        # Serve the HTML cleanly from the static directory
        html_path = Path(__file__).parent / "static" / "index.html"
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="Web UI template not found")
        return FileResponse(html_path)

    @app.get("/api/options")
    def get_options(refresh: bool = False):
        """Auto-discovers available LoRAs, style folders/files, and ABC directories for the Web UI."""
        with config_lock:
            abc_base_dir_str = shared_state.get("abc_base_dir", "sheetsage/output")

        cache_key = str(abc_base_dir_str)
        now = time.monotonic()

        with options_cache_lock:
            if (
                not refresh
                and options_cache["data"] is not None
                and options_cache["key"] == cache_key
                and (now - options_cache["time"]) <= OPTIONS_CACHE_TTL
            ):
                return options_cache["data"]

        lora_dir = base_path / "loras"
        loras = []
        if lora_dir.is_dir():
            loras = sorted(
                p.relative_to(base_path).as_posix() for p in lora_dir.rglob("*.safetensors")
            )

        styles_info = _discover_styles(base_path)
        abc_info = _discover_abc(base_path, abc_base_dir_str)
        
        # Inject the virtual favorites directory at the top of the collections list
        abc_info["collections"].insert(0, "⭐ Favorites (SQLite)")

        result = {
            "loras": loras,
            "styles": styles_info,
            "abc": abc_info,
        }

        with options_cache_lock:
            options_cache["key"] = cache_key
            options_cache["time"] = time.monotonic()
            options_cache["data"] = result

        return result

    @app.get("/api/config")
    def get_config():
        with config_lock:
            return shared_state.copy()

    @app.post("/api/config")
    def update_config(payload: ConfigUpdatePayload):
        with config_lock:
            old_abc_base = shared_state.get("abc_base_dir")
            vol = shared_state.get("volume", 0.4)
            paused = shared_state.get("pause", False)
            old_order = shared_state.get("abc_order", "random")

            shared_state.clear()
            shared_state.update(payload.settings)

            shared_state["volume"] = vol
            shared_state["pause"] = paused
            shared_state.setdefault("abc_order", old_order)

            new_abc_base = shared_state.get("abc_base_dir")
            snapshot = shared_state.copy()

        if old_abc_base != new_abc_base:
            with options_cache_lock:
                options_cache["data"] = None

        api_logger.info("⚙️ Generation settings updated via Web UI.")
        return {"status": "success", "config": snapshot}

    @app.post("/api/dsp")
    def update_dsp(payload: DSPUpdatePayload):
        with config_lock:
            shared_state["enable_normalization"] = payload.enable_normalization
            shared_state["enable_reverb"] = payload.enable_reverb
            shared_state["reverb_room_size"] = payload.reverb_room_size
            shared_state["reverb_damping"] = payload.reverb_damping
            shared_state["reverb_wet_level"] = payload.reverb_wet_level
            shared_state["reverb_dry_level"] = payload.reverb_dry_level
        return {"status": "success"}

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

    @app.get("/api/stream")
    async def stream_state():
        """Streams state updates to the UI using Server-Sent Events (SSE)."""
        async def event_generator():
            while True:
                # Briefly acquire the lock to read shared state securely
                with config_lock:
                    vol = float(shared_state.get("volume", 0.4))
                    paused = bool(shared_state.get("pause", False))

                metadata = playback_info.get("metadata", {})
                track_num = playback_info.get("track_num")

                track_name = f"Track {track_num}" if track_num else None
                style_name = _format_short_path(metadata.get("_source_style"), is_abc=False)
                abc_name = _format_short_path(metadata.get("_source_abc"), is_abc=True)

                state = {
                    "volume": vol,
                    "paused": paused,
                    "track": track_name,
                    "style": style_name,
                    "abc": abc_name,
                    "queued": play_queue.qsize(),
                }
                
                yield f"data: {json.dumps(state)}\n\n"
                # Push an update every 1 second
                await asyncio.sleep(1.0)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/api/state")
    def get_state():
        """Fallback REST state endpoint (still useful for CLI tools or debugging)."""
        with config_lock:
            vol = float(shared_state.get("volume", 0.4))
            paused = bool(shared_state.get("pause", False))

        metadata = playback_info.get("metadata", {})
        track_num = playback_info.get("track_num")

        track_name = f"Track {track_num}" if track_num else None
        style_name = _format_short_path(metadata.get("_source_style"), is_abc=False)
        abc_name = _format_short_path(metadata.get("_source_abc"), is_abc=True)

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
        return {"status": "success", "volume": clamped}

    @app.post("/api/skip")
    def skip_track():
        skip_event.set()
        return {"status": "success", "message": "Track skipped"}

    @app.post("/api/pause")
    def toggle_pause(state: bool):
        with config_lock:
            shared_state["pause"] = state
        if state:
            pause_event.set()
        else:
            pause_event.clear()
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
            
            # 1. Save to the new SQLite Database
            with sqlite3.connect(db_path) as conn:
                conn.execute('''
                    INSERT INTO votes (timestamp, vote, feature, value, track, lyrics)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    vote_type,
                    feature,
                    feature_val,
                    f"Track {track_num}",
                    metadata.get("lyrics") if feature == "style" else None
                ))
                
                # If it's an ABC upvote, add it to the virtual directory table
                if feature == "abc" and vote_type == "up" and feature_val:
                    conn.execute('''
                        INSERT OR IGNORE INTO favorite_abcs (file_path)
                        VALUES (?)
                    ''', (feature_val,))
                conn.commit()

            # 2. Maintain JSONL fallback for existing scripts
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
            api_logger.info(f"{icon} {vote_type.title()}voted {feature}! Saved to DB and {votes_file.name}")
            return {"status": "success", "record": vote_record}

        except Exception as e:
            api_logger.error(f"❌ Failed to process vote: {e}")
            raise HTTPException(status_code=500, detail="Vote failed")

    return app
