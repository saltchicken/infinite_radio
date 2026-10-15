import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

api_logger = logging.getLogger("RemoteAPI")

# Mobile-friendly HTML UI embedded directly to avoid needing static file hosting
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Infinite Radio Remote</title>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; background: #121212; color: #fff; text-align: center; padding: 20px; margin: 0; }
        .container { max-width: 400px; margin: 0 auto; display: flex; flex-direction: column; gap: 14px; }
        button { background: #2c2c2e; color: white; border: none; padding: 16px 20px; font-size: 17px; border-radius: 12px; cursor: pointer; transition: background 0.1s, transform 0.1s; display: flex; align-items: center; justify-content: center; gap: 8px; font-weight: 600; }
        button:active { background: #3a3a3c; transform: scale(0.98); }
        button.active-state { outline: 2px solid #0a84ff; background: #1c2c3e; }
        .primary { background: #0a84ff; }
        .primary:active { background: #0070e0; }
        .success { background: #30d158; }
        .success:active { background: #28a745; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .title { font-size: 26px; font-weight: 800; margin-bottom: 4px; letter-spacing: -0.5px; }
        .now-playing { background: #1c1c1e; border: 1px solid #2c2c2e; border-radius: 14px; padding: 14px 16px; text-align: left; display: flex; flex-direction: column; gap: 8px; }
        .np-header { display: flex; justify-content: space-between; align-items: center; font-size: 12px; font-weight: 700; color: #0a84ff; text-transform: uppercase; letter-spacing: 0.8px; }
        .np-row { font-size: 14px; color: #e5e5ea; display: flex; flex-direction: column; gap: 2px; word-break: break-all; }
        .np-sublabel { font-size: 11px; color: #8e8e93; text-transform: uppercase; letter-spacing: 0.5px; }
        .label { font-size: 13px; color: #8e8e93; text-transform: uppercase; letter-spacing: 1px; margin-top: 6px; text-align: left; display: flex; justify-content: space-between; }
        .slider-box { background: #2c2c2e; padding: 12px 16px; border-radius: 12px; display: flex; align-items: center; gap: 12px; }
        input[type="range"] { flex: 1; accent-color: #0a84ff; height: 6px; cursor: pointer; }
        hr { width: 100%; border: 0; height: 1px; background: #38383a; margin: 6px 0; }
        #toast { position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%); background: #ffffff; color: #000; padding: 12px 24px; border-radius: 24px; font-weight: bold; opacity: 0; transition: opacity 0.3s ease; pointer-events: none; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
    </style>
</head>
<body>
    <div class="container">
        <div class="title">📻 Infinite Radio</div>

        <div class="now-playing">
            <div class="np-header">
                <span id="np-status">NOW PLAYING</span>
                <span id="np-track">Waiting...</span>
            </div>
            <div class="np-row">
                <span class="np-sublabel">Style</span>
                <span id="np-style">—</span>
            </div>
            <div class="np-row">
                <span class="np-sublabel">ABC Sheet</span>
                <span id="np-abc">—</span>
            </div>
        </div>
        
        <div class="grid-2">
            <button id="btn-play" onclick="setPause(false)">▶️ Play</button>
            <button id="btn-pause" onclick="setPause(true)">⏸️ Pause</button>
        </div>

        <div class="label">
            <span>Volume</span>
            <span id="vol-label">40%</span>
        </div>
        <div class="slider-box">
            <span>🔈</span>
            <input id="vol-slider" type="range" min="0" max="1" step="0.02" value="0.4" oninput="setVolume(this.value)">
            <span>🔊</span>
        </div>
        
        <button class="primary" onclick="skipTrack()">⏭️ Skip Track</button>
        <button class="success" onclick="post('/api/save')">💾 Save Track</button>
        
        <hr>
        
        <div class="label">Style Reference</div>
        <div class="grid-2">
            <button onclick="post('/api/vote/style/up')">👍 Upvote</button>
            <button onclick="post('/api/vote/style/down')">👎 Downvote</button>
        </div>

        <div class="label">ABC Sheet</div>
        <div class="grid-2">
            <button onclick="post('/api/vote/abc/up')">👍 Upvote</button>
            <button onclick="post('/api/vote/abc/down')">👎 Downvote</button>
        </div>
    </div>
    
    <div id="toast">Action sent</div>

    <script>
        let isDraggingVolume = false;
        const slider = document.getElementById('vol-slider');
        slider.addEventListener('pointerdown', () => isDraggingVolume = true);
        slider.addEventListener('pointerup', () => isDraggingVolume = false);

        async function syncState() {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                if (!isDraggingVolume) {
                    slider.value = data.volume;
                    document.getElementById('vol-label').innerText = Math.round(data.volume * 100) + '%';
                }
                document.getElementById('btn-play').classList.toggle('active-state', !data.paused);
                document.getElementById('btn-pause').classList.toggle('active-state', data.paused);

                document.getElementById('np-status').innerText = data.paused ? 'PAUSED' : 'NOW PLAYING';
                document.getElementById('np-track').innerText = data.track || 'Buffering...';
                document.getElementById('np-style').innerText = data.style || '—';
                document.getElementById('np-abc').innerText = data.abc || '—';
            } catch (err) {}
        }

        async function setPause(state) {
            await post('/api/pause?state=' + state);
            syncState();
        }

        async function skipTrack() {
            await post('/api/skip');
            setTimeout(syncState, 700);
        }

        async function setVolume(val) {
            document.getElementById('vol-label').innerText = Math.round(val * 100) + '%';
            try {
                await fetch('/api/volume?level=' + val, { method: 'POST' });
            } catch (err) {}
        }

        async function post(url) {
            try {
                const res = await fetch(url, { method: 'POST' });
                const data = await res.json();
                if (data.status === 'success') {
                    showToast('✅ Success');
                } else {
                    showToast('❌ ' + (data.detail || 'Failed'));
                }
            } catch (err) {
                showToast('❌ Network Error');
            }
        }

        function showToast(msg) {
            const t = document.getElementById('toast');
            t.innerText = msg;
            t.style.opacity = 1;
            setTimeout(() => t.style.opacity = 0, 2000);
        }

        syncState();
        setInterval(syncState, 2000);
    </script>
</body>
</html>
"""


def _format_short_path(raw_path: str | None) -> str | None:
    """Formats a full path into 'parent_folder/filename' for clean mobile display."""
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.parent and p.parent.name:
        return f"{p.parent.name}/{p.name}"
    return p.name


def create_remote_api(shared_state, pause_event, skip_event, config_lock, playback_info, base_path: Path):
    app = FastAPI(title="Infinite Radio Remote Control")

    @app.get("/", response_class=HTMLResponse)
    def serve_ui():
        return HTML_UI

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
