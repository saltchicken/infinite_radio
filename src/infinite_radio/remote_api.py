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
        .container { max-width: 400px; margin: 0 auto; display: flex; flex-direction: column; gap: 15px; }
        button { background: #2c2c2e; color: white; border: none; padding: 18px 20px; font-size: 18px; border-radius: 12px; cursor: pointer; transition: background 0.1s, transform 0.1s; display: flex; align-items: center; justify-content: center; gap: 8px; font-weight: 600; }
        button:active { background: #3a3a3c; transform: scale(0.98); }
        .primary { background: #0a84ff; }
        .primary:active { background: #0070e0; }
        .success { background: #30d158; }
        .success:active { background: #28a745; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .title { font-size: 28px; font-weight: 800; margin-bottom: 10px; letter-spacing: -0.5px; }
        .label { font-size: 14px; color: #8e8e93; text-transform: uppercase; letter-spacing: 1px; margin-top: 10px; text-align: left; }
        hr { width: 100%; border: 0; height: 1px; background: #38383a; margin: 10px 0; }
        #toast { position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%); background: #ffffff; color: #000; padding: 12px 24px; border-radius: 24px; font-weight: bold; opacity: 0; transition: opacity 0.3s ease; pointer-events: none; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
    </style>
</head>
<body>
    <div class="container">
        <div class="title">📻 Infinite Radio</div>
        
        <div class="grid-2">
            <button onclick="post('/api/pause?state=false')">▶️ Play</button>
            <button onclick="post('/api/pause?state=true')">⏸️ Pause</button>
        </div>
        
        <button class="primary" onclick="post('/api/skip')">⏭️ Skip Track</button>
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
    </script>
</body>
</html>
"""

def create_remote_api(shared_state, pause_event, skip_event, config_lock, playback_info, base_path: Path):
    app = FastAPI(title="Infinite Radio Remote Control")

    @app.get("/", response_class=HTMLResponse)
    def serve_ui():
        return HTML_UI

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
