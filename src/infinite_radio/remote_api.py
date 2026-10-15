import json
import queue
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

api_logger = logging.getLogger("RemoteAPI")

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Infinite Radio Remote</title>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; background: #121212; color: #fff; text-align: center; padding: 20px; margin: 0; }
        .container { max-width: 420px; margin: 0 auto; display: flex; flex-direction: column; gap: 14px; padding-bottom: 40px; }
        button { background: #2c2c2e; color: white; border: none; padding: 15px 16px; font-size: 15px; border-radius: 12px; cursor: pointer; transition: background 0.1s, transform 0.1s; display: flex; align-items: center; justify-content: center; gap: 6px; font-weight: 600; }
        button:active { background: #3a3a3c; transform: scale(0.98); }
        button.active-state { outline: 2px solid #0a84ff; background: #1c2c3e; }
        .primary { background: #0a84ff; }
        .primary:active { background: #0070e0; }
        .warning { background: #ff9f0a; color: #000; }
        .warning:active { background: #d98200; }
        .success { background: #30d158; }
        .success:active { background: #28a745; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
        .title { font-size: 26px; font-weight: 800; margin-bottom: 4px; letter-spacing: -0.5px; }
        .card { background: #1c1c1e; border: 1px solid #2c2c2e; border-radius: 14px; padding: 14px 16px; text-align: left; display: flex; flex-direction: column; gap: 10px; }
        .np-header { display: flex; justify-content: space-between; align-items: center; font-size: 12px; font-weight: 700; color: #0a84ff; text-transform: uppercase; letter-spacing: 0.8px; }
        .np-meta { display: flex; gap: 10px; align-items: center; color: #8e8e93; }
        .badge { background: #2c2c2e; color: #e5e5ea; padding: 3px 8px; border-radius: 8px; font-size: 11px; font-weight: 600; }
        .np-row { font-size: 14px; color: #e5e5ea; display: flex; flex-direction: column; gap: 2px; word-break: break-all; }
        .np-sublabel { font-size: 11px; color: #8e8e93; text-transform: uppercase; letter-spacing: 0.5px; }
        .label { font-size: 12px; color: #8e8e93; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 4px; text-align: left; display: flex; justify-content: space-between; font-weight: 600; }
        .slider-box { background: #2c2c2e; padding: 10px 14px; border-radius: 10px; display: flex; align-items: center; gap: 12px; }
        input[type="range"] { flex: 1; accent-color: #0a84ff; height: 6px; cursor: pointer; }
        input[type="text"], select, textarea { width: 100%; box-sizing: border-box; background: #2c2c2e; color: #fff; border: 1px solid #3a3a3c; border-radius: 10px; padding: 10px 12px; font-size: 14px; font-family: inherit; }
        textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; min-height: 90px; resize: vertical; }
        details { background: #1c1c1e; border: 1px solid #2c2c2e; border-radius: 14px; padding: 12px 16px; text-align: left; }
        summary { font-weight: 700; font-size: 15px; cursor: pointer; color: #e5e5ea; outline: none; }
        .form-stack { display: flex; flex-direction: column; gap: 12px; margin-top: 14px; }
        hr { width: 100%; border: 0; height: 1px; background: #38383a; margin: 6px 0; }
        #toast { position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%); background: #ffffff; color: #000; padding: 12px 24px; border-radius: 24px; font-weight: bold; opacity: 0; transition: opacity 0.3s ease; pointer-events: none; box-shadow: 0 4px 12px rgba(0,0,0,0.5); z-index: 100; }
    </style>
</head>
<body>
    <div class="container">
        <div class="title">📻 Infinite Radio</div>

        <div class="card">
            <div class="np-header">
                <span id="np-status">NOW PLAYING</span>
                <div class="np-meta">
                    <span id="np-track">Waiting...</span>
                    <span class="badge" id="np-queue">Queue: 0</span>
                </div>
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
        
        <div class="grid-3">
            <button class="primary" onclick="skipTrack()">⏭️ Skip</button>
            <button class="warning" onclick="flushQueue()">🧹 Flush</button>
            <button class="success" onclick="post('/api/save')">💾 Save</button>
        </div>
        
        <div class="label">Style & ABC Voting</div>
        <div class="grid-2">
            <button onclick="post('/api/vote/style/up')">👍 Style</button>
            <button onclick="post('/api/vote/style/down')">👎 Style</button>
            <button onclick="post('/api/vote/abc/up')">👍 ABC</button>
            <button onclick="post('/api/vote/abc/down')">👎 ABC</button>
        </div>

        <hr>

        <!-- Generation Settings Panel -->
        <details open>
            <summary>⚙️ Generation Settings (Next Tracks)</summary>
            <div class="form-stack">
                <div>
                    <div class="label"><span>Active Style Path(s) (comma-separated)</span></div>
                    <input id="cfg-style" type="text" placeholder="styles/upvoted/">
                </div>

                <div>
                    <div class="label"><span>ABC Directory or File Path</span></div>
                    <input id="cfg-abc" type="text" placeholder="/path/to/sheetsage/vgm/...">
                </div>

                <div>
                    <div class="label"><span>CoT Mode</span></div>
                    <select id="cfg-cot">
                        <option value="melody">melody (*-melody.abc)</option>
                        <option value="full">full (*-full.abc)</option>
                        <option value="any">any (*.abc)</option>
                    </select>
                </div>

                <div>
                    <div class="label"><span>AR LoRA</span></div>
                    <select id="cfg-ar-lora"></select>
                </div>

                <div>
                    <div class="label">
                        <span>AR LoRA Scale</span>
                        <span id="lbl-ar-scale">0.50</span>
                    </div>
                    <div class="slider-box">
                        <input id="cfg-ar-scale" type="range" min="0" max="2" step="0.05" value="0.5"
                               oninput="document.getElementById('lbl-ar-scale').innerText = Number(this.value).toFixed(2)">
                    </div>
                </div>

                <div>
                    <div class="label"><span>NAR LoRA</span></div>
                    <select id="cfg-nar-lora"></select>
                </div>

                <div>
                    <div class="label">
                        <span>NAR LoRA Scale</span>
                        <span id="lbl-nar-scale">1.00</span>
                    </div>
                    <div class="slider-box">
                        <input id="cfg-nar-scale" type="range" min="0" max="2" step="0.05" value="1.0"
                               oninput="document.getElementById('lbl-nar-scale').innerText = Number(this.value).toFixed(2)">
                    </div>
                </div>

                <div>
                    <div class="label"><span>Extra YuE2 Parameters (JSON)</span></div>
                    <textarea id="cfg-extra" placeholder='{"abc_instrument": "Lead Synth", "abc_midi_program": 81}'></textarea>
                </div>

                <div class="grid-2">
                    <button class="primary" onclick="saveConfig(false)">💾 Apply Settings</button>
                    <button class="warning" onclick="saveConfig(true)">⚡ Apply & Flush</button>
                </div>
            </div>
        </details>
    </div>
    
    <div id="toast">Action sent</div>

    <script>
        let isDraggingVolume = false;
        const slider = document.getElementById('vol-slider');
        slider.addEventListener('pointerdown', () => isDraggingVolume = true);
        slider.addEventListener('pointerup', () => isDraggingVolume = false);

        const CORE_KEYS = new Set([
            'volume', 'pause', 'active_style', 'abc', 'cot',
            'ar_lora', 'nar_lora', 'ar_scale', 'nar_scale'
        ]);

        function populateSelect(selectEl, options, currentVal) {
            selectEl.innerHTML = '';
            const noneOpt = document.createElement('option');
            noneOpt.value = '';
            noneOpt.textContent = 'None (Base Model)';
            selectEl.appendChild(noneOpt);

            const allOpts = new Set(options);
            if (currentVal) allOpts.add(currentVal);

            allOpts.forEach(opt => {
                const el = document.createElement('option');
                el.value = opt;
                el.textContent = opt;
                if (opt === currentVal) el.selected = true;
                selectEl.appendChild(el);
            });
        }

        async function loadConfigForm() {
            try {
                const [optRes, cfgRes] = await Promise.all([
                    fetch('/api/options'),
                    fetch('/api/config')
                ]);
                const options = await optRes.json();
                const cfg = await cfgRes.json();

                const styles = Array.isArray(cfg.active_style) ? cfg.active_style.join(', ') : (cfg.active_style || '');
                document.getElementById('cfg-style').value = styles;

                const abcs = Array.isArray(cfg.abc) ? cfg.abc.join(', ') : (cfg.abc || '');
                document.getElementById('cfg-abc').value = abcs;

                document.getElementById('cfg-cot').value = cfg.cot || 'melody';

                populateSelect(document.getElementById('cfg-ar-lora'), options.loras, cfg.ar_lora || '');
                populateSelect(document.getElementById('cfg-nar-lora'), options.loras, cfg.nar_lora || '');

                const arScale = cfg.ar_scale !== undefined ? cfg.ar_scale : 1.0;
                document.getElementById('cfg-ar-scale').value = arScale;
                document.getElementById('lbl-ar-scale').innerText = Number(arScale).toFixed(2);

                const narScale = cfg.nar_scale !== undefined ? cfg.nar_scale : 1.0;
                document.getElementById('cfg-nar-scale').value = narScale;
                document.getElementById('lbl-nar-scale').innerText = Number(narScale).toFixed(2);

                const extras = {};
                Object.keys(cfg).forEach(k => {
                    if (!CORE_KEYS.has(k)) extras[k] = cfg[k];
                });
                document.getElementById('cfg-extra').value = Object.keys(extras).length
                    ? JSON.stringify(extras, null, 2)
                    : '{}';
            } catch (err) {}
        }

        async function saveConfig(flushAfter = false) {
            let extras = {};
            const rawExtra = document.getElementById('cfg-extra').value.trim();
            if (rawExtra) {
                try {
                    extras = JSON.parse(rawExtra);
                } catch (e) {
                    showToast('❌ Invalid JSON in Extra Parameters');
                    return;
                }
            }

            const styleStr = document.getElementById('cfg-style').value;
            const activeStyle = styleStr.split(',').map(s => s.trim()).filter(Boolean);

            const payload = {
                ...extras,
                active_style: activeStyle,
                abc: document.getElementById('cfg-abc').value.trim(),
                cot: document.getElementById('cfg-cot').value,
                ar_lora: document.getElementById('cfg-ar-lora').value || null,
                nar_lora: document.getElementById('cfg-nar-lora').value || null,
                ar_scale: parseFloat(document.getElementById('cfg-ar-scale').value),
                nar_scale: parseFloat(document.getElementById('cfg-nar-scale').value),
            };

            try {
                const res = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ settings: payload })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    if (flushAfter) {
                        const flushRes = await fetch('/api/flush', { method: 'POST' });
                        const flushData = await flushRes.json();
                        showToast(`⚡ Applied & Flushed (${flushData.flushed} tracks)`);
                        syncState();
                    } else {
                        showToast('✅ Settings Applied & Saved');
                    }
                } else {
                    showToast('❌ Failed to save');
                }
            } catch (err) {
                showToast('❌ Network Error');
            }
        }

        async function flushQueue() {
            try {
                const res = await fetch('/api/flush', { method: 'POST' });
                const data = await res.json();
                if (data.status === 'success') {
                    showToast(`🧹 Flushed ${data.flushed} queued track(s)`);
                    syncState();
                } else {
                    showToast('❌ Flush Failed');
                }
            } catch (err) {
                showToast('❌ Network Error');
            }
        }

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
                document.getElementById('np-queue').innerText = 'Queue: ' + (data.queued ?? 0);
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

        loadConfigForm();
        syncState();
        setInterval(syncState, 2000);
    </script>
</body>
</html>
"""


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
        return HTML_UI

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
