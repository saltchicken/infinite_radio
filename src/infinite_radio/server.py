import os
import sys
import gc
import io
import time
import zipfile
import tempfile
import argparse
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict
from yue2 import YuE2Pipeline

from .utils.lora_utils import load_ckpt, merge_lora

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
server_logger = logging.getLogger("YuE2Server")

# Evict uncollected or failed tasks after 30 minutes to prevent RAM leaks
TASK_TTL_SECONDS = 1800


class FilteredStream:
    """Lightweight stream proxy that filters noisy strings without buffering in RAM."""
    def __init__(self, original_stream, target="[YuE2]"):
        self.original_stream = original_stream
        self.target = target

    def write(self, s):
        if self.target not in s:
            self.original_stream.write(s)
        return len(s)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        self.original_stream.flush()

    def isatty(self):
        return getattr(self.original_stream, "isatty", lambda: False)()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)


@contextmanager
def suppress_noisy_logs(target="[YuE2]"):
    """Context manager to suppress specific strings from third-party libraries locally."""
    filtered_out = FilteredStream(sys.stdout, target)
    filtered_err = FilteredStream(sys.stderr, target)
    with redirect_stdout(filtered_out), redirect_stderr(filtered_err):
        yield


class GenerateRequest(BaseModel):
    ar_lora: Optional[str] = None
    nar_lora: Optional[str] = None
    ar_scale: float = 1.0
    nar_scale: float = 1.0
    yue_kwargs: Dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class TaskResponse(BaseModel):
    task_id: str
    status: str


def _load_module_state_device_safe(
    module: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = False
):
    """
    Loads a state_dict directly in-place into the target module.
    PyTorch's load_state_dict automatically handles device/dtype copying into existing
    parameters without duplicating the entire state_dict in memory first.
    """
    module.load_state_dict(state_dict, strict=strict)


class ModelManager:
    def __init__(
        self,
        model_name: str,
        vae_name: str,
        default_ar_lora: Optional[Path] = None,
        default_nar_lora: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.vae_name = vae_name
        self.default_ar_lora = default_ar_lora
        self.default_nar_lora = default_nar_lora

        self.pipe = None
        self.top_model = None
        self.bb = None

        # Cached state on CPU for fast in-place LoRA updates and unmerging
        self.base_vae2llm_state = None
        self.base_llm2vae_state = None
        self.pristine_ar_state = None
        self.pristine_nar_state = None

        self.ar_data = None
        self.nar_data = None

        self.current_ar_lora: Optional[Path] = None
        self.current_nar_lora: Optional[Path] = None
        self.current_ar_scale: float = 0.0
        self.current_nar_scale: float = 0.0

        self.lock = threading.Lock()

    def _ensure_base_model(self):
        if self.pipe is None:
            server_logger.info(f"Loading YuE2 Base Model ({self.model_name}) into VRAM...")
            self.pipe = YuE2Pipeline.from_pretrained(self.model_name, vae=self.vae_name, device="cuda")
            self.top_model = self.pipe._load_model()
            self.bb = self.top_model.model

            # Cache clean copies of crossover layers on CPU in case NAR LoRA is removed later
            self.base_vae2llm_state = {
                k: v.detach().cpu().clone() for k, v in self.top_model.vae2llm.state_dict().items()
            }
            self.base_llm2vae_state = {
                k: v.detach().cpu().clone() for k, v in self.top_model.llm2vae.state_dict().items()
            }

            # Cache only the LoRA-targeted backbone weights on CPU to minimize system RAM usage
            server_logger.info("Caching pristine LoRA-target weights in system RAM for 1:1 reproducibility...")
            bb_state = self.bb.state_dict()
            self.pristine_ar_state = {
                k: v.detach().cpu().clone()
                for k, v in bb_state.items()
                if (".self_attn." in k or ".mlp." in k) and ".nar_" not in k
            }
            self.pristine_nar_state = {
                k: v.detach().cpu().clone()
                for k, v in bb_state.items()
                if ".nar_self_attn." in k or ".nar_mlp." in k
            }
            del bb_state
            gc.collect()

    def ensure_model_loaded(
        self,
        req_ar_lora: Optional[str],
        req_nar_lora: Optional[str],
        ar_scale: float,
        nar_scale: float,
    ):
        self._ensure_base_model()

        requested_ar_lora = Path(req_ar_lora) if req_ar_lora else self.default_ar_lora
        requested_nar_lora = Path(req_nar_lora) if req_nar_lora else self.default_nar_lora

        if requested_ar_lora and not requested_ar_lora.exists():
            server_logger.warning(f"Requested AR LoRA not found at: {requested_ar_lora}. Ignoring.")
            requested_ar_lora = None

        if requested_nar_lora and not requested_nar_lora.exists():
            server_logger.warning(f"Requested NAR LoRA not found at: {requested_nar_lora}. Ignoring.")
            requested_nar_lora = None

        # --- 1. Handle AR LoRA (Pristine Restore + Merge) ---
        ar_changed = (requested_ar_lora != self.current_ar_lora) or (ar_scale != self.current_ar_scale)

        if ar_changed:
            if self.current_ar_lora is not None:
                server_logger.info("Restoring pristine AR weights from CPU cache...")
                _load_module_state_device_safe(self.bb, self.pristine_ar_state, strict=False)
                if requested_ar_lora != self.current_ar_lora:
                    self.ar_data = None
                self.current_ar_lora = None
                self.current_ar_scale = 0.0

            if requested_ar_lora is not None and ar_scale != 0.0:
                server_logger.info(f"Loading AR LoRA from {requested_ar_lora.name}...")
                if self.ar_data is None:
                    self.ar_data = load_ckpt(str(requested_ar_lora), "cpu")

                n_merged = merge_lora(self.bb, "self_attn", "mlp", self.ar_data["lora"], scale=ar_scale, dev="cuda")
                self.current_ar_lora = requested_ar_lora
                self.current_ar_scale = ar_scale
                server_logger.info(f"Merged {n_merged} AR LoRA linear layers (scale: {ar_scale}).")

        # --- 2. Handle NAR LoRA (Pristine Restore + Merge) ---
        nar_changed = (requested_nar_lora != self.current_nar_lora) or (nar_scale != self.current_nar_scale)

        if nar_changed:
            if self.current_nar_lora is not None:
                server_logger.info("Restoring pristine NAR weights from CPU cache...")
                _load_module_state_device_safe(self.bb, self.pristine_nar_state, strict=False)

                # Restore clean crossover weights
                if requested_nar_lora is None or nar_scale == 0.0:
                    _load_module_state_device_safe(self.top_model.vae2llm, self.base_vae2llm_state, strict=True)
                    _load_module_state_device_safe(self.top_model.llm2vae, self.base_llm2vae_state, strict=True)

                if requested_nar_lora != self.current_nar_lora:
                    self.nar_data = None
                self.current_nar_lora = None
                self.current_nar_scale = 0.0

            if requested_nar_lora is not None and nar_scale != 0.0:
                server_logger.info(f"Loading NAR LoRA from {requested_nar_lora.name}...")
                if self.nar_data is None:
                    self.nar_data = load_ckpt(str(requested_nar_lora), "cpu")

                n_merged = merge_lora(self.bb, "nar_self_attn", "nar_mlp", self.nar_data["lora"], scale=nar_scale, dev="cuda")
                _load_module_state_device_safe(self.top_model.vae2llm, self.nar_data["io"]["vae2llm"], strict=True)
                _load_module_state_device_safe(self.top_model.llm2vae, self.nar_data["io"]["llm2vae"], strict=True)
                self.current_nar_lora = requested_nar_lora
                self.current_nar_scale = nar_scale
                server_logger.info(f"Merged {n_merged} NAR LoRA linear layers (scale: {nar_scale}) & injected crossovers.")

        if ar_changed or nar_changed:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_artifacts_zip(self, req: GenerateRequest) -> bytes:
        with self.lock:
            self.ensure_model_loaded(req.ar_lora, req.nar_lora, req.ar_scale, req.nar_scale)

            song = None
            try:
                with suppress_noisy_logs("[YuE2]"):
                    with torch.inference_mode():
                        song = self.pipe(**req.yue_kwargs)

                # Save artifacts to a temporary directory and bundle into an in-memory ZIP
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    song.save_artifacts(tmp_path)

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_STORED) as zf:
                        for file_path in tmp_path.rglob("*"):
                            if file_path.is_file():
                                zf.write(file_path, arcname=file_path.relative_to(tmp_path))

                    return zip_buffer.getvalue()
            finally:
                # Explicitly release references and flush CPU/GPU memory caches after every generation
                del song
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


app = FastAPI(title="YuE2 Inference Server")
manager: Optional[ModelManager] = None
generation_tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)


def cleanup_stale_tasks():
    """Evict completed or failed tasks older than TASK_TTL_SECONDS to prevent memory leaks."""
    now = time.time()
    stale_ids = []
    with tasks_lock:
        stale_ids = [
            tid
            for tid, task in generation_tasks.items()
            if task["status"] in ("completed", "failed") and (now - task["created_at"]) > TASK_TTL_SECONDS
        ]
        for tid in stale_ids:
            del generation_tasks[tid]
    if stale_ids:
        gc.collect()
        server_logger.info(f"🧹 Cleaned up {len(stale_ids)} stale task(s) from memory.")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": manager.pipe is not None if manager else False}


def process_generation_task(task_id: str, req: GenerateRequest):
    with tasks_lock:
        if task_id in generation_tasks:
            generation_tasks[task_id]["status"] = "processing"
    try:
        zip_bytes = manager.generate_artifacts_zip(req)
        with tasks_lock:
            if task_id in generation_tasks:
                generation_tasks[task_id]["result"] = zip_bytes
                generation_tasks[task_id]["status"] = "completed"
    except Exception as e:
        server_logger.error(f"Generation failed for task {task_id}: {e}", exc_info=True)
        with tasks_lock:
            if task_id in generation_tasks:
                generation_tasks[task_id]["error"] = str(e)
                generation_tasks[task_id]["status"] = "failed"


@app.post("/generate", response_model=TaskResponse)
def generate_music(req: GenerateRequest):
    if manager is None:
        raise HTTPException(status_code=500, detail="ModelManager is not initialized.")

    cleanup_stale_tasks()
    server_logger.info(f"Received generation request (seed: {req.yue_kwargs.get('seed')})")

    task_id = str(uuid.uuid4())
    with tasks_lock:
        generation_tasks[task_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": time.time(),
        }

    executor.submit(process_generation_task, task_id, req)

    return TaskResponse(task_id=task_id, status="pending")


@app.get("/status/{task_id}")
def get_status(task_id: str):
    cleanup_stale_tasks()
    with tasks_lock:
        if task_id not in generation_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        task = generation_tasks[task_id]
        return {"task_id": task_id, "status": task["status"], "error": task.get("error")}


@app.get("/download/{task_id}")
def download_result(task_id: str):
    with tasks_lock:
        if task_id not in generation_tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        task = generation_tasks[task_id]
        if task["status"] != "completed":
            raise HTTPException(status_code=400, detail="Result is not ready or failed")

        zip_bytes = task["result"]
        del generation_tasks[task_id]  # Free memory immediately after download

    return Response(content=zip_bytes, media_type="application/zip")


def main():
    global manager
    parser = argparse.ArgumentParser(description="Host the YuE2 model API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--model", default="m-a-p/YuE2-3B", help="HuggingFace model ID or local path")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae", help="HuggingFace VAE ID or local path")
    parser.add_argument("--ar-lora", type=Path, help="Default path to the AR LoRA safetensors file")
    parser.add_argument("--nar-lora", type=Path, help="Default path to the NAR LoRA safetensors file")
    parser.add_argument("--preload", action="store_true", help="Preload the base model on startup before first request")
    args = parser.parse_args()

    manager = ModelManager(
        model_name=args.model,
        vae_name=args.vae,
        default_ar_lora=args.ar_lora,
        default_nar_lora=args.nar_lora,
    )

    if args.preload:
        manager.ensure_model_loaded(
            str(args.ar_lora) if args.ar_lora else None,
            str(args.nar_lora) if args.nar_lora else None,
            1.0,
            1.0,
        )

    try:
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    finally:
        server_logger.info("Forcing process exit to terminate background inference threads...")
        os._exit(0)


if __name__ == "__main__":
    main()
