import sys
import io
import zipfile
import tempfile
import argparse
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
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


class ThreadSafeStreamFilter:
    """Filter to intercept and drop noisy logs containing [YuE2] safely across multiple threads"""
    def __init__(self, stream, target="[YuE2]"):
        self.stream = stream
        self.target = target
        self.local = threading.local()

    def write(self, data):
        # Isolate the buffer per-thread to prevent concurrent log mangling
        if not hasattr(self.local, 'buf'):
            self.local.buf = ""
            
        self.local.buf += data
        while True:
            if "\n" in self.local.buf:
                splitter = "\n"
            elif "\r" in self.local.buf:
                splitter = "\r"
            else:
                break
            line, self.local.buf = self.local.buf.split(splitter, 1)
            if self.target not in line:
                self.stream.write(line + splitter)
                self.stream.flush()

    def flush(self):
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


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


class ModelManager:
    def __init__(self, model_name: str, vae_name: str, default_ar_lora: Optional[Path] = None, default_nar_lora: Optional[Path] = None):
        self.model_name = model_name
        self.vae_name = vae_name
        self.default_ar_lora = default_ar_lora
        self.default_nar_lora = default_nar_lora

        self.pipe = None
        self.current_ar_scale = None
        self.current_nar_scale = None
        self.current_ar_lora = None
        self.current_nar_lora = None
        self.lock = threading.Lock()

    def ensure_model_loaded(self, req_ar_lora: Optional[str], req_nar_lora: Optional[str], ar_scale: float, nar_scale: float):
        requested_ar_lora = Path(req_ar_lora) if req_ar_lora else self.default_ar_lora
        requested_nar_lora = Path(req_nar_lora) if req_nar_lora else self.default_nar_lora

        if (self.pipe is None or
                self.current_ar_scale != ar_scale or
                self.current_nar_scale != nar_scale or
                self.current_ar_lora != requested_ar_lora or
                self.current_nar_lora != requested_nar_lora):

            if self.pipe is not None:
                server_logger.info("🔄 LoRA configuration change detected. Reloading model...")
                del self.pipe
                torch.cuda.empty_cache()
            else:
                server_logger.info(f"Loading YuE2 Model ({self.model_name}) into VRAM...")

            self.pipe = YuE2Pipeline.from_pretrained(self.model_name, vae=self.vae_name, device="cuda")
            top_model = self.pipe._load_model()
            bb = top_model.model

            if requested_ar_lora and requested_ar_lora.exists():
                server_logger.info(f"Loading AR LoRA from {requested_ar_lora.name}...")
                ar_data = load_ckpt(str(requested_ar_lora), "cuda")
                n_merged = merge_lora(bb, "self_attn", "mlp", ar_data["lora"], scale=ar_scale, dev="cuda")
                server_logger.info(f"Merged {n_merged} AR LoRA linear layers (scale: {ar_scale}).")
            elif requested_ar_lora:
                server_logger.warning(f"Requested AR LoRA not found at: {requested_ar_lora}")

            if requested_nar_lora and requested_nar_lora.exists():
                server_logger.info(f"Loading NAR LoRA from {requested_nar_lora.name}...")
                nar_data = load_ckpt(str(requested_nar_lora), "cuda")
                n_merged = merge_lora(bb, "nar_self_attn", "nar_mlp", nar_data["lora"], scale=nar_scale, dev="cuda")
                top_model.vae2llm.load_state_dict({k: v.to(torch.bfloat16) for k, v in nar_data["io"]["vae2llm"].items()})
                top_model.llm2vae.load_state_dict({k: v.to(torch.bfloat16) for k, v in nar_data["io"]["llm2vae"].items()})
                server_logger.info(f"Merged {n_merged} NAR LoRA linear layers (scale: {nar_scale}) & injected custom VAE/LLM crossovers.")
            elif requested_nar_lora:
                server_logger.warning(f"Requested NAR LoRA not found at: {requested_nar_lora}")

            self.current_ar_scale = ar_scale
            self.current_nar_scale = nar_scale
            self.current_ar_lora = requested_ar_lora
            self.current_nar_lora = requested_nar_lora

    def generate_artifacts_zip(self, req: GenerateRequest) -> bytes:
        with self.lock:
            self.ensure_model_loaded(req.ar_lora, req.nar_lora, req.ar_scale, req.nar_scale)

            original_stdout, original_stderr = sys.stdout, sys.stderr
            sys.stdout = ThreadSafeStreamFilter(sys.stdout)
            sys.stderr = ThreadSafeStreamFilter(sys.stderr)

            try:
                song = self.pipe(**req.yue_kwargs)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

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


app = FastAPI(title="YuE2 Inference Server")
manager: Optional[ModelManager] = None
generation_tasks: Dict[str, Any] = {}
executor = ThreadPoolExecutor(max_workers=1)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": manager.pipe is not None if manager else False}


def process_generation_task(task_id: str, req: GenerateRequest):
    generation_tasks[task_id]["status"] = "processing"
    try:
        zip_bytes = manager.generate_artifacts_zip(req)
        generation_tasks[task_id]["result"] = zip_bytes
        generation_tasks[task_id]["status"] = "completed"
    except Exception as e:
        server_logger.error(f"Generation failed for task {task_id}: {e}", exc_info=True)
        generation_tasks[task_id]["error"] = str(e)
        generation_tasks[task_id]["status"] = "failed"


@app.post("/generate", response_model=TaskResponse)
def generate_music(req: GenerateRequest):
    if manager is None:
        raise HTTPException(status_code=500, detail="ModelManager is not initialized.")
    
    server_logger.info(f"Received generation request (seed: {req.yue_kwargs.get('seed')})")
    
    task_id = str(uuid.uuid4())
    generation_tasks[task_id] = {"status": "pending", "result": None, "error": None}
    
    executor.submit(process_generation_task, task_id, req)
    
    return TaskResponse(task_id=task_id, status="pending")


@app.get("/status/{task_id}")
def get_status(task_id: str):
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = generation_tasks[task_id]
    return {"task_id": task_id, "status": task["status"], "error": task.get("error")}


@app.get("/download/{task_id}")
def download_result(task_id: str):
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = generation_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Result is not ready or failed")
    
    zip_bytes = task["result"]
    del generation_tasks[task_id]  # Free memory after download
    
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

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
