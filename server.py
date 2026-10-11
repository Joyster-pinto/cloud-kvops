"""
LLM-KVOps: Real-Time Dual LLM Web Server
FastAPI backend orchestrating side-by-side real GPU execution and telemetry streaming.
"""

import os
import sys
import json
import time
from typing import Optional, List, Dict, Any, Union

# Python 3.12 Threading Fix
import threading
if hasattr(threading, "_register_atexit"):
    _orig_reg = threading._register_atexit
    def _safe_register_atexit(*args, **kwargs):
        try:
            return _orig_reg(*args, **kwargs)
        except RuntimeError:
            pass
    threading._register_atexit = _safe_register_atexit

# Starlette 1.6+ / FastAPI Router Compatibility Patch
import starlette.routing
_orig_router_init = starlette.routing.Router.__init__
def _safe_router_init(self, *args, **kwargs):
    kwargs.pop("on_startup", None)
    kwargs.pop("on_shutdown", None)
    return _orig_router_init(self, *args, **kwargs)
starlette.routing.Router.__init__ = _safe_router_init

import io
try:
    import fitz
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
FastAPI.max_body_size = None
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.engine.real_llm_runner import RealLLMRunner

app = FastAPI(title="LLM-KVOps Dual Profiler Server")

# Global runner cache
GLOBAL_RUNNER: Optional[RealLLMRunner] = None
CURRENT_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

def get_runner(model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct", hf_token: Optional[str] = None) -> RealLLMRunner:
    global GLOBAL_RUNNER, CURRENT_MODEL_ID
    if GLOBAL_RUNNER is None or CURRENT_MODEL_ID != model_name:
        print(f"[*] Initializing model {model_name}...")
        GLOBAL_RUNNER = RealLLMRunner(model_name=model_name, hf_token=hf_token)
        CURRENT_MODEL_ID = model_name
    return GLOBAL_RUNNER

class GenerateRequest(BaseModel):
    prompt: str
    mode: str = "dual" # "baseline", "kvops", or "dual"
    max_tokens: int = 1024
    budget_slots: int = 0
    temperature: float = 0.7
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    hf_token: Optional[str] = None

SESSIONS_FILE = os.path.join(PROJECT_ROOT, "chat_sessions.json")

def load_sessions_data():
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_sessions_data(data):
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("[!] Error saving sessions:", e)

@app.get("/api/sessions")
def get_sessions():
    return load_sessions_data()

@app.post("/api/sessions")
def save_session(session: dict):
    sessions = load_sessions_data()
    sess_id = session.get("id")
    idx = next((i for i, s in enumerate(sessions) if s.get("id") == sess_id), None)
    if idx is not None:
        sessions[idx] = session
    else:
        sessions.insert(0, session)
    sessions = sessions[:50]
    save_sessions_data(sessions)
    return {"status": "ok", "count": len(sessions)}

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    sessions = load_sessions_data()
    sessions = [s for s in sessions if s.get("id") != session_id]
    save_sessions_data(sessions)
    return {"status": "ok"}

@app.post("/api/extract_text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename or "uploaded_file"
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    pages = 1
    
    if ext == ".pdf":
        if fitz is not None:
            try:
                doc = fitz.open(stream=contents, filetype="pdf")
                pages = len(doc)
                parts = []
                for i, p in enumerate(doc):
                    pt = p.get_text()
                    if pt.strip():
                        parts.append(f"--- Page {i+1} ---\n{pt.strip()}")
                doc.close()
                text = "\n\n".join(parts)
            except Exception as e:
                text = f"[Error reading PDF: {e}]"
        else:
            text = "[Error: PyMuPDF not available on server]"
    elif ext in [".docx", ".doc"]:
        if docx is not None:
            try:
                doc = docx.Document(io.BytesIO(contents))
                paras = [p.text for p in doc.paragraphs if p.text.strip()]
                text = "\n".join(paras)
            except Exception as e:
                text = f"[Error reading Word document: {e}]"
        else:
            text = "[Error: python-docx not available on server]"
    else:
        # Standard code and text files
        try:
            text = contents.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = contents.decode("latin-1")
            except Exception:
                text = contents.decode("utf-8", errors="ignore")
                
    return {
        "filename": filename,
        "size": len(contents),
        "pages": pages,
        "type": ext,
        "content": text
    }

@app.get("/api/hardware_info")
def get_hardware_info():
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    total_vram_mb = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)) if has_cuda else 0.0
    allocated_vram_mb = (torch.cuda.memory_allocated() / (1024 * 1024)) if has_cuda else 0.0
    reserved_vram_mb = (torch.cuda.memory_reserved() / (1024 * 1024)) if has_cuda else 0.0

    return {
        "has_cuda": has_cuda,
        "device_name": device_name,
        "total_vram_mb": round(total_vram_mb, 2),
        "allocated_vram_mb": round(allocated_vram_mb, 2),
        "reserved_vram_mb": round(reserved_vram_mb, 2),
        "current_model": CURRENT_MODEL_ID,
        "cached_models": [
            {"id": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "name": "Qwen2.5-Coder (0.5B Instruct) • Fast Code & Math", "ready": True},
            {"id": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "name": "Qwen2.5-Coder (1.5B Instruct) • Expert Code & Engineering Math", "ready": True},
            {"id": "HuggingFaceTB/SmolLM2-135M-Instruct", "name": "SmolLM2 (135M Instruct) • Lightweight", "ready": True},
            {"id": "gpt2", "name": "OpenAI GPT-2 (124M Base) • Text Continuation", "ready": True},
            {"id": "distilbert/distilgpt2", "name": "DistilGPT-2 (82M Base) • Fast Continuation", "ready": True}
        ]
    }

@app.post("/api/load_model")
def load_model_endpoint(req: dict):
    model_name = req.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    hf_token = req.get("hf_token")
    try:
        get_runner(model_name, hf_token=hf_token)
        return {"status": "ok", "model": model_name}
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

@app.get("/api/stream_dual")
def stream_dual(
    prompt: str,
    max_tokens: int = 1024,
    budget_slots: int = 0,
    temperature: float = 0.7,
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    hf_token: Optional[str] = None
):
    """
    Streams side-by-side generation via Server-Sent Events (SSE).
    Interleaves Baseline and KVOps token generation step-by-step.
    """
    def event_generator():
        try:
            runner = get_runner(model_name, hf_token=hf_token)
        except Exception as e:
            err_payload = json.dumps({"error": f"Failed to load model: {str(e)}"})
            yield f"data: {err_payload}\n\n"
            return

        # Start generators
        gen_base = runner.generate_streaming_baseline(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        gen_kvops = runner.generate_streaming_kvops(
            prompt=prompt,
            max_new_tokens=max_tokens,
            budget_capacity=budget_slots,
            temperature=temperature
        )

        step = 0
        base_done = False
        kvops_done = False

        base_accum_text = ""
        kvops_accum_text = ""

        while not (base_done and kvops_done):
            step += 1
            base_data = None
            kvops_data = None

            if not base_done:
                try:
                    base_data = next(gen_base)
                    base_accum_text = base_data["full_text"]
                    if base_data.get("is_eos"):
                        base_done = True
                except StopIteration:
                    base_done = True

            if not kvops_done:
                try:
                    kvops_data = next(gen_kvops)
                    kvops_accum_text = kvops_data["full_text"]
                    if kvops_data.get("is_eos"):
                        kvops_done = True
                except StopIteration:
                    kvops_done = True

            if base_data is None and kvops_data is None:
                break

            # Calculate savings percentage
            b_kv = base_data["kv_cache_mb"] if base_data else 0.0
            k_kv = kvops_data["kv_cache_mb"] if kvops_data else 0.0
            saved_pct = 0.0
            if b_kv > 0:
                saved_pct = max(0.0, ((b_kv - k_kv) / b_kv) * 100.0)

            payload = {
                "step": step,
                "baseline": {
                    "token": base_data["token_text"] if base_data else "",
                    "full_text": base_accum_text,
                    "kv_cache_mb": round(b_kv, 4),
                    "gpu_allocated_mb": round(base_data["gpu_allocated_mb"], 1) if base_data else 0.0,
                    "step_time_ms": round(base_data["step_time_ms"], 2) if base_data else 0.0,
                    "total_tokens": base_data["total_context_tokens"] if base_data else 0,
                    "is_done": base_done
                },
                "kvops": {
                    "token": kvops_data["token_text"] if kvops_data else "",
                    "full_text": kvops_accum_text,
                    "kv_cache_mb": round(k_kv, 4),
                    "gpu_allocated_mb": round(kvops_data["gpu_allocated_mb"], 1) if kvops_data else 0.0,
                    "step_time_ms": round(kvops_data["step_time_ms"], 2) if kvops_data else 0.0,
                    "retained_slots": kvops_data["retained_slots"] if kvops_data else 0,
                    "tokens_evicted": kvops_data["tokens_evicted"] if kvops_data else 0,
                    "saved_pct": round(saved_pct, 1),
                    "is_done": kvops_done
                },
                "budget_slots": budget_slots,
                "is_complete": base_done and kvops_done
            }

            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.005)

        yield f"data: {json.dumps({'is_complete': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

class SingleStreamRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    mode: str = "baseline"
    max_tokens: int = 1024
    budget_slots: int = 0
    temperature: float = 0.7
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    hf_token: Optional[str] = None

def make_single_stream_generator(
    input_data: Any,
    mode: str = "baseline",
    max_tokens: int = 1024,
    budget_slots: int = 0,
    temperature: float = 0.7,
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    hf_token: Optional[str] = None
):
    try:
        runner = get_runner(model_name, hf_token=hf_token)
    except Exception as e:
        err_payload = json.dumps({"error": f"Failed to load model: {str(e)}"})
        yield f"data: {err_payload}\n\n"
        return

    if mode == "baseline":
        gen = runner.generate_streaming_baseline(
            prompt=input_data,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        for step_data in gen:
            payload = {
                "step": step_data["step"],
                "mode": "baseline",
                "token": step_data["token_text"],
                "full_text": step_data["full_text"],
                "kv_cache_mb": round(step_data["kv_cache_mb"], 4),
                "gpu_allocated_mb": round(step_data["gpu_allocated_mb"], 1),
                "step_time_ms": round(step_data["step_time_ms"], 2),
                "total_tokens": step_data["total_context_tokens"],
                "is_eos": step_data["is_eos"]
            }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.005)
    else:
        gen = runner.generate_streaming_kvops(
            prompt=input_data,
            max_new_tokens=max_tokens,
            budget_capacity=budget_slots,
            temperature=temperature
        )
        for step_data in gen:
            tot_tok = step_data.get("total_context_tokens", step_data["step"])
            evicted = step_data.get("tokens_evicted", 0)
            saved_pct = round((evicted / tot_tok * 100.0) if tot_tok > 0 else 0.0, 1)
            payload = {
                "step": step_data["step"],
                "mode": "kvops",
                "token": step_data["token_text"],
                "full_text": step_data["full_text"],
                "kv_cache_mb": round(step_data["kv_cache_mb"], 4),
                "gpu_allocated_mb": round(step_data["gpu_allocated_mb"], 1),
                "step_time_ms": round(step_data["step_time_ms"], 2),
                "total_tokens": tot_tok,
                "retained_slots": step_data["retained_slots"],
                "tokens_evicted": evicted,
                "saved_pct": saved_pct,
                "budget_slots": budget_slots,
                "daa_telemetry": step_data.get("daa_telemetry", {}),
                "is_eos": step_data["is_eos"]
            }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.005)

    yield f"data: {json.dumps({'is_complete': True})}\n\n"

@app.get("/api/stream_single")
def stream_single_get(
    prompt: Optional[str] = None,
    messages: Optional[str] = None,
    mode: str = "baseline",
    max_tokens: int = 1024,
    budget_slots: int = 0,
    temperature: float = 0.7,
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    hf_token: Optional[str] = None
):
    inp: Any = prompt or ""
    if messages:
        try:
            inp = json.loads(messages)
        except Exception:
            pass
    return StreamingResponse(
        make_single_stream_generator(inp, mode, max_tokens, budget_slots, temperature, model_name, hf_token),
        media_type="text/event-stream"
    )

@app.post("/api/stream_single")
def stream_single_post(req: SingleStreamRequest):
    inp: Any = req.messages if req.messages else (req.prompt or "")
    return StreamingResponse(
        make_single_stream_generator(inp, req.mode, req.max_tokens, req.budget_slots, req.temperature, req.model_name, req.hf_token),
        media_type="text/event-stream"
    )

# Serve Frontend HTML
static_dir = os.path.join(PROJECT_ROOT, "static")
os.makedirs(static_dir, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def index_page():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h3>LLM-KVOps Server Running. static/index.html not found.</h3>")

app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    port = 8000
    print(f"[*] Starting LLM-KVOps Dual Profiler Server on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
