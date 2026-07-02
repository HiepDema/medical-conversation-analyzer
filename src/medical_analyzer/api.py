"""FastAPI app — upload audio file and get structured medical JSON back."""

from __future__ import annotations

import io
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .config import load_config
from .pipeline import MedicalPipeline

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    log.info("initializing pipeline")
    app.state.pipeline = MedicalPipeline(cfg)
    log.info("pipeline ready")
    yield
    app.state.pipeline.close()


app = FastAPI(
    title="Medical Conversation Analyzer",
    description="Upload Vietnamese medical conversations, get structured JSON output.",
    lifespan=lifespan,
)


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile) -> JSONResponse:
    """Upload an audio file and analyze the medical conversation.

    Returns JSON with transcript and 3 classified fields:
    - qua_trinh_benh_ly
    - tien_su_benh_nhan
    - tien_su_gia_dinh
    """
    if not file.filename:
        raise HTTPException(400, "No file provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        raise HTTPException(400, f"Unsupported format: {suffix}. Use .wav, .mp3, .flac, .ogg, .m4a")

    # Save uploaded file to temp location
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pipeline: MedicalPipeline = app.state.pipeline
        result = pipeline.analyze_file(tmp_path)
        return JSONResponse(content=result.to_dict())
    except Exception as e:
        log.exception("analysis failed")
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
