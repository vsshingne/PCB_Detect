from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .config import (
    ALLOWED_ORIGINS,
    CLASS_NAMES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    FRIENDLY_CLASS_NAMES,
    MAX_UPLOAD_SIZE_BYTES,
    MODEL_PATH,
    SUPPORTED_EXTENSIONS,
)
from .inference import InvalidImageError, ModelService, ModelUnavailableError
from .schemas import ClassesResponse, ErrorResponse, HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO)
service = ModelService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load()
    yield


app = FastAPI(
    title="PCB Defect Detection API",
    description="Local inference API for the six-class custom YOLOv8 PCB defect model.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health", response_model=HealthResponse, summary="Check API and model readiness")
async def health() -> HealthResponse:
    loaded = service.ready
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_name=MODEL_PATH.name,
        prediction_available=loaded,
        model_error=service.error,
    )


@app.get("/api/classes", response_model=ClassesResponse, summary="List configured PCB defect classes")
async def classes() -> ClassesResponse:
    return ClassesResponse(classes=[
        {"class_id": index, "model_name": model_name, "display_name": FRIENDLY_CLASS_NAMES[index]}
        for index, model_name in enumerate(CLASS_NAMES)
    ])


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or unsupported image."},
        413: {"model": ErrorResponse, "description": "Image exceeds the configured upload limit."},
        503: {"model": ErrorResponse, "description": "The trained model is unavailable."},
        500: {"model": ErrorResponse, "description": "Inference failed."},
    },
    summary="Detect PCB defects in an uploaded image",
)
async def predict(
    file: UploadFile = File(..., description="JPEG, PNG, or WEBP PCB image"),
    confidence_threshold: float = Form(DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
) -> PredictionResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image format. Upload a JPEG, PNG, or WEBP image.")
    content = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    await file.close()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Image exceeds the {max_mb} MB upload limit.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    try:
        result = await run_in_threadpool(service.predict, content, confidence_threshold)
        return PredictionResponse(**result)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"message": "PCB Defect Detection API", "docs": "/docs"}
