"""FastAPI inference endpoint with guardrails, monitoring, and self-recovery.

Production-grade serving with:
- Real-time system monitoring and health scoring
- Circuit breaker pattern for fault tolerance
- Automatic model reloading on failures
- PII guardrails and confidence thresholding
- Batch prediction with throughput tracking
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .guardrails import apply_guardrails
from .monitor import SystemMonitor
from .self_recovery import CircuitBreaker, RecoveryManager, RecoveryPolicy

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("FINTUNE_MODEL_PATH", "outputs/fintune-financial")
FALLBACK_MODEL_PATH = os.getenv("FINTUNE_FALLBACK_MODEL_PATH", "outputs/fintune-financial-quantized")

# Global state
_classifier = None
_monitor = SystemMonitor()
_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout_sec=30.0)
_recovery_manager = RecoveryManager(policy=RecoveryPolicy())


def _load_model(model_path: str) -> Any:
    """Load model pipeline with error handling."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup with recovery, cleanup on shutdown."""
    global _classifier

    logger.info("Loading model from %s...", MODEL_PATH)
    try:
        _classifier = _load_model(MODEL_PATH)
        logger.info("Primary model loaded successfully.")
    except Exception as e:
        logger.error("Primary model failed: %s. Trying fallback...", e)
        try:
            _classifier = _load_model(FALLBACK_MODEL_PATH)
            logger.warning("Fallback model loaded from %s", FALLBACK_MODEL_PATH)
        except Exception as fallback_err:
            logger.critical("All model loading failed: %s", fallback_err)
            _classifier = None

    _monitor.reset()
    yield
    _classifier = None
    _monitor.reset()


app = FastAPI(
    title="FinTune API",
    description="Financial text sentiment classification with AI guardrails, monitoring, and self-recovery",
    version="0.2.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """Input schema for prediction endpoint."""
    text: str = Field(..., min_length=1, max_length=2048, description="Financial text to classify")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """Output schema with guardrail metadata."""
    label: str
    confidence: float
    guardrails_passed: bool
    flags: list[str]
    pii_detected: list[str]
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response with monitoring metrics."""
    status: str
    model_loaded: bool
    model_path: str
    health_score: float
    circuit_breaker_state: str
    total_requests: int
    error_rate: float


class MetricsResponse(BaseModel):
    """System metrics for monitoring dashboards."""
    metrics: dict


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with monitoring and circuit breaker state."""
    health_score = _monitor.compute_health_score()
    return HealthResponse(
        status="healthy" if _classifier and health_score > 50 else "degraded",
        model_loaded=_classifier is not None,
        model_path=MODEL_PATH,
        health_score=health_score,
        circuit_breaker_state=_circuit_breaker.state.value,
        total_requests=_monitor.request_count,
        error_rate=_monitor.get_error_rate(),
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Export monitoring metrics for dashboards."""
    return MetricsResponse(metrics=_monitor.export_metrics())


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Classify financial text with guardrails and monitoring."""
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not _circuit_breaker.allow_request():
        _monitor.record_error()
        raise HTTPException(
            status_code=503,
            detail="Circuit breaker OPEN - service recovering",
        )

    start = time.perf_counter()

    try:
        results = _classifier(request.text)
        top = max(results[0], key=lambda x: x["score"])

        guard_result = apply_guardrails(
            text=request.text,
            label=top["label"],
            confidence=top["score"],
            confidence_threshold=request.confidence_threshold,
        )

        latency = (time.perf_counter() - start) * 1000

        _circuit_breaker.record_success()
        _monitor.record_request(latency_ms=latency)
        _monitor.record_prediction(guard_result.label)

        return PredictResponse(
            label=guard_result.label,
            confidence=round(top["score"], 4),
            guardrails_passed=guard_result.passed,
            flags=guard_result.flags,
            pii_detected=guard_result.pii_detected,
            latency_ms=round(latency, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        _circuit_breaker.record_failure()
        _monitor.record_error()
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Inference error") from e


@app.post("/predict/batch", response_model=list[PredictResponse])
async def predict_batch(requests: list[PredictRequest]):
    """Batch prediction with per-request monitoring and circuit breaking."""
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not _circuit_breaker.allow_request():
        raise HTTPException(status_code=503, detail="Circuit breaker OPEN")

    responses = []
    for req in requests:
        start = time.perf_counter()
        try:
            results = _classifier(req.text)
            top = max(results[0], key=lambda x: x["score"])
            guard_result = apply_guardrails(
                text=req.text,
                label=top["label"],
                confidence=top["score"],
                confidence_threshold=req.confidence_threshold,
            )
            latency = (time.perf_counter() - start) * 1000
            _circuit_breaker.record_success()
            _monitor.record_request(latency_ms=latency)
            _monitor.record_prediction(guard_result.label)
            responses.append(PredictResponse(
                label=guard_result.label,
                confidence=round(top["score"], 4),
                guardrails_passed=guard_result.passed,
                flags=guard_result.flags,
                pii_detected=guard_result.pii_detected,
                latency_ms=round(latency, 2),
            ))
        except Exception as e:
            _circuit_breaker.record_failure()
            _monitor.record_error()
            logger.error("Batch item failed: %s", e)
            responses.append(PredictResponse(
                label="error",
                confidence=0.0,
                guardrails_passed=False,
                flags=["INFERENCE_ERROR: " + str(e)],
                pii_detected=[],
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            ))

    return responses
