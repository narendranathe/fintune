"""Tests for FastAPI serving endpoint schemas."""

import pytest
from pydantic import ValidationError

from src.serve import HealthResponse, PredictRequest, PredictResponse


class TestPredictRequest:
    def test_valid_request(self):
        req = PredictRequest(text="Revenue increased 20%")
        assert req.text == "Revenue increased 20%"
        assert req.confidence_threshold == 0.7

    def test_custom_threshold(self):
        req = PredictRequest(text="Test", confidence_threshold=0.9)
        assert req.confidence_threshold == 0.9

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(text="")

    def test_threshold_bounds(self):
        with pytest.raises(ValidationError):
            PredictRequest(text="Test", confidence_threshold=1.5)
        with pytest.raises(ValidationError):
            PredictRequest(text="Test", confidence_threshold=-0.1)


class TestPredictResponse:
    def test_valid_response(self):
        resp = PredictResponse(
            label="positive",
            confidence=0.95,
            guardrails_passed=True,
            flags=[],
            pii_detected=[],
            latency_ms=12.5,
        )
        assert resp.label == "positive"
        assert resp.guardrails_passed is True

    def test_failed_guardrails(self):
        resp = PredictResponse(
            label="neutral",
            confidence=0.3,
            guardrails_passed=False,
            flags=["LOW_CONFIDENCE: 0.300 < 0.700"],
            pii_detected=["ssn"],
            latency_ms=15.0,
        )
        assert resp.guardrails_passed is False
        assert len(resp.flags) == 1
        assert len(resp.pii_detected) == 1


class TestHealthResponse:
    def test_healthy(self):
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_path="outputs/model",
            health_score=100.0,
            circuit_breaker_state="CLOSED",
            total_requests=0,
            error_rate=0.0,
        )
        assert resp.status == "healthy"
        assert resp.health_score == 100.0
        assert resp.circuit_breaker_state == "CLOSED"

    def test_degraded(self):
        resp = HealthResponse(
            status="degraded",
            model_loaded=False,
            model_path="outputs/model",
            health_score=25.0,
            circuit_breaker_state="OPEN",
            total_requests=100,
            error_rate=0.15,
        )
        assert resp.status == "degraded"
        assert resp.error_rate == 0.15
