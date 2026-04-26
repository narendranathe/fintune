"""AI guardrails: output validation, PII redaction, and compliance checks.

Implements responsible AI practices for financial text classification:
- PII detection and redaction (SSN, credit card, email, phone)
- Confidence thresholding to avoid low-certainty predictions
- Output schema validation
- Audit logging for compliance
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# PII patterns for financial text
PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+1[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b"),
    "account_number": re.compile(r"\b\d{8,17}\b"),
}


@dataclass
class GuardrailResult:
    """Result of guardrail checks on a prediction."""

    text: str
    label: str
    confidence: float
    passed: bool
    redacted_text: str
    pii_detected: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Scan text for PII and replace with redaction markers.

    Args:
        text: Input text to scan.

    Returns:
        Tuple of (redacted text, list of PII types found).
    """
    detected = []
    redacted = text

    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(redacted):
            detected.append(pii_type)
            redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)

    if detected:
        logger.warning("PII detected and redacted: %s", detected)

    return redacted, detected


def check_confidence(confidence: float, threshold: float = 0.7) -> list[str]:
    """Flag predictions below confidence threshold.

    In financial contexts, low-confidence predictions should be
    escalated to human review rather than served automatically.

    Args:
        confidence: Model prediction confidence (0-1).
        threshold: Minimum acceptable confidence.

    Returns:
        List of flag strings (empty if passed).
    """
    flags = []
    if confidence < threshold:
        flags.append(f"LOW_CONFIDENCE: {confidence:.3f} < {threshold}")
    return flags


def validate_output(label: str, valid_labels: set[str] | None = None) -> list[str]:
    """Ensure model output is a valid label.

    Args:
        label: Predicted label string.
        valid_labels: Set of acceptable labels.

    Returns:
        List of flag strings.
    """
    if valid_labels is None:
        valid_labels = {"negative", "neutral", "positive"}

    flags = []
    if label not in valid_labels:
        flags.append(f"INVALID_LABEL: '{label}' not in {valid_labels}")

    return flags


def apply_guardrails(
    text: str,
    label: str,
    confidence: float,
    confidence_threshold: float = 0.7,
) -> GuardrailResult:
    """Run all guardrail checks on a single prediction.

    Pipeline:
        1. PII redaction on input text
        2. Confidence threshold check
        3. Output label validation
        4. Aggregate pass/fail decision

    Args:
        text: Original input text.
        label: Model predicted label.
        confidence: Prediction confidence score.
        confidence_threshold: Minimum confidence to pass.

    Returns:
        GuardrailResult with pass/fail and details.
    """
    redacted_text, pii_detected = redact_pii(text)

    flags = []
    flags.extend(check_confidence(confidence, confidence_threshold))
    flags.extend(validate_output(label))

    passed = len(flags) == 0

    result = GuardrailResult(
        text=text,
        label=label,
        confidence=confidence,
        passed=passed,
        redacted_text=redacted_text,
        pii_detected=pii_detected,
        flags=flags,
    )

    if not passed:
        logger.warning("Guardrail FAILED: %s", flags)

    return result
