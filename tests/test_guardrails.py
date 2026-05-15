"""Tests for AI guardrails: PII redaction, confidence checks, validation."""

from src.guardrails import apply_guardrails, redact_pii, check_confidence, validate_output


class TestPIIRedaction:
    def test_ssn_redacted(self):
        text = "Client SSN is 123-45-6789 for account."
        redacted, detected = redact_pii(text)
        assert "123-45-6789" not in redacted
        assert "[REDACTED_SSN]" in redacted
        assert "ssn" in detected

    def test_credit_card_redacted(self):
        text = "Card number 4111-1111-1111-1111 on file."
        redacted, detected = redact_pii(text)
        assert "4111" not in redacted
        assert "credit_card" in detected

    def test_email_redacted(self):
        text = "Contact john.doe@example.com for details."
        redacted, detected = redact_pii(text)
        assert "john.doe@example.com" not in redacted
        assert "email" in detected

    def test_clean_text_passes(self):
        text = "Revenue increased by 15% in Q3."
        redacted, detected = redact_pii(text)
        assert redacted == text
        assert detected == []

    def test_multiple_pii_types(self):
        text = "SSN 123-45-6789, email test@test.com"
        _, detected = redact_pii(text)
        assert "ssn" in detected
        assert "email" in detected


class TestConfidenceCheck:
    def test_high_confidence_passes(self):
        assert check_confidence(0.95, threshold=0.7) == []

    def test_low_confidence_flagged(self):
        flags = check_confidence(0.3, threshold=0.7)
        assert len(flags) == 1
        assert "LOW_CONFIDENCE" in flags[0]

    def test_boundary_passes(self):
        assert check_confidence(0.7, threshold=0.7) == []


class TestOutputValidation:
    def test_valid_label(self):
        assert validate_output("positive") == []
        assert validate_output("negative") == []
        assert validate_output("neutral") == []

    def test_invalid_label(self):
        flags = validate_output("INVALID")
        assert len(flags) == 1
        assert "INVALID_LABEL" in flags[0]


class TestFullGuardrails:
    def test_clean_high_confidence_passes(self):
        result = apply_guardrails("Revenue grew 20%", "positive", 0.95)
        assert result.passed is True
        assert result.flags == []
        assert result.pii_detected == []

    def test_pii_still_passes_but_detected(self):
        result = apply_guardrails("SSN 123-45-6789", "neutral", 0.9)
        assert result.passed is True  # PII is redacted, not a failure
        assert "ssn" in result.pii_detected
        assert "123-45-6789" not in result.redacted_text

    def test_low_confidence_fails(self):
        result = apply_guardrails("Ambiguous statement", "neutral", 0.3)
        assert result.passed is False
        assert any("LOW_CONFIDENCE" in f for f in result.flags)
