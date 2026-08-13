"""Self-recovery system for autonomous fault detection and remediation.

Implements circuit breaker pattern, automatic model reloading,
graceful degradation, and recovery workflows without human intervention.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not available; GPU monitoring disabled")


class CircuitState(str, Enum):
    """Circuit breaker state machine."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RecoveryPolicy:
    """Configurable recovery thresholds and parameters."""

    # Model loading
    model_load_max_retries: int = 3
    model_load_initial_backoff_ms: int = 1000
    model_load_max_backoff_ms: int = 30000

    # Latency spike detection
    latency_spike_percentile: float = 95.0  # Use p95 as baseline
    latency_spike_multiplier: float = 2.5  # Alert if p99 > 2.5 * baseline_p95
    latency_spike_duration_samples: int = 10  # Consecutive spikes

    # OOM detection and batch size reduction
    oom_batch_size_reduction_factor: float = 0.5  # Reduce batch by 50%
    oom_min_batch_size: int = 1
    oom_recovery_cooldown_sec: int = 60

    # Error rate spike detection
    error_rate_threshold: float = 0.10  # 10% error rate alert
    error_rate_window_samples: int = 100

    # Circuit breaker
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout_sec: int = 30  # Time before half-open attempt
    half_open_success_threshold: int = 2  # Successes to close circuit

    # Quality degradation
    quality_degradation_threshold: float = 0.15  # 15% drop in avg confidence
    quality_check_window_samples: int = 50

    # Background health check
    health_check_interval_sec: int = 5


class CircuitBreaker:
    """Circuit breaker for fault isolation.

    State transitions:
        CLOSED -> OPEN: When failure_threshold exceeded
        OPEN -> HALF_OPEN: After recovery_timeout
        HALF_OPEN -> CLOSED: On successful requests
        HALF_OPEN -> OPEN: On failure in half-open state
    """

    def __init__(
        self,
        policy: RecoveryPolicy | None = None,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 30.0,
    ):
        """Initialize circuit breaker.

        Args:
            policy: RecoveryPolicy with thresholds. If None, uses keyword args.
            failure_threshold: Failures before opening (used if policy is None).
            recovery_timeout_sec: Seconds before half-open (used if policy is None).
        """
        if policy is None:
            policy = RecoveryPolicy(
                failure_threshold=failure_threshold,
                recovery_timeout_sec=int(recovery_timeout_sec),
            )
        self.policy = policy
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.RLock()
        logger.info("CircuitBreaker initialized: failure_threshold=%d, recovery_timeout=%ds",
                    self.policy.failure_threshold, self.policy.recovery_timeout_sec)

    def record_success(self) -> None:
        """Record successful request."""
        with self._lock:
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.policy.half_open_success_threshold:
                    self._state = CircuitState.CLOSED
                    self._success_count = 0
                    logger.info("Circuit closed: recovery successful")

    def record_failure(self) -> None:
        """Record failed request."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.policy.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning("Circuit opened: failure threshold (%d) exceeded",
                             self.policy.failure_threshold)

    def check_half_open_eligible(self) -> bool:
        """Check if circuit can attempt recovery (OPEN -> HALF_OPEN)."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                return False

            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.policy.recovery_timeout_sec:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("Circuit half-open: attempting recovery")
                return True

            return False

    def allow_request(self) -> bool:
        """Check if circuit allows requests; auto-transition OPEN -> HALF_OPEN.

        Returns:
            True if request is allowed, False if circuit is open.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.HALF_OPEN:
                return True
            # OPEN state: check if recovery timeout has elapsed
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.policy.recovery_timeout_sec:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("Circuit half-open: attempting recovery")
                return True
            return False

    def is_healthy(self) -> bool:
        """Alias for allow_request for backward compatibility."""
        return self.allow_request()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state


@dataclass
class RecoveryAction:
    """Logged recovery action with metadata."""
    timestamp: float = field(default_factory=time.time)
    action_type: str = ""  # e.g., "model_reload", "batch_size_reduction"
    reason: str = ""
    success: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'timestamp': self.timestamp,
            'action_type': self.action_type,
            'reason': self.reason,
            'success': self.success,
            'details': self.details,
        }


class RecoveryManager:
    """Autonomous recovery system with health monitoring and remediation.

    Detects faults and triggers corrective actions:
    - Model loading failures: Retry with exponential backoff
    - Latency spikes: Switch to quantized model
    - OOM errors: Reduce batch size dynamically
    - Quality degradation: Reload base model
    """

    def __init__(
        self,
        policy: RecoveryPolicy | None = None,
        model_loader: Callable[[], Any] | None = None,
        quantized_model_loader: Callable[[], Any] | None = None,
    ):
        """Initialize recovery manager.

        Args:
            policy: RecoveryPolicy. Uses defaults if None.
            model_loader: Callable to load full model.
            quantized_model_loader: Callable to load quantized fallback.
        """
        self.policy = policy or RecoveryPolicy()
        self.model_loader = model_loader
        self.quantized_model_loader = quantized_model_loader

        self._lock = threading.RLock()
        self._circuit_breaker = CircuitBreaker(self.policy)
        self._recovery_actions: list[RecoveryAction] = []

        # Batch size tracking
        self._current_batch_size = 32
        self._oom_last_recovery_time = 0.0

        # Latency spike tracking
        self._latency_spike_count = 0
        self._baseline_latency_p95 = 100.0  # ms

        # Quality degradation tracking
        self._confidence_history: list[float] = []
        self._baseline_confidence = 0.8

        # Health check thread
        self._health_thread: threading.Thread | None = None
        self._health_thread_stop = threading.Event()

        logger.info("RecoveryManager initialized with policy: %s", self.policy)

    def start_health_monitoring(self) -> None:
        """Start background health check thread."""
        if self._health_thread is not None and self._health_thread.is_alive():
            logger.warning("Health monitoring already running")
            return

        self._health_thread_stop.clear()
        self._health_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
        )
        self._health_thread.start()
        logger.info("Health monitoring thread started")

    def stop_health_monitoring(self) -> None:
        """Stop background health check thread."""
        self._health_thread_stop.set()
        if self._health_thread:
            self._health_thread.join(timeout=5.0)
            logger.info("Health monitoring thread stopped")

    def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._health_thread_stop.wait(self.policy.health_check_interval_sec):
            try:
                # Attempt circuit recovery
                self._circuit_breaker.check_half_open_eligible()

                # Check for GPU OOM
                if TORCH_AVAILABLE:
                    self._check_gpu_memory()

            except Exception:
                logger.exception("Error in health check loop")

    def handle_model_loading_failure(self, error: Exception) -> bool:
        """Handle model loading failure with retry logic.

        Args:
            error: Exception from model loading.

        Returns:
            True if recovery succeeded, False otherwise.
        """
        with self._lock:
            backoff_ms = self.policy.model_load_initial_backoff_ms

            for attempt in range(self.policy.model_load_max_retries):
                logger.warning("Model loading failed (attempt %d/%d): %s",
                             attempt + 1, self.policy.model_load_max_retries, error)

                time.sleep(backoff_ms / 1000.0)

                try:
                    if self.model_loader:
                        self.model_loader()
                    self._log_recovery_action(
                        action_type="model_reload",
                        reason=f"Loading failure after {attempt} retries",
                        success=True,
                        details={"attempt": attempt + 1},
                    )
                    logger.info("Model loaded successfully on attempt %d", attempt + 1)
                    return True
                except Exception as e:
                    logger.error("Retry attempt %d failed: %s", attempt + 1, e)
                    backoff_ms = min(
                        backoff_ms * 2,
                        self.policy.model_load_max_backoff_ms,
                    )

            # All retries exhausted
            self._log_recovery_action(
                action_type="model_reload",
                reason="All retry attempts exhausted",
                success=False,
                details={"attempts": self.policy.model_load_max_retries},
            )
            return False

    def handle_latency_spike(self, current_p99_ms: float, baseline_p95_ms: float) -> None:
        """Detect and respond to latency spikes.

        If p99 consistently exceeds baseline_p95 * multiplier,
        switch to quantized model for degraded but faster inference.

        Args:
            current_p99_ms: Current P99 latency.
            baseline_p95_ms: Baseline P95 latency.
        """
        with self._lock:
            threshold = baseline_p95_ms * self.policy.latency_spike_multiplier

            if current_p99_ms > threshold:
                self._latency_spike_count += 1
                logger.warning("Latency spike detected: p99=%.1fms > threshold=%.1fms (count=%d)",
                             current_p99_ms, threshold, self._latency_spike_count)

                if (self._latency_spike_count >= self.policy.latency_spike_duration_samples
                        and self.quantized_model_loader):
                    try:
                        self.quantized_model_loader()
                        self._log_recovery_action(
                            action_type="quantized_model_switch",
                            reason="Persistent latency spikes detected",
                            success=True,
                            details={
                                "p99_ms": round(current_p99_ms, 2),
                                "threshold_ms": round(threshold, 2),
                                "spike_count": self._latency_spike_count,
                            },
                        )
                        logger.info("Switched to quantized model due to latency spikes")
                        self._latency_spike_count = 0
                    except Exception as e:
                        logger.error("Failed to load quantized model: %s", e)
                        self._circuit_breaker.record_failure()
            else:
                self._latency_spike_count = max(0, self._latency_spike_count - 1)

    def handle_oom_error(self) -> None:
        """Handle out-of-memory error by reducing batch size.

        Implements exponential backoff in batch size with cooldown.
        """
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self._oom_last_recovery_time

            # Cooldown prevents aggressive thrashing
            if elapsed < self.policy.oom_recovery_cooldown_sec:
                logger.warning("OOM recovery in cooldown period (%.1fs remaining)",
                             self.policy.oom_recovery_cooldown_sec - elapsed)
                return

            old_batch_size = self._current_batch_size
            self._current_batch_size = max(
                self.policy.oom_min_batch_size,
                int(self._current_batch_size * self.policy.oom_batch_size_reduction_factor),
            )

            self._log_recovery_action(
                action_type="batch_size_reduction",
                reason="Out-of-memory detected",
                success=True,
                details={
                    "old_batch_size": old_batch_size,
                    "new_batch_size": self._current_batch_size,
                    "reduction_factor": self.policy.oom_batch_size_reduction_factor,
                },
            )

            self._oom_last_recovery_time = current_time
            logger.warning("Reduced batch size: %d -> %d",
                         old_batch_size, self._current_batch_size)

    def handle_quality_degradation(self, avg_confidence: float) -> None:
        """Detect and respond to prediction quality degradation.

        Triggers model reload if average confidence drops beyond threshold.

        Args:
            avg_confidence: Average prediction confidence (0-1).
        """
        with self._lock:
            self._confidence_history.append(avg_confidence)
            if len(self._confidence_history) > self.policy.quality_check_window_samples:
                self._confidence_history.pop(0)

            if len(self._confidence_history) < 10:
                return

            recent_avg = sum(self._confidence_history[-10:]) / 10.0
            degradation = (self._baseline_confidence - recent_avg) / self._baseline_confidence

            if degradation > self.policy.quality_degradation_threshold:
                logger.warning("Quality degradation detected: %.3f -> %.3f (degradation=%.1f%%)",
                             self._baseline_confidence, recent_avg, degradation * 100)

                if self.model_loader:
                    try:
                        self.model_loader()
                        self._log_recovery_action(
                            action_type="model_reload_quality",
                            reason="Prediction quality degradation detected",
                            success=True,
                            details={
                                "baseline_confidence": round(self._baseline_confidence, 3),
                                "recent_confidence": round(recent_avg, 3),
                                "degradation_percent": round(degradation * 100, 2),
                            },
                        )
                        logger.info("Reloaded model due to quality degradation")
                    except Exception as e:
                        logger.error("Failed to reload model: %s", e)
                        self._circuit_breaker.record_failure()

    def _check_gpu_memory(self) -> None:
        """Check GPU memory and trigger OOM recovery if needed."""
        if not TORCH_AVAILABLE:
            return

        try:
            if not torch.cuda.is_available():
                return

            used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            if used > 0.95:  # 95% utilization threshold
                logger.warning("GPU memory utilization high: %.1f%%", used * 100)
                self.handle_oom_error()
        except Exception as e:
            logger.debug("GPU memory check failed: %s", e)

    def _log_recovery_action(
        self,
        action_type: str,
        reason: str,
        success: bool,
        details: dict[str, Any],
    ) -> None:
        """Log recovery action for audit trail.

        Args:
            action_type: Type of recovery action.
            reason: Human-readable reason.
            success: Whether action succeeded.
            details: Action-specific details.
        """
        action = RecoveryAction(
            action_type=action_type,
            reason=reason,
            success=success,
            details=details,
        )
        with self._lock:
            self._recovery_actions.append(action)
            logger.info("Recovery action logged: %s", action.to_dict())

    def get_batch_size(self) -> int:
        """Get current recommended batch size."""
        with self._lock:
            return self._current_batch_size

    def is_circuit_healthy(self) -> bool:
        """Check if circuit breaker allows requests."""
        return self._circuit_breaker.is_healthy()

    def get_circuit_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._circuit_breaker.state

    def get_recovery_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent recovery actions.

        Args:
            limit: Maximum number of actions to return.

        Returns:
            List of recovery action dictionaries.
        """
        with self._lock:
            return [action.to_dict() for action in self._recovery_actions[-limit:]]

    def reset(self) -> None:
        """Reset recovery manager state."""
        with self._lock:
            self._circuit_breaker = CircuitBreaker(self.policy)
            self._recovery_actions.clear()
            self._current_batch_size = 32
            self._latency_spike_count = 0
            self._confidence_history.clear()
            logger.info("RecoveryManager reset to initial state")


__all__ = [
    'CircuitBreaker',
    'CircuitState',
    'RecoveryAction',
    'RecoveryManager',
    'RecoveryPolicy',
]
