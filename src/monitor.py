"""Real-time system monitoring for FinTune inference pipeline.

Production-grade observability: health checks, latency tracking,
throughput metrics, drift detection, and alerting hooks.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Latency percentile tracking."""
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0


@dataclass
class HealthMetrics:
    """Aggregated health indicators."""
    health_score: float = 100.0  # 0-100
    request_count: int = 0
    error_count: int = 0
    error_rate: float = 0.0  # 0-1
    throughput_req_sec: float = 0.0
    latency_metrics: LatencyMetrics = field(default_factory=LatencyMetrics)
    model_drift_detected: bool = False
    drift_kl_divergence: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)


class SystemMonitor:
    """Singleton system monitoring with real-time metrics tracking.

    Thread-safe metrics collection for latency, throughput, errors,
    drift detection, and resource utilization.
    """

    _instance: Optional[SystemMonitor] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> SystemMonitor:
        """Enforce singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        window_size: int = 3600,  # 1 hour in seconds
        percentile_window: int = 1000,  # samples for p50/p95/p99
        drift_detection_window: int = 500,  # samples for distribution
        kl_divergence_threshold: float = 0.15,
    ):
        """Initialize monitor with configurable windows.

        Args:
            window_size: Time window in seconds for time-series metrics.
            percentile_window: Max samples to track for latency percentiles.
            drift_detection_window: Max samples for prediction drift detection.
            kl_divergence_threshold: Alert threshold for model drift.
        """
        if self._initialized:
            return

        self._window_size = window_size
        self._percentile_window = percentile_window
        self._drift_detection_window = drift_detection_window
        self._kl_divergence_threshold = kl_divergence_threshold

        # Thread-safe counters
        self._lock_data = threading.RLock()
        self._request_count = 0
        self._error_count = 0

        # Latency tracking (sliding window)
        self._latencies: deque[tuple[float, float]] = deque(maxlen=percentile_window)

        # Throughput tracking (requests per time window)
        self._request_times: deque[float] = deque()

        # Prediction distribution for drift detection
        self._prediction_distribution: deque[tuple[str, float]] = deque(
            maxlen=drift_detection_window
        )
        self._baseline_distribution: dict[str, float] = {}

        # Resource tracking
        self._memory_percent = 0.0
        self._gpu_memory_percent = 0.0

        # Recovery hooks
        self._on_drift_detected: Optional[Callable[[HealthMetrics], None]] = None
        self._on_latency_spike: Optional[Callable[[HealthMetrics], None]] = None
        self._on_error_spike: Optional[Callable[[HealthMetrics], None]] = None

        self._initialized = True
        logger.info("SystemMonitor initialized: window_size=%ds, drift_threshold=%.3f",
                   window_size, kl_divergence_threshold)

    @property
    def request_count(self) -> int:
        """Total requests recorded."""
        return self._request_count

    def record_request(
        self,
        latency_ms: float,
        label: str = "unknown",
        confidence: float = 0.0,
        error: bool = False,
    ) -> None:
        """Record a single prediction request.

        Args:
            latency_ms: Request latency in milliseconds.
            label: Predicted label.
            confidence: Prediction confidence (0-1).
            error: Whether request failed.
        """
        with self._lock_data:
            current_time = time.time()

            # Update counters
            self._request_count += 1
            if error:
                self._error_count += 1

            # Track latency
            self._latencies.append((current_time, latency_ms))

            # Track request timing for throughput
            self._request_times.append(current_time)

            # Track prediction distribution for drift
            if label != "unknown":
                self._prediction_distribution.append((label, confidence))

    def record_error(self) -> None:
        """Record an error without a full request context."""
        with self._lock_data:
            self._error_count += 1

    def record_prediction(self, label: str, confidence: float = 1.0) -> None:
        """Record a prediction label for drift detection.

        Args:
            label: Predicted label string.
            confidence: Prediction confidence (0-1).
        """
        with self._lock_data:
            self._prediction_distribution.append((label, confidence))

    def update_resource_usage(self, memory_percent: float, gpu_memory_percent: float = 0.0) -> None:
        """Update system resource utilization.

        Args:
            memory_percent: RAM usage percentage (0-100).
            gpu_memory_percent: GPU memory usage percentage (0-100).
        """
        with self._lock_data:
            self._memory_percent = min(100.0, max(0.0, memory_percent))
            self._gpu_memory_percent = min(100.0, max(0.0, gpu_memory_percent))

    def set_baseline_distribution(self, distribution: dict[str, float]) -> None:
        """Set expected prediction distribution for drift detection.

        Args:
            distribution: Baseline class distribution {label: probability}.
        """
        with self._lock_data:
            self._baseline_distribution = distribution.copy()
            logger.info("Baseline distribution set: %s", distribution)

    def set_recovery_hooks(
        self,
        on_drift: Optional[Callable[[HealthMetrics], None]] = None,
        on_latency_spike: Optional[Callable[[HealthMetrics], None]] = None,
        on_error_spike: Optional[Callable[[HealthMetrics], None]] = None,
    ) -> None:
        """Register recovery action callbacks.

        Args:
            on_drift: Called when model drift detected.
            on_latency_spike: Called when latency exceeds threshold.
            on_error_spike: Called when error rate spikes.
        """
        with self._lock_data:
            self._on_drift_detected = on_drift
            self._on_latency_spike = on_latency_spike
            self._on_error_spike = on_error_spike
            logger.info("Recovery hooks registered")

    def _calculate_latency_percentiles(self) -> LatencyMetrics:
        """Calculate latency metrics from recorded samples."""
        if not self._latencies:
            return LatencyMetrics()

        latencies = sorted([lat for _, lat in self._latencies])
        n = len(latencies)

        def percentile(p: float) -> float:
            idx = int(n * p / 100.0)
            return latencies[min(idx, n - 1)]

        return LatencyMetrics(
            p50=percentile(50),
            p95=percentile(95),
            p99=percentile(99),
            min=min(latencies),
            max=max(latencies),
            mean=sum(latencies) / n,
        )

    def _calculate_error_rate(self) -> float:
        """Calculate error rate (0-1)."""
        if self._request_count == 0:
            return 0.0
        return self._error_count / self._request_count

    def _calculate_throughput(self) -> float:
        """Calculate throughput in requests per second."""
        if not self._request_times:
            return 0.0

        # Clean old timestamps outside window
        current_time = time.time()
        while self._request_times and (current_time - self._request_times[0]) > self._window_size:
            self._request_times.popleft()

        if not self._request_times:
            return 0.0

        time_span = current_time - self._request_times[0]
        if time_span == 0:
            return 0.0

        return len(self._request_times) / time_span

    def _detect_drift(self) -> tuple[bool, float]:
        """Detect model drift using KL-divergence from baseline.

        Returns:
            (is_drifted, kl_divergence_value)
        """
        if not self._baseline_distribution or len(self._prediction_distribution) < 100:
            return False, 0.0

        # Compute empirical distribution from recent predictions
        counts: dict[str, int] = {}
        for label, _ in self._prediction_distribution:
            counts[label] = counts.get(label, 0) + 1

        total = len(self._prediction_distribution)
        empirical = {label: count / total for label, count in counts.items()}

        # KL-divergence: sum(P(x) * log(P(x) / Q(x)))
        kl_divergence = 0.0
        for label, p in self._baseline_distribution.items():
            q = empirical.get(label, 1e-6)
            if p > 0:
                kl_divergence += p * (math.log(p / max(q, 1e-6)))

        is_drifted = kl_divergence > self._kl_divergence_threshold
        return is_drifted, kl_divergence

    def _compute_health_score(
        self,
        error_rate: float,
        latency_p99: float,
        throughput: float,
    ) -> float:
        """Compute composite health score (0-100).

        Higher error rate, latency, and lower throughput reduce score.

        Args:
            error_rate: Error rate (0-1).
            latency_p99: P99 latency in milliseconds.
            throughput: Throughput in requests/sec.

        Returns:
            Health score 0-100.
        """
        # Start with 100
        score = 100.0

        # Error rate penalty: 0 error = 0 penalty, 5% error = -25 penalty
        score -= min(50.0, error_rate * 500)

        # Latency penalty: 100ms = 0 penalty, 500ms = -25 penalty
        score -= min(25.0, max(0.0, (latency_p99 - 100) / 16))

        # Throughput penalty: < 1 req/sec = -25 penalty
        if throughput < 1.0:
            score -= 25.0 * (1.0 - throughput)

        return max(0.0, min(100.0, score))

    def get_metrics(self) -> HealthMetrics:
        """Get current health metrics snapshot.

        Returns:
            HealthMetrics with all current measurements.
        """
        with self._lock_data:
            latency_metrics = self._calculate_latency_percentiles()
            error_rate = self._calculate_error_rate()
            throughput = self._calculate_throughput()
            is_drifted, kl_div = self._detect_drift()

            health_score = self._compute_health_score(
                error_rate=error_rate,
                latency_p99=latency_metrics.p99,
                throughput=throughput,
            )

            metrics = HealthMetrics(
                health_score=health_score,
                request_count=self._request_count,
                error_count=self._error_count,
                error_rate=error_rate,
                throughput_req_sec=throughput,
                latency_metrics=latency_metrics,
                model_drift_detected=is_drifted,
                drift_kl_divergence=kl_div,
                memory_percent=self._memory_percent,
                gpu_memory_percent=self._gpu_memory_percent,
                timestamp=time.time(),
            )

            return metrics

    def compute_health_score(self) -> float:
        """Public API: compute current health score (0-100).

        Returns:
            Composite health score.
        """
        metrics = self.get_metrics()
        return metrics.health_score

    def get_error_rate(self) -> float:
        """Public API: get current error rate (0-1).

        Returns:
            Error rate as fraction.
        """
        with self._lock_data:
            return self._calculate_error_rate()

    def export_metrics(self) -> dict:
        """Public API: export metrics as dict for JSON serialization.

        Returns:
            Dict of current metrics.
        """
        return json.loads(self.export_metrics_json())

    def export_metrics_json(self) -> str:
        """Export metrics as JSON for dashboard integration.

        Returns:
            JSON string of current metrics.
        """
        metrics = self.get_metrics()
        data = {
            'health_score': round(metrics.health_score, 2),
            'request_count': metrics.request_count,
            'error_count': metrics.error_count,
            'error_rate': round(metrics.error_rate, 4),
            'throughput_req_sec': round(metrics.throughput_req_sec, 2),
            'latency_ms': {
                'p50': round(metrics.latency_metrics.p50, 2),
                'p95': round(metrics.latency_metrics.p95, 2),
                'p99': round(metrics.latency_metrics.p99, 2),
                'min': round(metrics.latency_metrics.min, 2),
                'max': round(metrics.latency_metrics.max, 2),
                'mean': round(metrics.latency_metrics.mean, 2),
            },
            'drift': {
                'detected': metrics.model_drift_detected,
                'kl_divergence': round(metrics.drift_kl_divergence, 4),
            },
            'resources': {
                'memory_percent': round(metrics.memory_percent, 2),
                'gpu_memory_percent': round(metrics.gpu_memory_percent, 2),
            },
            'timestamp': metrics.timestamp,
        }
        return json.dumps(data, indent=2)

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock_data:
            self._request_count = 0
            self._error_count = 0
            self._latencies.clear()
            self._request_times.clear()
            self._prediction_distribution.clear()
            self._memory_percent = 0.0
            self._gpu_memory_percent = 0.0
            logger.info("SystemMonitor metrics reset")


def monitor_request(
    monitor: SystemMonitor,
    latency_spike_threshold_ms: float = 500.0,
    error_spike_threshold: float = 0.05,
) -> Callable:
    """Decorator to monitor FastAPI endpoint requests.

    Records latency, success/failure, and triggers alerts on anomalies.

    Args:
        monitor: SystemMonitor instance.
        latency_spike_threshold_ms: Latency alert threshold.
        error_spike_threshold: Error rate alert threshold.

    Returns:
        Decorated async function.
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            error = False
            label = "unknown"
            confidence = 0.0

            try:
                result = await func(*args, **kwargs)
                # Extract label and confidence from response
                if hasattr(result, 'label'):
                    label = result.label
                if hasattr(result, 'confidence'):
                    confidence = result.confidence
                return result
            except Exception as e:
                error = True
                logger.exception("Request failed in monitored endpoint")
                raise
            finally:
                latency_ms = (time.perf_counter() - start) * 1000
                monitor.record_request(
                    latency_ms=latency_ms,
                    label=label,
                    confidence=confidence,
                    error=error,
                )

                # Check for anomalies
                metrics = monitor.get_metrics()
                if latency_ms > latency_spike_threshold_ms and monitor._on_latency_spike:
                    monitor._on_latency_spike(metrics)
                if metrics.error_rate > error_spike_threshold and monitor._on_error_spike:
                    monitor._on_error_spike(metrics)

        return wrapper
    return decorator


# Import math for KL-divergence calculation
import math

__all__ = ['SystemMonitor', 'HealthMetrics', 'LatencyMetrics', 'monitor_request']
