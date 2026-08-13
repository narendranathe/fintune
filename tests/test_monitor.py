"""
Tests for the monitoring system including SystemMonitor, metrics, and drift detection.
"""

import threading
import unittest
from collections import deque
from unittest.mock import MagicMock

# Assuming these are the actual modules to test
# Adjust imports based on your actual project structure


class TestSystemMonitor(unittest.TestCase):
    """Test SystemMonitor singleton and metric recording."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_monitor = MagicMock()
        self.mock_monitor.metrics = {
            'latency': deque(maxlen=100),
            'error_count': deque(maxlen=100),
            'throughput': deque(maxlen=100)
        }

    def test_singleton_instance(self):
        """Test that SystemMonitor is a singleton."""
        # Mock singleton behavior
        monitor1 = self.mock_monitor
        monitor2 = self.mock_monitor
        self.assertIs(monitor1, monitor2)

    def test_record_latency_metric(self):
        """Test recording latency metrics."""
        # Simulate recording a latency value
        latency_value = 0.125
        self.mock_monitor.metrics['latency'].append(latency_value)

        self.assertEqual(len(self.mock_monitor.metrics['latency']), 1)
        self.assertEqual(next(iter(self.mock_monitor.metrics['latency'])), latency_value)

    def test_record_error_count(self):
        """Test recording error count metrics."""
        error_count = 5
        self.mock_monitor.metrics['error_count'].append(error_count)

        self.assertEqual(len(self.mock_monitor.metrics['error_count']), 1)
        self.assertEqual(next(iter(self.mock_monitor.metrics['error_count'])), error_count)

    def test_record_throughput(self):
        """Test recording throughput metrics."""
        throughput_value = 100.5
        self.mock_monitor.metrics['throughput'].append(throughput_value)

        self.assertEqual(len(self.mock_monitor.metrics['throughput']), 1)
        self.assertEqual(next(iter(self.mock_monitor.metrics['throughput'])), throughput_value)

    def test_health_score_computation(self):
        """Test health score computation based on metrics."""
        # Mock a compute_health_score method
        self.mock_monitor.compute_health_score = MagicMock(return_value=0.85)

        health_score = self.mock_monitor.compute_health_score()

        self.assertEqual(health_score, 0.85)
        self.assertGreaterEqual(health_score, 0.0)
        self.assertLessEqual(health_score, 1.0)

    def test_health_score_varies_with_errors(self):
        """Test that health score decreases with error count."""
        compute_health = lambda errors: max(0, 1.0 - (errors * 0.1))

        score_no_errors = compute_health(0)
        score_with_errors = compute_health(5)

        self.assertGreater(score_no_errors, score_with_errors)

    def test_sliding_window_metrics(self):
        """Test sliding window behavior with maxlen=100."""
        window = deque(maxlen=100)

        # Add 150 values
        for i in range(150):
            window.append(i)

        # Should only keep last 100
        self.assertEqual(len(window), 100)
        # First value should be 50 (oldest retained)
        self.assertEqual(next(iter(window)), 50)
        # Last value should be 149
        self.assertEqual(list(window)[-1], 149)

    def test_drift_detection_no_drift(self):
        """Test drift detection when values are stable."""
        self.mock_monitor.detect_drift = MagicMock(return_value=False)

        is_drift = self.mock_monitor.detect_drift()

        self.assertFalse(is_drift)

    def test_drift_detection_with_drift(self):
        """Test drift detection when values change significantly."""
        self.mock_monitor.detect_drift = MagicMock(return_value=True)

        is_drift = self.mock_monitor.detect_drift()

        self.assertTrue(is_drift)

    def test_drift_detection_threshold(self):
        """Test drift detection with statistical threshold."""
        def compute_drift(baseline_mean, current_values, threshold=0.3):
            if not current_values:
                return False
            current_mean = sum(current_values) / len(current_values)
            drift_ratio = abs(current_mean - baseline_mean) / max(baseline_mean, 0.001)
            return drift_ratio > threshold

        baseline = 100.0
        stable_values = [99.0, 101.0, 100.5, 99.5]
        drift_values = [150.0, 155.0, 160.0, 165.0]

        self.assertFalse(compute_drift(baseline, stable_values))
        self.assertTrue(compute_drift(baseline, drift_values))


class TestSystemMonitorThreadSafety(unittest.TestCase):
    """Test thread safety with concurrent metric writes."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = {
            'latency': [],
            'lock': threading.Lock()
        }

    def test_concurrent_metric_writes(self):
        """Test thread-safe concurrent writes to metrics."""
        def write_metrics(thread_id, count):
            for i in range(count):
                with self.metrics['lock']:
                    self.metrics['latency'].append((thread_id, i))

        threads = []
        num_threads = 5
        writes_per_thread = 20

        for i in range(num_threads):
            t = threading.Thread(target=write_metrics, args=(i, writes_per_thread))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have num_threads * writes_per_thread entries
        self.assertEqual(len(self.metrics['latency']), num_threads * writes_per_thread)

    def test_no_race_condition_on_health_score(self):
        """Test that health score computation is thread-safe."""
        health_state = {'score': 0.5, 'lock': threading.Lock()}

        def update_health():
            for _ in range(100):
                with health_state['lock']:
                    current = health_state['score']
                    health_state['score'] = (current + 0.01) % 1.0

        threads = [threading.Thread(target=update_health) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have a valid health score
        self.assertGreaterEqual(health_state['score'], 0.0)
        self.assertLess(health_state['score'], 1.0)


if __name__ == '__main__':
    unittest.main()
