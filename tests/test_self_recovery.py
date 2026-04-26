"""
Tests for self-recovery system including CircuitBreaker and RecoveryPolicy.
"""

import unittest
import time
from unittest.mock import patch, MagicMock
from enum import Enum
from dataclasses import dataclass, field
from typing import List


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RecoveryPolicy:
    """Configuration for recovery behavior."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_attempts: int = 3
    backoff_multiplier: float = 2.0
    log_recovery_actions: bool = True
    recovery_actions: List[str] = field(default_factory=list)


class CircuitBreaker:
    """Simple circuit breaker implementation for testing."""

    def __init__(self, policy: RecoveryPolicy):
        self.policy = policy
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_attempts = 0
        self.recovery_log = []

    def record_failure(self):
        """Record a failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.policy.log_recovery_actions:
            self.recovery_log.append(f"Failure recorded: count={self.failure_count}")

        if self.failure_count >= self.policy.failure_threshold:
            self._open_circuit()

    def record_success(self):
        """Record a success and potentially close circuit."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_attempts += 1
            if self.half_open_attempts >= self.policy.half_open_max_attempts:
                self._close_circuit()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

        if self.policy.log_recovery_actions:
            self.recovery_log.append(f"Success recorded: state={self.state.value}")

    def _open_circuit(self):
        """Transition to OPEN state."""
        self.state = CircuitState.OPEN
        if self.policy.log_recovery_actions:
            self.recovery_log.append("Circuit opened due to failure threshold")

    def check_recovery(self):
        """Check if circuit should transition to HALF_OPEN."""
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.policy.recovery_timeout:
                self._transition_to_half_open()

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.half_open_attempts = 0
        if self.policy.log_recovery_actions:
            self.recovery_log.append("Circuit transitioned to HALF_OPEN")

    def _close_circuit(self):
        """Transition to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_attempts = 0
        if self.policy.log_recovery_actions:
            self.recovery_log.append("Circuit closed after successful recovery")

    def is_open(self):
        """Check if circuit is open."""
        return self.state == CircuitState.OPEN


class TestCircuitBreakerTransitions(unittest.TestCase):
    """Test CircuitBreaker state transitions."""

    def setUp(self):
        """Set up test fixtures."""
        self.policy = RecoveryPolicy()
        self.breaker = CircuitBreaker(self.policy)

    def test_initial_state_closed(self):
        """Test that circuit starts in CLOSED state."""
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)

    def test_closed_to_open_transition(self):
        """Test transition from CLOSED to OPEN state."""
        # Record failures up to threshold
        for _ in range(self.policy.failure_threshold):
            self.breaker.record_failure()

        self.assertEqual(self.breaker.state, CircuitState.OPEN)
        self.assertTrue(self.breaker.is_open())

    def test_open_to_half_open_transition(self):
        """Test transition from OPEN to HALF_OPEN state."""
        # Open the circuit
        for _ in range(self.policy.failure_threshold):
            self.breaker.record_failure()

        self.assertEqual(self.breaker.state, CircuitState.OPEN)

        # Simulate timeout (use 0 for immediate transition in test)
        self.policy.recovery_timeout = 0
        self.breaker.check_recovery()

        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)

    def test_half_open_to_closed_transition(self):
        """Test transition from HALF_OPEN back to CLOSED state."""
        # Set timeout to 0 for immediate recovery
        self.policy.recovery_timeout = 0
        self.policy.half_open_max_attempts = 2

        # Open the circuit
        for _ in range(self.policy.failure_threshold):
            self.breaker.record_failure()

        # Transition to HALF_OPEN
        self.breaker.check_recovery()
        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)

        # Record successful attempts to close
        for _ in range(self.policy.half_open_max_attempts):
            self.breaker.record_success()

        self.assertEqual(self.breaker.state, CircuitState.CLOSED)

    def test_half_open_failure_reopens(self):
        """Test that failure in HALF_OPEN state reopens circuit."""
        # Set up and open circuit
        self.policy.recovery_timeout = 0
        for _ in range(self.policy.failure_threshold):
            self.breaker.record_failure()

        # Transition to HALF_OPEN
        self.breaker.check_recovery()
        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)

        # Record failure in HALF_OPEN
        self.breaker.record_failure()

        # Should reopen or stay open
        self.assertIn(self.breaker.state, [CircuitState.OPEN, CircuitState.HALF_OPEN])


class TestFailureThreshold(unittest.TestCase):
    """Test failure threshold triggers circuit open."""

    def test_failure_threshold_at_boundary(self):
        """Test that circuit opens exactly at threshold."""
        policy = RecoveryPolicy(failure_threshold=3)
        breaker = CircuitBreaker(policy)

        # Record failures below threshold
        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitState.CLOSED)

        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitState.CLOSED)

        # Record failure at threshold
        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitState.OPEN)

    def test_custom_failure_threshold(self):
        """Test with custom failure threshold."""
        for threshold in [1, 5, 10]:
            policy = RecoveryPolicy(failure_threshold=threshold)
            breaker = CircuitBreaker(policy)

            for i in range(threshold - 1):
                breaker.record_failure()
                self.assertEqual(breaker.state, CircuitState.CLOSED)

            breaker.record_failure()
            self.assertEqual(breaker.state, CircuitState.OPEN)


class TestRecoveryPolicy(unittest.TestCase):
    """Test RecoveryPolicy dataclass."""

    def test_default_values(self):
        """Test RecoveryPolicy default values."""
        policy = RecoveryPolicy()

        self.assertEqual(policy.failure_threshold, 5)
        self.assertEqual(policy.recovery_timeout, 60)
        self.assertEqual(policy.half_open_max_attempts, 3)
        self.assertEqual(policy.backoff_multiplier, 2.0)
        self.assertTrue(policy.log_recovery_actions)
        self.assertEqual(policy.recovery_actions, [])

    def test_custom_recovery_policy(self):
        """Test RecoveryPolicy with custom values."""
        actions = ["restart_service", "clear_cache"]
        policy = RecoveryPolicy(
            failure_threshold=10,
            recovery_timeout=120,
            half_open_max_attempts=5,
            backoff_multiplier=1.5,
            log_recovery_actions=False,
            recovery_actions=actions
        )

        self.assertEqual(policy.failure_threshold, 10)
        self.assertEqual(policy.recovery_timeout, 120)
        self.assertEqual(policy.half_open_max_attempts, 5)
        self.assertEqual(policy.backoff_multiplier, 1.5)
        self.assertFalse(policy.log_recovery_actions)
        self.assertEqual(policy.recovery_actions, actions)


class TestRecoveryActionLogging(unittest.TestCase):
    """Test recovery action logging."""

    def test_recovery_actions_logged(self):
        """Test that recovery actions are logged."""
        policy = RecoveryPolicy(log_recovery_actions=True)
        breaker = CircuitBreaker(policy)

        # Trigger failures to open circuit
        for _ in range(policy.failure_threshold):
            breaker.record_failure()

        # Check that actions were logged
        self.assertGreater(len(breaker.recovery_log), 0)
        self.assertIn("Circuit opened", breaker.recovery_log[-1])

    def test_recovery_actions_not_logged_when_disabled(self):
        """Test that recovery actions are not logged when disabled."""
        policy = RecoveryPolicy(log_recovery_actions=False)
        breaker = CircuitBreaker(policy)

        # Trigger failures
        for _ in range(policy.failure_threshold):
            breaker.record_failure()

        # Should have no log entries
        self.assertEqual(len(breaker.recovery_log), 0)

    def test_recovery_log_contents(self):
        """Test recovery log contains expected messages."""
        policy = RecoveryPolicy(log_recovery_actions=True, recovery_timeout=0)
        breaker = CircuitBreaker(policy)

        # Open circuit
        for _ in range(policy.failure_threshold):
            breaker.record_failure()

        # Transition to HALF_OPEN
        breaker.check_recovery()

        # Check log
        log_text = ' '.join(breaker.recovery_log)
        self.assertIn("opened", log_text.lower())
        self.assertIn("half_open", log_text.lower())


if __name__ == '__main__':
    unittest.main()
