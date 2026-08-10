"""Execution context shared across one agent run."""

import math
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from threading import Event, Lock
from types import MappingProxyType
from uuid import uuid4

from dqagent.errors import RunCancelledError, RunDeadlineExceededError


class RunContext:
    """Carries identity, deadline, cancellation, and metadata through a run."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        timeout_seconds: float | None = None,
        metadata: Mapping[str, object] | None = None,
        _parent: "RunContext | None" = None,
    ) -> None:
        resolved_run_id = str(uuid4()) if run_id is None else run_id
        if not resolved_run_id.strip():
            raise ValueError("run ID must not be empty")
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise ValueError("run timeout must be a finite number greater than zero")

        self.run_id = resolved_run_id
        self.started_at = datetime.now(UTC)
        self.deadline = (
            self.started_at + timedelta(seconds=timeout_seconds)
            if timeout_seconds is not None
            else None
        )
        self.metadata: Mapping[str, object] = MappingProxyType(dict(metadata or {}))
        self._parent = _parent
        self._started_monotonic = time.perf_counter()
        self._deadline_monotonic = (
            self._started_monotonic + timeout_seconds
            if timeout_seconds is not None
            else None
        )
        self._cancelled = Event()
        self._cancel_lock = Lock()
        self._cancel_reason: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self._started_monotonic)

    @property
    def remaining_seconds(self) -> float | None:
        if self._deadline_monotonic is None:
            return None
        return max(0.0, self._deadline_monotonic - time.perf_counter())

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set() or (
            self._parent is not None and self._parent.is_cancelled
        )

    @property
    def cancel_reason(self) -> str | None:
        if self._cancel_reason is not None:
            return self._cancel_reason
        return self._parent.cancel_reason if self._parent is not None else None

    def cancel(self, reason: str = "run cancelled") -> bool:
        """Request cancellation and return whether this call won the race."""
        resolved_reason = reason.strip()
        if not resolved_reason:
            raise ValueError("cancellation reason must not be empty")
        with self._cancel_lock:
            if self._cancelled.is_set():
                return False
            self._cancel_reason = resolved_reason
            self._cancelled.set()
            return True

    def check_active(self) -> None:
        if self._parent is not None:
            self._parent.check_active()
        if self._cancelled.is_set():
            raise RunCancelledError(
                self._cancel_reason or "run cancelled",
                run_id=self.run_id,
            )
        if self.remaining_seconds == 0:
            raise RunDeadlineExceededError(
                "agent run deadline exceeded",
                run_id=self.run_id,
            )

    def wait(self, delay_seconds: float) -> None:
        """Wait for retry backoff while remaining responsive to cancellation."""
        if not math.isfinite(delay_seconds) or delay_seconds < 0:
            raise ValueError("wait delay must be a finite non-negative number")
        wait_deadline = time.perf_counter() + delay_seconds
        while True:
            self.check_active()
            delay_remaining = wait_deadline - time.perf_counter()
            if delay_remaining <= 0:
                self.check_active()
                return
            run_remaining = self.remaining_seconds
            wait_seconds = (
                delay_remaining
                if run_remaining is None
                else min(delay_remaining, run_remaining)
            )
            if self._parent is not None:
                wait_seconds = min(wait_seconds, 0.05)
            self._cancelled.wait(wait_seconds)

    def child(self, *, metadata: Mapping[str, object] | None = None) -> "RunContext":
        """Create an independently cancellable context bounded by this context."""
        self.check_active()
        merged_metadata = {**self.metadata, **dict(metadata or {})}
        return RunContext(
            run_id=self.run_id,
            timeout_seconds=self.remaining_seconds,
            metadata=merged_metadata,
            _parent=self,
        )
