"""End-to-end lifecycle coordination for one agent run."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar

from dqagent.errors import (
    DQAgentError,
    RunCancelledError,
    RunDeadlineExceededError,
    RunExecutionError,
)
from dqagent.events import EventSink, RunEvent, RunEventEmitter, RunEventType, RunState
from dqagent.execution import RunContext

T = TypeVar("T")

_RUN_LIFECYCLE_EVENTS = frozenset(
    {
        RunEventType.RUN_STARTED,
        RunEventType.RUN_RESUMED,
        RunEventType.RUN_INTERRUPTED,
        RunEventType.RUN_COMPLETED,
        RunEventType.RUN_FAILED,
        RunEventType.RUN_CANCELLED,
        RunEventType.RUN_TIMED_OUT,
    }
)


@dataclass(frozen=True, slots=True)
class RunRecord:
    """The immutable lifecycle record produced by a coordinated run."""

    run_id: str
    state: RunState
    events: tuple[RunEvent, ...]
    started_at: datetime
    completed_at: datetime


@dataclass(frozen=True, slots=True)
class CoordinatedRun(Generic[T]):
    """A successful operation value paired with its completed lifecycle record."""

    value: T
    record: RunRecord


class RunScope:
    """Stage-facing event capability without run terminal transition methods."""

    def __init__(self, context: RunContext, emitter: RunEventEmitter) -> None:
        self._context = context
        self._emitter = emitter
        self._terminal_state: RunState | None = None

    @property
    def context(self) -> RunContext:
        return self._context

    def emit(
        self,
        event_type: RunEventType,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        """Record a non-terminal stage event while the run is active."""
        self._validate_stage_event(event_type)
        self._ensure_running()
        self._emitter.emit(event_type, RunState.RUNNING, attributes)

    def emit_error(
        self,
        event_type: RunEventType,
        error: DQAgentError,
        attributes: Mapping[str, object] | None = None,
        *,
        cause_type: str | None = None,
    ) -> None:
        """Record a non-terminal stage failure with normalized error attributes."""
        self._validate_stage_event(event_type)
        _bind_error(error, self._context)
        details = {
            **(attributes or {}),
            **_error_attributes(error),
        }
        if cause_type is not None:
            details["cause_type"] = cause_type
        self.emit(event_type, details)

    def _terminate(
        self,
        event_type: RunEventType,
        state: RunState,
        attributes: Mapping[str, object] | None,
    ) -> None:
        self._ensure_running()
        self._emitter.emit(event_type, state, attributes)
        self._terminal_state = state

    def _record(self) -> RunRecord:
        if self._terminal_state is None:
            raise RuntimeError("run scope has not reached a terminal state")
        terminal_event = self._emitter.events[-1]
        return RunRecord(
            run_id=self._context.run_id,
            state=self._terminal_state,
            events=self._emitter.events,
            started_at=self._context.started_at,
            completed_at=terminal_event.occurred_at,
        )

    def _ensure_running(self) -> None:
        if self._terminal_state is not None:
            raise RuntimeError(f"run scope is already terminal: {self._terminal_state.value}")

    @staticmethod
    def _validate_stage_event(event_type: RunEventType) -> None:
        if event_type in _RUN_LIFECYCLE_EVENTS:
            raise ValueError("run lifecycle events are owned by RunCoordinator")


class RunCoordinator:
    """Owns start, terminal classification, and closure for end-to-end agent runs."""

    def __init__(
        self,
        *,
        default_timeout_seconds: float | None = 120.0,
        event_sinks: Sequence[EventSink] = (),
    ) -> None:
        if default_timeout_seconds is not None and (
            not math.isfinite(default_timeout_seconds) or default_timeout_seconds <= 0
        ):
            raise ValueError("default run timeout must be a finite number greater than zero")
        self._default_timeout_seconds = default_timeout_seconds
        self._event_sinks = tuple(event_sinks)

    def create_context(
        self,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> RunContext:
        return RunContext(
            timeout_seconds=self._default_timeout_seconds,
            metadata=metadata,
        )

    def execute(
        self,
        operation: Callable[[RunScope], T],
        *,
        context: RunContext | None = None,
        completion_attributes: Callable[[T], Mapping[str, object] | None] | None = None,
    ) -> CoordinatedRun[T]:
        """Run one operation inside a lifecycle that this coordinator closes."""
        run_context = context or self.create_context()
        emitter = RunEventEmitter(run_context, self._event_sinks)
        scope = RunScope(run_context, emitter)
        emitter.emit(
            RunEventType.RUN_STARTED,
            RunState.RUNNING,
            {
                "deadline": (
                    run_context.deadline.isoformat() if run_context.deadline else None
                ),
                "metadata": dict(run_context.metadata),
            },
        )

        try:
            run_context.check_active()
            value = operation(scope)
            run_context.check_active()
            attributes = (
                completion_attributes(value) if completion_attributes is not None else None
            )
        except DQAgentError as error:
            self._fail(scope, error)
            raise
        except Exception as exc:
            unexpected_error = RunExecutionError(
                "unexpected run execution failure",
                run_id=run_context.run_id,
            )
            self._fail(scope, unexpected_error, cause_type=type(exc).__name__)
            raise unexpected_error from exc

        scope._terminate(RunEventType.RUN_COMPLETED, RunState.COMPLETED, attributes)
        return CoordinatedRun(value, scope._record())

    @staticmethod
    def _fail(
        scope: RunScope,
        error: DQAgentError,
        *,
        cause_type: str | None = None,
    ) -> None:
        _bind_error(error, scope.context)
        if isinstance(error, RunCancelledError):
            event_type = RunEventType.RUN_CANCELLED
            state = RunState.CANCELLED
        elif isinstance(error, RunDeadlineExceededError):
            event_type = RunEventType.RUN_TIMED_OUT
            state = RunState.TIMED_OUT
        else:
            event_type = RunEventType.RUN_FAILED
            state = RunState.FAILED
        details = dict(_error_attributes(error))
        resolved_cause_type = cause_type or (
            type(error.__cause__).__name__ if error.__cause__ is not None else None
        )
        if resolved_cause_type is not None:
            details["cause_type"] = resolved_cause_type
        scope._terminate(event_type, state, details)


def _bind_error(error: DQAgentError, context: RunContext) -> None:
    error.run_id = context.run_id


def _error_attributes(error: DQAgentError) -> Mapping[str, object]:
    return {
        "error_type": type(error).__name__,
        "error_category": error.category.value,
        "retryable": error.retryable,

        "message": str(error),
    }
