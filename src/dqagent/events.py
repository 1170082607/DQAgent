"""Execution lifecycle events shared by agent and workflow runtimes."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from dqagent.execution import RunContext

logger = logging.getLogger(__name__)


class RunState(StrEnum):
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class RunEventType(StrEnum):
    RUN_STARTED = "run_started"
    RUN_RESUMED = "run_resumed"
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    RETRIEVAL_FAILED = "retrieval_failed"
    MEMORY_RECALL_STARTED = "memory_recall_started"
    MEMORY_RECALL_COMPLETED = "memory_recall_completed"
    MEMORY_RECALL_FAILED = "memory_recall_failed"
    MEMORY_EXTRACTION_STARTED = "memory_extraction_started"
    MEMORY_EXTRACTION_COMPLETED = "memory_extraction_completed"
    MEMORY_EXTRACTION_FAILED = "memory_extraction_failed"
    CONTEXT_ASSEMBLED = "context_assembled"
    MODEL_REQUEST_STARTED = "model_request_started"
    MODEL_REQUEST_COMPLETED = "model_request_completed"
    MODEL_REQUEST_FAILED = "model_request_failed"
    RETRY_SCHEDULED = "retry_scheduled"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    WORKFLOW_NODE_STARTED = "workflow_node_started"
    WORKFLOW_NODE_COMPLETED = "workflow_node_completed"
    WORKFLOW_NODE_FAILED = "workflow_node_failed"
    WORKFLOW_TRANSITION_SELECTED = "workflow_transition_selected"
    CHECKPOINT_SAVED = "checkpoint_saved"
    RUN_INTERRUPTED = "run_interrupted"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    RUN_TIMED_OUT = "run_timed_out"


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    sequence: int
    type: RunEventType
    state: RunState
    occurred_at: datetime
    elapsed_seconds: float
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))


class EventSink(Protocol):
    """Receives ordered execution events for tracing, metrics, or audit adapters."""

    def emit(self, event: RunEvent) -> None:
        """Consume one immutable event without raising into the run."""
        ...


class RunEventEmitter:
    """Creates one ordered best-effort event stream for a run."""

    def __init__(self, context: RunContext, sinks: Sequence[EventSink]) -> None:
        self._context = context
        self._sinks = tuple(sinks)
        self._events: list[RunEvent] = []

    @property
    def events(self) -> tuple[RunEvent, ...]:
        return tuple(self._events)

    def emit(
        self,
        event_type: RunEventType,
        state: RunState,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        self._notify(self._append(event_type, state, attributes))

    def emit_if_active(
        self,
        event_type: RunEventType,
        state: RunState,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        """Append one event while its context remains active, then notify sinks."""
        event = self._context.run_if_active(
            lambda: self._append(event_type, state, attributes)
        )
        self._notify(event)

    def _append(
        self,
        event_type: RunEventType,
        state: RunState,
        attributes: Mapping[str, object] | None,
    ) -> RunEvent:
        event = RunEvent(
            run_id=self._context.run_id,
            sequence=len(self._events) + 1,
            type=event_type,
            state=state,
            occurred_at=datetime.now(UTC),
            elapsed_seconds=self._context.elapsed_seconds,
            attributes=attributes or {},
        )
        self._events.append(event)
        return event

    def _notify(self, event: RunEvent) -> None:
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception:  # Observability must not change business execution semantics.
                logger.exception("execution event sink failed", extra={"run_id": event.run_id})
