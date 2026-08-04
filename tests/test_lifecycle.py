from collections.abc import Mapping

import pytest

from dqagent.errors import (
    DQAgentError,
    RetrievalError,
    RunCancelledError,
    RunDeadlineExceededError,
    RunExecutionError,
)
from dqagent.events import RunEvent, RunEventType, RunState
from dqagent.execution import RunContext
from dqagent.lifecycle import RunCoordinator, RunScope


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


def test_coordinator_exclusively_owns_start_and_terminal_transitions() -> None:
    sink = RecordingSink()
    coordinator = RunCoordinator(event_sinks=(sink,))
    captured_scopes: list[RunScope] = []

    def operation(scope: RunScope) -> str:
        captured_scopes.append(scope)
        with pytest.raises(ValueError, match="owned by RunCoordinator"):
            scope.emit(RunEventType.RUN_COMPLETED)
        scope.emit(RunEventType.CONTEXT_ASSEMBLED, {"item_count": 2})
        return "done"

    result = coordinator.execute(
        operation,
        context=RunContext(run_id="coordinated-run"),
        completion_attributes=lambda value: {"result": value},
    )

    assert result.value == "done"
    assert result.record.state is RunState.COMPLETED
    assert result.record.events == tuple(sink.events)
    assert [event.type for event in sink.events] == [
        RunEventType.RUN_STARTED,
        RunEventType.CONTEXT_ASSEMBLED,
        RunEventType.RUN_COMPLETED,
    ]
    assert sink.events[-1].attributes == {"result": "done"}
    assert not hasattr(captured_scopes[0], "complete")
    assert not hasattr(captured_scopes[0], "fail")
    with pytest.raises(RuntimeError, match="already terminal"):
        captured_scopes[0].emit(RunEventType.CONTEXT_ASSEMBLED)


@pytest.mark.parametrize(
    ("error", "terminal_type", "terminal_state"),
    [
        (RetrievalError("index unavailable"), RunEventType.RUN_FAILED, RunState.FAILED),
        (
            RunCancelledError("caller cancelled"),
            RunEventType.RUN_CANCELLED,
            RunState.CANCELLED,
        ),
        (
            RunDeadlineExceededError("deadline exceeded"),
            RunEventType.RUN_TIMED_OUT,
            RunState.TIMED_OUT,
        ),
    ],
)
def test_coordinator_classifies_terminal_errors(
    error: DQAgentError,
    terminal_type: RunEventType,
    terminal_state: RunState,
) -> None:
    sink = RecordingSink()
    coordinator = RunCoordinator(event_sinks=(sink,))

    def fail(scope: RunScope) -> None:
        scope.emit_error(RunEventType.RETRIEVAL_FAILED, error)
        raise error

    with pytest.raises(type(error), match=str(error)):
        coordinator.execute(fail, context=RunContext(run_id="failed-run"))

    assert error.run_id == "failed-run"
    assert sink.events[-1].type is terminal_type
    assert sink.events[-1].state is terminal_state
    assert [event.type for event in sink.events[:3]] == [
        RunEventType.RUN_STARTED,
        RunEventType.RETRIEVAL_FAILED,
        terminal_type,
    ]


def test_coordinator_wraps_unexpected_failures_and_records_the_cause() -> None:
    sink = RecordingSink()
    coordinator = RunCoordinator(event_sinks=(sink,))

    def fail(scope: RunScope) -> Mapping[str, object]:
        del scope
        raise RuntimeError("broken stage")

    with pytest.raises(RunExecutionError, match="unexpected run execution failure") as error:
        coordinator.execute(fail, context=RunContext(run_id="unexpected-run"))

    assert isinstance(error.value.__cause__, RuntimeError)
    assert error.value.run_id == "unexpected-run"
    assert sink.events[-1].type is RunEventType.RUN_FAILED
    assert sink.events[-1].attributes["cause_type"] == "RuntimeError"
