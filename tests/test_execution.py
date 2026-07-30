import time
from threading import Timer

import pytest

from dqagent.errors import RunCancelledError, RunDeadlineExceededError
from dqagent.execution import RunContext


def test_context_exposes_identity_deadline_and_read_only_metadata() -> None:
    context = RunContext(
        run_id="run-123",
        timeout_seconds=1,
        metadata={"tenant": "demo"},
    )

    assert context.run_id == "run-123"
    assert context.deadline is not None
    assert context.remaining_seconds is not None
    assert 0 < context.remaining_seconds <= 1
    assert context.metadata == {"tenant": "demo"}
    with pytest.raises(TypeError):
        context.metadata["tenant"] = "changed"  # type: ignore[index]


def test_context_cancellation_is_idempotent_and_interrupts_wait() -> None:
    context = RunContext(run_id="run-cancelled")

    assert context.cancel("caller stopped the run") is True
    assert context.cancel("ignored second reason") is False
    assert context.is_cancelled is True
    assert context.cancel_reason == "caller stopped the run"
    with pytest.raises(RunCancelledError, match="caller stopped") as error:
        context.wait(1)
    assert error.value.run_id == "run-cancelled"


def test_context_cancellation_interrupts_an_active_wait() -> None:
    context = RunContext(run_id="run-active-cancel")
    timer = Timer(0.01, context.cancel, args=("caller stopped the wait",))
    started = time.monotonic()
    timer.start()

    try:
        with pytest.raises(RunCancelledError, match="stopped the wait"):
            context.wait(1)
    finally:
        timer.join()

    assert time.monotonic() - started < 0.5


def test_context_enforces_deadline() -> None:
    context = RunContext(run_id="run-timeout", timeout_seconds=0.001)
    time.sleep(0.005)

    with pytest.raises(RunDeadlineExceededError) as error:
        context.check_active()

    assert error.value.run_id == "run-timeout"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_id": ""},
        {"run_id": " "},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
    ],
)
def test_context_rejects_invalid_construction(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RunContext(**kwargs)  # type: ignore[arg-type]


def test_context_rejects_invalid_cancel_and_wait_values() -> None:
    context = RunContext()

    with pytest.raises(ValueError, match="reason"):
        context.cancel(" ")
    with pytest.raises(ValueError, match="delay"):
        context.wait(-1)


def test_child_context_inherits_parent_cancellation_but_can_cancel_independently() -> None:
    parent = RunContext(run_id="run-parent", metadata={"scope": "workflow"})
    child = parent.child(metadata={"branch": "left"})

    assert child.run_id == parent.run_id
    assert child.metadata == {"scope": "workflow", "branch": "left"}
    assert child.cancel("stop branch") is True
    assert parent.is_cancelled is False

    active_child = parent.child()
    parent.cancel("stop workflow")

    with pytest.raises(RunCancelledError, match="stop workflow"):
        active_child.check_active()


def test_parent_cancellation_interrupts_child_wait() -> None:
    parent = RunContext(run_id="run-parent-wait")
    child = parent.child()
    timer = Timer(0.01, parent.cancel, args=("stop child wait",))
    started = time.monotonic()
    timer.start()

    try:
        with pytest.raises(RunCancelledError, match="stop child wait"):
            child.wait(1)
    finally:
        timer.join()

    assert child.is_cancelled is True
    assert child.cancel_reason == "stop child wait"
    assert time.monotonic() - started < 0.5
