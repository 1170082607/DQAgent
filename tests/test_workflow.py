from collections.abc import Mapping
from threading import Event

import pytest

from dqagent.checkpoint import InMemoryCheckpointStore, WorkflowCheckpoint
from dqagent.errors import (
    CheckpointError,
    RunCancelledError,
    WorkflowDefinitionError,
    WorkflowExecutionError,
)
from dqagent.events import RunEvent, RunEventType, RunState
from dqagent.execution import RunContext
from dqagent.workflow import (
    END,
    ConditionalTransition,
    NextTransition,
    NodeResult,
    ParallelTransition,
    WorkflowDefinition,
    WorkflowNode,
    WorkflowRunner,
)


def update(**values: object):
    def handler(state: Mapping[str, object], context: RunContext) -> NodeResult:
        context.check_active()
        return NodeResult(values)

    return handler


def test_sequential_and_conditional_workflow_commits_each_selected_node() -> None:
    definition = WorkflowDefinition(
        "routing",
        "1",
        "prepare",
        (
            WorkflowNode("prepare", update(prepared=True), NextTransition("route")),
            WorkflowNode(
                "route",
                update(routed=True),
                ConditionalTransition(
                    lambda state, context: "large" if state["amount"] > 10 else "small",
                    {"small": "small", "large": "large"},
                ),
            ),
            WorkflowNode("small", update(decision="small"), END),
            WorkflowNode("large", update(decision="large"), END),
        ),
    )
    store = InMemoryCheckpointStore()

    result = WorkflowRunner(store).start(
        definition,
        {"amount": 20},
        workflow_id="routing-1",
        context=RunContext(run_id="run-routing"),
    )

    assert result.state is RunState.COMPLETED
    assert result.data == {
        "amount": 20,
        "prepared": True,
        "routed": True,
        "decision": "large",
    }
    assert result.checkpoint.completed_nodes == ("prepare", "route", "large")
    assert result.checkpoint.revision == 4
    assert result.events[0].type is RunEventType.RUN_STARTED
    assert result.events[-1].type is RunEventType.RUN_COMPLETED


def test_parallel_branches_share_snapshot_and_merge_in_declaration_order() -> None:
    seen: dict[str, Mapping[str, object]] = {}

    def branch(name: str, key: str, value: object):
        def handler(state: Mapping[str, object], context: RunContext) -> NodeResult:
            seen[name] = state
            return NodeResult({key: value})

        return handler

    definition = WorkflowDefinition(
        "parallel",
        "1",
        "fanout",
        (
            WorkflowNode(
                "fanout",
                update(shared="ready"),
                ParallelTransition(("left", "right"), "join"),
            ),
            WorkflowNode("left", branch("left", "left_value", 1), END),
            WorkflowNode("right", branch("right", "right_value", 2), END),
            WorkflowNode("join", update(joined=True), END),
        ),
    )

    result = WorkflowRunner(InMemoryCheckpointStore(), max_parallelism=2).start(
        definition,
        {"input": "value"},
        workflow_id="parallel-1",
    )

    expected_snapshot = {"input": "value", "shared": "ready"}
    assert seen == {"left": expected_snapshot, "right": expected_snapshot}
    assert result.data == {
        **expected_snapshot,
        "left_value": 1,
        "right_value": 2,
        "joined": True,
    }
    assert result.checkpoint.completed_nodes == ("fanout", "left", "right", "join")
    branch_completions = [
        event.attributes["node_id"]
        for event in result.events
        if event.type is RunEventType.WORKFLOW_NODE_COMPLETED
        and event.attributes.get("parallel") is True
    ]
    assert branch_completions == ["left", "right"]


def test_parallel_conflict_fails_without_committing_partial_branch_state() -> None:
    definition = WorkflowDefinition(
        "conflict",
        "1",
        "fanout",
        (
            WorkflowNode(
                "fanout",
                update(prepared=True),
                ParallelTransition(("left", "right")),
            ),
            WorkflowNode("left", update(value=1), END),
            WorkflowNode("right", update(value=2), END),
        ),
    )
    store = InMemoryCheckpointStore()

    with pytest.raises(WorkflowExecutionError, match="both update state key 'value'"):
        WorkflowRunner(store, max_parallelism=2).start(
            definition,
            {},
            workflow_id="conflict-1",
        )

    checkpoint = store.load("conflict-1")
    assert checkpoint is not None
    assert checkpoint.status is RunState.FAILED
    assert checkpoint.current_node == "fanout"
    assert checkpoint.state == {}
    assert checkpoint.completed_nodes == ()


def test_parallel_failure_cooperatively_cancels_sibling() -> None:
    slow_started = Event()
    sibling_cancelled = Event()

    def slow(state: Mapping[str, object], context: RunContext) -> NodeResult:
        slow_started.set()
        try:
            context.wait(1)
        except Exception:
            sibling_cancelled.set()
            raise
        return NodeResult({"slow": True})

    def fail(state: Mapping[str, object], context: RunContext) -> NodeResult:
        assert slow_started.wait(0.5)
        raise RuntimeError("branch failed")

    definition = WorkflowDefinition(
        "cancel-sibling",
        "1",
        "fanout",
        (
            WorkflowNode(
                "fanout",
                update(),
                ParallelTransition(("slow", "fail")),
            ),
            WorkflowNode("slow", slow, END),
            WorkflowNode("fail", fail, END),
        ),
    )

    with pytest.raises(WorkflowExecutionError, match="node 'fail' failed"):
        WorkflowRunner(InMemoryCheckpointStore(), max_parallelism=2).start(
            definition,
            {},
            workflow_id="cancel-sibling-1",
        )

    assert sibling_cancelled.is_set()


def test_interrupt_resume_and_replay_have_explicit_idempotency_boundaries() -> None:
    store = InMemoryCheckpointStore()
    observed_keys: list[str] = []

    def prepare(state: Mapping[str, object], context: RunContext) -> NodeResult:
        observed_keys.append(str(context.metadata["idempotency_key"]))
        return NodeResult.interrupt("approval required", {"prepared": True})

    def apply(state: Mapping[str, object], context: RunContext) -> NodeResult:
        observed_keys.append(str(context.metadata["idempotency_key"]))
        return NodeResult({"applied": True})

    definition = WorkflowDefinition(
        "approval",
        "1",
        "prepare",
        (
            WorkflowNode("prepare", prepare, NextTransition("apply")),
            WorkflowNode("apply", apply, END),
        ),
    )
    runner = WorkflowRunner(store)

    interrupted = runner.start(definition, {"request": 1}, workflow_id="approval-1")
    resumed = runner.resume(definition, "approval-1")
    replayed = runner.replay(definition, "approval-1", workflow_id="approval-replay")

    assert interrupted.state is RunState.INTERRUPTED
    assert interrupted.checkpoint.current_node == "apply"
    assert resumed.state is RunState.COMPLETED
    assert resumed.data == {"request": 1, "prepared": True, "applied": True}
    assert RunEventType.RUN_RESUMED in [event.type for event in resumed.events]
    assert replayed.state is RunState.INTERRUPTED
    assert replayed.data == {"request": 1, "prepared": True}
    assert observed_keys == [
        "approval-1:approval:1:prepare",
        "approval-1:approval:1:apply",
        "approval-replay:approval:1:prepare",
    ]


def test_failed_node_resumes_with_same_idempotency_key() -> None:
    attempts = 0
    keys: list[str] = []

    def flaky(state: Mapping[str, object], context: RunContext) -> NodeResult:
        nonlocal attempts
        attempts += 1
        keys.append(str(context.metadata["idempotency_key"]))
        if attempts == 1:
            raise RuntimeError("temporary failure")
        return NodeResult({"done": True})

    definition = WorkflowDefinition(
        "recovery",
        "1",
        "flaky",
        (WorkflowNode("flaky", flaky, END),),
    )
    store = InMemoryCheckpointStore()
    runner = WorkflowRunner(store)

    with pytest.raises(WorkflowExecutionError, match="node 'flaky' failed"):
        runner.start(definition, {}, workflow_id="recovery-1")
    result = runner.resume(definition, "recovery-1")

    assert result.state is RunState.COMPLETED
    assert keys == [
        "recovery-1:recovery:1:flaky",
        "recovery-1:recovery:1:flaky",
    ]


def test_cancelled_workflow_persists_uncompleted_node() -> None:
    definition = WorkflowDefinition(
        "cancelled",
        "1",
        "start",
        (WorkflowNode("start", update(done=True), END),),
    )
    store = InMemoryCheckpointStore()
    context = RunContext(run_id="run-cancelled-workflow")
    context.cancel("caller cancelled workflow")

    with pytest.raises(RunCancelledError, match="caller cancelled workflow"):
        WorkflowRunner(store).start(
            definition,
            {},
            workflow_id="cancelled-1",
            context=context,
        )

    checkpoint = store.load("cancelled-1")
    assert checkpoint is not None
    assert checkpoint.status is RunState.CANCELLED
    assert checkpoint.current_node == "start"
    assert checkpoint.completed_nodes == ()


def test_conditional_transition_rejects_unknown_runtime_route() -> None:
    definition = WorkflowDefinition(
        "unknown-route",
        "1",
        "route",
        (
            WorkflowNode(
                "route",
                update(),
                ConditionalTransition(
                    lambda state, context: "unexpected",
                    {"known": "finish"},
                ),
            ),
            WorkflowNode("finish", update(done=True), END),
        ),
    )

    with pytest.raises(WorkflowExecutionError, match="unknown route 'unexpected'"):
        WorkflowRunner(InMemoryCheckpointStore()).start(
            definition,
            {},
            workflow_id="unknown-route-1",
        )


def test_parallel_transition_enforces_runner_concurrency_limit() -> None:
    definition = WorkflowDefinition(
        "parallel-limit",
        "1",
        "fanout",
        (
            WorkflowNode(
                "fanout",
                update(),
                ParallelTransition(("left", "right")),
            ),
            WorkflowNode("left", update(left=True), END),
            WorkflowNode("right", update(right=True), END),
        ),
    )

    with pytest.raises(WorkflowExecutionError, match="exceeds configured maximum 1"):
        WorkflowRunner(InMemoryCheckpointStore(), max_parallelism=1).start(
            definition,
            {},
            workflow_id="parallel-limit-1",
        )


def test_checkpoint_failure_emits_terminal_failure_event() -> None:
    class FailAfterInitialSave(InMemoryCheckpointStore):
        def save(
            self,
            checkpoint: WorkflowCheckpoint,
            *,
            expected_revision: int | None,
        ) -> WorkflowCheckpoint:
            if expected_revision is not None:
                raise CheckpointError("checkpoint storage unavailable")
            return super().save(checkpoint, expected_revision=expected_revision)

    class Sink:
        def __init__(self) -> None:
            self.events: list[RunEvent] = []

        def emit(self, event: RunEvent) -> None:
            self.events.append(event)

    definition = WorkflowDefinition(
        "checkpoint-failure",
        "1",
        "start",
        (WorkflowNode("start", update(done=True), END),),
    )
    sink = Sink()

    with pytest.raises(CheckpointError, match="storage unavailable"):
        WorkflowRunner(FailAfterInitialSave(), event_sinks=(sink,)).start(
            definition,
            {},
            workflow_id="checkpoint-failure-1",
        )

    assert sink.events[-1].type is RunEventType.RUN_FAILED


@pytest.mark.parametrize(
    "factory, message",
    [
        (
            lambda: WorkflowDefinition(
                "invalid",
                "1",
                "start",
                (WorkflowNode("start", update(), NextTransition("missing")),),
            ),
            "unknown nodes",
        ),
        (
            lambda: WorkflowDefinition(
                "cycle",
                "1",
                "first",
                (
                    WorkflowNode("first", update(), NextTransition("second")),
                    WorkflowNode("second", update(), NextTransition("first")),
                ),
            ),
            "acyclic",
        ),
    ],
)
def test_definition_rejects_invalid_graphs(factory, message: str) -> None:
    with pytest.raises(WorkflowDefinitionError, match=message):
        factory()
