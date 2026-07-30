"""Deterministic workflow graph with checkpointed execution and recovery."""

import json
import math
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Protocol, TypeAlias, cast
from uuid import uuid4

from dqagent.checkpoint import CheckpointStore, WorkflowCheckpoint
from dqagent.errors import (
    CheckpointError,
    DQAgentError,
    RunCancelledError,
    RunDeadlineExceededError,
    WorkflowDefinitionError,
    WorkflowExecutionError,
)
from dqagent.events import (
    EventSink,
    RunEvent,
    RunEventEmitter,
    RunEventType,
    RunState,
)
from dqagent.execution import RunContext

_NODE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


def _json_mapping(value: Mapping[str, object], *, label: str) -> dict[str, object]:
    try:
        encoded = json.dumps(dict(value), ensure_ascii=True, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise WorkflowExecutionError(
            f"{label} must contain only JSON-compatible values"
        ) from exc
    if not isinstance(decoded, dict):
        raise WorkflowExecutionError(f"{label} must be a JSON object")
    return cast(dict[str, object], decoded)


@dataclass(frozen=True, slots=True)
class NodeResult:
    """State updates and an optional durable interruption after one node."""

    updates: Mapping[str, object] = field(default_factory=dict)
    interrupt_reason: str | None = None

    def __post_init__(self) -> None:
        updates = _json_mapping(self.updates, label="node updates")
        object.__setattr__(self, "updates", MappingProxyType(updates))
        if self.interrupt_reason is not None and not self.interrupt_reason.strip():
            raise ValueError("interrupt reason must not be empty")

    @classmethod
    def interrupt(
        cls,
        reason: str,
        updates: Mapping[str, object] | None = None,
    ) -> "NodeResult":
        return cls(updates or {}, interrupt_reason=reason)


class NodeHandler(Protocol):
    def __call__(self, state: Mapping[str, object], context: RunContext) -> NodeResult: ...


class RouteSelector(Protocol):
    def __call__(self, state: Mapping[str, object], context: RunContext) -> str: ...


@dataclass(frozen=True, slots=True)
class EndTransition:
    """Complete the workflow after the current node commits."""


@dataclass(frozen=True, slots=True)
class NextTransition:
    target: str

    def __post_init__(self) -> None:
        _validate_node_id(self.target, "transition target")


@dataclass(frozen=True, slots=True)
class ConditionalTransition:
    selector: RouteSelector
    routes: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.routes:
            raise ValueError("conditional transition requires at least one route")
        copied = dict(self.routes)
        for route, target in copied.items():
            if not route.strip():
                raise ValueError("conditional route must not be empty")
            _validate_node_id(target, "conditional target")
        object.__setattr__(self, "routes", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class ParallelTransition:
    branches: tuple[str, ...]
    next_node: str | None = None

    def __post_init__(self) -> None:
        if len(self.branches) < 2:
            raise ValueError("parallel transition requires at least two branches")
        if len(self.branches) != len(set(self.branches)):
            raise ValueError("parallel branch node IDs must be unique")
        for branch in self.branches:
            _validate_node_id(branch, "parallel branch")
        if self.next_node is not None:
            _validate_node_id(self.next_node, "parallel continuation")


WorkflowTransition: TypeAlias = (
    EndTransition | NextTransition | ConditionalTransition | ParallelTransition
)
END = EndTransition()


@dataclass(frozen=True, slots=True)
class WorkflowNode:
    node_id: str
    handler: NodeHandler
    transition: WorkflowTransition = END

    def __post_init__(self) -> None:
        _validate_node_id(self.node_id, "node ID")


def _validate_node_id(value: str, label: str) -> None:
    if not _NODE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must start with a lowercase letter and contain only "
            "lowercase letters, digits, underscores, or hyphens"
        )


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    definition_id: str
    version: str
    start_node: str
    nodes: tuple[WorkflowNode, ...]
    _node_map: Mapping[str, WorkflowNode] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_node_id(self.definition_id, "definition ID")
        _validate_node_id(self.start_node, "start node")
        if not self.version.strip():
            raise WorkflowDefinitionError("workflow definition version must not be empty")
        if not self.nodes:
            raise WorkflowDefinitionError("workflow definition requires at least one node")
        node_map = {node.node_id: node for node in self.nodes}
        if len(node_map) != len(self.nodes):
            raise WorkflowDefinitionError("workflow node IDs must be unique")
        if self.start_node not in node_map:
            raise WorkflowDefinitionError(
                f"workflow start node '{self.start_node}' is not defined"
            )
        object.__setattr__(self, "_node_map", MappingProxyType(node_map))
        self._validate_graph()

    def node(self, node_id: str) -> WorkflowNode:
        return self._node_map[node_id]

    def _validate_graph(self) -> None:
        branch_nodes: set[str] = set()
        regular_targets: set[str] = {self.start_node}
        for node in self.nodes:
            transition = node.transition
            targets = self._targets(transition)
            missing = [target for target in targets if target not in self._node_map]
            if missing:
                raise WorkflowDefinitionError(
                    f"node '{node.node_id}' references unknown nodes: {missing}"
                )
            if isinstance(transition, NextTransition):
                regular_targets.add(transition.target)
            elif isinstance(transition, ConditionalTransition):
                regular_targets.update(transition.routes.values())
            elif isinstance(transition, ParallelTransition):
                overlap = branch_nodes.intersection(transition.branches)
                if overlap:
                    raise WorkflowDefinitionError(
                        f"parallel branch nodes may have only one owner: {sorted(overlap)}"
                    )
                branch_nodes.update(transition.branches)
                if transition.next_node is not None:
                    regular_targets.add(transition.next_node)

        invalid_branch_entries = branch_nodes.intersection(regular_targets)
        if invalid_branch_entries:
            raise WorkflowDefinitionError(
                "parallel branch nodes cannot be start or regular transition targets: "
                f"{sorted(invalid_branch_entries)}"
            )
        for branch in branch_nodes:
            if not isinstance(self.node(branch).transition, EndTransition):
                raise WorkflowDefinitionError(
                    f"parallel branch node '{branch}' must use EndTransition"
                )

        visited: set[str] = set()
        active: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in active:
                raise WorkflowDefinitionError("workflow graph must be acyclic")
            if node_id in visited:
                return
            active.add(node_id)
            for target in self._targets(self.node(node_id).transition):
                visit(target)
            active.remove(node_id)
            visited.add(node_id)

        visit(self.start_node)
        unreachable = set(self._node_map).difference(visited)
        if unreachable:
            raise WorkflowDefinitionError(
                f"workflow contains unreachable nodes: {sorted(unreachable)}"
            )

    @staticmethod
    def _targets(transition: WorkflowTransition) -> tuple[str, ...]:
        if isinstance(transition, EndTransition):
            return ()
        if isinstance(transition, NextTransition):
            return (transition.target,)
        if isinstance(transition, ConditionalTransition):
            return tuple(transition.routes.values())
        continuation = (transition.next_node,) if transition.next_node is not None else ()
        return (*transition.branches, *continuation)


@dataclass(frozen=True, slots=True)
class WorkflowRunResult:
    workflow_id: str
    run_id: str
    state: RunState
    data: Mapping[str, object]
    checkpoint: WorkflowCheckpoint
    events: tuple[RunEvent, ...]
    started_at: datetime
    completed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "data",
            MappingProxyType(_json_mapping(self.data, label="workflow result")),
        )


class WorkflowRunner:
    """Executes validated workflows with checkpoint-after-node semantics."""

    def __init__(
        self,
        checkpoint_store: CheckpointStore,
        *,
        max_parallelism: int = 4,
        default_timeout_seconds: float | None = 120.0,
        event_sinks: Sequence[EventSink] = (),
    ) -> None:
        if max_parallelism < 1:
            raise ValueError("maximum parallelism must be at least one")
        if default_timeout_seconds is not None and (
            not math.isfinite(default_timeout_seconds) or default_timeout_seconds <= 0
        ):
            raise ValueError("default workflow timeout must be finite and greater than zero")
        self._store = checkpoint_store
        self._max_parallelism = max_parallelism
        self._default_timeout_seconds = default_timeout_seconds
        self._event_sinks = tuple(event_sinks)

    def start(
        self,
        definition: WorkflowDefinition,
        initial_state: Mapping[str, object],
        *,
        workflow_id: str | None = None,
        context: RunContext | None = None,
    ) -> WorkflowRunResult:
        resolved_id = workflow_id or str(uuid4())
        if not resolved_id.strip():
            raise ValueError("workflow ID must not be empty")
        run_context = context or RunContext(timeout_seconds=self._default_timeout_seconds)
        emitter = RunEventEmitter(run_context, self._event_sinks)
        state = _json_mapping(initial_state, label="initial workflow state")
        checkpoint = WorkflowCheckpoint(
            workflow_id=resolved_id,
            definition_id=definition.definition_id,
            definition_version=definition.version,
            initial_state=state,
            state=state,
            current_node=definition.start_node,
            completed_nodes=(),
            status=RunState.RUNNING,
        )
        emitter.emit(
            RunEventType.RUN_STARTED,
            RunState.RUNNING,
            self._run_attributes(checkpoint, resumed=False),
        )
        try:
            checkpoint = self._save(checkpoint, expected_revision=None, emitter=emitter)
            return self._execute(definition, checkpoint, run_context, emitter)
        except Exception as exc:
            self._emit_start_failure(exc, run_context, emitter)
            raise

    def resume(
        self,
        definition: WorkflowDefinition,
        workflow_id: str,
        *,
        context: RunContext | None = None,
    ) -> WorkflowRunResult:
        checkpoint = self._store.load(workflow_id)
        if checkpoint is None:
            raise CheckpointError(f"workflow checkpoint '{workflow_id}' does not exist")
        self._validate_compatibility(definition, checkpoint)
        if checkpoint.status is RunState.COMPLETED:
            raise WorkflowExecutionError(f"workflow '{workflow_id}' is already completed")
        run_context = context or RunContext(timeout_seconds=self._default_timeout_seconds)
        emitter = RunEventEmitter(run_context, self._event_sinks)
        emitter.emit(
            RunEventType.RUN_STARTED,
            RunState.RUNNING,
            self._run_attributes(checkpoint, resumed=True),
        )
        try:
            checkpoint = self._save(
                replace(checkpoint, status=RunState.RUNNING, last_error=None),
                expected_revision=checkpoint.revision,
                emitter=emitter,
            )
            emitter.emit(
                RunEventType.RUN_RESUMED,
                RunState.RUNNING,
                {
                    "workflow_id": workflow_id,
                    "checkpoint_revision": checkpoint.revision,
                    "current_node": checkpoint.current_node,
                },
            )
            return self._execute(definition, checkpoint, run_context, emitter)
        except Exception as exc:
            self._emit_start_failure(exc, run_context, emitter)
            raise

    def replay(
        self,
        definition: WorkflowDefinition,
        source_workflow_id: str,
        *,
        workflow_id: str | None = None,
        context: RunContext | None = None,
    ) -> WorkflowRunResult:
        source = self._store.load(source_workflow_id)
        if source is None:
            raise CheckpointError(
                f"source workflow checkpoint '{source_workflow_id}' does not exist"
            )
        self._validate_compatibility(definition, source)
        replay_id = workflow_id or str(uuid4())
        if replay_id == source_workflow_id:
            raise ValueError("replay requires a new workflow ID")
        return self.start(
            definition,
            source.initial_state,
            workflow_id=replay_id,
            context=context,
        )

    def _execute(
        self,
        definition: WorkflowDefinition,
        checkpoint: WorkflowCheckpoint,
        context: RunContext,
        emitter: RunEventEmitter,
    ) -> WorkflowRunResult:
        current = checkpoint
        try:
            while current.current_node is not None:
                context.check_active()
                node = definition.node(current.current_node)
                state = dict(current.state)
                result = self._execute_node(
                    definition,
                    current.workflow_id,
                    node,
                    state,
                    context,
                    emitter,
                )
                state.update(result.updates)
                completed = (*current.completed_nodes, node.node_id)
                next_node: str | None
                transition = node.transition
                if isinstance(transition, ParallelTransition):
                    if result.interrupt_reason is not None:
                        raise WorkflowExecutionError(
                            "parallel coordinator cannot interrupt before its branches"
                        )
                    branch_updates = self._execute_parallel(
                        definition,
                        current.workflow_id,
                        transition,
                        state,
                        context,
                        emitter,
                    )
                    state.update(branch_updates)
                    completed = (*completed, *transition.branches)
                    next_node = transition.next_node
                    self._emit_transition(
                        emitter,
                        node.node_id,
                        "parallel",
                        next_node,
                        branches=transition.branches,
                    )
                else:
                    next_node = self._select_next(
                        node,
                        transition,
                        state,
                        context,
                        emitter,
                    )

                status = (
                    RunState.INTERRUPTED
                    if result.interrupt_reason is not None
                    else RunState.COMPLETED
                    if next_node is None
                    else RunState.RUNNING
                )
                if status is RunState.INTERRUPTED and next_node is None:
                    raise WorkflowExecutionError(
                        f"terminal node '{node.node_id}' cannot interrupt the workflow"
                    )
                current = self._save(
                    replace(
                        current,
                        state=state,
                        current_node=next_node,
                        completed_nodes=completed,
                        status=status,
                        last_error=None,
                    ),
                    expected_revision=current.revision,
                    emitter=emitter,
                )
                if status is RunState.INTERRUPTED:
                    emitter.emit(
                        RunEventType.RUN_INTERRUPTED,
                        RunState.INTERRUPTED,
                        {
                            "workflow_id": current.workflow_id,
                            "reason": result.interrupt_reason,
                            "next_node": next_node,
                        },
                    )
                    return self._result(current, context, emitter)

            emitter.emit(
                RunEventType.RUN_COMPLETED,
                RunState.COMPLETED,
                {
                    "workflow_id": current.workflow_id,
                    "completed_nodes": list(current.completed_nodes),
                },
            )
            return self._result(current, context, emitter)
        except RunCancelledError as exc:
            self._save_terminal_failure(current, RunState.CANCELLED, exc, emitter)
            emitter.emit(
                RunEventType.RUN_CANCELLED,
                RunState.CANCELLED,
                self._error_attributes(current.workflow_id, exc),
            )
            raise
        except RunDeadlineExceededError as exc:
            self._save_terminal_failure(current, RunState.TIMED_OUT, exc, emitter)
            emitter.emit(
                RunEventType.RUN_TIMED_OUT,
                RunState.TIMED_OUT,
                self._error_attributes(current.workflow_id, exc),
            )
            raise
        except CheckpointError as exc:
            emitter.emit(
                RunEventType.RUN_FAILED,
                RunState.FAILED,
                self._error_attributes(current.workflow_id, exc),
            )
            raise
        except DQAgentError as exc:
            self._save_terminal_failure(current, RunState.FAILED, exc, emitter)
            emitter.emit(
                RunEventType.RUN_FAILED,
                RunState.FAILED,
                self._error_attributes(current.workflow_id, exc),
            )
            raise
        except Exception as exc:
            error = WorkflowExecutionError(
                "unexpected workflow execution failure",
                run_id=context.run_id,
            )
            self._save_terminal_failure(current, RunState.FAILED, error, emitter)
            emitter.emit(
                RunEventType.RUN_FAILED,
                RunState.FAILED,
                {
                    **self._error_attributes(current.workflow_id, error),
                    "cause_type": type(exc).__name__,
                },
            )
            raise error from exc

    def _execute_node(
        self,
        definition: WorkflowDefinition,
        workflow_id: str,
        node: WorkflowNode,
        state: Mapping[str, object],
        context: RunContext,
        emitter: RunEventEmitter,
    ) -> NodeResult:
        idempotency_key = self._idempotency_key(definition, workflow_id, node.node_id)
        emitter.emit(
            RunEventType.WORKFLOW_NODE_STARTED,
            RunState.RUNNING,
            {
                "workflow_id": workflow_id,
                "node_id": node.node_id,
                "idempotency_key": idempotency_key,
            },
        )
        child = context.child(
            metadata={
                "workflow_id": workflow_id,
                "workflow_node": node.node_id,
                "idempotency_key": idempotency_key,
            }
        )
        try:
            result = node.handler(
                MappingProxyType(_json_mapping(state, label="workflow state")),
                child,
            )
            if not isinstance(result, NodeResult):
                raise TypeError("workflow node handler must return NodeResult")
            child.check_active()
        except (RunCancelledError, RunDeadlineExceededError):
            emitter.emit(
                RunEventType.WORKFLOW_NODE_FAILED,
                RunState.RUNNING,
                {"workflow_id": workflow_id, "node_id": node.node_id},
            )
            raise
        except Exception as exc:
            emitter.emit(
                RunEventType.WORKFLOW_NODE_FAILED,
                RunState.RUNNING,
                {
                    "workflow_id": workflow_id,
                    "node_id": node.node_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            if isinstance(exc, DQAgentError):
                raise
            raise WorkflowExecutionError(
                f"workflow node '{node.node_id}' failed",
                run_id=context.run_id,
            ) from exc
        emitter.emit(
            RunEventType.WORKFLOW_NODE_COMPLETED,
            RunState.RUNNING,
            {
                "workflow_id": workflow_id,
                "node_id": node.node_id,
                "updated_keys": sorted(result.updates),
                "interrupted": result.interrupt_reason is not None,
            },
        )
        return result

    def _execute_parallel(
        self,
        definition: WorkflowDefinition,
        workflow_id: str,
        transition: ParallelTransition,
        state: Mapping[str, object],
        context: RunContext,
        emitter: RunEventEmitter,
    ) -> dict[str, object]:
        if len(transition.branches) > self._max_parallelism:
            raise WorkflowExecutionError(
                f"parallel branch count {len(transition.branches)} exceeds configured maximum "
                f"{self._max_parallelism}"
            )
        contexts: dict[str, RunContext] = {}
        futures: dict[str, Future[NodeResult]] = {}
        executor = ThreadPoolExecutor(
            max_workers=len(transition.branches),
            thread_name_prefix="dqagent-workflow",
        )
        try:
            for branch_id in transition.branches:
                branch = definition.node(branch_id)
                idempotency_key = self._idempotency_key(
                    definition, workflow_id, branch_id
                )
                emitter.emit(
                    RunEventType.WORKFLOW_NODE_STARTED,
                    RunState.RUNNING,
                    {
                        "workflow_id": workflow_id,
                        "node_id": branch_id,
                        "parallel": True,
                        "idempotency_key": idempotency_key,
                    },
                )
                branch_context = context.child(
                    metadata={
                        "workflow_id": workflow_id,
                        "workflow_node": branch_id,
                        "idempotency_key": idempotency_key,
                        "parallel": True,
                    }
                )
                contexts[branch_id] = branch_context
                snapshot = _json_mapping(state, label="parallel input state")
                futures[branch_id] = executor.submit(
                    branch.handler,
                    MappingProxyType(snapshot),
                    branch_context,
                )

            done, _ = wait(futures.values(), return_when=FIRST_EXCEPTION)
            if any(future.exception() is not None for future in done):
                for branch_context in contexts.values():
                    branch_context.cancel("parallel sibling failed")
            wait(futures.values())
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        results: dict[str, NodeResult] = {}
        errors: list[tuple[str, BaseException]] = []
        for branch_id in transition.branches:
            future = futures[branch_id]
            try:
                result = future.result()
                if not isinstance(result, NodeResult):
                    raise TypeError("workflow node handler must return NodeResult")
                contexts[branch_id].check_active()
                if result.interrupt_reason is not None:
                    raise WorkflowExecutionError(
                        f"parallel branch '{branch_id}' cannot interrupt the workflow"
                    )
                results[branch_id] = result
                emitter.emit(
                    RunEventType.WORKFLOW_NODE_COMPLETED,
                    RunState.RUNNING,
                    {
                        "workflow_id": workflow_id,
                        "node_id": branch_id,
                        "parallel": True,
                        "updated_keys": sorted(result.updates),
                    },
                )
            except BaseException as exc:
                errors.append((branch_id, exc))
                emitter.emit(
                    RunEventType.WORKFLOW_NODE_FAILED,
                    RunState.RUNNING,
                    {
                        "workflow_id": workflow_id,
                        "node_id": branch_id,
                        "parallel": True,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
        if errors:
            primary = next(
                (error for _, error in errors if not isinstance(error, RunCancelledError)),
                errors[0][1],
            )
            if isinstance(primary, (RunCancelledError, RunDeadlineExceededError)):
                raise primary
            if isinstance(primary, DQAgentError):
                raise primary
            failed_node = next(node_id for node_id, error in errors if error is primary)
            raise WorkflowExecutionError(
                f"parallel workflow node '{failed_node}' failed",
                run_id=context.run_id,
            ) from primary

        merged: dict[str, object] = {}
        owners: dict[str, str] = {}
        for branch_id in transition.branches:
            for key, value in results[branch_id].updates.items():
                if key in owners:
                    raise WorkflowExecutionError(
                        f"parallel branches '{owners[key]}' and '{branch_id}' both update "
                        f"state key '{key}'"
                    )
                owners[key] = branch_id
                merged[key] = value
        return merged

    def _select_next(
        self,
        node: WorkflowNode,
        transition: EndTransition | NextTransition | ConditionalTransition,
        state: Mapping[str, object],
        context: RunContext,
        emitter: RunEventEmitter,
    ) -> str | None:
        if isinstance(transition, EndTransition):
            self._emit_transition(emitter, node.node_id, "end", None)
            return None
        if isinstance(transition, NextTransition):
            self._emit_transition(emitter, node.node_id, "next", transition.target)
            return transition.target
        route_context = context.child(
            metadata={"workflow_node": node.node_id, "transition": "conditional"}
        )
        try:
            route = transition.selector(
                MappingProxyType(_json_mapping(state, label="conditional state")),
                route_context,
            )
        except (RunCancelledError, RunDeadlineExceededError):
            raise
        except Exception as exc:
            raise WorkflowExecutionError(
                f"conditional transition for node '{node.node_id}' failed",
                run_id=context.run_id,
            ) from exc
        target = transition.routes.get(route)
        if target is None:
            raise WorkflowExecutionError(
                f"conditional transition for node '{node.node_id}' returned unknown route "
                f"'{route}'",
                run_id=context.run_id,
            )
        self._emit_transition(
            emitter,
            node.node_id,
            "conditional",
            target,
            route=route,
        )
        return target

    @staticmethod
    def _emit_transition(
        emitter: RunEventEmitter,
        node_id: str,
        kind: str,
        target: str | None,
        **attributes: object,
    ) -> None:
        emitter.emit(
            RunEventType.WORKFLOW_TRANSITION_SELECTED,
            RunState.RUNNING,
            {"node_id": node_id, "kind": kind, "target": target, **attributes},
        )

    def _save(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        expected_revision: int | None,
        emitter: RunEventEmitter,
    ) -> WorkflowCheckpoint:
        saved = self._store.save(checkpoint, expected_revision=expected_revision)
        emitter.emit(
            RunEventType.CHECKPOINT_SAVED,
            saved.status,
            {
                "workflow_id": saved.workflow_id,
                "revision": saved.revision,
                "current_node": saved.current_node,
                "status": saved.status.value,
            },
        )
        return saved

    def _save_terminal_failure(
        self,
        checkpoint: WorkflowCheckpoint,
        status: RunState,
        error: BaseException,
        emitter: RunEventEmitter,
    ) -> None:
        try:
            self._save(
                replace(
                    checkpoint,
                    status=status,
                    last_error={"type": type(error).__name__, "message": str(error)},
                ),
                expected_revision=checkpoint.revision,
                emitter=emitter,
            )
        except CheckpointError as checkpoint_error:
            emitter.emit(
                RunEventType.RUN_FAILED,
                RunState.FAILED,
                {
                    **self._error_attributes(checkpoint.workflow_id, checkpoint_error),
                    "original_error_type": type(error).__name__,
                },
            )
            raise

    @staticmethod
    def _validate_compatibility(
        definition: WorkflowDefinition,
        checkpoint: WorkflowCheckpoint,
    ) -> None:
        if (
            checkpoint.definition_id != definition.definition_id
            or checkpoint.definition_version != definition.version
        ):
            raise WorkflowDefinitionError(
                f"checkpoint expects workflow {checkpoint.definition_id}@"
                f"{checkpoint.definition_version}, got "
                f"{definition.definition_id}@{definition.version}"
            )
        if checkpoint.current_node is not None:
            try:
                definition.node(checkpoint.current_node)
            except KeyError as exc:
                raise WorkflowDefinitionError(
                    f"checkpoint current node '{checkpoint.current_node}' is not defined"
                ) from exc

    @staticmethod
    def _idempotency_key(
        definition: WorkflowDefinition,
        workflow_id: str,
        node_id: str,
    ) -> str:
        return f"{workflow_id}:{definition.definition_id}:{definition.version}:{node_id}"

    @staticmethod
    def _run_attributes(
        checkpoint: WorkflowCheckpoint,
        *,
        resumed: bool,
    ) -> Mapping[str, object]:
        return {
            "workflow_id": checkpoint.workflow_id,
            "definition_id": checkpoint.definition_id,
            "definition_version": checkpoint.definition_version,
            "checkpoint_revision": checkpoint.revision,
            "resumed": resumed,
        }

    @staticmethod
    def _error_attributes(
        workflow_id: str,
        error: DQAgentError,
    ) -> Mapping[str, object]:
        return {
            "workflow_id": workflow_id,
            "error_type": type(error).__name__,
            "error_category": error.category.value,
            "retryable": error.retryable,
            "message": str(error),
        }

    @staticmethod
    def _result(
        checkpoint: WorkflowCheckpoint,
        context: RunContext,
        emitter: RunEventEmitter,
    ) -> WorkflowRunResult:
        return WorkflowRunResult(
            workflow_id=checkpoint.workflow_id,
            run_id=context.run_id,
            state=checkpoint.status,
            data=checkpoint.state,
            checkpoint=checkpoint,
            events=emitter.events,
            started_at=context.started_at,
            completed_at=datetime.now(UTC),
        )

    @staticmethod
    def _emit_start_failure(
        error: BaseException,
        context: RunContext,
        emitter: RunEventEmitter,
    ) -> None:
        if any(
            event.type
            in {
                RunEventType.RUN_FAILED,
                RunEventType.RUN_CANCELLED,
                RunEventType.RUN_TIMED_OUT,
            }
            for event in emitter.events
        ):
            return
        emitter.emit(
            RunEventType.RUN_FAILED,
            RunState.FAILED,
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "run_id": context.run_id,
            },
        )
