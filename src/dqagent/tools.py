"""Application-owned legacy and governed tool definitions and execution."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
from typing import Protocol

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError

from dqagent.errors import RunCancelledError, RunDeadlineExceededError
from dqagent.events import RunEventType
from dqagent.execution import RunContext
from dqagent.models import ToolCall, ToolDefinition, ToolErrorCode, ToolOutcome, ToolResult
from dqagent.tool_governance import (
    HARD_GUARD_ORDER,
    ActionExecutionResult,
    ActionPolicy,
    ActionRecord,
    ApprovalDecision,
    ApprovalOutcome,
    ApprovalProvider,
    DefaultActionPolicy,
    EffectState,
    GovernanceDecision,
    GuardContext,
    GuardName,
    GuardResult,
    HookResult,
    PolicyDecision,
    PolicyOutcome,
    PostActionHook,
    PreActionHook,
    PreActionHookSpec,
    PreparedAction,
    authorize_action,
    build_action_record,
    build_post_action_hook_input,
    build_pre_action_hook_input,
    evaluate_action,
    evaluate_guards,
    run_post_hooks,
    run_pre_hooks,
)

logger = logging.getLogger(__name__)

DEFAULT_GOVERNED_ARGUMENT_BYTES = 64_000
PHASE9_RESERVED_TOOL_NAMES = frozenset(
    {
        "workspace_read",
        "workspace_search",
        "workspace_patch",
        "workspace_command",
    }
)


class ToolHandler(Protocol):
    def __call__(self, arguments: Mapping[str, object], context: RunContext) -> str: ...


class ActionPreparer(Protocol):
    def __call__(
        self,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> PreparedAction: ...


class ActionExecutor(Protocol):
    def __call__(
        self,
        action: PreparedAction,
        context: ToolExecutionContext,
    ) -> ActionExecutionResult | str: ...


class ActionCleanup(Protocol):
    def __call__(
        self,
        action: PreparedAction | None,
        context: ToolExecutionContext,
    ) -> None: ...


class ActionOutputSanitizer(Protocol):
    def __call__(
        self,
        guard_context: GuardContext,
        output: str,
        max_characters: int,
    ) -> tuple[str, bool, bool]: ...


class ActionPreparationError(Exception):
    """A bounded, model-visible failure while constructing an action."""

    def __init__(self, code: ToolErrorCode, message: str) -> None:
        if not isinstance(code, ToolErrorCode):
            raise TypeError("action preparation error code must be a ToolErrorCode")
        super().__init__(message)
        self.code = code


class ActionExecutionError(Exception):
    """A typed executor failure that preserves its observed effect state."""

    def __init__(
        self,
        code: ToolErrorCode,
        message: str,
        *,
        effect_state: EffectState,
        diagnostics: Sequence[str] = (),
    ) -> None:
        if not isinstance(code, ToolErrorCode):
            raise TypeError("action execution error code must be a ToolErrorCode")
        if not isinstance(effect_state, EffectState):
            raise TypeError("action execution error effect state must be an EffectState")
        if not isinstance(diagnostics, tuple):
            diagnostics = tuple(diagnostics)
        if len(diagnostics) > 8 or any(not isinstance(item, str) for item in diagnostics):
            raise ValueError("action execution error diagnostics are malformed or unbounded")
        super().__init__(message)
        self.code = code
        self.effect_state = effect_state
        self.diagnostics = diagnostics


class GuardContextFactory(Protocol):
    def __call__(
        self,
        action: PreparedAction,
        context: ToolExecutionContext,
    ) -> GuardContext: ...


class _ActionRecordCollector(Protocol):
    run_id: str
    max_records: int

    def reserved_count(self, run_id: str) -> int: ...

    def reserve(self, run_id: str) -> object: ...

    def append(self, run_id: str, record: ActionRecord, reservation: object) -> None: ...


class _ActionRecordCollectorError(RuntimeError):
    """Internal run-scoped action-record collection failure."""


class _CollectorCapacityError(_ActionRecordCollectorError):
    pass


class _CollectorObservationError(_ActionRecordCollectorError):
    pass


class _RunActionRecordCollector:
    """Private synchronous retention for one active run."""

    def __init__(self, run_id: str, max_records: int) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("collector run ID must be non-empty text")
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1:
            raise ValueError("collector capacity must be a positive integer")
        self.run_id = run_id
        self.max_records = max_records
        self._records: list[ActionRecord] = []
        self._reservations: dict[int, bool] = {}
        self._next_reservation = 1
        self._closed = False

    @property
    def records(self) -> tuple[ActionRecord, ...]:
        return tuple(self._records)

    @property
    def closed(self) -> bool:
        return self._closed

    def reserved_count(self, run_id: str) -> int:
        self._check_run(run_id)
        return len(self._reservations)

    def reserve(self, run_id: str) -> object:
        self._check_run(run_id)
        if len(self._reservations) >= self.max_records:
            raise _CollectorCapacityError("action record collector capacity exhausted")
        token = self._next_reservation
        self._next_reservation += 1
        self._reservations[token] = False
        return token

    def append(self, run_id: str, record: ActionRecord, reservation: object) -> None:
        self._check_run(run_id)
        if not isinstance(record, ActionRecord):
            raise _CollectorObservationError("collector received an invalid action record")
        if not isinstance(reservation, int) or reservation not in self._reservations:
            raise _CollectorObservationError("collector reservation does not belong to the run")
        if self._reservations[reservation]:
            raise _CollectorObservationError("collector reservation was appended twice")
        self._reservations[reservation] = True
        self._records.append(record)

    def close(self, run_id: str) -> tuple[ActionRecord, ...]:
        self._check_run(run_id)
        retained = tuple(self._records)
        self._records.clear()
        self._reservations.clear()
        self._closed = True
        return retained

    def _check_run(self, run_id: str) -> None:
        if self._closed:
            raise _CollectorObservationError("action record collector is closed")
        if run_id != self.run_id:
            raise _CollectorObservationError("action record collector run ID mismatch")


@dataclass
class _ToolRunState:
    collector: _ActionRecordCollector | None
    owns_collector: bool = False
    local_reservations: set[object] = field(default_factory=set)


class ToolExecutionContext:
    """Minimal provider-neutral context passed through the tool execution port."""

    def __init__(
        self,
        run_context: RunContext,
        *,
        emit_stage: Callable[[str, Mapping[str, object]], None] | None = None,
        record_collector: _ActionRecordCollector | None = None,
        max_governed_calls: int = 1,
        _state: _ToolRunState | None = None,
    ) -> None:
        if not isinstance(run_context, RunContext):
            raise TypeError("tool execution context requires a RunContext")
        self._run_context = run_context
        self._emit_stage_callback = emit_stage
        if _state is not None and record_collector is not None:
            raise ValueError("a derived tool execution context cannot replace its collector")
        if _state is not None:
            self._state = _state
        else:
            owns_collector = record_collector is None
            collector = record_collector or _RunActionRecordCollector(
                run_context.run_id,
                max_governed_calls,
            )
            self._state = _ToolRunState(collector, owns_collector=owns_collector)

    @property
    def run_context(self) -> RunContext:
        return self._run_context

    @property
    def context(self) -> RunContext:
        """Compatibility alias for callers that use the shorter context name."""

        return self._run_context

    def with_stage_emitter(
        self,
        emit_stage: Callable[[str, Mapping[str, object]], None] | None,
    ) -> ToolExecutionContext:
        return ToolExecutionContext(
            self._run_context,
            emit_stage=emit_stage,
            _state=self._state,
        )

    def _close_owned_records(self) -> None:
        if not self._state.owns_collector:
            return
        collector = self._state.collector
        if isinstance(collector, _RunActionRecordCollector):
            collector.close(self._run_context.run_id)

    def emit_stage(self, event_type: str, attributes: Mapping[str, object]) -> None:
        callback = self._emit_stage_callback
        if callback is None:
            return
        try:
            callback(event_type, attributes)
        except Exception:
            logger.exception(
                "tool stage event emission failed",
                extra={"run_id": self._run_context.run_id, "event_type": event_type},
            )

    def governed_call_count(
        self,
        base_count: int,
        *,
        excluding: object | None = None,
    ) -> int:
        if isinstance(base_count, bool) or not isinstance(base_count, int) or base_count < 0:
            raise _CollectorObservationError("malformed governed call count")
        collector = self._state.collector
        if collector is None:
            raise _CollectorObservationError("governed call collector is unavailable")
        count = collector.reserved_count(self._run_context.run_id)
        if excluding is not None:
            if excluding not in self._state.local_reservations:
                raise _CollectorObservationError("unknown governed call reservation")
            count -= 1
        return base_count + count

    def reserve_governed_call(self, max_calls: int, base_count: int) -> object:
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
            raise _CollectorObservationError("malformed governed call capacity")
        if isinstance(base_count, bool) or not isinstance(base_count, int) or base_count < 0:
            raise _CollectorObservationError("malformed governed call count")
        if base_count >= max_calls:
            raise _CollectorCapacityError("max governed calls exhausted")
        collector = self._state.collector
        if collector is None:
            raise _CollectorObservationError("governed call collector is unavailable")
        if collector.max_records != max_calls:
            raise _CollectorObservationError("collector capacity does not match governed limit")
        token = collector.reserve(self._run_context.run_id)
        self._state.local_reservations.add(token)
        return token

    def append_action_record(self, record: ActionRecord, reservation: object) -> None:
        collector = self._state.collector
        if collector is None:
            raise _CollectorObservationError("governed call collector is unavailable")
        collector.append(self._run_context.run_id, record, reservation)
        self._state.local_reservations.discard(reservation)


@dataclass(frozen=True, slots=True)
class Tool:
    definition: ToolDefinition
    handler: ToolHandler
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("tool timeout must be a finite number greater than zero")


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """Separates the model-visible result from internal diagnostic details."""

    result: ToolResult
    error_type: str | None = None
    error_message: str | None = None
    action_record: ActionRecord | None = None


class ToolRegistry:
    """Explicit name-to-tool registry and the single tool execution boundary."""

    def __init__(self, tools: Sequence[Tool | ActionTool] = ()) -> None:
        self._tools: dict[str, Tool | ActionTool] = {}
        for tool in tools:
            self.register(tool)

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._tools.values())

    def register(self, tool: Tool | ActionTool) -> None:
        if isinstance(tool, ActionTool):
            self.register_action(tool)
            return
        if not isinstance(tool, Tool):
            raise TypeError("registry accepts Tool or ActionTool values")
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"tool '{name}' is already registered")
        if name in PHASE9_RESERVED_TOOL_NAMES:
            raise ValueError(f"tool '{name}' is reserved for governed action registration")
        self._validate_definition(tool.definition)
        self._tools[name] = tool

    def register_action(self, tool: ActionTool) -> None:
        if not isinstance(tool, ActionTool):
            raise TypeError("governed registration requires an ActionTool")
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"tool '{name}' is already registered")
        self._validate_definition(tool.definition)
        self._tools[name] = tool

    @staticmethod
    def _validate_definition(definition: ToolDefinition) -> None:
        name = definition.name
        if definition.input_schema.get("type") != "object":
            raise ValueError(f"tool '{name}' input schema must describe an object")
        try:
            Draft202012Validator.check_schema(definition.input_schema)
        except SchemaError as exc:
            raise ValueError(f"tool '{name}' has an invalid input schema: {exc.message}") from exc

    def execute(
        self,
        call: ToolCall,
        context: RunContext | None = None,
        *,
        execution_context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        run_context = context or RunContext()
        return self.execute_detailed(
            call,
            run_context,
            execution_context=execution_context,
        ).result

    def execute_detailed(
        self,
        call: ToolCall,
        context: RunContext,
        *,
        execution_context: ToolExecutionContext | None = None,
    ) -> ToolExecution:
        context.check_active()
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolExecution(
                self._error(call, ToolErrorCode.UNKNOWN_TOOL, f"unknown tool: {call.name}")
            )

        if isinstance(tool, ActionTool):
            if execution_context is None:
                return ToolExecution(
                    self._error(
                        call,
                        ToolErrorCode.OBSERVATION_FAILURE,
                        "governed execution requires an explicit ToolExecutionContext",
                    ),
                    "ActionExecutionContextError",
                    "governed execution requires an explicit ToolExecutionContext",
                )
            action_context = execution_context
            if action_context.run_context is not context:
                return ToolExecution(
                    self._error(
                        call,
                        ToolErrorCode.OBSERVATION_FAILURE,
                        "governed execution context is not the active RunContext",
                    ),
                    "ActionExecutionContextError",
                    "governed execution context is not the active RunContext",
                )
            return tool.execute_detailed(call, action_context)

        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as exc:
            return ToolExecution(
                self._error(
                    call,
                    ToolErrorCode.INVALID_ARGUMENTS,
                    f"invalid JSON arguments: {exc.msg}",
                ),
                type(exc).__name__,
                str(exc),
            )
        if not isinstance(arguments, dict):
            return ToolExecution(
                self._error(
                    call,
                    ToolErrorCode.INVALID_ARGUMENTS,
                    "tool arguments must be a JSON object",
                )
            )

        try:
            Draft202012Validator(
                tool.definition.input_schema,
                format_checker=FormatChecker(),
            ).validate(arguments)
        except ValidationError as exc:
            return ToolExecution(
                self._error(
                    call,
                    ToolErrorCode.INVALID_ARGUMENTS,
                    f"arguments do not match the tool schema: {exc.message}",
                ),
                type(exc).__name__,
                str(exc),
            )

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"dqagent-{call.name}")
        future = executor.submit(tool.handler, arguments, context)
        tool_deadline = time.perf_counter() + tool.timeout_seconds
        try:
            while True:
                context.check_active()
                tool_remaining = tool_deadline - time.perf_counter()
                if tool_remaining <= 0:
                    future.cancel()
                    return ToolExecution(
                        self._error(
                            call,
                            ToolErrorCode.TIMEOUT,
                            f"tool timed out after {tool.timeout_seconds:g} seconds",
                        ),
                        "TimeoutError",
                        f"tool exceeded {tool.timeout_seconds:g} second timeout",
                    )
                context_remaining = context.remaining_seconds
                wait_seconds = min(0.05, tool_remaining)
                if context_remaining is not None:
                    wait_seconds = min(wait_seconds, context_remaining)
                try:
                    output = future.result(timeout=wait_seconds)
                    break
                except FutureTimeoutError:
                    continue
            if not isinstance(output, str) or not output.strip():
                return ToolExecution(
                    self._error(
                        call,
                        ToolErrorCode.EXECUTION_ERROR,
                        "tool returned an empty or non-text result",
                    ),
                    "InvalidToolOutput",
                    f"handler returned {type(output).__name__}",
                )
            return ToolExecution(ToolResult(call.call_id, call.name, output))
        except (RunCancelledError, RunDeadlineExceededError):
            future.cancel()
            raise
        except Exception as exc:  # Tool handlers are an untrusted application boundary.
            return ToolExecution(
                self._error(call, ToolErrorCode.EXECUTION_ERROR, "tool execution failed"),
                type(exc).__name__,
                str(exc),
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _error(call: ToolCall, code: ToolErrorCode, message: str) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            output=message,
            outcome=ToolOutcome.ERROR,
            error_code=code,
        )


@dataclass(frozen=True, slots=True)
class ActionTool:
    """Explicit governed adapter for one model-reachable action."""

    definition: ToolDefinition
    prepare: ActionPreparer
    executor: ActionExecutor
    guard_context: GuardContext | None = None
    guard_context_factory: GuardContextFactory | None = None
    policy: ActionPolicy = field(default_factory=DefaultActionPolicy)
    approval_provider: ApprovalProvider | None = None
    pre_hooks: tuple[PreActionHook | PreActionHookSpec, ...] = ()
    post_hooks: tuple[PostActionHook, ...] = ()
    secret_values: tuple[str, ...] = ()
    max_argument_bytes: int = DEFAULT_GOVERNED_ARGUMENT_BYTES
    output_sanitizer: ActionOutputSanitizer | None = None
    action_cleanup: ActionCleanup | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.definition, ToolDefinition):
            raise TypeError("action tool definition must be a ToolDefinition")
        if not callable(self.prepare) or not callable(self.executor):
            raise TypeError("action tool prepare and executor must be callable")
        if self.guard_context is None and self.guard_context_factory is None:
            raise ValueError("action tool requires a GuardContext or guard context factory")
        if self.guard_context is not None and not isinstance(self.guard_context, GuardContext):
            raise TypeError("action tool guard_context must be a GuardContext")
        if self.guard_context_factory is not None and not callable(self.guard_context_factory):
            raise TypeError("action tool guard_context_factory must be callable")
        if self.output_sanitizer is not None and not callable(self.output_sanitizer):
            raise TypeError("action output sanitizer must be callable")
        if self.action_cleanup is not None and not callable(self.action_cleanup):
            raise TypeError("action cleanup must be callable")
        if (
            isinstance(self.max_argument_bytes, bool)
            or not isinstance(self.max_argument_bytes, int)
            or self.max_argument_bytes < 1
        ):
            raise ValueError("action argument byte limit must be a positive integer")
        object.__setattr__(self, "pre_hooks", tuple(self.pre_hooks))
        object.__setattr__(self, "post_hooks", tuple(self.post_hooks))
        if isinstance(self.secret_values, (str, bytes)):
            raise TypeError("action secret values must be an iterable of strings")
        secrets = tuple(self.secret_values)
        if len(secrets) > 64 or any(not isinstance(value, str) or not value for value in secrets):
            raise ValueError("action secret values are malformed or exceed the bound")
        object.__setattr__(self, "secret_values", secrets)

    def execute_detailed(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecution:
        """Run the fixed governed path exactly once for one tool call."""

        run_context = context.run_context
        run_context.check_active()
        action: PreparedAction | None = None
        decision: GovernanceDecision | None = None
        current_guard_context: GuardContext | None = None
        reservation: object | None = None
        approval: ApprovalDecision | None = None
        pre_hook_results: tuple[HookResult, ...] = ()
        post_hook_results: tuple[HookResult, ...] = ()
        executor_attempts = 0
        effect_state = EffectState.NONE
        diagnostics: list[str] = []
        execution_error_code: ToolErrorCode | None = None
        revalidated_guard_count = 0
        expected_workspace = (
            self.guard_context.workspace
            if self.guard_context is not None and self.guard_context_factory is None
            else None
        )

        try:
            try:
                raw_bytes = call.arguments.encode("utf-8")
            except UnicodeEncodeError:
                self._emit(context, RunEventType.ACTION_OBSERVED, reason="invalid_utf8")
                return self._error_execution(
                    call,
                    ToolErrorCode.INVALID_ARGUMENTS,
                    "governed action arguments are not valid UTF-8",
                )
            if len(raw_bytes) > self.max_argument_bytes:
                self._emit(context, RunEventType.ACTION_OBSERVED, reason="argument_too_large")
                return self._error_execution(
                    call,
                    ToolErrorCode.ARGUMENT_TOO_LARGE,
                    "governed action arguments exceed the byte limit",
                )

            try:
                arguments = json.loads(raw_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._emit(context, RunEventType.ACTION_OBSERVED, reason="invalid_json")
                return self._error_execution(
                    call,
                    ToolErrorCode.INVALID_ARGUMENTS,
                    "governed action arguments are invalid JSON",
                )
            if not isinstance(arguments, dict):
                self._emit(context, RunEventType.ACTION_OBSERVED, reason="arguments_not_object")
                return self._error_execution(
                    call,
                    ToolErrorCode.INVALID_ARGUMENTS,
                    "governed action arguments must be a JSON object",
                )

            try:
                Draft202012Validator(
                    self.definition.input_schema,
                    format_checker=FormatChecker(),
                ).validate(arguments)
            except ValidationError:
                self._emit(context, RunEventType.ACTION_OBSERVED, reason="schema_invalid")
                return self._error_execution(
                    call,
                    ToolErrorCode.INVALID_ARGUMENTS,
                    "governed action arguments do not match the tool schema",
                )

            try:
                action = self.prepare(arguments, context)
                if not isinstance(action, PreparedAction):
                    raise TypeError("action preparer returned an invalid PreparedAction")
            except (RunCancelledError, RunDeadlineExceededError):
                raise
            except ActionPreparationError as error:
                self._emit(context, RunEventType.ACTION_OBSERVED, reason=error.code.value)
                return self._error_execution(
                    call,
                    error.code,
                    "governed action preparation failed",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            except Exception as error:
                self._emit(context, RunEventType.ACTION_OBSERVED, reason="preparation_failed")
                return self._error_execution(
                    call,
                    ToolErrorCode.GOVERNANCE_FAILURE,
                    "governed action preparation failed",
                    error_type=type(error).__name__,
                    error_message="governed action preparation failed",
                )

            self._emit(context, RunEventType.ACTION_PREPARED, action=action)
            try:
                current_guard_context = self._guard_context(action, context)
                if expected_workspace is None:
                    expected_workspace = current_guard_context.workspace
                decision = evaluate_action(
                    action,
                    current_guard_context,
                    self.policy,
                    expected_workspace=expected_workspace,
                )
            except (RunCancelledError, RunDeadlineExceededError):
                raise
            except _ActionRecordCollectorError:
                self._emit(
                    context,
                    RunEventType.ACTION_RECORD_FAILED,
                    action=action,
                    reason="collector_failure",
                )
                return self._error_execution(
                    call,
                    ToolErrorCode.OBSERVATION_FAILURE,
                    "governed action observation failed",
                )
            except Exception:
                self._emit(
                    context,
                    RunEventType.ACTION_OBSERVED,
                    action=action,
                    reason="governance_failed",
                )
                return self._error_execution(
                    call,
                    ToolErrorCode.GOVERNANCE_FAILURE,
                    "governed action evaluation failed",
                )

            failed_guard = next(
                (result for result in decision.guard_results if not result.passed),
                None,
            )
            self._emit(
                context,
                RunEventType.ACTION_GUARDS_EVALUATED,
                action=action,
                guard_count=len(decision.guard_results),
                guards_passed=decision.guards_passed,
                reason=failed_guard.reason if failed_guard is not None else None,
                guard_context=current_guard_context,
            )
            if decision.policy_evaluated:
                self._emit(
                    context,
                    RunEventType.ACTION_POLICY_DECIDED,
                    action=action,
                    policy_outcome=decision.policy_decision.outcome.value,
                    reason=decision.policy_decision.reason,
                    guard_context=current_guard_context,
                )

            max_guard_failed = (
                failed_guard is not None
                and failed_guard.name is GuardName.MAX_GOVERNED_CALLS
            )
            if not max_guard_failed:
                try:
                    reservation = context.reserve_governed_call(
                        current_guard_context.max_governed_calls,
                        current_guard_context.governed_call_count,
                    )
                except _CollectorCapacityError:
                    self._emit(
                        context,
                        RunEventType.ACTION_OBSERVED,
                        action=action,
                        reason="max_governed_calls_exhausted",
                        guard_context=current_guard_context,
                    )
                    return self._error_execution(
                        call,
                        ToolErrorCode.GOVERNED_CALL_LIMIT,
                        "maximum governed action calls exhausted",
                    )
                except Exception as error:
                    self._emit(
                        context,
                        RunEventType.ACTION_RECORD_FAILED,
                        action=action,
                        reason="collector_failure",
                    )
                    return self._error_execution(
                        call,
                        ToolErrorCode.OBSERVATION_FAILURE,
                        "governed action observation failed",
                        error_type=type(error).__name__,
                        error_message="governed action observation failed",
                    )

            if not decision.guards_passed:
                code = self._guard_error_code(failed_guard)
                record = self._build_record(
                    decision,
                    current_guard_context,
                    approval=approval,
                    pre_hook_results=pre_hook_results,
                    post_hook_results=post_hook_results,
                    executor_attempts=executor_attempts,
                    effect_state=effect_state,
                    diagnostics=diagnostics,
                    observation_failure=self._has_scope_binding_failure(
                        decision.guard_results
                    ),
                )
                return self._finish_recorded_error(
                    call,
                    context,
                    action,
                    reservation,
                    record,
                    code,
                    "governed action denied by a technical guard",
                )

            if decision.policy_decision.outcome is PolicyOutcome.DENY:
                record = self._build_record(
                    decision,
                    current_guard_context,
                    approval=approval,
                    pre_hook_results=pre_hook_results,
                    post_hook_results=post_hook_results,
                    executor_attempts=executor_attempts,
                    effect_state=effect_state,
                    diagnostics=diagnostics,
                )
                return self._finish_recorded_error(
                    call,
                    context,
                    action,
                    reservation,
                    record,
                    ToolErrorCode.POLICY_DENIED,
                    "governed action denied by policy",
                )

            if decision.requires_approval:
                self._emit(context, RunEventType.ACTION_APPROVAL_REQUESTED, action=action)
                try:
                    revalidation_context = self._guard_context(
                        action,
                        context,
                        excluding=reservation,
                    )
                    approval = authorize_action(
                        decision,
                        provider=self.approval_provider or _non_interactive_provider(),
                        context=run_context,
                        guard_context=current_guard_context,
                        execution_guard_context=revalidation_context,
                        secret_values=self.secret_values,
                    )
                except (RunCancelledError, RunDeadlineExceededError):
                    raise
                except _ActionRecordCollectorError:
                    self._emit(
                        context,
                        RunEventType.ACTION_RECORD_FAILED,
                        action=action,
                        reason="collector_failure",
                    )
                    return self._error_execution(
                        call,
                        ToolErrorCode.OBSERVATION_FAILURE,
                        "governed action observation failed",
                    )
                revalidated_guard_count = len(HARD_GUARD_ORDER)
                current_guard_context = revalidation_context
                self._emit(
                    context,
                    RunEventType.ACTION_APPROVAL_DECIDED,
                    action=action,
                    approval_outcome=approval.outcome.value,
                    reason=approval.reason,
                )
                if approval.outcome is not ApprovalOutcome.APPROVE:
                    record = self._build_record(
                        decision,
                        revalidation_context,
                        approval=approval,
                        pre_hook_results=pre_hook_results,
                        post_hook_results=post_hook_results,
                        executor_attempts=executor_attempts,
                        effect_state=effect_state,
                        diagnostics=diagnostics,
                    )
                    return self._finish_recorded_error(
                        call,
                        context,
                        action,
                        reservation,
                        record,
                        self._approval_error_code(approval),
                        "governed action approval was not granted",
                    )
            else:
                try:
                    revalidation_context = self._guard_context(
                        action,
                        context,
                        excluding=reservation,
                    )
                    fresh_guards = evaluate_guards(
                        action,
                        revalidation_context,
                        expected_workspace=expected_workspace,
                    )
                except (RunCancelledError, RunDeadlineExceededError):
                    raise
                except _ActionRecordCollectorError:
                    self._emit(
                        context,
                        RunEventType.ACTION_RECORD_FAILED,
                        action=action,
                        reason="collector_failure",
                    )
                    return self._error_execution(
                        call,
                        ToolErrorCode.OBSERVATION_FAILURE,
                        "governed action observation failed",
                    )
                revalidated_guard_count = len(fresh_guards)
                current_guard_context = revalidation_context
                if not all(result.passed for result in fresh_guards):
                    decision = self._decision_after_revalidation(
                        decision,
                        fresh_guards,
                        "action_revalidation_failed",
                    )
                    self._emit(
                        context,
                        RunEventType.ACTION_REVALIDATED,
                        action=action,
                        guards_passed=False,
                        guard_count=len(fresh_guards),
                        reason="action_revalidation_failed",
                    )
                    record = self._build_record(
                        decision,
                        revalidation_context,
                        approval=approval,
                        pre_hook_results=pre_hook_results,
                        post_hook_results=post_hook_results,
                        executor_attempts=executor_attempts,
                        effect_state=effect_state,
                        diagnostics=diagnostics,
                        observation_failure=self._has_scope_binding_failure(fresh_guards),
                    )
                    return self._finish_recorded_error(
                        call,
                        context,
                        action,
                        reservation,
                        record,
                        self._guard_error_code(
                            next(result for result in fresh_guards if not result.passed)
                        ),
                        "governed action changed before execution",
                    )

            self._emit(
                context,
                RunEventType.ACTION_REVALIDATED,
                action=action,
                guards_passed=True,
                guard_count=revalidated_guard_count,
            )
            pre_input = build_pre_action_hook_input(
                decision,
                run_context,
                revalidation_context,
                approval_decision=approval,
                secret_values=self.secret_values,
            )
            pre_result = run_pre_hooks(
                self.pre_hooks,
                pre_input,
                context=run_context,
                workspace=revalidation_context.workspace,
                secret_values=self.secret_values,
            )
            pre_hook_results = pre_result.results
            self._emit(
                context,
                RunEventType.ACTION_PRE_HOOKS_COMPLETED,
                action=action,
                hook_count=len(pre_hook_results),
                hook_failures=sum(result.failed for result in pre_hook_results),
                proceeded=pre_result.proceeded,
            )
            if not pre_result.proceeded:
                record = self._build_record(
                    decision,
                    revalidation_context,
                    approval=approval,
                    pre_hook_results=pre_hook_results,
                    post_hook_results=post_hook_results,
                    executor_attempts=executor_attempts,
                    effect_state=effect_state,
                    diagnostics=diagnostics,
                )
                return self._finish_recorded_error(
                    call,
                    context,
                    action,
                    reservation,
                    record,
                    ToolErrorCode.GOVERNANCE_FAILURE,
                    "required pre-hook blocked the governed action",
                )

            try:
                effect_guard_context = self._guard_context(
                    action,
                    context,
                    excluding=reservation,
                )
                effect_guards = evaluate_guards(
                    action,
                    effect_guard_context,
                    expected_workspace=expected_workspace,
                )
            except (RunCancelledError, RunDeadlineExceededError):
                raise
            except _ActionRecordCollectorError:
                self._emit(
                    context,
                    RunEventType.ACTION_RECORD_FAILED,
                    action=action,
                    reason="collector_failure",
                )
                return self._error_execution(
                    call,
                    ToolErrorCode.OBSERVATION_FAILURE,
                    "governed action observation failed",
                )
            effect_guards_passed = len(effect_guards) == len(HARD_GUARD_ORDER) and all(
                result.passed for result in effect_guards
            )
            current_guard_context = effect_guard_context
            self._emit(
                context,
                RunEventType.ACTION_EFFECT_REVALIDATED,
                action=action,
                guards_passed=effect_guards_passed,
                guard_count=len(effect_guards),
            )
            if not effect_guards_passed:
                decision = self._decision_after_revalidation(
                    decision,
                    effect_guards,
                    "effect_boundary_revalidation_failed",
                )
                record = self._build_record(
                    decision,
                    effect_guard_context,
                    approval=approval,
                    pre_hook_results=pre_hook_results,
                    post_hook_results=post_hook_results,
                    executor_attempts=executor_attempts,
                    effect_state=effect_state,
                    diagnostics=diagnostics,
                    observation_failure=self._has_scope_binding_failure(effect_guards),
                )
                return self._finish_recorded_error(
                    call,
                    context,
                    action,
                    reservation,
                    record,
                    self._guard_error_code(
                        next(result for result in effect_guards if not result.passed)
                    ),
                    "governed action changed at the effect boundary",
                )

            run_context.check_active()
            executor_attempts = 1
            self._emit(context, RunEventType.ACTION_EXECUTOR_STARTED, action=action, attempt=1)
            executor_error: Exception | None = None
            execution_result: ActionExecutionResult | None = None
            output = "governed action execution produced no observation"
            output_observation_failure = False
            output_truncated = False
            try:
                raw_execution = self.executor(action, context)
                execution_result = self._normalize_execution_result(raw_execution)
                effect_state = execution_result.effect_state
                diagnostics.extend(execution_result.diagnostics)
                run_context.check_active()
                output, output_truncated, output_observation_failure = self._sanitize_output(
                    effect_guard_context,
                    execution_result.output,
                    action.limits.max_output_characters,
                )
                if output_observation_failure:
                    diagnostics.append("action_output_sanitization_failed")
                elif output_truncated:
                    diagnostics.append("action_output_truncated")
            except (RunCancelledError, RunDeadlineExceededError):
                effect_state = EffectState.UNKNOWN
                raise
            except ActionExecutionError as error:
                executor_error = error
                effect_state = error.effect_state
                execution_error_code = error.code
                diagnostics.extend(error.diagnostics)
                output_observation_failure = False
            except Exception as error:
                executor_error = error
                effect_state = EffectState.UNKNOWN
                diagnostics.append(self._safe_text(effect_guard_context, str(error)))
                output_observation_failure = False
            self._emit(
                context,
                RunEventType.ACTION_EXECUTOR_COMPLETED,
                action=action,
                attempt=executor_attempts,
                effect_state=effect_state.value,
                executor_failed=executor_error is not None,
            )

            effect_diagnostic = diagnostics[0] if diagnostics else ""
            post_input = build_post_action_hook_input(
                pre_input,
                effect_state=effect_state,
                executor_attempts=executor_attempts,
                workspace=effect_guard_context.workspace,
                secret_values=self.secret_values,
                effect_diagnostic=effect_diagnostic,
            )
            post_result = run_post_hooks(
                self.post_hooks,
                post_input,
                context=run_context,
                workspace=effect_guard_context.workspace,
                secret_values=self.secret_values,
            )
            post_hook_results = post_result.results
            self._emit(
                context,
                RunEventType.ACTION_POST_HOOKS_COMPLETED,
                action=action,
                hook_count=len(post_hook_results),
                hook_failures=sum(result.failed for result in post_hook_results),
            )
            observation_failure = output_observation_failure
            record = self._build_record(
                decision,
                effect_guard_context,
                approval=approval,
                pre_hook_results=pre_hook_results,
                post_hook_results=post_hook_results,
                executor_attempts=executor_attempts,
                effect_state=effect_state,
                diagnostics=diagnostics,
                observation_failure=observation_failure,
            )
            stored_error = self._store_record(
                context,
                call,
                action,
                reservation,
                record,
            )
            if stored_error is not None:
                return stored_error
            self._emit(
                context,
                RunEventType.ACTION_OBSERVED,
                action=action,
                effect_state=effect_state.value,
                executor_attempts=executor_attempts,
                observation_failure=observation_failure,
            )
            if executor_error is not None:
                code = execution_error_code or (
                    ToolErrorCode.PROCESS_FAILURE
                    if action.effect_kind.value == "process_execution"
                    else ToolErrorCode.EXECUTION_ERROR
                )
                return self._execution_error(
                    call,
                    code,
                    "governed action executor failed",
                    record,
                    executor_error,
                    effect_guard_context,
                )
            if observation_failure:
                return self._error_execution(
                    call,
                    ToolErrorCode.OBSERVATION_FAILURE,
                    "governed action observation failed",
                    error_type="ActionObservationError",
                    error_message="governed action observation failed",
                    record=record,
                )
            return ToolExecution(
                ToolResult(call.call_id, call.name, output),
                action_record=record,
            )
        except (RunCancelledError, RunDeadlineExceededError) as control_error:
            partial_post_hook_results = getattr(
                control_error,
                "_dqagent_partial_post_hook_results",
                (),
            )
            if isinstance(partial_post_hook_results, tuple) and all(
                isinstance(result, HookResult) for result in partial_post_hook_results
            ):
                post_hook_results = partial_post_hook_results
            if action is not None and decision is not None and current_guard_context is not None:
                try:
                    control_effect = EffectState.UNKNOWN if executor_attempts else EffectState.NONE
                    record = self._build_record(
                        decision,
                        current_guard_context,
                        approval=approval,
                        pre_hook_results=pre_hook_results,
                        post_hook_results=post_hook_results,
                        executor_attempts=executor_attempts,
                        effect_state=control_effect,
                        diagnostics=diagnostics,
                    )
                    self._store_record(context, call, action, reservation, record)
                except Exception:
                    logger.exception(
                        "governed action control evidence failed",
                        extra={"run_id": run_context.run_id},
                    )
            raise
        finally:
            if self.action_cleanup is not None:
                self.action_cleanup(action, context)

    def _guard_context(
        self,
        action: PreparedAction,
        context: ToolExecutionContext,
        *,
        excluding: object | None = None,
    ) -> GuardContext:
        raw_context = (
            self.guard_context_factory(action, context)
            if self.guard_context_factory is not None
            else self.guard_context
        )
        if not isinstance(raw_context, GuardContext):
            raise TypeError("action guard context factory returned an invalid GuardContext")
        count = context.governed_call_count(
            raw_context.governed_call_count,
            excluding=excluding,
        )
        return replace(raw_context, governed_call_count=count)

    def _build_record(
        self,
        decision: GovernanceDecision,
        guard_context: GuardContext,
        *,
        approval: ApprovalDecision | None,
        pre_hook_results: tuple[HookResult, ...],
        post_hook_results: tuple[HookResult, ...],
        executor_attempts: int,
        effect_state: EffectState,
        diagnostics: Sequence[str],
        observation_failure: bool = False,
    ) -> ActionRecord:
        return build_action_record(
            decision,
            workspace=guard_context.workspace,
            secret_values=self.secret_values,
            backend_identity=guard_context.backend_identity,
            backend_capabilities=guard_context.available_capabilities,
            executor_attempts=executor_attempts,
            effect_state=effect_state,
            diagnostics=tuple(diagnostics),
            approval_decision=approval,
            pre_hook_results=pre_hook_results,
            post_hook_results=post_hook_results,
            observation_failure=observation_failure,
        )

    def _store_record(
        self,
        context: ToolExecutionContext,
        call: ToolCall,
        action: PreparedAction,
        reservation: object | None,
        record: ActionRecord,
    ) -> ToolExecution | None:
        if reservation is None:
            return None
        try:
            context.append_action_record(record, reservation)
        except Exception as error:
            failed_record = replace(
                record,
                observation_failure=True,
                sanitized_diagnostics=tuple(
                    (*record.sanitized_diagnostics, "action_record_collector_failed")[:8]
                ),
                diagnostics_truncated=record.diagnostics_truncated
                or len(record.sanitized_diagnostics) >= 8,
            )
            self._emit(
                context,
                RunEventType.ACTION_RECORD_FAILED,
                action=action,
                reason="collector_failure",
            )
            return ToolExecution(
                ToolRegistry._error(
                    call,
                    ToolErrorCode.OBSERVATION_FAILURE,
                    "governed action observation failed",
                ),
                type(error).__name__,
                "governed action record collection failed",
                failed_record,
            )
        return None

    def _finish_recorded_error(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        action: PreparedAction,
        reservation: object | None,
        record: ActionRecord,
        code: ToolErrorCode,
        message: str,
    ) -> ToolExecution:
        stored_error = self._store_record(context, call, action, reservation, record)
        if stored_error is not None:
            return ToolExecution(
                ToolRegistry._error(
                    call,
                    ToolErrorCode.OBSERVATION_FAILURE,
                    stored_error.result.output,
                ),
                stored_error.error_type,
                stored_error.error_message,
                stored_error.action_record,
            )
        self._emit(
            context,
            RunEventType.ACTION_OBSERVED,
            action=action,
            effect_state=record.effect_state.value,
            executor_attempts=record.executor_attempts,
        )
        return self._error_execution(call, code, message, record=record)

    def _decision_after_revalidation(
        self,
        decision: GovernanceDecision,
        guard_results: Sequence[GuardResult],
        reason: str,
    ) -> GovernanceDecision:
        results = tuple(guard_results)
        if all(result.passed for result in results) and len(results) == 7:
            return GovernanceDecision(
                decision.action,
                results,
                decision.policy_decision,
                decision.policy_evaluated,
            )
        return GovernanceDecision(
            decision.action,
            results,
            PolicyDecision(PolicyOutcome.DENY, reason, "hard-guards"),
            False,
        )

    def _emit(
        self,
        context: ToolExecutionContext,
        event_type: RunEventType,
        *,
        action: PreparedAction | None = None,
        reason: str | None = None,
        guard_context: GuardContext | None = None,
        **attributes: object,
    ) -> None:
        values: dict[str, object] = {"tool": self.definition.name, **attributes}
        if action is not None:
            values.update(
                {
                    "action_digest": action.canonical_digest,
                    "action_kind": action.action_kind.value,
                    "effect_kind": action.effect_kind.value,
                }
            )
            if reason is not None:
                values["reason"] = self._safe_text_from_action(
                    action,
                    reason,
                    guard_context=guard_context,
                )
        elif reason is not None:
            values["reason"] = reason[:160]
        context.emit_stage(event_type.value, values)

    def _safe_text_from_action(
        self,
        action: PreparedAction,
        value: str,
        *,
        guard_context: GuardContext | None = None,
    ) -> str:
        del action
        selected_context = guard_context
        if selected_context is None and self.guard_context_factory is None:
            selected_context = self.guard_context
        if selected_context is None:
            return "[unavailable]"
        return self._safe_text(selected_context, value)

    def _safe_text(self, guard_context: GuardContext, value: str) -> str:
        try:
            return guard_context.workspace.sanitize(
                value,
                secrets=self.secret_values,
                max_characters=160,
            )
        except Exception:
            return "[unavailable]"

    def _sanitize_output(
        self,
        guard_context: GuardContext,
        output: str,
        max_characters: int,
    ) -> tuple[str, bool, bool]:
        try:
            if self.output_sanitizer is not None:
                return self.output_sanitizer(guard_context, output, max_characters)
            rendered = guard_context.workspace.sanitizer(
                secrets=self.secret_values,
            ).sanitize_with_evidence(output, max_characters=max_characters)
            return rendered.text, rendered.truncated, False
        except Exception:
            return "governed action output unavailable", False, True

    @staticmethod
    def _has_scope_binding_failure(results: Sequence[GuardResult]) -> bool:
        return any(
            result.name is GuardName.WORKSPACE_IDENTITY
            and result.reason
            in {
                "workspace_scope_binding_mismatch",
                "workspace_identity_dependency_failure",
            }
            for result in results
        )

    @staticmethod
    def _normalize_execution_result(raw: ActionExecutionResult | str) -> ActionExecutionResult:
        if isinstance(raw, ActionExecutionResult):
            return raw
        if isinstance(raw, str):
            return ActionExecutionResult(raw)
        raise TypeError("action executor returned an invalid result")

    @staticmethod
    def _guard_error_code(result: object | None) -> ToolErrorCode:
        name = result.name if isinstance(result, object) and hasattr(result, "name") else None
        if name is GuardName.MAX_GOVERNED_CALLS:
            return ToolErrorCode.GOVERNED_CALL_LIMIT
        if name is GuardName.WORKSPACE_IDENTITY:
            return ToolErrorCode.CONTAINMENT_DENIED
        if name is GuardName.CURRENT_CONTAINMENT:
            return ToolErrorCode.CONTAINMENT_DENIED
        if name is GuardName.PROTECTED_SECRET:
            return ToolErrorCode.PROTECTED_RESOURCE_DENIED
        if name is GuardName.LIMITS:
            return ToolErrorCode.RESOURCE_LIMIT
        if name is GuardName.PRECONDITIONS:
            return ToolErrorCode.PRECONDITION_CONFLICT
        if name is GuardName.CAPABILITIES:
            return ToolErrorCode.CAPABILITY_MISSING
        return ToolErrorCode.GOVERNANCE_FAILURE

    @staticmethod
    def _approval_error_code(decision: ApprovalDecision) -> ToolErrorCode:
        if decision.outcome is ApprovalOutcome.REJECT:
            return ToolErrorCode.APPROVAL_REJECTED
        if decision.outcome in {
            ApprovalOutcome.IDENTITY_MISMATCH,
            ApprovalOutcome.DRIFT,
        }:
            return ToolErrorCode.APPROVAL_MISMATCH
        return ToolErrorCode.APPROVAL_UNAVAILABLE

    @staticmethod
    def _error_execution(
        call: ToolCall,
        code: ToolErrorCode,
        message: str,
        *,
        error_type: str | None = None,
        error_message: str | None = None,
        record: ActionRecord | None = None,
    ) -> ToolExecution:
        return ToolExecution(
            ToolRegistry._error(call, code, message),
            error_type or code.value,
            error_message or message,
            record,
        )

    def _execution_error(
        self,
        call: ToolCall,
        code: ToolErrorCode,
        message: str,
        record: ActionRecord,
        error: Exception,
        guard_context: GuardContext,
    ) -> ToolExecution:
        diagnostic = self._safe_text(guard_context, str(error))
        return self._error_execution(
            call,
            code,
            message,
            error_type=type(error).__name__,
            error_message=diagnostic,
            record=record,
        )


def _non_interactive_provider() -> ApprovalProvider:
    from dqagent.tool_governance import NonInteractiveApprovalProvider

    return NonInteractiveApprovalProvider()
