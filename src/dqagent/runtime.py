"""Observable runtime for one bounded agent execution."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from dqagent.errors import (
    AgentLoopError,
    AgentRuntimeError,
    DQAgentError,
    LLMProviderError,
)
from dqagent.events import EventSink, RunEvent, RunEventType, RunState
from dqagent.execution import RunContext
from dqagent.lifecycle import RunCoordinator, RunRecord, RunScope
from dqagent.llm import LLMClient
from dqagent.models import (
    Completion,
    ConversationItem,
    Message,
    Role,
    ToolCall,
    ToolErrorCode,
    ToolOutcome,
    ToolResult,
)
from dqagent.tools import ToolExecutionContext, ToolRegistry

__all__ = [
    "AgentRunResult",
    "AgentExecutionResult",
    "AgentRuntime",
    "EventSink",
    "RetryPolicy",
    "RunEvent",
    "RunEventType",
    "RunState",
]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_delay_seconds: float = 0.25
    multiplier: float = 2.0
    max_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max attempts must be at least one")
        for name, value in (
            ("initial delay", self.initial_delay_seconds),
            ("maximum delay", self.max_delay_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if not math.isfinite(self.multiplier) or self.multiplier < 1:
            raise ValueError("retry multiplier must be a finite number at least one")

    def delay_after(self, failed_attempt: int) -> float:
        if failed_attempt < 1:
            raise ValueError("failed attempt must be at least one")
        if (
            self.initial_delay_seconds == 0
            or self.multiplier == 1
            or self.initial_delay_seconds >= self.max_delay_seconds
        ):
            return min(self.initial_delay_seconds, self.max_delay_seconds)
        exponent = failed_attempt - 1
        exponent_to_cap = (
            math.log(self.max_delay_seconds) - math.log(self.initial_delay_seconds)
        ) / math.log(self.multiplier)
        if exponent >= exponent_to_cap:
            return self.max_delay_seconds
        try:
            delay = self.initial_delay_seconds * self.multiplier**exponent
        except OverflowError:
            return self.max_delay_seconds
        return min(delay, self.max_delay_seconds)


@dataclass(frozen=True, slots=True)
class AgentExecutionResult:
    """Model/tool loop output before end-to-end lifecycle metadata is attached."""

    output: Message
    conversation: tuple[ConversationItem, ...]
    new_items: tuple[ConversationItem, ...]
    iterations: int


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    run_id: str
    state: RunState
    output: Message
    conversation: tuple[ConversationItem, ...]
    new_items: tuple[ConversationItem, ...]
    events: tuple[RunEvent, ...]
    started_at: datetime
    completed_at: datetime

    @classmethod
    def from_execution(
        cls,
        execution: AgentExecutionResult,
        record: RunRecord,
    ) -> "AgentRunResult":
        if record.state is not RunState.COMPLETED:
            raise ValueError("an agent result requires a completed run record")
        return cls(
            run_id=record.run_id,
            state=record.state,
            output=execution.output,
            conversation=execution.conversation,
            new_items=execution.new_items,
            events=record.events,
            started_at=record.started_at,
            completed_at=record.completed_at,
        )


class AgentRuntime:
    """Executes the bounded model/tool stage of an agent run."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        *,
        max_iterations: int = 8,
        max_governed_calls: int = 1,
        retry_policy: RetryPolicy | None = None,
        default_timeout_seconds: float | None = 120.0,
        event_sinks: Sequence[EventSink] = (),
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least one")
        if (
            isinstance(max_governed_calls, bool)
            or not isinstance(max_governed_calls, int)
            or max_governed_calls < 1
        ):
            raise ValueError("max governed calls must be a positive integer")
        self._llm = llm
        self._tools = tools
        self._max_iterations = max_iterations
        self._max_governed_calls = max_governed_calls
        self._retry_policy = retry_policy or RetryPolicy()
        self._run_coordinator = RunCoordinator(
            default_timeout_seconds=default_timeout_seconds,
            event_sinks=event_sinks,
        )

    def run(
        self,
        conversation: Sequence[ConversationItem],
        *,
        context: RunContext | None = None,
        tool_context: ToolExecutionContext | None = None,
    ) -> AgentRunResult:
        if tool_context is not None and not isinstance(tool_context, ToolExecutionContext):
            raise TypeError("tool_context must be a ToolExecutionContext")
        if (
            tool_context is not None
            and context is not None
            and tool_context.run_context is not context
        ):
            raise ValueError("tool context and run context must be the same object")
        coordinated_context = context or (
            tool_context.run_context if tool_context is not None else None
        )
        coordinated = self._run_coordinator.execute(
            lambda scope: self.execute(
                conversation,
                scope=scope,
                tool_context=tool_context,
            ),
            context=coordinated_context,
            completion_attributes=lambda result: {"iterations": result.iterations},
        )
        return AgentRunResult.from_execution(coordinated.value, coordinated.record)

    def execute(
        self,
        conversation: Sequence[ConversationItem],
        *,
        scope: RunScope,
        tool_context: ToolExecutionContext | None = None,
    ) -> AgentExecutionResult:
        """Execute the model/tool loop inside an externally coordinated run."""
        run_context = scope.context
        if tool_context is not None and not isinstance(tool_context, ToolExecutionContext):
            raise TypeError("tool_context must be a ToolExecutionContext")
        if tool_context is not None and tool_context.run_context is not run_context:
            raise ValueError("tool context and run scope must share the same RunContext")
        owns_execution_context = tool_context is None
        execution_context = (
            tool_context
            or ToolExecutionContext(
                run_context,
                max_governed_calls=self._max_governed_calls,
            )
        ).with_stage_emitter(
            lambda event_type, attributes: scope.emit(RunEventType(event_type), attributes)
        )

        try:
            pending = list(conversation)
            initial_item_count = len(pending)
            seen_call_ids: set[str] = set()
            seen_executions: set[tuple[str, str]] = set()
            run_context.check_active()
            for iteration in range(1, self._max_iterations + 1):
                completion = self._complete_with_retry(
                    pending,
                    run_context,
                    scope,
                    iteration,
                )
                assistant_message = (
                    Message(Role.ASSISTANT, completion.content)
                    if completion.content is not None
                    else None
                )
                if assistant_message is not None:
                    pending.append(assistant_message)

                if not completion.tool_calls:
                    if assistant_message is None:
                        raise AgentLoopError("model returned neither text nor tool calls")
                    return AgentExecutionResult(
                        output=assistant_message,
                        conversation=tuple(pending),
                        new_items=tuple(pending[initial_item_count:]),
                        iterations=iteration,
                    )

                pending.extend(completion.tool_calls)
                for call in completion.tool_calls:
                    run_context.check_active()
                    scope.emit(
                        RunEventType.TOOL_CALL_STARTED,
                        {"iteration": iteration, "call_id": call.call_id, "tool": call.name},
                    )
                    execution_key = self._execution_key(call)
                    if call.call_id in seen_call_ids or execution_key in seen_executions:
                        result = ToolResult(
                            call_id=call.call_id,
                            name=call.name,
                            output="repeated tool call rejected",
                            outcome=ToolOutcome.ERROR,
                            error_code=ToolErrorCode.REPEATED_CALL,
                        )
                        diagnostic: Mapping[str, object] = {}
                    else:
                        seen_call_ids.add(call.call_id)
                        seen_executions.add(execution_key)
                        tool_execution = self._tools.execute_detailed(
                            call,
                            run_context,
                            execution_context=execution_context,
                        )
                        result = tool_execution.result
                        diagnostic = {
                            key: value
                            for key, value in (
                                ("error_type", tool_execution.error_type),
                                ("error_message", tool_execution.error_message),
                            )
                            if value is not None
                        }
                    pending.append(result)
                    scope.emit(
                        RunEventType.TOOL_CALL_COMPLETED,
                        {
                            "iteration": iteration,
                            "call_id": call.call_id,
                            "tool": call.name,
                            "outcome": result.outcome.value,
                            "error_code": result.error_code.value if result.error_code else None,
                            **diagnostic,
                        },
                    )

            raise AgentLoopError(
                f"agent did not produce a final answer within {self._max_iterations} model calls"
            )
        except DQAgentError:
            raise
        except Exception as exc:
            error = AgentRuntimeError(
                "unexpected agent runtime failure",
                run_id=run_context.run_id,
            )
            raise error from exc
        finally:
            if owns_execution_context:
                execution_context._close_owned_records()

    @property
    def run_coordinator(self) -> RunCoordinator:
        """Return the default coordinator used by standalone runtime runs."""
        return self._run_coordinator

    def create_context(
        self,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> RunContext:
        """Create a context using the configured coordinator's timeout policy."""
        return self._run_coordinator.create_context(metadata=metadata)

    def _complete_with_retry(
        self,
        pending: Sequence[ConversationItem],
        context: RunContext,
        scope: RunScope,
        iteration: int,
    ) -> Completion:
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            context.check_active()
            scope.emit(
                RunEventType.MODEL_REQUEST_STARTED,
                {"iteration": iteration, "attempt": attempt},
            )
            try:
                completion = self._llm.complete(
                    pending,
                    self._tools.definitions,
                    context=context,
                )
                context.check_active()
            except LLMProviderError as exc:
                scope.emit_error(
                    RunEventType.MODEL_REQUEST_FAILED,
                    exc,
                    {"iteration": iteration, "attempt": attempt},
                )
                if not exc.retryable or attempt >= self._retry_policy.max_attempts:
                    raise
                delay = self._retry_policy.delay_after(attempt)
                scope.emit(
                    RunEventType.RETRY_SCHEDULED,
                    {"iteration": iteration, "attempt": attempt, "delay_seconds": delay},
                )
                context.wait(delay)
                continue
            except DQAgentError as exc:
                scope.emit_error(
                    RunEventType.MODEL_REQUEST_FAILED,
                    exc,
                    {"iteration": iteration, "attempt": attempt},
                )
                raise
            except Exception as exc:
                error = AgentRuntimeError(
                    "unexpected model client failure",
                    run_id=context.run_id,
                )
                scope.emit_error(
                    RunEventType.MODEL_REQUEST_FAILED,
                    error,
                    {"iteration": iteration, "attempt": attempt},
                    cause_type=type(exc).__name__,
                )
                raise error from exc
            scope.emit(
                RunEventType.MODEL_REQUEST_COMPLETED,
                {
                    "iteration": iteration,
                    "attempt": attempt,
                    "response_id": completion.response_id,
                    "model": completion.model,
                    "tool_call_count": len(completion.tool_calls),
                    "input_tokens": (
                        completion.usage.input_tokens if completion.usage is not None else None
                    ),
                    "output_tokens": (
                        completion.usage.output_tokens if completion.usage is not None else None
                    ),
                    "total_tokens": (
                        completion.usage.total_tokens if completion.usage is not None else None
                    ),
                },
            )
            return completion
        raise AssertionError("retry loop must return or raise")

    @staticmethod
    def _execution_key(call: ToolCall) -> tuple[str, str]:
        try:
            arguments = json.loads(call.arguments)
            canonical_arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError, ValueError):
            canonical_arguments = call.arguments
        return call.name, canonical_arguments
