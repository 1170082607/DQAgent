"""Observable runtime for one bounded agent execution."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from dqagent.errors import (
    AgentLoopError,
    AgentRuntimeError,
    DQAgentError,
    LLMProviderError,
    RunCancelledError,
    RunDeadlineExceededError,
)
from dqagent.events import EventSink, RunEvent, RunEventEmitter, RunEventType, RunState
from dqagent.execution import RunContext
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
from dqagent.tools import ToolRegistry

__all__ = [
    "AgentRunResult",
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
class AgentRunResult:
    run_id: str
    state: RunState
    output: Message
    conversation: tuple[ConversationItem, ...]
    new_items: tuple[ConversationItem, ...]
    events: tuple[RunEvent, ...]
    started_at: datetime
    completed_at: datetime


class AgentRuntime:
    """Owns one run's loop, retry policy, lifecycle, and event stream."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        *,
        max_iterations: int = 8,
        retry_policy: RetryPolicy | None = None,
        default_timeout_seconds: float | None = 120.0,
        event_sinks: Sequence[EventSink] = (),
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least one")
        if default_timeout_seconds is not None and (
            not math.isfinite(default_timeout_seconds) or default_timeout_seconds <= 0
        ):
            raise ValueError("default run timeout must be a finite number greater than zero")
        self._llm = llm
        self._tools = tools
        self._max_iterations = max_iterations
        self._retry_policy = retry_policy or RetryPolicy()
        self._default_timeout_seconds = default_timeout_seconds
        self._event_sinks = tuple(event_sinks)

    def run(
        self,
        conversation: Sequence[ConversationItem],
        *,
        context: RunContext | None = None,
        context_attributes: Mapping[str, object] | None = None,
    ) -> AgentRunResult:
        run_context = context or RunContext(timeout_seconds=self._default_timeout_seconds)
        emitter = RunEventEmitter(run_context, self._event_sinks)
        pending = list(conversation)
        initial_item_count = len(pending)
        seen_call_ids: set[str] = set()
        seen_executions: set[tuple[str, str]] = set()
        emitter.emit(
            RunEventType.RUN_STARTED,
            RunState.RUNNING,
            {
                "deadline": run_context.deadline.isoformat() if run_context.deadline else None,
                "metadata": dict(run_context.metadata),
            },
        )
        if context_attributes is not None:
            emitter.emit(
                RunEventType.CONTEXT_ASSEMBLED,
                RunState.RUNNING,
                context_attributes,
            )

        try:
            run_context.check_active()
            for iteration in range(1, self._max_iterations + 1):
                completion = self._complete_with_retry(
                    pending,
                    run_context,
                    emitter,
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
                    emitter.emit(
                        RunEventType.RUN_COMPLETED,
                        RunState.COMPLETED,
                        {"iterations": iteration},
                    )
                    return AgentRunResult(
                        run_id=run_context.run_id,
                        state=RunState.COMPLETED,
                        output=assistant_message,
                        conversation=tuple(pending),
                        new_items=tuple(pending[initial_item_count:]),
                        events=emitter.events,
                        started_at=run_context.started_at,
                        completed_at=datetime.now(UTC),
                    )

                pending.extend(completion.tool_calls)
                for call in completion.tool_calls:
                    run_context.check_active()
                    emitter.emit(
                        RunEventType.TOOL_CALL_STARTED,
                        RunState.RUNNING,
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
                        execution = self._tools.execute_detailed(call, run_context)
                        result = execution.result
                        diagnostic = {
                            key: value
                            for key, value in (
                                ("error_type", execution.error_type),
                                ("error_message", execution.error_message),
                            )
                            if value is not None
                        }
                    pending.append(result)
                    emitter.emit(
                        RunEventType.TOOL_CALL_COMPLETED,
                        RunState.RUNNING,
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
        except RunCancelledError as exc:
            self._bind_error_to_run(exc, run_context)
            emitter.emit(
                RunEventType.RUN_CANCELLED,
                RunState.CANCELLED,
                self._error_attributes(exc),
            )
            raise
        except RunDeadlineExceededError as exc:
            self._bind_error_to_run(exc, run_context)
            emitter.emit(
                RunEventType.RUN_TIMED_OUT,
                RunState.TIMED_OUT,
                self._error_attributes(exc),
            )
            raise
        except DQAgentError as exc:
            self._bind_error_to_run(exc, run_context)
            emitter.emit(
                RunEventType.RUN_FAILED,
                RunState.FAILED,
                self._error_attributes(exc),
            )
            raise
        except Exception as exc:
            error = AgentRuntimeError(
                "unexpected agent runtime failure",
                run_id=run_context.run_id,
            )
            emitter.emit(
                RunEventType.RUN_FAILED,
                RunState.FAILED,
                {
                    **self._error_attributes(error),
                    "cause_type": type(exc).__name__,
                },
            )
            raise error from exc

    def create_context(
        self,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> RunContext:
        """Create a context using this runtime's end-to-end timeout policy."""
        return RunContext(
            timeout_seconds=self._default_timeout_seconds,
            metadata=metadata,
        )

    def _complete_with_retry(
        self,
        pending: Sequence[ConversationItem],
        context: RunContext,
        emitter: RunEventEmitter,
        iteration: int,
    ) -> Completion:
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            context.check_active()
            emitter.emit(
                RunEventType.MODEL_REQUEST_STARTED,
                RunState.RUNNING,
                {"iteration": iteration, "attempt": attempt},
            )
            try:
                completion = self._llm.complete(
                    pending,
                    self._tools.definitions,
                    context=context,
                )
            except LLMProviderError as exc:
                self._bind_error_to_run(exc, context)
                emitter.emit(
                    RunEventType.MODEL_REQUEST_FAILED,
                    RunState.RUNNING,
                    {
                        "iteration": iteration,
                        "attempt": attempt,
                        **self._error_attributes(exc),
                    },
                )
                if not exc.retryable or attempt >= self._retry_policy.max_attempts:
                    raise
                delay = self._retry_policy.delay_after(attempt)
                emitter.emit(
                    RunEventType.RETRY_SCHEDULED,
                    RunState.RUNNING,
                    {"iteration": iteration, "attempt": attempt, "delay_seconds": delay},
                )
                context.wait(delay)
                continue
            except DQAgentError as exc:
                self._bind_error_to_run(exc, context)
                emitter.emit(
                    RunEventType.MODEL_REQUEST_FAILED,
                    RunState.RUNNING,
                    {
                        "iteration": iteration,
                        "attempt": attempt,
                        **self._error_attributes(exc),
                    },
                )
                raise
            except Exception as exc:
                error = AgentRuntimeError(
                    "unexpected model client failure",
                    run_id=context.run_id,
                )
                emitter.emit(
                    RunEventType.MODEL_REQUEST_FAILED,
                    RunState.RUNNING,
                    {
                        "iteration": iteration,
                        "attempt": attempt,
                        **self._error_attributes(error),
                        "cause_type": type(exc).__name__,
                    },
                )
                raise error from exc
            emitter.emit(
                RunEventType.MODEL_REQUEST_COMPLETED,
                RunState.RUNNING,
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
    def _error_attributes(error: DQAgentError) -> Mapping[str, object]:
        return {
            "error_type": type(error).__name__,
            "error_category": error.category.value,
            "retryable": error.retryable,
            "message": str(error),
        }

    @staticmethod
    def _bind_error_to_run(error: DQAgentError, context: RunContext) -> None:
        error.run_id = context.run_id

    @staticmethod
    def _execution_key(call: ToolCall) -> tuple[str, str]:
        try:
            arguments = json.loads(call.arguments)
            canonical_arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError, ValueError):
            canonical_arguments = call.arguments
        return call.name, canonical_arguments
