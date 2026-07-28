from collections.abc import Mapping, Sequence

import pytest

from dqagent.errors import (
    AgentLoopError,
    AgentRuntimeError,
    DQAgentError,
    ErrorCategory,
    LLMProviderError,
    RunCancelledError,
    RunDeadlineExceededError,
)
from dqagent.execution import RunContext
from dqagent.models import (
    Completion,
    ConversationItem,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from dqagent.runtime import (
    AgentRuntime,
    RetryPolicy,
    RunEvent,
    RunEventType,
    RunState,
)
from dqagent.tools import Tool, ToolRegistry


class ScriptedLLM:
    def __init__(self, outcomes: Sequence[Completion | Exception]) -> None:
        self._outcomes = iter(outcomes)
        self.requests: list[tuple[tuple[ConversationItem, ...], str, int]] = []

    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolDefinition] = (),
        *,
        context: RunContext | None = None,
    ) -> Completion:
        assert context is not None
        context.check_active()
        self.requests.append((tuple(messages), context.run_id, len(tools)))
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class CollectingSink:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


class FailingSink:
    def emit(self, event: RunEvent) -> None:
        raise RuntimeError("telemetry unavailable")


def make_registry(
    handler: object | None = None,
    seen_contexts: list[str] | None = None,
) -> ToolRegistry:
    def greet(arguments: Mapping[str, object], context: RunContext) -> str:
        if seen_contexts is not None:
            seen_contexts.append(context.run_id)
        return f"hello {arguments['name']}"

    return ToolRegistry(
        (
            Tool(
                ToolDefinition(
                    "greet",
                    "Greet a person.",
                    {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                ),
                handler or greet,  # type: ignore[arg-type]
            ),
        )
    )


def test_runtime_propagates_context_and_returns_ordered_events() -> None:
    call = ToolCall("call-1", "greet", '{"name":"Ada"}')
    llm = ScriptedLLM([Completion(tool_calls=(call,)), Completion("Done")])
    sink = CollectingSink()
    handler_contexts: list[str] = []
    runtime = AgentRuntime(llm, make_registry(seen_contexts=handler_contexts), event_sinks=(sink,))
    context = RunContext(run_id="run-success", timeout_seconds=1)

    result = runtime.run([Message(Role.USER, "Say hello")], context=context)

    assert result.run_id == "run-success"
    assert result.state is RunState.COMPLETED
    assert result.output == Message(Role.ASSISTANT, "Done")
    assert handler_contexts == ["run-success"]
    assert [request[1] for request in llm.requests] == ["run-success", "run-success"]
    assert result.events == tuple(sink.events)
    assert [event.sequence for event in result.events] == list(range(1, len(result.events) + 1))
    assert result.events[0].type is RunEventType.RUN_STARTED
    assert result.events[0].attributes["deadline"] == context.deadline.isoformat()
    assert result.events[-1].type is RunEventType.RUN_COMPLETED
    assert all(event.run_id == "run-success" for event in result.events)


def test_runtime_retries_only_retryable_provider_failures() -> None:
    transient = LLMProviderError(
        "service unavailable",
        category=ErrorCategory.UNAVAILABLE,
        retryable=True,
    )
    llm = ScriptedLLM([transient, Completion("Recovered")])
    runtime = AgentRuntime(
        llm,
        ToolRegistry(),
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_seconds=0),
    )

    result = runtime.run([Message(Role.USER, "Hi")])

    assert result.output.content == "Recovered"
    assert len(llm.requests) == 2
    assert [event.type for event in result.events].count(RunEventType.RETRY_SCHEDULED) == 1


def test_runtime_does_not_retry_non_retryable_provider_failure() -> None:
    sink = CollectingSink()
    llm = ScriptedLLM([LLMProviderError("invalid request")])
    runtime = AgentRuntime(llm, ToolRegistry(), event_sinks=(sink,))

    with pytest.raises(LLMProviderError) as error:
        runtime.run([Message(Role.USER, "Hi")], context=RunContext(run_id="run-failed"))

    assert len(llm.requests) == 1
    assert error.value.run_id == "run-failed"
    assert sink.events[-1].type is RunEventType.RUN_FAILED
    assert sink.events[-1].attributes["error_category"] == ErrorCategory.PROVIDER.value


def test_runtime_records_preexisting_cancellation() -> None:
    sink = CollectingSink()
    context = RunContext(run_id="run-cancelled")
    context.cancel("caller cancelled")
    runtime = AgentRuntime(ScriptedLLM([]), ToolRegistry(), event_sinks=(sink,))

    with pytest.raises(RunCancelledError):
        runtime.run([Message(Role.USER, "Hi")], context=context)

    assert [event.type for event in sink.events] == [
        RunEventType.RUN_STARTED,
        RunEventType.RUN_CANCELLED,
    ]


@pytest.mark.parametrize(
    ("error", "terminal_event"),
    [
        (RunCancelledError("adapter cancelled"), RunEventType.RUN_CANCELLED),
        (RunDeadlineExceededError("adapter deadline"), RunEventType.RUN_TIMED_OUT),
    ],
)
def test_runtime_closes_interrupted_model_attempt_and_binds_run_id(
    error: DQAgentError,
    terminal_event: RunEventType,
) -> None:
    sink = CollectingSink()
    runtime = AgentRuntime(ScriptedLLM([error]), ToolRegistry(), event_sinks=(sink,))

    with pytest.raises(type(error)) as raised:
        runtime.run([Message(Role.USER, "Hi")], context=RunContext(run_id="run-control"))

    assert raised.value.run_id == "run-control"
    assert [event.type for event in sink.events] == [
        RunEventType.RUN_STARTED,
        RunEventType.MODEL_REQUEST_STARTED,
        RunEventType.MODEL_REQUEST_FAILED,
        terminal_event,
    ]


def test_runtime_deadline_interrupts_retry_backoff() -> None:
    transient = LLMProviderError("retry", retryable=True)
    sink = CollectingSink()
    runtime = AgentRuntime(
        ScriptedLLM([transient]),
        ToolRegistry(),
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_seconds=0.05),
        event_sinks=(sink,),
    )

    with pytest.raises(RunDeadlineExceededError):
        runtime.run(
            [Message(Role.USER, "Hi")],
            context=RunContext(run_id="run-timeout", timeout_seconds=0.005),
        )

    assert sink.events[-1].type is RunEventType.RUN_TIMED_OUT
    assert RunEventType.RETRY_SCHEDULED in [event.type for event in sink.events]


def test_runtime_keeps_tool_diagnostics_out_of_model_observation() -> None:
    def fail(arguments: Mapping[str, object], context: RunContext) -> str:
        raise RuntimeError("database host db.internal unavailable")

    call = ToolCall("call-1", "greet", '{"name":"Ada"}')
    llm = ScriptedLLM([Completion(tool_calls=(call,)), Completion("Cannot greet")])
    runtime = AgentRuntime(llm, make_registry(fail))

    result = runtime.run([Message(Role.USER, "Greet Ada")])

    observation = llm.requests[1][0][-1]
    assert isinstance(observation, ToolResult)
    assert observation.output == "tool execution failed"
    tool_event = next(
        event for event in result.events if event.type is RunEventType.TOOL_CALL_COMPLETED
    )
    assert tool_event.attributes["error_type"] == "RuntimeError"
    assert "db.internal" in str(tool_event.attributes["error_message"])


def test_event_sink_failure_does_not_change_run_result() -> None:
    runtime = AgentRuntime(
        ScriptedLLM([Completion("Done")]),
        ToolRegistry(),
        event_sinks=(FailingSink(),),
    )

    result = runtime.run([Message(Role.USER, "Hi")])

    assert result.output.content == "Done"


def test_runtime_classifies_loop_limit_and_unexpected_failures() -> None:
    call = ToolCall("call-1", "missing", "{}")
    sink = CollectingSink()
    runtime = AgentRuntime(
        ScriptedLLM([Completion(tool_calls=(call,))]),
        ToolRegistry(),
        max_iterations=1,
        event_sinks=(sink,),
    )

    with pytest.raises(AgentLoopError):
        runtime.run([Message(Role.USER, "Loop")])
    assert sink.events[-1].attributes["error_category"] == ErrorCategory.LOOP_LIMIT.value

    broken_sink = CollectingSink()
    broken = AgentRuntime(
        ScriptedLLM([RuntimeError("bug")]),
        ToolRegistry(),
        event_sinks=(broken_sink,),
    )
    with pytest.raises(AgentRuntimeError) as error:
        broken.run([Message(Role.USER, "Hi")])
    assert isinstance(error.value.__cause__, RuntimeError)
    assert [event.type for event in broken_sink.events] == [
        RunEventType.RUN_STARTED,
        RunEventType.MODEL_REQUEST_STARTED,
        RunEventType.MODEL_REQUEST_FAILED,
        RunEventType.RUN_FAILED,
    ]


@pytest.mark.parametrize(
    "policy",
    [
        {"max_attempts": 0},
        {"initial_delay_seconds": -1},
        {"multiplier": 0.5},
    ],
)
def test_retry_policy_rejects_invalid_values(policy: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**policy)  # type: ignore[arg-type]


def test_retry_policy_caps_exponential_delay() -> None:
    policy = RetryPolicy(initial_delay_seconds=1, multiplier=3, max_delay_seconds=2)

    assert policy.delay_after(1) == 1
    assert policy.delay_after(2) == 2


def test_retry_policy_caps_delay_without_overflow() -> None:
    policy = RetryPolicy(initial_delay_seconds=1, multiplier=10, max_delay_seconds=2)

    assert policy.delay_after(10_000) == 2
    with pytest.raises(ValueError, match="at least one"):
        policy.delay_after(0)
