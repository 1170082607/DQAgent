import time
from collections.abc import Mapping

import pytest

from dqagent.errors import RunDeadlineExceededError
from dqagent.execution import RunContext
from dqagent.models import ToolCall, ToolDefinition, ToolErrorCode, ToolOutcome
from dqagent.tools import Tool, ToolRegistry


def make_tool(
    handler: object | None = None,
    *,
    name: str = "greet",
    timeout_seconds: float = 1.0,
) -> Tool:
    def default_handler(arguments: Mapping[str, object], context: RunContext) -> str:
        context.check_active()
        return f"hello {arguments['name']}"

    return Tool(
        definition=ToolDefinition(
            name=name,
            description="Greet a person.",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        handler=handler or default_handler,  # type: ignore[arg-type]
        timeout_seconds=timeout_seconds,
    )


def test_registry_executes_valid_call() -> None:
    registry = ToolRegistry((make_tool(),))

    result = registry.execute(ToolCall("call-1", "greet", '{"name":"Ada"}'))

    assert result.output == "hello Ada"
    assert result.outcome is ToolOutcome.SUCCESS
    assert result.error_code is None


@pytest.mark.parametrize(
    ("call", "error_code"),
    [
        (ToolCall("call-1", "missing", "{}"), ToolErrorCode.UNKNOWN_TOOL),
        (ToolCall("call-1", "greet", "not-json"), ToolErrorCode.INVALID_ARGUMENTS),
        (ToolCall("call-1", "greet", "[]"), ToolErrorCode.INVALID_ARGUMENTS),
        (ToolCall("call-1", "greet", '{}'), ToolErrorCode.INVALID_ARGUMENTS),
    ],
)
def test_registry_returns_structured_call_errors(
    call: ToolCall, error_code: ToolErrorCode
) -> None:
    result = ToolRegistry((make_tool(),)).execute(call)

    assert result.outcome is ToolOutcome.ERROR
    assert result.error_code is error_code


def test_registry_translates_handler_failure() -> None:
    def fail(arguments: Mapping[str, object], context: RunContext) -> str:
        raise RuntimeError("backend unavailable")

    result = ToolRegistry((make_tool(fail),)).execute(
        ToolCall("call-1", "greet", '{"name":"Ada"}')
    )

    assert result.error_code is ToolErrorCode.EXECUTION_ERROR
    assert result.output == "tool execution failed"


def test_registry_times_out_slow_handler() -> None:
    def slow(arguments: Mapping[str, object], context: RunContext) -> str:
        time.sleep(0.05)
        return "late"

    result = ToolRegistry((make_tool(slow, timeout_seconds=0.001),)).execute(
        ToolCall("call-1", "greet", '{"name":"Ada"}')
    )

    assert result.error_code is ToolErrorCode.TIMEOUT


def test_run_deadline_interrupts_waiting_for_tool_handler() -> None:
    def slow(arguments: Mapping[str, object], context: RunContext) -> str:
        time.sleep(0.05)
        return "late"

    context = RunContext(run_id="run-tool-timeout", timeout_seconds=0.001)

    with pytest.raises(RunDeadlineExceededError) as error:
        ToolRegistry((make_tool(slow),)).execute(
            ToolCall("call-1", "greet", '{"name":"Ada"}'),
            context,
        )

    assert error.value.run_id == "run-tool-timeout"


def test_registry_rejects_duplicate_names_and_invalid_schema() -> None:
    registry = ToolRegistry((make_tool(),))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_tool())

    invalid = Tool(
        ToolDefinition("invalid", "Invalid schema.", {"type": "array"}),
        lambda arguments, context: "unused",
    )
    with pytest.raises(ValueError, match="must describe an object"):
        registry.register(invalid)


@pytest.mark.parametrize("timeout", [0, float("nan"), float("inf")])
def test_tool_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite number greater than zero"):
        make_tool(timeout_seconds=timeout)
