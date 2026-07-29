import json
from datetime import datetime

import pytest

from dqagent.builtin_tools import create_builtin_tool_registry
from dqagent.execution import RunContext
from dqagent.models import ToolCall, ToolErrorCode, ToolOutcome


def test_current_time_returns_an_offset_aware_timestamp() -> None:
    registry = create_builtin_tool_registry()

    result = registry.execute(
        ToolCall("call-1", "current_time", '{"utc_offset_hours":8}'),
        RunContext(run_id="run-time"),
    )

    parsed = datetime.fromisoformat(result.output)
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset().total_seconds() == 8 * 60 * 60


def test_current_time_rejects_out_of_range_offset() -> None:
    registry = create_builtin_tool_registry()

    result = registry.execute(
        ToolCall("call-1", "current_time", '{"utc_offset_hours":15}')
    )

    assert result.error_code is ToolErrorCode.INVALID_ARGUMENTS


def test_get_weather_returns_deterministic_demo_weather() -> None:
    registry = create_builtin_tool_registry()

    result = registry.execute(
        ToolCall(
            "call-weather",
            "get_weather",
            '{"city":"Shanghai","date":"2026-07-29"}',
        ),
        RunContext(run_id="run-weather"),
    )

    assert json.loads(result.output) == {
        "city": "Shanghai",
        "condition": "sunny",
        "date": "2026-07-29",
        "is_demo": True,
    }
    assert result.outcome is ToolOutcome.SUCCESS
    assert result.error_code is None


@pytest.mark.parametrize(
    "arguments",
    [
        '{"city":"Shanghai"}',
        '{"city":123,"date":"2026-07-29"}',
        '{"city":"","date":"2026-07-29"}',
        '{"city":"Shanghai","date":20260729}',
        '{"city":"Shanghai","date":"2026-02-30"}',
        '{"city":"Shanghai","date":"2026-07-29","units":"celsius"}',
    ],
)
def test_get_weather_rejects_arguments_that_do_not_match_its_schema(arguments: str) -> None:
    registry = create_builtin_tool_registry()

    result = registry.execute(ToolCall("call-invalid", "get_weather", arguments))

    assert result.outcome is ToolOutcome.ERROR
    assert result.error_code is ToolErrorCode.INVALID_ARGUMENTS
