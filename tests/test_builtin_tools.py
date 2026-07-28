from datetime import datetime

from dqagent.builtin_tools import create_builtin_tool_registry
from dqagent.execution import RunContext
from dqagent.models import ToolCall, ToolErrorCode


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
