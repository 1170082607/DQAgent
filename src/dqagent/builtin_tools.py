"""Small built-in tools used by the command-line agent."""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from dqagent.execution import RunContext
from dqagent.models import ToolDefinition
from dqagent.tools import Tool, ToolRegistry


def _current_time(arguments: Mapping[str, object], context: RunContext) -> str:
    context.check_active()
    raw_offset = arguments["utc_offset_hours"]
    if isinstance(raw_offset, bool) or not isinstance(raw_offset, (int, float)):
        raise ValueError("utc_offset_hours must be a number")
    zone = timezone(timedelta(hours=float(raw_offset)))
    return datetime.now(zone).isoformat(timespec="seconds")


def create_builtin_tool_registry() -> ToolRegistry:
    """Create the concrete tool set exposed by the CLI."""

    current_time = Tool(
        definition=ToolDefinition(
            name="current_time",
            description="Return the current time for a numeric UTC offset.",
            input_schema={
                "type": "object",
                "properties": {
                    "utc_offset_hours": {
                        "type": "number",
                        "minimum": -12,
                        "maximum": 14,
                    }
                },
                "required": ["utc_offset_hours"],
                "additionalProperties": False,
            },
        ),
        handler=_current_time,
    )
    return ToolRegistry((current_time,))
