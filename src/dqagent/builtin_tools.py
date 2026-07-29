"""Small built-in tools used by the command-line agent."""

import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import cast

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


def _get_weather(arguments: Mapping[str, object], context: RunContext) -> str:
    """Return deterministic weather data for tool-calling demonstrations."""

    context.check_active()
    city = cast(str, arguments["city"])
    forecast_date = cast(str, arguments["date"])
    return json.dumps(
        {
            "city": city,
            "condition": "sunny",
            "date": forecast_date,
            "is_demo": True,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


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

    get_weather = Tool(
        definition=ToolDefinition(
            name="get_weather",
            description="Return deterministic demo weather for a city and date.",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Non-empty city name.",
                        "minLength": 1,
                        "maxLength": 100,
                    },
                    "date": {
                        "type": "string",
                        "description": "Forecast date in YYYY-MM-DD format.",
                        "format": "date",
                    },
                },
                "required": ["city", "date"],
                "additionalProperties": False,
            },
        ),
        handler=_get_weather,
    )

    return ToolRegistry(
        (
            current_time,
            get_weather,
        )
    )
