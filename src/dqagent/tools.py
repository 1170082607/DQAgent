"""Application-owned tool definitions, registration, and execution."""

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from dqagent.models import ToolCall, ToolDefinition, ToolErrorCode, ToolOutcome, ToolResult

ToolHandler = Callable[[Mapping[str, object]], str]


@dataclass(frozen=True, slots=True)
class Tool:
    definition: ToolDefinition
    handler: ToolHandler
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("tool timeout must be greater than zero")


class ToolRegistry:
    """Explicit name-to-tool registry and the single tool execution boundary."""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._tools.values())

    def register(self, tool: Tool) -> None:
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"tool '{name}' is already registered")
        if tool.definition.input_schema.get("type") != "object":
            raise ValueError(f"tool '{name}' input schema must describe an object")
        try:
            Draft202012Validator.check_schema(tool.definition.input_schema)
        except SchemaError as exc:
            raise ValueError(f"tool '{name}' has an invalid input schema: {exc.message}") from exc
        self._tools[name] = tool

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return self._error(
                call,
                ToolErrorCode.UNKNOWN_TOOL,
                f"unknown tool: {call.name}",
            )

        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as exc:
            return self._error(
                call,
                ToolErrorCode.INVALID_ARGUMENTS,
                f"invalid JSON arguments: {exc.msg}",
            )
        if not isinstance(arguments, dict):
            return self._error(
                call,
                ToolErrorCode.INVALID_ARGUMENTS,
                "tool arguments must be a JSON object",
            )

        try:
            Draft202012Validator(tool.definition.input_schema).validate(arguments)
        except ValidationError as exc:
            return self._error(
                call,
                ToolErrorCode.INVALID_ARGUMENTS,
                f"arguments do not match the tool schema: {exc.message}",
            )

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"dqagent-{call.name}")
        future = executor.submit(tool.handler, arguments)
        try:
            output = future.result(timeout=tool.timeout_seconds)
            if not isinstance(output, str) or not output.strip():
                return self._error(
                    call,
                    ToolErrorCode.EXECUTION_ERROR,
                    "tool returned an empty or non-text result",
                )
            return ToolResult(call.call_id, call.name, output)
        except FutureTimeoutError:
            future.cancel()
            return self._error(
                call,
                ToolErrorCode.TIMEOUT,
                f"tool timed out after {tool.timeout_seconds:g} seconds",
            )
        except Exception:  # Tool handlers are an untrusted application boundary.
            return self._error(
                call,
                ToolErrorCode.EXECUTION_ERROR,
                "tool execution failed",
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
