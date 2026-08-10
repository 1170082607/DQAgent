"""Application-owned tool definitions, registration, and execution."""

import json
import math
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Protocol

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError

from dqagent.errors import RunCancelledError, RunDeadlineExceededError
from dqagent.execution import RunContext
from dqagent.models import ToolCall, ToolDefinition, ToolErrorCode, ToolOutcome, ToolResult


class ToolHandler(Protocol):
    def __call__(self, arguments: Mapping[str, object], context: RunContext) -> str: ...


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

    def execute(self, call: ToolCall, context: RunContext | None = None) -> ToolResult:
        run_context = context or RunContext()
        return self.execute_detailed(call, run_context).result

    def execute_detailed(self, call: ToolCall, context: RunContext) -> ToolExecution:
        context.check_active()
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolExecution(
                self._error(call, ToolErrorCode.UNKNOWN_TOOL, f"unknown tool: {call.name}")
            )

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
