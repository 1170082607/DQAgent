"""Provider-neutral conversation and tool models."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("message content must not be empty")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Metadata exposed to a model for one application-owned tool."""

    name: str
    description: str
    input_schema: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        if not self.description.strip():
            raise ValueError("tool description must not be empty")


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A provider-neutral function call requested by the model."""

    call_id: str
    name: str
    arguments: str

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("tool call ID must not be empty")
        if not self.name.strip():
            raise ValueError("tool call name must not be empty")
        if not self.arguments.strip():
            raise ValueError("tool call arguments must not be empty")


class ToolOutcome(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class ToolErrorCode(StrEnum):
    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"
    REPEATED_CALL = "repeated_call"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """An observation produced by attempting a model-requested tool call."""

    call_id: str
    name: str
    output: str
    outcome: ToolOutcome = ToolOutcome.SUCCESS
    error_code: ToolErrorCode | None = None

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("tool result call ID must not be empty")
        if not self.name.strip():
            raise ValueError("tool result name must not be empty")
        if not self.output.strip():
            raise ValueError("tool result output must not be empty")
        if (self.outcome is ToolOutcome.ERROR) != (self.error_code is not None):
            raise ValueError("tool result errors must include an error code")


ConversationItem: TypeAlias = Message | ToolCall | ToolResult


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider-neutral token counts for one model response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        for name, value in (
            ("input tokens", self.input_tokens),
            ("output tokens", self.output_tokens),
            ("total tokens", self.total_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class Completion:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    response_id: str | None = None
    model: str | None = None
    usage: TokenUsage | None = None

    def __post_init__(self) -> None:
        if self.content is not None and not self.content.strip():
            raise ValueError("completion content must not be empty")
        if self.content is None and not self.tool_calls:
            raise ValueError("completion must contain text or tool calls")
