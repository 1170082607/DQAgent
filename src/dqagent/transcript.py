"""Conversation-turn parsing and durable transcript invariants."""

from collections.abc import Sequence

from dqagent.models import ConversationItem, Message, Role, ToolCall, ToolResult

ConversationTurn = tuple[ConversationItem, ...]


class TranscriptValidationError(ValueError):
    """Raised when conversation items cannot form a valid durable transcript."""


def split_turns(items: Sequence[ConversationItem]) -> tuple[ConversationTurn, ...]:
    """Split user-led conversation items while allowing one active incomplete turn."""
    turns: list[ConversationTurn] = []
    current: list[ConversationItem] = []
    for item in items:
        if isinstance(item, Message) and item.role is Role.SYSTEM:
            raise TranscriptValidationError("durable transcript must not contain system messages")
        if isinstance(item, Message) and item.role is Role.USER:
            if current:
                turns.append(tuple(current))
            current = [item]
        elif not current:
            raise TranscriptValidationError("durable transcript must start with a user message")
        else:
            current.append(item)
    if current:
        turns.append(tuple(current))
    return tuple(turns)


def validate_complete_transcript(items: Sequence[ConversationItem]) -> None:
    """Require complete user/assistant turns and paired tool calls and results."""
    turns = split_turns(items) if items else ()
    for turn in turns:
        outstanding: dict[str, str] = {}
        for item in turn:
            if isinstance(item, ToolCall):
                if item.call_id in outstanding:
                    raise TranscriptValidationError(
                        f"duplicate tool call ID in transcript: {item.call_id!r}"
                    )
                outstanding[item.call_id] = item.name
            elif isinstance(item, ToolResult):
                expected_name = outstanding.pop(item.call_id, None)
                if expected_name is None or expected_name != item.name:
                    raise TranscriptValidationError(
                        f"tool result {item.call_id!r} has no matching tool call"
                    )
        if outstanding:
            raise TranscriptValidationError(
                f"transcript has tool calls without results: {sorted(outstanding)!r}"
            )
        if not isinstance(turn[-1], Message) or turn[-1].role is not Role.ASSISTANT:
            raise TranscriptValidationError(
                "each durable transcript turn must end with an assistant message"
            )
