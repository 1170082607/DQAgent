"""LLM boundary used by the application layer."""

from collections.abc import Sequence
from typing import Protocol

from dqagent.execution import RunContext
from dqagent.models import Completion, ConversationItem, ToolDefinition


class LLMClient(Protocol):
    """Provider-neutral interface for generating one assistant response."""

    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolDefinition] = (),
        *,
        context: RunContext | None = None,
    ) -> Completion:
        """Generate text or tool calls within an optional execution context."""
        ...
