"""LLM boundary used by the application layer."""

from collections.abc import Sequence
from typing import Protocol

from dqagent.models import Completion, ConversationItem, ToolDefinition


class LLMClient(Protocol):
    """Provider-neutral interface for generating one assistant response."""

    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolDefinition] = (),
    ) -> Completion:
        """Generate text or tool calls from an ordered conversation history."""
        ...
