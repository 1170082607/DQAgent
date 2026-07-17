"""LLM boundary used by the application layer."""

from collections.abc import Sequence
from typing import Protocol

from dqagent.models import Completion, Message


class LLMClient(Protocol):
    """Provider-neutral interface for generating one assistant response."""

    def complete(self, messages: Sequence[Message]) -> Completion:
        """Generate a completion from an ordered conversation history."""
        ...
