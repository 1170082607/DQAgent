"""Chat use case orchestration."""

from dqagent.llm import LLMClient
from dqagent.models import Message, Role


class ChatApplication:
    """Owns an in-memory conversation and coordinates LLM requests."""

    def __init__(self, llm: LLMClient, system_prompt: str | None = None) -> None:
        self._llm = llm
        self._system_message = (
            Message(Role.SYSTEM, system_prompt) if system_prompt and system_prompt.strip() else None
        )
        self._messages: list[Message] = []
        self.reset()

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    def send(self, user_input: str) -> Message:
        user_message = Message(Role.USER, user_input)
        pending_messages = [*self._messages, user_message]

        completion = self._llm.complete(pending_messages)
        assistant_message = Message(Role.ASSISTANT, completion.content)

        # Commit both messages only after the provider returns a valid response.
        self._messages.extend((user_message, assistant_message))
        return assistant_message

    def reset(self) -> None:
        self._messages = [self._system_message] if self._system_message else []
