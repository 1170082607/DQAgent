"""Direct chat and stateful agent application use cases."""

from dqagent.errors import LLMProviderError
from dqagent.execution import RunContext
from dqagent.llm import LLMClient
from dqagent.models import ConversationItem, Message, Role
from dqagent.runtime import AgentRunResult, AgentRuntime


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
        if completion.content is None or completion.tool_calls:
            raise LLMProviderError("direct chat requires a text completion")
        assistant_message = Message(Role.ASSISTANT, completion.content)

        # Commit both messages only after the provider returns a valid response.
        self._messages.extend((user_message, assistant_message))
        return assistant_message

    def reset(self) -> None:
        self._messages = [self._system_message] if self._system_message else []


class AgentApplication:
    """Owns conversation state and commits successful runtime results."""

    def __init__(
        self,
        runtime: AgentRuntime,
        system_prompt: str | None = None,
    ) -> None:
        self._runtime = runtime
        self._system_message = (
            Message(Role.SYSTEM, system_prompt) if system_prompt and system_prompt.strip() else None
        )
        self._messages: list[ConversationItem] = []
        self.reset()

    @property
    def messages(self) -> tuple[ConversationItem, ...]:
        return tuple(self._messages)

    def send(self, user_input: str) -> Message:
        return self.run(user_input).output

    def run(
        self,
        user_input: str,
        *,
        context: RunContext | None = None,
    ) -> AgentRunResult:
        user_message = Message(Role.USER, user_input)
        pending: list[ConversationItem] = [*self._messages, user_message]
        result = self._runtime.run(pending, context=context)
        self._messages = list(result.conversation)
        return result

    def reset(self) -> None:
        self._messages = [self._system_message] if self._system_message else []
