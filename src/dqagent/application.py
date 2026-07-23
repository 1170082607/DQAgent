"""Direct chat and bounded agent-loop orchestration."""

import json

from dqagent.errors import AgentLoopError, LLMProviderError
from dqagent.llm import LLMClient
from dqagent.models import (
    ConversationItem,
    Message,
    Role,
    ToolCall,
    ToolErrorCode,
    ToolOutcome,
    ToolResult,
)
from dqagent.tools import ToolRegistry


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
    """Runs model/tool turns until the model produces a bounded final answer."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        system_prompt: str | None = None,
        max_iterations: int = 8,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least one")
        self._llm = llm
        self._tools = tools
        self._max_iterations = max_iterations
        self._system_message = (
            Message(Role.SYSTEM, system_prompt) if system_prompt and system_prompt.strip() else None
        )
        self._messages: list[ConversationItem] = []
        self.reset()

    @property
    def messages(self) -> tuple[ConversationItem, ...]:
        return tuple(self._messages)

    def send(self, user_input: str) -> Message:
        user_message = Message(Role.USER, user_input)
        pending: list[ConversationItem] = [*self._messages, user_message]
        seen_call_ids: set[str] = set()
        seen_executions: set[tuple[str, str]] = set()

        for _ in range(self._max_iterations):
            completion = self._llm.complete(pending, self._tools.definitions)
            assistant_message = (
                Message(Role.ASSISTANT, completion.content)
                if completion.content is not None
                else None
            )
            if assistant_message is not None:
                pending.append(assistant_message)

            if not completion.tool_calls:
                if assistant_message is None:
                    raise AgentLoopError("model returned neither text nor tool calls")
                self._messages = pending
                return assistant_message

            pending.extend(completion.tool_calls)
            for call in completion.tool_calls:
                execution_key = self._execution_key(call)
                if call.call_id in seen_call_ids or execution_key in seen_executions:
                    result = ToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        output="repeated tool call rejected",
                        outcome=ToolOutcome.ERROR,
                        error_code=ToolErrorCode.REPEATED_CALL,
                    )
                else:
                    seen_call_ids.add(call.call_id)
                    seen_executions.add(execution_key)
                    result = self._tools.execute(call)
                pending.append(result)

        raise AgentLoopError(
            f"agent did not produce a final answer within {self._max_iterations} model calls"
        )

    def reset(self) -> None:
        self._messages = [self._system_message] if self._system_message else []

    @staticmethod
    def _execution_key(call: ToolCall) -> tuple[str, str]:
        try:
            arguments = json.loads(call.arguments)
            canonical_arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError, ValueError):
            canonical_arguments = call.arguments
        return call.name, canonical_arguments
