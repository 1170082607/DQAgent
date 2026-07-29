"""llama.cpp llama-server adapter using its OpenAI-compatible Chat Completions API."""

from collections.abc import Sequence
from typing import Any

from openai import OpenAI, OpenAIError

from dqagent.config import Settings
from dqagent.errors import LLMProviderError
from dqagent.execution import RunContext
from dqagent.models import (
    Completion,
    ConversationItem,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from dqagent.providers._openai_sdk import map_token_usage, translate_openai_sdk_error


class LlamaCppChatClient:
    """Maps DQAgent values to llama-server's `/v1/chat/completions` endpoint."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if settings.base_url is None:
            raise ValueError("llama.cpp provider requires a base URL")
        self._model = settings.model
        self._timeout_seconds = settings.timeout_seconds
        self._client: Any = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=0,
        )

    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolDefinition] = (),
        *,
        context: RunContext | None = None,
    ) -> Completion:
        if context is not None:
            context.check_active()
        request_messages = self._map_messages(messages)
        request_tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.input_schema),
                },
            }
            for tool in tools
        ]
        request_options: dict[str, Any] = {}
        if context is not None:
            remaining_seconds = context.remaining_seconds
            if remaining_seconds is not None:
                if remaining_seconds <= 0:
                    context.check_active()
                request_options["timeout"] = min(self._timeout_seconds, remaining_seconds)

        try:
            if request_tools:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=request_messages,
                    tools=request_tools,
                    **request_options,
                )
            else:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=request_messages,
                    **request_options,
                )
        except OpenAIError as exc:
            if context is not None:
                context.check_active()
            raise translate_openai_sdk_error(exc, provider_label="llama.cpp") from exc

        if context is not None:
            context.check_active()
        choices = getattr(response, "choices", None)
        if not isinstance(choices, Sequence) or not choices:
            raise LLMProviderError("llama.cpp returned no chat completion choices")
        response_message = getattr(choices[0], "message", None)
        if response_message is None:
            raise LLMProviderError("llama.cpp returned an invalid chat completion")

        content = self._map_content(getattr(response_message, "content", None))
        tool_calls = self._map_tool_calls(getattr(response_message, "tool_calls", None))
        if content is None and not tool_calls:
            raise LLMProviderError("llama.cpp returned neither text nor function calls")
        return Completion(
            content=content,
            tool_calls=tool_calls,
            response_id=getattr(response, "id", None),
            model=getattr(response, "model", None),
            usage=map_token_usage(
                getattr(response, "usage", None),
                input_field="prompt_tokens",
                output_field="completion_tokens",
                provider_label="llama.cpp",
            ),
        )

    @classmethod
    def _map_messages(cls, items: Sequence[ConversationItem]) -> list[dict[str, Any]]:
        mapped: list[dict[str, Any]] = []
        index = 0
        while index < len(items):
            item = items[index]
            if isinstance(item, Message):
                calls, next_index = cls._following_calls(items, index + 1)
                if item.role is Role.ASSISTANT and calls:
                    mapped.append(cls._assistant_tool_message(item.content, calls))
                    index = next_index
                    continue
                mapped.append({"role": item.role.value, "content": item.content})
                index += 1
                continue
            if isinstance(item, ToolCall):
                calls, next_index = cls._following_calls(items, index)
                mapped.append(cls._assistant_tool_message(None, calls))
                index = next_index
                continue
            if isinstance(item, ToolResult):
                mapped.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.call_id,
                        "content": item.output,
                    }
                )
                index += 1
                continue
            raise TypeError(f"unsupported conversation item: {type(item).__name__}")
        return mapped

    @staticmethod
    def _following_calls(
        items: Sequence[ConversationItem], start: int
    ) -> tuple[tuple[ToolCall, ...], int]:
        index = start
        calls: list[ToolCall] = []
        while index < len(items):
            item = items[index]
            if not isinstance(item, ToolCall):
                break
            calls.append(item)
            index += 1
        return tuple(calls), index

    @staticmethod
    def _assistant_tool_message(
        content: str | None, calls: Sequence[ToolCall]
    ) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in calls
            ],
        }

    @staticmethod
    def _map_content(raw_content: object | None) -> str | None:
        if raw_content is None:
            return None
        if not isinstance(raw_content, str):
            raise LLMProviderError("llama.cpp returned invalid text output")
        return raw_content.strip() or None

    @staticmethod
    def _map_tool_calls(raw_calls: object | None) -> tuple[ToolCall, ...]:
        if raw_calls is None:
            return ()
        if not isinstance(raw_calls, Sequence):
            raise LLMProviderError("llama.cpp returned invalid function calls")
        try:
            return tuple(
                ToolCall(
                    call_id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
                for call in raw_calls
                if getattr(call, "type", "function") == "function"
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise LLMProviderError("llama.cpp returned invalid function calls") from exc
