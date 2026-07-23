"""OpenAI Responses API adapter."""

from collections.abc import Sequence
from typing import Any

from openai import OpenAI, OpenAIError
from openai.types.responses import (
    FunctionToolParam,
    ResponseInputItemParam,
    ResponseInputParam,
)

from dqagent.config import Settings
from dqagent.errors import LLMProviderError
from dqagent.models import (
    Completion,
    ConversationItem,
    Message,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class OpenAIResponsesClient:
    """Maps DQAgent messages to the OpenAI Responses API."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._model = settings.model
        self._client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
        )

    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolDefinition] = (),
    ) -> Completion:
        request_messages: ResponseInputParam = [self._map_input(item) for item in messages]
        request_tools: list[FunctionToolParam] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.input_schema),
                "strict": False,
            }
            for tool in tools
        ]

        try:
            if request_tools:
                response = self._client.responses.create(
                    model=self._model,
                    input=request_messages,
                    tools=request_tools,
                )
            else:
                response = self._client.responses.create(
                    model=self._model,
                    input=request_messages,
                )
        except OpenAIError as exc:
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc

        raw_output_text = getattr(response, "output_text", None)
        if raw_output_text is not None and not isinstance(raw_output_text, str):
            raise LLMProviderError("OpenAI returned invalid text output")
        output_text = raw_output_text.strip() or None if raw_output_text is not None else None
        try:
            tool_calls = tuple(
                ToolCall(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=item.arguments,
                )
                for item in (getattr(response, "output", None) or ())
                if getattr(item, "type", None) == "function_call"
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise LLMProviderError("OpenAI returned an invalid function call") from exc

        if output_text is None and not tool_calls:
            raise LLMProviderError("OpenAI returned neither text nor function calls")

        return Completion(
            content=output_text,
            tool_calls=tool_calls,
            response_id=getattr(response, "id", None),
            model=getattr(response, "model", None),
        )

    @staticmethod
    def _map_input(item: ConversationItem) -> ResponseInputItemParam:
        if isinstance(item, Message):
            return {"role": item.role.value, "content": item.content}
        if isinstance(item, ToolCall):
            return {
                "type": "function_call",
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
            }
        if isinstance(item, ToolResult):
            return {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": item.output,
            }
        raise TypeError(f"unsupported conversation item: {type(item).__name__}")
