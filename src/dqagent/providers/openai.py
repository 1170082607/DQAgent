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
from dqagent.execution import RunContext
from dqagent.models import (
    Completion,
    ConversationItem,
    Message,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from dqagent.providers._openai_sdk import map_token_usage, translate_openai_sdk_error


class OpenAIResponsesClient:
    """Maps DQAgent messages to the OpenAI Responses API."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._model = settings.model
        self._timeout_seconds = settings.timeout_seconds
        self._client = client or OpenAI(
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

        request_options: dict[str, Any] = {}
        if context is not None:
            remaining_seconds = context.remaining_seconds
            if remaining_seconds is not None:
                if remaining_seconds <= 0:
                    context.check_active()
                request_options["timeout"] = min(
                    self._timeout_seconds,
                    remaining_seconds,
                )

        try:
            if request_tools:
                response = self._client.responses.create(
                    model=self._model,
                    input=request_messages,
                    tools=request_tools,
                    **request_options,
                )
            else:
                response = self._client.responses.create(
                    model=self._model,
                    input=request_messages,
                    **request_options,
                )
        except OpenAIError as exc:
            if context is not None:
                context.check_active()
            raise self._translate_error(exc) from exc

        if context is not None:
            context.check_active()

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
            usage=map_token_usage(
                getattr(response, "usage", None),
                input_field="input_tokens",
                output_field="output_tokens",
                provider_label="OpenAI",
            ),
        )

    @staticmethod
    def _translate_error(error: OpenAIError) -> LLMProviderError:
        return translate_openai_sdk_error(error, provider_label="OpenAI")

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
