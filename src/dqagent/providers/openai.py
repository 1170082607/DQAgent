"""OpenAI Responses API adapter."""

from collections.abc import Sequence
from typing import Any

from openai import OpenAI, OpenAIError
from openai.types.responses import ResponseInputParam

from dqagent.config import Settings
from dqagent.errors import LLMProviderError
from dqagent.models import Completion, Message


class OpenAIResponsesClient:
    """Maps DQAgent messages to the OpenAI Responses API."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._model = settings.model
        self._client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
        )

    def complete(self, messages: Sequence[Message]) -> Completion:
        request_messages: ResponseInputParam = [
            {"role": message.role.value, "content": message.content} for message in messages
        ]

        try:
            response = self._client.responses.create(
                model=self._model,
                input=request_messages,
            )
        except OpenAIError as exc:
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc

        output_text = str(getattr(response, "output_text", "")).strip()
        if not output_text:
            raise LLMProviderError("OpenAI returned an empty text response")

        return Completion(
            content=output_text,
            response_id=getattr(response, "id", None),
            model=getattr(response, "model", None),
        )
