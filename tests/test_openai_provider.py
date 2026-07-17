from types import SimpleNamespace
from typing import Any

import pytest
from openai import OpenAIError

from dqagent.config import Settings
from dqagent.errors import LLMProviderError
from dqagent.models import Completion, Message, Role
from dqagent.providers.openai import OpenAIResponsesClient


class FakeResponses:
    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self._response


class FakeOpenAI:
    def __init__(self, response: object) -> None:
        self.responses = FakeResponses(response)


class FailingResponses:
    def create(self, **kwargs: Any) -> object:
        raise OpenAIError("provider unavailable")


class FailingOpenAI:
    responses = FailingResponses()


def make_settings() -> Settings:
    return Settings(api_key="secret", model="test-model")


def test_complete_maps_messages_and_response() -> None:
    sdk = FakeOpenAI(
        SimpleNamespace(output_text="  Hello  ", id="response-1", model="resolved-model")
    )
    client = OpenAIResponsesClient(make_settings(), client=sdk)

    completion = client.complete(
        [Message(Role.SYSTEM, "Be concise."), Message(Role.USER, "Hi")]
    )

    assert completion == Completion(
        content="Hello", response_id="response-1", model="resolved-model"
    )
    assert sdk.responses.calls == [
        {
            "model": "test-model",
            "input": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
            ],
        }
    ]


def test_complete_rejects_empty_provider_response() -> None:
    client = OpenAIResponsesClient(
        make_settings(),
        client=FakeOpenAI(SimpleNamespace(output_text="", id="response-1")),
    )

    with pytest.raises(LLMProviderError, match="empty text response"):
        client.complete([Message(Role.USER, "Hi")])


def test_complete_translates_openai_errors() -> None:
    client = OpenAIResponsesClient(make_settings(), client=FailingOpenAI())

    with pytest.raises(LLMProviderError, match="provider unavailable"):
        client.complete([Message(Role.USER, "Hi")])
