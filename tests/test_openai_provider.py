from types import SimpleNamespace
from typing import Any

import pytest
from openai import OpenAIError

from dqagent.config import Settings
from dqagent.errors import LLMProviderError
from dqagent.models import (
    Completion,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
    ToolErrorCode,
    ToolOutcome,
    ToolResult,
)
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


def test_complete_rejects_response_without_text_or_tool_calls() -> None:
    client = OpenAIResponsesClient(
        make_settings(),
        client=FakeOpenAI(SimpleNamespace(output_text="", id="response-1")),
    )

    with pytest.raises(LLMProviderError, match="neither text nor function calls"):
        client.complete([Message(Role.USER, "Hi")])


def test_complete_rejects_non_text_output() -> None:
    client = OpenAIResponsesClient(
        make_settings(),
        client=FakeOpenAI(SimpleNamespace(output_text=123, output=[], id="response-1")),
    )

    with pytest.raises(LLMProviderError, match="invalid text output"):
        client.complete([Message(Role.USER, "Hi")])


def test_complete_translates_openai_errors() -> None:
    client = OpenAIResponsesClient(make_settings(), client=FailingOpenAI())

    with pytest.raises(LLMProviderError, match="provider unavailable"):
        client.complete([Message(Role.USER, "Hi")])


def test_complete_maps_tools_calls_and_observations() -> None:
    function_call = SimpleNamespace(
        type="function_call",
        call_id="call-1",
        name="weather",
        arguments='{"city":"Beijing"}',
    )
    sdk = FakeOpenAI(
        SimpleNamespace(output_text="", output=[function_call], id="response-1", model="model")
    )
    client = OpenAIResponsesClient(make_settings(), client=sdk)
    definition = ToolDefinition(
        name="weather",
        description="Get weather by city.",
        input_schema={"type": "object"},
    )

    completion = client.complete([Message(Role.USER, "Weather?")], [definition])

    call = ToolCall("call-1", "weather", '{"city":"Beijing"}')
    assert completion == Completion(
        tool_calls=(call,), response_id="response-1", model="model"
    )
    assert sdk.responses.calls == [
        {
            "model": "test-model",
            "input": [{"role": "user", "content": "Weather?"}],
            "tools": [
                {
                    "type": "function",
                    "name": "weather",
                    "description": "Get weather by city.",
                    "parameters": {"type": "object"},
                    "strict": False,
                }
            ],
        }
    ]

    sdk.responses._response = SimpleNamespace(
        output_text="Sunny", output=[], id="response-2", model="model"
    )
    error_result = ToolResult(
        "call-1",
        "weather",
        "service unavailable",
        ToolOutcome.ERROR,
        ToolErrorCode.EXECUTION_ERROR,
    )

    client.complete([Message(Role.USER, "Weather?"), call, error_result], [definition])

    assert sdk.responses.calls[1]["input"] == [
        {"role": "user", "content": "Weather?"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "weather",
            "arguments": '{"city":"Beijing"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "service unavailable",
        },
    ]
