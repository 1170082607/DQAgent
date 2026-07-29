from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APITimeoutError, OpenAI, OpenAIError

from dqagent.config import ModelProvider, Settings
from dqagent.errors import ErrorCategory, LLMProviderError
from dqagent.execution import RunContext
from dqagent.models import (
    Completion,
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from dqagent.providers import create_llm_client
from dqagent.providers.llama_cpp import LlamaCppChatClient
from dqagent.providers.openai import OpenAIResponsesClient


class FakeCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.response


class FakeLlamaCpp:
    def __init__(self, response: object) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(response))


class FailingCompletions:
    def __init__(self, error: OpenAIError) -> None:
        self._error = error

    def create(self, **kwargs: Any) -> object:
        raise self._error


class FailingLlamaCpp:
    def __init__(self, error: OpenAIError) -> None:
        self.chat = SimpleNamespace(completions=FailingCompletions(error))


def make_settings() -> Settings:
    return Settings(
        api_key="local",
        model="local-model",
        provider=ModelProvider.LLAMA_CPP,
        base_url="http://127.0.0.1:8080/v1",
    )


def chat_response(
    *,
    content: object | None = "Hello",
    tool_calls: object | None = None,
    usage: object | None = None,
) -> object:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        id="chatcmpl-1",
        model="local-model",
        choices=[SimpleNamespace(message=message)],
        usage=usage,
    )


def test_complete_maps_chat_messages_response_and_usage() -> None:
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=4, total_tokens=16)
    sdk = FakeLlamaCpp(chat_response(content="  Hello locally  ", usage=usage))
    client = LlamaCppChatClient(make_settings(), client=sdk)

    completion = client.complete(
        [Message(Role.SYSTEM, "Be concise."), Message(Role.USER, "Hello")]
    )

    assert completion == Completion(
        content="Hello locally",
        response_id="chatcmpl-1",
        model="local-model",
        usage=TokenUsage(12, 4, 16),
    )
    assert sdk.chat.completions.calls == [
        {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ],
        }
    ]


def test_real_sdk_transport_calls_llama_server_chat_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-http-1",
                "object": "chat.completion",
                "created": 1,
                "model": "local-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Local response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle))
    sdk = OpenAI(
        api_key="local",
        base_url="http://127.0.0.1:8080/v1",
        http_client=http_client,
        max_retries=0,
    )
    try:
        completion = LlamaCppChatClient(make_settings(), client=sdk).complete(
            [Message(Role.USER, "Hello")]
        )
    finally:
        http_client.close()

    assert completion.content == "Local response"
    assert completion.usage == TokenUsage(5, 2, 7)
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/chat/completions"


def test_complete_maps_tool_schema_calls_and_observations() -> None:
    raw_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name="weather", arguments='{"city":"Beijing"}'),
    )
    sdk = FakeLlamaCpp(chat_response(content="", tool_calls=[raw_call]))
    client = LlamaCppChatClient(make_settings(), client=sdk)
    definition = ToolDefinition(
        name="weather",
        description="Get weather by city.",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )

    completion = client.complete([Message(Role.USER, "Weather?")], [definition])

    call = ToolCall("call-1", "weather", '{"city":"Beijing"}')
    assert completion.tool_calls == (call,)
    assert sdk.chat.completions.calls[0] == {
        "model": "local-model",
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather by city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
    }

    sdk.chat.completions.response = chat_response(content="Sunny")
    client.complete(
        [
            Message(Role.USER, "Weather?"),
            Message(Role.ASSISTANT, "I will check."),
            call,
            ToolResult("call-1", "weather", "sunny"),
        ],
        [definition],
    )

    assert sdk.chat.completions.calls[1]["messages"] == [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "I will check.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city":"Beijing"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
    ]


def test_complete_caps_timeout_and_classifies_sdk_errors() -> None:
    sdk = FakeLlamaCpp(chat_response())
    context = RunContext(run_id="llama-run", timeout_seconds=5)

    LlamaCppChatClient(make_settings(), client=sdk).complete(
        [Message(Role.USER, "Hello")], context=context
    )

    timeout = sdk.chat.completions.calls[0]["timeout"]
    assert isinstance(timeout, float)
    assert 0 < timeout <= 5

    error_client = LlamaCppChatClient(
        make_settings(),
        client=FailingLlamaCpp(OpenAIError("local server failed")),
    )
    with pytest.raises(LLMProviderError, match="llama.cpp request failed") as error:
        error_client.complete([Message(Role.USER, "Hello")])
    assert error.value.category is ErrorCategory.PROVIDER

    timeout_client = LlamaCppChatClient(
        make_settings(),
        client=FailingLlamaCpp(
            APITimeoutError(request=httpx.Request("POST", "http://127.0.0.1:8080/v1"))
        ),
    )
    with pytest.raises(LLMProviderError) as timeout_error:
        timeout_client.complete([Message(Role.USER, "Hello")])
    assert timeout_error.value.category is ErrorCategory.TIMEOUT
    assert timeout_error.value.retryable is True


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[]),
        chat_response(content=123),
        chat_response(content="", tool_calls=[]),
        chat_response(
            content=None,
            tool_calls=[SimpleNamespace(id="", type="function", function=SimpleNamespace())],
        ),
    ],
)
def test_complete_rejects_invalid_responses(response: object) -> None:
    client = LlamaCppChatClient(make_settings(), client=FakeLlamaCpp(response))

    with pytest.raises(LLMProviderError):
        client.complete([Message(Role.USER, "Hello")])


def test_provider_factory_selects_configured_adapter() -> None:
    assert isinstance(create_llm_client(make_settings()), LlamaCppChatClient)
    assert isinstance(
        create_llm_client(Settings(api_key="secret", model="gpt-test")),
        OpenAIResponsesClient,
    )
