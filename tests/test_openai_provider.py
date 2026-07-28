import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)

from dqagent.config import Settings
from dqagent.errors import ErrorCategory, LLMProviderError, RunDeadlineExceededError
from dqagent.execution import RunContext
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


class ErrorResponses:
    def __init__(self, error: OpenAIError) -> None:
        self._error = error

    def create(self, **kwargs: Any) -> object:
        raise self._error


class ErrorOpenAI:
    def __init__(self, error: OpenAIError) -> None:
        self.responses = ErrorResponses(error)


class TimeoutResponses:
    def create(self, **kwargs: Any) -> object:
        raise APITimeoutError(request=httpx.Request("POST", "https://example.test"))


class TimeoutOpenAI:
    responses = TimeoutResponses()


class DelayedTimeoutResponses:
    def create(self, **kwargs: Any) -> object:
        time.sleep(0.005)
        raise APITimeoutError(request=httpx.Request("POST", "https://example.test"))


class DelayedTimeoutOpenAI:
    responses = DelayedTimeoutResponses()


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

    with pytest.raises(LLMProviderError, match="provider unavailable") as error:
        client.complete([Message(Role.USER, "Hi")])

    assert error.value.category is ErrorCategory.PROVIDER
    assert error.value.retryable is False


def test_complete_classifies_timeout_as_retryable() -> None:
    client = OpenAIResponsesClient(make_settings(), client=TimeoutOpenAI())

    with pytest.raises(LLMProviderError) as error:
        client.complete([Message(Role.USER, "Hi")])

    assert error.value.category is ErrorCategory.TIMEOUT
    assert error.value.retryable is True


@pytest.mark.parametrize(
    ("provider_error", "category"),
    [
        (
            RateLimitError(
                "rate limited",
                response=httpx.Response(
                    429,
                    request=httpx.Request("POST", "https://example.test"),
                ),
                body=None,
            ),
            ErrorCategory.RATE_LIMIT,
        ),
        (
            APIConnectionError(
                request=httpx.Request("POST", "https://example.test"),
            ),
            ErrorCategory.UNAVAILABLE,
        ),
        (
            InternalServerError(
                "server error",
                response=httpx.Response(
                    500,
                    request=httpx.Request("POST", "https://example.test"),
                ),
                body=None,
            ),
            ErrorCategory.UNAVAILABLE,
        ),
    ],
)
def test_complete_classifies_retryable_provider_errors(
    provider_error: OpenAIError,
    category: ErrorCategory,
) -> None:
    client = OpenAIResponsesClient(make_settings(), client=ErrorOpenAI(provider_error))

    with pytest.raises(LLMProviderError) as error:
        client.complete([Message(Role.USER, "Hi")])

    assert error.value.category is category
    assert error.value.retryable is True


def test_complete_prefers_exhausted_run_deadline_over_provider_timeout() -> None:
    client = OpenAIResponsesClient(make_settings(), client=DelayedTimeoutOpenAI())
    context = RunContext(run_id="run-timeout", timeout_seconds=0.001)

    with pytest.raises(RunDeadlineExceededError) as error:
        client.complete([Message(Role.USER, "Hi")], context=context)

    assert error.value.run_id == "run-timeout"


def test_complete_caps_provider_timeout_by_run_deadline() -> None:
    sdk = FakeOpenAI(SimpleNamespace(output_text="Done", output=[]))
    client = OpenAIResponsesClient(make_settings(), client=sdk)
    context = RunContext(run_id="run-1", timeout_seconds=5)

    client.complete([Message(Role.USER, "Hi")], context=context)

    request_timeout = sdk.responses.calls[0]["timeout"]
    assert isinstance(request_timeout, float)
    assert 0 < request_timeout <= 5


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
