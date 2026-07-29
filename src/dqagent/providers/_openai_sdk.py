"""Shared mapping for adapters that use the OpenAI Python SDK transport."""

from typing import cast

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)

from dqagent.errors import ErrorCategory, LLMProviderError
from dqagent.models import TokenUsage


def map_token_usage(
    usage: object | None,
    *,
    input_field: str,
    output_field: str,
    provider_label: str,
) -> TokenUsage | None:
    if usage is None:
        return None
    input_tokens = getattr(usage, input_field, None)
    output_tokens = getattr(usage, output_field, None)
    total_tokens = getattr(usage, "total_tokens", None)
    values = (input_tokens, output_tokens, total_tokens)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise LLMProviderError(f"{provider_label} returned invalid token usage")
    try:
        return TokenUsage(
            cast(int, input_tokens),
            cast(int, output_tokens),
            cast(int, total_tokens),
        )
    except ValueError as exc:
        raise LLMProviderError(f"{provider_label} returned invalid token usage") from exc


def translate_openai_sdk_error(error: OpenAIError, *, provider_label: str) -> LLMProviderError:
    if isinstance(error, APITimeoutError):
        return LLMProviderError(
            f"{provider_label} request timed out: {error}",
            category=ErrorCategory.TIMEOUT,
            retryable=True,
        )
    if isinstance(error, RateLimitError):
        return LLMProviderError(
            f"{provider_label} rate limit exceeded: {error}",
            category=ErrorCategory.RATE_LIMIT,
            retryable=True,
        )
    if isinstance(error, (APIConnectionError, InternalServerError)):
        return LLMProviderError(
            f"{provider_label} service unavailable: {error}",
            category=ErrorCategory.UNAVAILABLE,
            retryable=True,
        )
    return LLMProviderError(f"{provider_label} request failed: {error}")
