"""LLM provider adapters and composition factory."""

from dqagent.config import ModelProvider, Settings
from dqagent.llm import LLMClient
from dqagent.providers.llama_cpp import LlamaCppChatClient
from dqagent.providers.openai import OpenAIResponsesClient


def create_llm_client(settings: Settings) -> LLMClient:
    if settings.provider is ModelProvider.LLAMA_CPP:
        return LlamaCppChatClient(settings)
    return OpenAIResponsesClient(settings)


__all__ = ["LlamaCppChatClient", "OpenAIResponsesClient", "create_llm_client"]
