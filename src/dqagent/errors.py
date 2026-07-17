"""Domain-level exceptions exposed by DQAgent."""


class DQAgentError(Exception):
    """Base exception for errors that can be shown to application users."""


class ConfigurationError(DQAgentError):
    """Raised when required runtime configuration is invalid or missing."""


class LLMProviderError(DQAgentError):
    """Raised when an LLM provider request fails or returns invalid data."""
