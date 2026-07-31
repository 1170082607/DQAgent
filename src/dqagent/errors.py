"""Stable error taxonomy exposed by DQAgent."""

from enum import StrEnum


class ErrorCategory(StrEnum):
    CONFIGURATION = "configuration"
    PROVIDER = "provider"
    RATE_LIMIT = "rate_limit"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    LOOP_LIMIT = "loop_limit"
    CONTEXT_LIMIT = "context_limit"
    INTERNAL = "internal"


class DQAgentError(Exception):
    """Base exception for errors that can be shown to application users."""

    category = ErrorCategory.INTERNAL
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory | None = None,
        retryable: bool | None = None,
        run_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category or type(self).category
        self.retryable = type(self).retryable if retryable is None else retryable
        self.run_id = run_id


class ConfigurationError(DQAgentError):
    """Raised when required runtime configuration is invalid or missing."""

    category = ErrorCategory.CONFIGURATION


class LLMProviderError(DQAgentError):
    """Raised when an LLM provider request fails or returns invalid data."""

    category = ErrorCategory.PROVIDER


class AgentLoopError(DQAgentError):
    """Raised when an agent cannot reach a final answer within its safety bound."""

    category = ErrorCategory.LOOP_LIMIT


class RunCancelledError(DQAgentError):
    """Raised when cooperative cancellation stops an agent run."""

    category = ErrorCategory.CANCELLED


class RunDeadlineExceededError(DQAgentError):
    """Raised when an agent run exhausts its wall-clock budget."""

    category = ErrorCategory.DEADLINE_EXCEEDED


class AgentRuntimeError(DQAgentError):
    """Raised when an unexpected failure escapes a runtime boundary."""


class WorkflowDefinitionError(DQAgentError):
    """Raised when a workflow graph violates a structural invariant."""

    category = ErrorCategory.CONFIGURATION


class WorkflowExecutionError(DQAgentError):
    """Raised when a workflow node or transition cannot complete."""


class CheckpointError(DQAgentError):
    """Raised when durable workflow state cannot be loaded or saved."""

    category = ErrorCategory.UNAVAILABLE


class CheckpointConflictError(CheckpointError):
    """Raised when a stale workflow owner attempts to overwrite a checkpoint."""


class SessionError(DQAgentError):
    """Raised when durable conversation state cannot be loaded or saved."""

    category = ErrorCategory.UNAVAILABLE


class SessionNotFoundError(SessionError):
    """Raised when a requested durable session does not exist."""


class SessionConflictError(SessionError):
    """Raised when a stale session owner attempts to overwrite a transcript."""


class ContextError(DQAgentError):
    """Raised when an active model context cannot be constructed safely."""

    category = ErrorCategory.CONFIGURATION


class ContextOverflowError(ContextError):
    """Raised when mandatory prompt content cannot fit the configured budget."""

    category = ErrorCategory.CONTEXT_LIMIT
