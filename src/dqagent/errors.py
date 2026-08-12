"""Stable error taxonomy exposed by DQAgent."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol


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


class RunExecutionError(DQAgentError):
    """Raised when an unexpected failure escapes the end-to-end run boundary."""


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


class RetrievalError(DQAgentError):
    """Raised when knowledge ingestion, indexing, or retrieval cannot complete."""

    category = ErrorCategory.UNAVAILABLE


class MemoryError(DQAgentError):
    """Raised when long-term memory processing cannot complete safely."""

    category = ErrorCategory.UNAVAILABLE


class MemoryExtractionError(MemoryError):
    """Raised when a transient memory extraction cannot produce safe candidates."""


class MemoryExtractionSourceError(MemoryExtractionError):
    """Raised when an extraction source is not an explicit bounded committed turn."""

    category = ErrorCategory.CONFIGURATION


class MemoryExtractionFormatError(MemoryExtractionError):
    """Raised when an extractor output violates the transient candidate contract."""

    category = ErrorCategory.CONFIGURATION


class MemoryValidationError(MemoryError, ValueError):
    """Raised when a long-term memory domain value violates an invariant."""

    category = ErrorCategory.CONFIGURATION


class MemoryNotFoundError(MemoryError):
    """Raised when a memory change targets a record that does not exist."""


class MemoryConflictError(MemoryError):
    """Raised when a stale scope revision attempts to overwrite newer memory state."""


class MemoryCorruptChangeError(MemoryError):
    """Raised when a memory change set violates the transactional store contract."""

    category = ErrorCategory.CONFIGURATION


class _MemoryServiceMetadataOwner(Protocol):
    operation: str
    scope_kind: str
    scope_id: str
    memory_id: str | None
    candidate_digest: str | None
    reason: str | None
    metadata: Mapping[str, object]


def _set_memory_service_metadata(
    error: _MemoryServiceMetadataOwner,
    *,
    operation: str,
    scope_kind: str,
    scope_id: str,
    memory_id: str | None,
    candidate_digest: str | None,
    reason: str | None,
) -> None:
    """Attach content-free fields suitable for logs and lifecycle events."""

    error.operation = operation
    error.scope_kind = scope_kind
    error.scope_id = scope_id
    error.memory_id = memory_id
    error.candidate_digest = candidate_digest
    error.reason = reason
    error.metadata = MappingProxyType(
        {
            "operation": operation,
            "scope_kind": scope_kind,
            "scope_id": scope_id,
            "memory_id": memory_id,
            "candidate_digest": candidate_digest,
            "reason": reason,
        }
    )


class MemoryServiceError(MemoryError):
    """Base error for explicit memory management operations."""

    operation: str
    scope_kind: str
    scope_id: str
    memory_id: str | None
    candidate_digest: str | None
    reason: str | None
    metadata: Mapping[str, object]

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        scope_kind: str,
        scope_id: str,
        memory_id: str | None = None,
        candidate_digest: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        _set_memory_service_metadata(
            self,
            operation=operation,
            scope_kind=scope_kind,
            scope_id=scope_id,
            memory_id=memory_id,
            candidate_digest=candidate_digest,
            reason=reason,
        )

    @property
    def event_attributes(self) -> Mapping[str, object]:
        return self.metadata


class MemoryDependencyError(MemoryServiceError):
    """Raised when a policy, clock, or store dependency fails unexpectedly."""


class MemoryAdmissionDeniedError(MemoryServiceError):
    """Raised when policy denies a candidate during a write operation."""


class MemoryDigestMismatchError(MemoryServiceError):
    """Raised when the candidate submitted for confirmation is not digest-identical."""


class MemoryServiceConflictError(MemoryConflictError):
    """A service-level conflict with content-free operation metadata."""

    operation: str
    scope_kind: str
    scope_id: str
    memory_id: str | None
    candidate_digest: str | None
    reason: str | None
    metadata: Mapping[str, object]

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        scope_kind: str,
        scope_id: str,
        memory_id: str | None = None,
        candidate_digest: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        _set_memory_service_metadata(
            self,
            operation=operation,
            scope_kind=scope_kind,
            scope_id=scope_id,
            memory_id=memory_id,
            candidate_digest=candidate_digest,
            reason=reason,
        )

    @property
    def event_attributes(self) -> Mapping[str, object]:
        return self.metadata


class MemoryTopicConflictError(MemoryServiceConflictError):
    """Raised when a candidate conflicts with an active proposition topic."""


class MemoryTargetStateError(MemoryServiceConflictError):
    """Raised when an explicit target is not in a state for the requested mutation."""


class MemoryServiceNotFoundError(MemoryNotFoundError):
    """A service lookup failure with content-free operation metadata."""

    operation: str
    scope_kind: str
    scope_id: str
    memory_id: str | None
    candidate_digest: str | None
    reason: str | None
    metadata: Mapping[str, object]

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        scope_kind: str,
        scope_id: str,
        memory_id: str | None = None,
        candidate_digest: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        _set_memory_service_metadata(
            self,
            operation=operation,
            scope_kind=scope_kind,
            scope_id=scope_id,
            memory_id=memory_id,
            candidate_digest=candidate_digest,
            reason=reason,
        )

    @property
    def event_attributes(self) -> Mapping[str, object]:
        return self.metadata


# Descriptive aliases keep the public vocabulary stable for callers that use either term.
MemoryPolicyDeniedError = MemoryAdmissionDeniedError
MemoryCandidateDigestMismatchError = MemoryDigestMismatchError
