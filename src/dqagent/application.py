"""Direct chat and stateful agent application use cases."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from threading import Lock
from uuid import uuid4

from dqagent.context import ContextBuilder, ContextWindow, MemoryProjectionEvidence
from dqagent.errors import (
    DQAgentError,
    LLMProviderError,
    RetrievalError,
    RunExecutionError,
    SessionNotFoundError,
)
from dqagent.errors import MemoryError as DQMemoryError
from dqagent.events import RunEventType
from dqagent.execution import RunContext
from dqagent.lifecycle import RunCoordinator, RunScope
from dqagent.llm import LLMClient
from dqagent.memory import MemoryScope
from dqagent.memory_recall import MemoryRecall, MemoryRecallRequest
from dqagent.memory_service import MemoryService
from dqagent.models import ConversationItem, Message, Role
from dqagent.retrieval import (
    CitationResolution,
    RetrievalResult,
    Retriever,
    resolve_answer_citations,
)
from dqagent.runtime import AgentExecutionResult, AgentRunResult, AgentRuntime
from dqagent.session import SessionSnapshot, SessionStore


class ChatApplication:
    """Owns an in-memory conversation and coordinates LLM requests."""

    def __init__(self, llm: LLMClient, system_prompt: str | None = None) -> None:
        self._llm = llm
        self._system_message = (
            Message(Role.SYSTEM, system_prompt) if system_prompt and system_prompt.strip() else None
        )
        self._messages: list[Message] = []
        self.reset()

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    def send(self, user_input: str) -> Message:
        user_message = Message(Role.USER, user_input)
        pending_messages = [*self._messages, user_message]

        completion = self._llm.complete(pending_messages)
        if completion.content is None or completion.tool_calls:
            raise LLMProviderError("direct chat requires a text completion")
        assistant_message = Message(Role.ASSISTANT, completion.content)

        # Commit both messages only after the provider returns a valid response.
        self._messages.extend((user_message, assistant_message))
        return assistant_message

    def reset(self) -> None:
        self._messages = [self._system_message] if self._system_message else []


class AgentApplication:
    """Owns conversation state and commits successful runtime results."""

    def __init__(
        self,
        runtime: AgentRuntime,
        system_prompt: str | None = None,
    ) -> None:
        self._runtime = runtime
        self._system_message = (
            Message(Role.SYSTEM, system_prompt) if system_prompt and system_prompt.strip() else None
        )
        self._messages: list[ConversationItem] = []
        self.reset()

    @property
    def messages(self) -> tuple[ConversationItem, ...]:
        return tuple(self._messages)

    def send(self, user_input: str) -> Message:
        return self.run(user_input).output

    def run(
        self,
        user_input: str,
        *,
        context: RunContext | None = None,
    ) -> AgentRunResult:
        user_message = Message(Role.USER, user_input)
        pending: list[ConversationItem] = [*self._messages, user_message]
        result = self._runtime.run(pending, context=context)
        self._messages = list(result.conversation)
        return result

    def reset(self) -> None:
        self._messages = [self._system_message] if self._system_message else []


@dataclass(frozen=True, slots=True)
class SessionRunResult:
    """One committed durable turn plus the transient context used to execute it."""

    agent: AgentRunResult
    session: SessionSnapshot
    context_window: ContextWindow
    retrieval: RetrievalResult | None = None
    memory_recall: MemoryRecall | None = None

    @property
    def output(self) -> Message:
        return self.agent.output

    @property
    def citations(self) -> CitationResolution | None:
        if self.retrieval is None:
            return None
        return resolve_answer_citations(self.output.content, self.retrieval)

    @property
    def memory(self) -> MemoryRecall | None:
        """Alias for callers that treat recall as the run's memory evidence."""

        return self.memory_recall

    @property
    def memory_projection(self) -> MemoryProjectionEvidence | None:
        """Content-free evidence describing what recall entered active context."""

        return self.context_window.memory_projection


@dataclass(frozen=True, slots=True)
class _SessionExecution:
    agent: AgentExecutionResult
    context_window: ContextWindow
    retrieval: RetrievalResult | None
    memory_recall: MemoryRecall | None


class SessionAgentApplication:
    """Coordinates durable transcripts, bounded context views, and agent runs."""

    def __init__(
        self,
        runtime: AgentRuntime,
        store: SessionStore,
        context_builder: ContextBuilder,
        session_id: str,
        *,
        retriever: Retriever | None = None,
        retrieval_limit: int = 5,
        retrieval_min_score: float = 0.05,
        memory_service: MemoryService | None = None,
        memory_scope: MemoryScope | None = None,
        run_coordinator: RunCoordinator | None = None,
    ) -> None:
        if not session_id.strip():
            raise ValueError("session ID must not be empty")
        _validate_memory_configuration(memory_service, memory_scope)
        self._runtime = runtime
        self._store = store
        self._context_builder = context_builder
        self._session_id = session_id
        if retrieval_limit < 1:
            raise ValueError("retrieval limit must be positive")
        self._retriever = retriever
        self._retrieval_limit = retrieval_limit
        self._retrieval_min_score = retrieval_min_score
        self._memory_service = memory_service
        self._memory_scope = memory_scope
        self._run_coordinator = run_coordinator or runtime.run_coordinator
        self._lock = Lock()

    @classmethod
    def create(
        cls,
        runtime: AgentRuntime,
        store: SessionStore,
        context_builder: ContextBuilder,
        *,
        session_id: str | None = None,
        retriever: Retriever | None = None,
        retrieval_limit: int = 5,
        retrieval_min_score: float = 0.05,
        memory_service: MemoryService | None = None,
        memory_scope: MemoryScope | None = None,
        run_coordinator: RunCoordinator | None = None,
    ) -> "SessionAgentApplication":
        _validate_memory_configuration(memory_service, memory_scope)
        resolved_id = str(uuid4()) if session_id is None else session_id
        store.save(SessionSnapshot(resolved_id), expected_revision=None)
        return cls(
            runtime,
            store,
            context_builder,
            resolved_id,
            retriever=retriever,
            retrieval_limit=retrieval_limit,
            retrieval_min_score=retrieval_min_score,
            memory_service=memory_service,
            memory_scope=memory_scope,
            run_coordinator=run_coordinator,
        )

    @classmethod
    def resume(
        cls,
        runtime: AgentRuntime,
        store: SessionStore,
        context_builder: ContextBuilder,
        session_id: str,
        *,
        retriever: Retriever | None = None,
        retrieval_limit: int = 5,
        retrieval_min_score: float = 0.05,
        memory_service: MemoryService | None = None,
        memory_scope: MemoryScope | None = None,
        run_coordinator: RunCoordinator | None = None,
    ) -> "SessionAgentApplication":
        _validate_memory_configuration(memory_service, memory_scope)
        if store.load(session_id) is None:
            raise SessionNotFoundError(f"session not found: {session_id!r}")
        return cls(
            runtime,
            store,
            context_builder,
            session_id,
            retriever=retriever,
            retrieval_limit=retrieval_limit,
            retrieval_min_score=retrieval_min_score,
            memory_service=memory_service,
            memory_scope=memory_scope,
            run_coordinator=run_coordinator,
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def snapshot(self) -> SessionSnapshot:
        snapshot = self._store.load(self._session_id)
        if snapshot is None:
            raise SessionNotFoundError(f"session not found: {self._session_id!r}")
        return snapshot

    @property
    def messages(self) -> tuple[ConversationItem, ...]:
        return self.snapshot.transcript

    def send(
        self,
        user_input: str,
        *,
        context: RunContext | None = None,
        knowledge_keys: Sequence[str] = (),
    ) -> Message:
        return self.run(
            user_input,
            context=context,
            knowledge_keys=knowledge_keys,
        ).output

    def run(
        self,
        user_input: str,
        *,
        context: RunContext | None = None,
        knowledge_keys: Sequence[str] = (),
    ) -> SessionRunResult:
        user_message = Message(Role.USER, user_input)
        # This lock serializes callers using one application instance. Store CAS remains the
        # authority when separate owners race on the same session revision.
        with self._lock:
            snapshot = self.snapshot
            run_context = (
                context.child(metadata={"session_id": self._session_id})
                if context is not None
                else self._run_coordinator.create_context(
                    metadata={"session_id": self._session_id}
                )
            )
            coordinated = self._run_coordinator.execute(
                lambda scope: self._execute_run_stages(
                    snapshot,
                    user_message,
                    knowledge_keys,
                    scope,
                ),
                context=run_context,
                completion_attributes=lambda result: {
                    "iterations": result.agent.iterations
                },
            )
            execution = coordinated.value
            agent_result = AgentRunResult.from_execution(
                execution.agent,
                coordinated.record,
            )
            candidate = replace(
                snapshot,
                transcript=(
                    *snapshot.transcript,
                    user_message,
                    *agent_result.new_items,
                ),
            )
            saved = self._store.save(candidate, expected_revision=snapshot.revision)
            return SessionRunResult(
                agent_result,
                saved,
                execution.context_window,
                execution.retrieval,
                execution.memory_recall,
            )

    def _execute_run_stages(
        self,
        snapshot: SessionSnapshot,
        user_message: Message,
        knowledge_keys: Sequence[str],
        scope: RunScope,
    ) -> _SessionExecution:
        retrieval: RetrievalResult | None = None
        memory_recall: MemoryRecall | None = None
        try:
            scope.context.check_active()
            if self._retriever is not None:
                retrieval = _retrieve(
                    self._retriever,
                    user_message.content,
                    limit=self._retrieval_limit,
                    min_score=self._retrieval_min_score,
                    scope=scope,
                )
            if self._memory_service is not None and self._memory_scope is not None:
                memory_recall = _recall_memory(
                    self._memory_service,
                    self._memory_scope,
                    user_message.content,
                    scope=scope,
                )
            if memory_recall is None:
                window = self._context_builder.build(
                    snapshot.transcript,
                    user_message,
                    knowledge_keys=knowledge_keys,
                    retrieval=retrieval,
                    context=scope.context,
                )
            else:
                window = self._context_builder.build(
                    snapshot.transcript,
                    user_message,
                    knowledge_keys=knowledge_keys,
                    retrieval=retrieval,
                    memory=memory_recall,
                    context=scope.context,
                )
            scope.emit(RunEventType.CONTEXT_ASSEMBLED, window.event_attributes())
        except DQAgentError:
            raise
        except Exception as exc:
            error = RunExecutionError(
                "unexpected pre-model application failure",
                run_id=scope.context.run_id,
            )
            raise error from exc

        agent = self._runtime.execute(window.items, scope=scope)
        return _SessionExecution(agent, window, retrieval, memory_recall)


def _validate_memory_configuration(
    memory_service: MemoryService | None,
    memory_scope: MemoryScope | None,
) -> None:
    if (memory_service is None) != (memory_scope is None):
        raise ValueError("memory service and memory scope must be provided together")
    if memory_scope is not None and not isinstance(memory_scope, MemoryScope):
        raise TypeError("memory scope must be a MemoryScope")
    if memory_service is not None and not callable(getattr(memory_service, "recall", None)):
        raise TypeError("memory service must provide a recall operation")


def _recall_memory(
    memory_service: MemoryService,
    memory_scope: MemoryScope,
    query: str,
    *,
    scope: RunScope,
) -> MemoryRecall | None:
    request_attributes = _memory_recall_request_attributes(memory_scope, query)
    scope.emit(RunEventType.MEMORY_RECALL_STARTED, request_attributes)
    try:
        request = MemoryRecallRequest(memory_scope, query)
        scope.context.check_active()
        recall = memory_service.recall(request)
        if not isinstance(recall, MemoryRecall):
            raise DQMemoryError("memory recall returned an invalid result")
        if recall.request != request:
            raise DQMemoryError("memory recall returned an invalid request result")
        scope.context.check_active()
    except DQMemoryError as error:
        try:
            _emit_memory_recall_failed(
                scope,
                request_attributes,
                error,
                fallback=True,
                require_active=True,
            )
        except DQAgentError as lifecycle_error:
            _emit_memory_recall_failed(
                scope,
                request_attributes,
                lifecycle_error,
                fallback=False,
            )
            raise
        return None
    except DQAgentError as error:
        _emit_memory_recall_failed(scope, request_attributes, error, fallback=False)
        raise
    except Exception as error:
        try:
            scope.context.check_active()
        except DQAgentError as lifecycle_error:
            _emit_memory_recall_failed(
                scope,
                request_attributes,
                lifecycle_error,
                fallback=False,
            )
            raise
        # Preserve the unknown exception for RunCoordinator's common terminal classification;
        # the stage event deliberately contains only sanitized diagnostic fields.
        failure = DQMemoryError("unexpected memory recall failure")
        _emit_memory_recall_failed(
            scope,
            request_attributes,
            failure,
            fallback=False,
            cause_type=type(error).__name__,
        )
        raise

    scope.emit(
        RunEventType.MEMORY_RECALL_COMPLETED,
        {
            **request_attributes,
            **_memory_recall_result_attributes(recall),
        },
    )
    return recall


def _memory_recall_request_attributes(
    memory_scope: MemoryScope,
    query: str,
) -> dict[str, object]:
    return {
        "scope_kind": memory_scope.kind.value,
        "scope_id_digest": _sha256(memory_scope.scope_id),
        "query_digest": _sha256(query),
        "query_characters": len(query),
        "min_score": 0.05,
        "max_records": 5,
        "max_characters": 8_000,
    }


def _memory_recall_result_attributes(recall: MemoryRecall) -> dict[str, object]:
    return {
        "memory_candidate_count": recall.candidate_count,
        "memory_recalled_count": len(recall.matches),
        "memory_omitted_count": len(recall.omitted),
        "memory_ids": [match.memory_id for match in recall.matches],
        "memory_kinds": [match.record.kind.value for match in recall.matches],
        "memory_scores": [match.score for match in recall.matches],
        "memory_omitted_ids": [match.memory_id for match in recall.omitted],
        "memory_omitted_kinds": [match.record.kind.value for match in recall.omitted],
        "memory_omitted_scores": [match.score for match in recall.omitted],
        "memory_omitted_reasons": [match.reason.value for match in recall.omitted],
        "memory_selector_identity": recall.selector_identity,
    }


def _emit_memory_recall_failed(
    scope: RunScope,
    request_attributes: dict[str, object],
    error: DQAgentError,
    *,
    fallback: bool,
    cause_type: str | None = None,
    require_active: bool = False,
) -> None:
    attributes: dict[str, object] = {
        **request_attributes,
        "error_type": type(error).__name__,
        "error_category": error.category.value,
        "retryable": error.retryable,
        "fallback": fallback,
    }
    if cause_type is not None:
        attributes["cause_type"] = cause_type
    if require_active:
        scope.emit_if_active(RunEventType.MEMORY_RECALL_FAILED, attributes)
    else:
        scope.emit(RunEventType.MEMORY_RECALL_FAILED, attributes)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _retrieve(
    retriever: Retriever,
    query: str,
    *,
    limit: int,
    min_score: float,
    scope: RunScope,
) -> RetrievalResult:
    context = scope.context
    request_attributes = {
        "query": query,
        "limit": limit,
        "min_score": min_score,
    }
    scope.emit(RunEventType.RETRIEVAL_STARTED, request_attributes)
    try:
        retrieval = retriever.retrieve(
            query,
            limit=limit,
            min_score=min_score,
            context=context,
        )
        context.check_active()
    except DQAgentError as exc:
        scope.emit_error(
            RunEventType.RETRIEVAL_FAILED,
            exc,
            request_attributes,
        )
        raise
    except Exception as exc:
        error = RetrievalError("retrieval failed", run_id=context.run_id)
        scope.emit_error(
            RunEventType.RETRIEVAL_FAILED,
            error,
            request_attributes,
            cause_type=type(exc).__name__,
        )
        raise error from exc
    scope.emit(
        RunEventType.RETRIEVAL_COMPLETED,
        _retrieval_event_attributes(retrieval),
    )
    return retrieval


def _retrieval_event_attributes(retrieval: RetrievalResult) -> dict[str, object]:
    return {
        "query": retrieval.query,
        "retrieved_chunk_count": len(retrieval.chunks),
        "retrieved_chunk_ids": [item.chunk.chunk_id for item in retrieval.chunks],
        "retrieval_scores": [item.score for item in retrieval.chunks],
        "retriever_identity": retrieval.retriever_identity,
        "candidate_count": retrieval.candidate_count,
    }
