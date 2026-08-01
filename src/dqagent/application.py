"""Direct chat and stateful agent application use cases."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from threading import Lock
from uuid import uuid4

from dqagent.context import ContextBuilder, ContextWindow
from dqagent.errors import LLMProviderError, SessionNotFoundError
from dqagent.execution import RunContext
from dqagent.llm import LLMClient
from dqagent.models import ConversationItem, Message, Role
from dqagent.runtime import AgentRunResult, AgentRuntime
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

    @property
    def output(self) -> Message:
        return self.agent.output


class SessionAgentApplication:
    """Coordinates durable transcripts, bounded context views, and agent runs."""

    def __init__(
        self,
        runtime: AgentRuntime,
        store: SessionStore,
        context_builder: ContextBuilder,
        session_id: str,
    ) -> None:
        if not session_id.strip():
            raise ValueError("session ID must not be empty")
        self._runtime = runtime
        self._store = store
        self._context_builder = context_builder
        self._session_id = session_id
        self._lock = Lock()

    @classmethod
    def create(
        cls,
        runtime: AgentRuntime,
        store: SessionStore,
        context_builder: ContextBuilder,
        *,
        session_id: str | None = None,
    ) -> "SessionAgentApplication":
        resolved_id = str(uuid4()) if session_id is None else session_id
        store.save(SessionSnapshot(resolved_id), expected_revision=None)
        return cls(runtime, store, context_builder, resolved_id)

    @classmethod
    def resume(
        cls,
        runtime: AgentRuntime,
        store: SessionStore,
        context_builder: ContextBuilder,
        session_id: str,
    ) -> "SessionAgentApplication":
        if store.load(session_id) is None:
            raise SessionNotFoundError(f"session not found: {session_id!r}")
        return cls(runtime, store, context_builder, session_id)

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
                else self._runtime.create_context(metadata={"session_id": self._session_id})
            )
            window = self._context_builder.build(
                snapshot.transcript,
                user_message,
                knowledge_keys=knowledge_keys,
                context=run_context,
            )
            agent_result = self._runtime.run(
                window.items,
                context=run_context,
                context_attributes=window.event_attributes(),
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
            return SessionRunResult(agent_result, saved, window)
