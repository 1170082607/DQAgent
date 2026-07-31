from collections.abc import Sequence
from dataclasses import replace

import pytest

from dqagent.application import SessionAgentApplication
from dqagent.context import ContextBudget, ContextBuilder, PromptAssembler, PromptSection
from dqagent.errors import LLMProviderError, SessionConflictError, SessionNotFoundError
from dqagent.events import RunEventType
from dqagent.execution import RunContext
from dqagent.models import Completion, ConversationItem, Message, Role, ToolDefinition
from dqagent.runtime import AgentRuntime, RetryPolicy
from dqagent.session import InMemorySessionStore, SessionSnapshot
from dqagent.tools import ToolRegistry


class StubLLM:
    def __init__(self, completions: Sequence[Completion]) -> None:
        self._completions = iter(completions)
        self.requests: list[tuple[ConversationItem, ...]] = []

    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolDefinition] = (),
        *,
        context: RunContext | None = None,
    ) -> Completion:
        del tools, context
        self.requests.append(tuple(messages))
        return next(self._completions)


class FailingLLM:
    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolDefinition] = (),
        *,
        context: RunContext | None = None,
    ) -> Completion:
        del messages, tools, context
        raise LLMProviderError("provider failed")


def make_runtime(llm: StubLLM | FailingLLM) -> AgentRuntime:
    return AgentRuntime(
        llm,
        ToolRegistry(),
        retry_policy=RetryPolicy(max_attempts=1),
    )


def make_builder(*, small: bool = False) -> ContextBuilder:
    return ContextBuilder(
        PromptAssembler((PromptSection("behavior", "Be precise."),)),
        ContextBudget(
            max_characters=500 if small else 4_000,
            reserved_characters=50,
            summary_max_characters=120,
        ),
    )


def test_session_application_persists_and_resumes_complete_transcript() -> None:
    store = InMemorySessionStore()
    first_llm = StubLLM([Completion("First answer")])
    app = SessionAgentApplication.create(
        make_runtime(first_llm), store, make_builder(), session_id="learning"
    )

    first = app.run("First question")
    second_llm = StubLLM([Completion("Second answer")])
    resumed = SessionAgentApplication.resume(
        make_runtime(second_llm), store, make_builder(), "learning"
    )
    second = resumed.run("Second question")

    assert first.session.revision == 2
    assert second.session.revision == 3
    assert second.session.transcript == (
        Message(Role.USER, "First question"),
        Message(Role.ASSISTANT, "First answer"),
        Message(Role.USER, "Second question"),
        Message(Role.ASSISTANT, "Second answer"),
    )
    assert second_llm.requests[0][-3:] == second.context_window.items[-3:]
    assert any(event.type is RunEventType.CONTEXT_ASSEMBLED for event in second.agent.events)
    assert second.agent.events[0].attributes["metadata"] == {"session_id": "learning"}


def test_compacted_system_summary_is_not_written_to_durable_transcript() -> None:
    store = InMemorySessionStore()
    transcript = (
        Message(Role.USER, "old " + "x" * 200),
        Message(Role.ASSISTANT, "constraint BETA " + "y" * 200),
    )
    store.save(SessionSnapshot("compact", transcript), expected_revision=None)
    llm = StubLLM([Completion("BETA")])
    app = SessionAgentApplication.resume(
        make_runtime(llm), store, make_builder(small=True), "compact"
    )

    result = app.run("recall?")

    assert result.context_window.summary is not None
    assert any(
        isinstance(item, Message) and item.role is Role.SYSTEM and "context-summary" in item.content
        for item in llm.requests[0]
    )
    assert not any(
        isinstance(item, Message) and item.role is Role.SYSTEM
        for item in result.session.transcript
    )


def test_failed_run_does_not_advance_session_revision() -> None:
    store = InMemorySessionStore()
    app = SessionAgentApplication.create(
        make_runtime(FailingLLM()), store, make_builder(), session_id="failure"
    )
    before = app.snapshot

    with pytest.raises(LLMProviderError, match="provider failed"):
        app.send("Do not commit")

    assert app.snapshot == before


def test_separate_owner_conflict_is_detected_after_model_run() -> None:
    store = InMemorySessionStore()
    initial = store.save(SessionSnapshot("race"), expected_revision=None)

    class RacingLLM(StubLLM):
        def complete(
            self,
            messages: Sequence[ConversationItem],
            tools: Sequence[ToolDefinition] = (),
            *,
            context: RunContext | None = None,
        ) -> Completion:
            current = store.load("race")
            assert current is not None
            store.save(replace(current), expected_revision=current.revision)
            return super().complete(messages, tools, context=context)

    app = SessionAgentApplication.resume(
        make_runtime(RacingLLM([Completion("losing answer")])),
        store,
        make_builder(),
        "race",
    )

    with pytest.raises(SessionConflictError, match="expected 1, found 2"):
        app.send("racing request")

    current = store.load("race")
    assert current is not None
    assert current.revision == initial.revision + 1
    assert current.transcript == ()


def test_resume_rejects_missing_session() -> None:
    with pytest.raises(SessionNotFoundError, match="session not found"):
        SessionAgentApplication.resume(
            make_runtime(StubLLM([])),
            InMemorySessionStore(),
            make_builder(),
            "missing",
        )
