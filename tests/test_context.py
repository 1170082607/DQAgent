from collections.abc import Sequence
from pathlib import Path

import pytest

from dqagent.context import (
    ContextBudget,
    ContextBuilder,
    FileProjectKnowledgeSource,
    InMemoryKnowledgeSource,
    LLMConversationSummarizer,
    PromptAssembler,
    PromptSection,
    SummaryMethod,
)
from dqagent.errors import ContextError, ContextOverflowError, LLMProviderError
from dqagent.execution import RunContext
from dqagent.models import (
    Completion,
    ConversationItem,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


def completed_turn(user: str, assistant: str) -> tuple[Message, Message]:
    return Message(Role.USER, user), Message(Role.ASSISTANT, assistant)


class SummaryLLM:
    def __init__(self, completion: Completion) -> None:
        self.completion = completion
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
        return self.completion


def test_prompt_sections_and_project_knowledge_are_loaded_on_demand() -> None:
    source = InMemoryKnowledgeSource({"style": "Use Python type hints.", "other": "unused"})
    builder = ContextBuilder(
        PromptAssembler(
            (PromptSection("identity", "You are DQAgent."),),
            knowledge_source=source,
        ),
        ContextBudget(max_characters=2_000, reserved_characters=100),
    )

    window = builder.build((), Message(Role.USER, "Help"), knowledge_keys=("style",))

    assert [item.role for item in window.items if isinstance(item, Message)] == [
        Role.SYSTEM,
        Role.SYSTEM,
        Role.USER,
    ]
    assert "identity" in window.items[0].content  # type: ignore[union-attr]
    assert "Use Python type hints" in window.items[1].content  # type: ignore[union-attr]
    assert window.knowledge_keys == ("style",)


def test_context_trims_only_complete_turns_and_preserves_tool_pairing() -> None:
    tool_turn: tuple[ConversationItem, ...] = (
        Message(Role.USER, "T" * 140),
        ToolCall("call-1", "lookup", '{"query":"value"}'),
        ToolResult("call-1", "lookup", "R" * 140),
        Message(Role.ASSISTANT, "A" * 140),
    )
    transcript = (
        *completed_turn("old " + "x" * 120, "constraint=KEEP " + "y" * 120),
        *tool_turn,
    )
    builder = ContextBuilder(
        PromptAssembler((PromptSection("rules", "Be concise."),)),
        ContextBudget(
            max_characters=850,
            reserved_characters=50,
            summary_max_characters=160,
            structural_input_max_characters=500,
        ),
    )

    window = builder.build(transcript, Message(Role.USER, "new question"))

    calls = [item for item in window.items if isinstance(item, ToolCall)]
    results = [item for item in window.items if isinstance(item, ToolResult)]
    assert len(calls) == len(results)
    assert [call.call_id for call in calls] == [result.call_id for result in results]
    assert window.omitted_turns >= 1
    assert window.summary is not None
    assert window.summary.method is SummaryMethod.STRUCTURAL
    assert window.estimated_characters <= window.max_characters


def test_model_summary_receives_bounded_structural_input_and_records_provenance() -> None:
    llm = SummaryLLM(
        Completion(
            "Keep constraint ALPHA.",
            response_id="summary-1",
            model="summary-model",
        )
    )
    builder = ContextBuilder(
        PromptAssembler(),
        ContextBudget(
            max_characters=450,
            reserved_characters=20,
            summary_max_characters=100,
            structural_input_max_characters=120,
        ),
        summarizer=LLMConversationSummarizer(llm),
    )
    transcript = (
        *completed_turn("remember ALPHA " + "x" * 180, "noted " + "y" * 180),
        *completed_turn("middle " + "m" * 100, "done " + "n" * 100),
    )

    window = builder.build(transcript, Message(Role.USER, "what is the constraint?"))

    assert len(llm.requests) == 1
    summary_input = llm.requests[0][1]
    assert isinstance(summary_input, Message)
    assert len(summary_input.content) <= 120
    assert window.summary is not None
    assert window.summary.method is SummaryMethod.MODEL
    assert window.summary.model == "summary-model"
    assert window.summary.response_id == "summary-1"
    assert "Keep constraint ALPHA" in window.items[-2].content  # type: ignore[union-attr]


def test_context_rejects_mandatory_overflow_and_invalid_tool_pairing() -> None:
    builder = ContextBuilder(
        PromptAssembler((PromptSection("large", "x" * 200),)),
        ContextBudget(max_characters=300, reserved_characters=50),
    )
    with pytest.raises(ContextOverflowError, match="required recent conversation"):
        builder.build((), Message(Role.USER, "y" * 200))

    invalid = (
        Message(Role.USER, "use tool"),
        ToolCall("call-1", "lookup", "{}"),
        Message(Role.ASSISTANT, "done"),
    )
    with pytest.raises(ContextError, match="without results"):
        builder.build(invalid, Message(Role.USER, "next"))


def test_model_summarizer_rejects_tool_call_completion() -> None:
    llm = SummaryLLM(Completion(tool_calls=(ToolCall("call-1", "bad", "{}"),)))

    with pytest.raises(LLMProviderError, match="requires a text completion"):
        LLMConversationSummarizer(llm).summarize("source", max_characters=100)


def test_file_knowledge_source_enforces_root_and_reports_missing_file(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "rules.md").write_text("Project rules", encoding="utf-8")
    source = FileProjectKnowledgeSource(
        tmp_path,
        {
            "rules": Path("docs/rules.md"),
            "escape": Path("../outside.md"),
            "missing": Path("docs/missing.md"),
        },
    )

    assert source.load("rules").content == "Project rules"
    with pytest.raises(ContextError, match="escapes root"):
        source.load("escape")
    with pytest.raises(ContextError, match="cannot load"):
        source.load("missing")
    with pytest.raises(ContextError, match="unknown project knowledge"):
        source.load("unknown")
