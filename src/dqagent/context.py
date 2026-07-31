"""Prompt assembly, on-demand knowledge, and bounded active model context."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from dqagent.errors import ContextError, ContextOverflowError, LLMProviderError
from dqagent.execution import RunContext
from dqagent.llm import LLMClient
from dqagent.models import (
    ConversationItem,
    Message,
    Role,
    ToolCall,
    ToolResult,
)


@dataclass(frozen=True, slots=True)
class PromptSection:
    """One application-owned, independently testable prompt section."""

    name: str
    content: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("prompt section name must not be empty")
        if not self.content.strip():
            raise ValueError("prompt section content must not be empty")


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    key: str
    content: str
    source: str

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("knowledge key must not be empty")
        if not self.content.strip():
            raise ValueError("knowledge content must not be empty")
        if not self.source.strip():
            raise ValueError("knowledge source must not be empty")


class KnowledgeSource(Protocol):
    """Loads explicitly requested project knowledge by stable key."""

    def load(self, key: str) -> KnowledgeDocument: ...


class InMemoryKnowledgeSource:
    def __init__(self, documents: Mapping[str, str]) -> None:
        self._documents = dict(documents)

    def load(self, key: str) -> KnowledgeDocument:
        try:
            content = self._documents[key]
        except KeyError as exc:
            raise ContextError(f"unknown project knowledge key: {key!r}") from exc
        return KnowledgeDocument(key, content, f"memory:{key}")


class FileProjectKnowledgeSource:
    """Reads only allowlisted files contained by one configured project root."""

    def __init__(self, root: Path, documents: Mapping[str, Path]) -> None:
        self._root = root.resolve()
        self._documents = dict(documents)
        for key in self._documents:
            if not key.strip():
                raise ValueError("knowledge key must not be empty")

    def load(self, key: str) -> KnowledgeDocument:
        try:
            configured = self._documents[key]
        except KeyError as exc:
            raise ContextError(f"unknown project knowledge key: {key!r}") from exc
        candidate = (self._root / configured).resolve()
        if not candidate.is_relative_to(self._root):
            raise ContextError(f"project knowledge path escapes root for key {key!r}")
        try:
            content = candidate.read_text(encoding="utf-8")
        except OSError as exc:
            raise ContextError(f"cannot load project knowledge {key!r}: {exc}") from exc
        return KnowledgeDocument(key, content, str(configured).replace("\\", "/"))


@dataclass(frozen=True, slots=True)
class PromptAssembly:
    messages: tuple[Message, ...]
    knowledge: tuple[KnowledgeDocument, ...]


class PromptAssembler:
    """Builds system messages from owned sections and requested knowledge."""

    def __init__(
        self,
        sections: Sequence[PromptSection] = (),
        *,
        knowledge_source: KnowledgeSource | None = None,
    ) -> None:
        names = [section.name for section in sections]
        if len(names) != len(set(names)):
            raise ValueError("prompt section names must be unique")
        self._sections = tuple(sections)
        self._knowledge_source = knowledge_source

    def assemble(self, knowledge_keys: Sequence[str] = ()) -> PromptAssembly:
        if len(knowledge_keys) != len(set(knowledge_keys)):
            raise ValueError("project knowledge keys must be unique")
        if knowledge_keys and self._knowledge_source is None:
            raise ContextError("project knowledge was requested without a knowledge source")
        documents = tuple(
            self._knowledge_source.load(key)  # type: ignore[union-attr]
            for key in knowledge_keys
        )
        messages = tuple(
            Message(Role.SYSTEM, f"[section:{section.name}]\n{section.content}")
            for section in self._sections
        ) + tuple(
            Message(
                Role.SYSTEM,
                f"[knowledge:{document.key} source={document.source}]\n{document.content}",
            )
            for document in documents
        )
        return PromptAssembly(messages, documents)


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Provider-neutral character estimate with explicit safety reserve."""

    max_characters: int = 32_000
    reserved_characters: int = 4_000
    summary_max_characters: int = 2_000
    structural_input_max_characters: int = 8_000
    min_recent_turns: int = 1

    def __post_init__(self) -> None:
        if self.max_characters < 1:
            raise ValueError("maximum context characters must be positive")
        if self.reserved_characters < 0:
            raise ValueError("reserved context characters must be non-negative")
        if self.reserved_characters >= self.max_characters:
            raise ValueError("reserved context characters must be below the maximum")
        if self.summary_max_characters < 0:
            raise ValueError("summary characters must be non-negative")
        if self.structural_input_max_characters < 1:
            raise ValueError("structural input characters must be positive")
        if self.min_recent_turns < 1:
            raise ValueError("minimum recent turns must be at least one")

    @property
    def active_characters(self) -> int:
        return self.max_characters - self.reserved_characters


class SummaryMethod(StrEnum):
    STRUCTURAL = "structural"
    MODEL = "model"


@dataclass(frozen=True, slots=True)
class SummaryDraft:
    content: str
    method: SummaryMethod
    model: str | None = None
    response_id: str | None = None


class ConversationSummarizer(Protocol):
    def summarize(
        self,
        structural_source: str,
        *,
        max_characters: int,
        context: RunContext | None = None,
    ) -> SummaryDraft: ...


class StructuralSummarizer:
    """Uses the bounded structural representation directly, without another model call."""

    def summarize(
        self,
        structural_source: str,
        *,
        max_characters: int,
        context: RunContext | None = None,
    ) -> SummaryDraft:
        del context
        return SummaryDraft(structural_source[:max_characters], SummaryMethod.STRUCTURAL)


class LLMConversationSummarizer:
    """Summarizes an already structurally compacted source through the LLM boundary."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def summarize(
        self,
        structural_source: str,
        *,
        max_characters: int,
        context: RunContext | None = None,
    ) -> SummaryDraft:
        completion = self._llm.complete(
            (
                Message(
                    Role.SYSTEM,
                    "Summarize durable conversation facts, decisions, unresolved work, and "
                    "constraints. Do not invent facts.",
                ),
                Message(Role.USER, structural_source),
            ),
            context=context,
        )
        if completion.content is None or completion.tool_calls:
            raise LLMProviderError("context summarization requires a text completion")
        return SummaryDraft(
            completion.content[:max_characters],
            SummaryMethod.MODEL,
            model=completion.model,
            response_id=completion.response_id,
        )


@dataclass(frozen=True, slots=True)
class SummaryProvenance:
    method: SummaryMethod
    source_digest: str
    source_item_count: int
    source_characters: int
    summary_characters: int
    structural_input_characters: int
    model: str | None = None
    response_id: str | None = None


@dataclass(frozen=True, slots=True)
class ContextWindow:
    items: tuple[ConversationItem, ...]
    estimated_characters: int
    max_characters: int
    retained_turns: int
    omitted_turns: int
    knowledge_keys: tuple[str, ...]
    summary: SummaryProvenance | None = None

    def event_attributes(self) -> Mapping[str, object]:
        summary = self.summary
        return MappingProxyType(
            {
                "estimated_characters": self.estimated_characters,
                "max_characters": self.max_characters,
                "retained_turns": self.retained_turns,
                "omitted_turns": self.omitted_turns,
                "knowledge_keys": list(self.knowledge_keys),
                "summary_method": summary.method.value if summary else None,
                "summary_source_digest": summary.source_digest if summary else None,
                "summary_source_item_count": summary.source_item_count if summary else 0,
            }
        )


class ContextBuilder:
    """Projects a durable transcript into one bounded, structurally valid model view."""

    _SUMMARY_HEADER_RESERVE = 180

    def __init__(
        self,
        prompt_assembler: PromptAssembler,
        budget: ContextBudget | None = None,
        *,
        summarizer: ConversationSummarizer | None = None,
    ) -> None:
        self._prompt_assembler = prompt_assembler
        self._budget = budget or ContextBudget()
        self._summarizer = summarizer or StructuralSummarizer()

    def build(
        self,
        transcript: Sequence[ConversationItem],
        user_message: Message,
        *,
        knowledge_keys: Sequence[str] = (),
        context: RunContext | None = None,
    ) -> ContextWindow:
        if user_message.role is not Role.USER:
            raise ValueError("active context requires a user message")
        transcript_items = tuple(transcript)
        _validate_complete_transcript(transcript_items)
        prompt = self._prompt_assembler.assemble(knowledge_keys)
        turns = _split_turns((*transcript_items, user_message))
        prompt_cost = _items_size(prompt.messages)
        all_cost = prompt_cost + sum(_items_size(turn) for turn in turns)
        limit = self._budget.active_characters
        if all_cost <= limit:
            items = (*prompt.messages, *(item for turn in turns for item in turn))
            return ContextWindow(
                items=items,
                estimated_characters=all_cost,
                max_characters=limit,
                retained_turns=len(turns),
                omitted_turns=0,
                knowledge_keys=tuple(document.key for document in prompt.knowledge),
            )

        recent_count = min(self._budget.min_recent_turns, len(turns))
        retained = list(turns[-recent_count:])
        retained_cost = sum(_items_size(turn) for turn in retained)
        if prompt_cost + retained_cost > limit:
            raise ContextOverflowError(
                "prompt sections and the required recent conversation exceed the active "
                "context budget"
            )

        older = list(turns[:-recent_count])
        summary_reserve = min(
            self._budget.summary_max_characters + self._SUMMARY_HEADER_RESERVE,
            max(0, limit - prompt_cost - retained_cost),
        )
        while older:
            candidate = older[-1]
            candidate_cost = _items_size(candidate)
            if prompt_cost + retained_cost + candidate_cost + summary_reserve > limit:
                break
            retained.insert(0, older.pop())
            retained_cost += candidate_cost

        omitted = tuple(item for turn in older for item in turn)
        summary_message: Message | None = None
        provenance: SummaryProvenance | None = None
        if omitted and self._budget.summary_max_characters > 0:
            structural_source = _structural_compact(
                older,
                self._budget.structural_input_max_characters,
            )
            available = max(0, limit - prompt_cost - retained_cost - self._SUMMARY_HEADER_RESERVE)
            summary_limit = min(self._budget.summary_max_characters, available)
            if summary_limit > 0:
                draft = self._summarizer.summarize(
                    structural_source,
                    max_characters=summary_limit,
                    context=context,
                )
                if not draft.content.strip():
                    raise ContextError("context summarizer returned empty content")
                digest = hashlib.sha256(_serialize_items(omitted).encode("utf-8")).hexdigest()
                header = (
                    f"[context-summary method={draft.method.value} "
                    f"source_sha256={digest} source_items={len(omitted)}]\n"
                )
                content_limit = max(
                    1,
                    limit - prompt_cost - retained_cost - len(header) - 16,
                )
                summary_content = draft.content[: min(summary_limit, content_limit)]
                while summary_content:
                    summary_candidate = Message(Role.SYSTEM, header + summary_content)
                    excess = (
                        prompt_cost
                        + retained_cost
                        + _item_size(summary_candidate)
                        - limit
                    )
                    if excess <= 0:
                        summary_message = summary_candidate
                        break
                    summary_content = summary_content[: max(0, len(summary_content) - excess)]
                if summary_message is not None:
                    provenance = SummaryProvenance(
                        method=draft.method,
                        source_digest=digest,
                        source_item_count=len(omitted),
                        source_characters=len(_serialize_items(omitted)),
                        summary_characters=len(summary_content),
                        structural_input_characters=len(structural_source),
                        model=draft.model,
                        response_id=draft.response_id,
                    )

        active: tuple[ConversationItem, ...] = (
            *prompt.messages,
            *((summary_message,) if summary_message is not None else ()),
            *(item for turn in retained for item in turn),
        )
        estimated = _items_size(active)
        if estimated > limit:
            raise ContextOverflowError("assembled context exceeds the active context budget")
        return ContextWindow(
            items=active,
            estimated_characters=estimated,
            max_characters=limit,
            retained_turns=len(retained),
            omitted_turns=len(older),
            knowledge_keys=tuple(document.key for document in prompt.knowledge),
            summary=provenance,
        )


def _validate_complete_transcript(items: Sequence[ConversationItem]) -> None:
    turns = _split_turns(items) if items else ()
    for turn in turns:
        outstanding: dict[str, str] = {}
        for item in turn:
            if isinstance(item, ToolCall):
                if item.call_id in outstanding:
                    raise ContextError(f"duplicate tool call ID in transcript: {item.call_id!r}")
                outstanding[item.call_id] = item.name
            elif isinstance(item, ToolResult):
                expected_name = outstanding.pop(item.call_id, None)
                if expected_name is None or expected_name != item.name:
                    raise ContextError(
                        f"tool result {item.call_id!r} has no matching tool call"
                    )
        if outstanding:
            raise ContextError(
                f"transcript has tool calls without results: {sorted(outstanding)!r}"
            )
        if not isinstance(turn[-1], Message) or turn[-1].role is not Role.ASSISTANT:
            raise ContextError("each durable transcript turn must end with an assistant message")


def _split_turns(
    items: Sequence[ConversationItem],
) -> tuple[tuple[ConversationItem, ...], ...]:
    turns: list[tuple[ConversationItem, ...]] = []
    current: list[ConversationItem] = []
    for item in items:
        if isinstance(item, Message) and item.role is Role.SYSTEM:
            raise ContextError("durable transcript must not contain system messages")
        if isinstance(item, Message) and item.role is Role.USER:
            if current:
                turns.append(tuple(current))
            current = [item]
        elif not current:
            raise ContextError("durable transcript must start with a user message")
        else:
            current.append(item)
    if current:
        turns.append(tuple(current))
    return tuple(turns)


def _item_size(item: ConversationItem) -> int:
    return len(json.dumps(_item_shape(item), ensure_ascii=False, separators=(",", ":")))


def _items_size(items: Sequence[ConversationItem]) -> int:
    return sum(_item_size(item) for item in items)


def _item_shape(item: ConversationItem) -> Mapping[str, object]:
    if isinstance(item, Message):
        return {"role": item.role.value, "content": item.content}
    if isinstance(item, ToolCall):
        return {
            "type": "tool_call",
            "call_id": item.call_id,
            "name": item.name,
            "arguments": item.arguments,
        }
    if isinstance(item, ToolResult):
        return {
            "type": "tool_result",
            "call_id": item.call_id,
            "name": item.name,
            "output": item.output,
            "outcome": item.outcome.value,
        }
    raise TypeError(f"unsupported conversation item: {type(item).__name__}")


def _serialize_items(items: Sequence[ConversationItem]) -> str:
    return json.dumps(
        [_item_shape(item) for item in items],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _structural_compact(
    turns: Sequence[Sequence[ConversationItem]], max_characters: int
) -> str:
    lines: list[str] = []
    for index, turn in enumerate(turns, start=1):
        parts: list[str] = []
        for item in turn:
            if isinstance(item, Message):
                parts.append(f"{item.role.value}={item.content}")
            elif isinstance(item, ToolCall):
                parts.append(f"tool_call={item.name}({item.arguments})")
            elif isinstance(item, ToolResult):
                parts.append(f"tool_result={item.name}:{item.outcome.value}:{item.output}")
        line = f"turn {index}: " + " | ".join(parts)
        remaining = max_characters - sum(len(value) + 1 for value in lines)
        if remaining <= 0:
            break
        lines.append(line[:remaining])
    return "\n".join(lines)
