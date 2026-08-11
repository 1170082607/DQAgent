"""Component tests for policy-filtered request-time memory recall."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import pytest

from dqagent.errors import MemoryDependencyError
from dqagent.memory import (
    MemoryCandidate,
    MemoryConfidence,
    MemoryKind,
    MemoryLifecycleStatus,
    MemoryProvenance,
    MemoryScope,
    MemoryScopeKind,
    MemorySensitivity,
    MemorySourceType,
)
from dqagent.memory_policy import DefaultMemoryPolicy
from dqagent.memory_recall import (
    MemoryMatch,
    MemoryMatchReason,
    MemoryRecall,
    MemoryRecallRequest,
    MemorySelector,
)
from dqagent.memory_service import MemoryService
from dqagent.memory_store import InMemoryMemoryStore

NOW = datetime(2026, 8, 11, 8, 0, tzinfo=UTC)
USER_SCOPE = MemoryScope(MemoryScopeKind.USER, "user-7")
OTHER_SCOPE = MemoryScope(MemoryScopeKind.USER, "user-8")
SOURCE_DIGEST = hashlib.sha256(b"recall test source").hexdigest()


@dataclass
class ManualClock:
    current: datetime

    def __call__(self) -> datetime:
        return self.current


class RecordingEmbeddingProvider:
    identity = "recording-memory-v1"

    def __init__(self, vectors: dict[str, tuple[float, ...]] | None = None) -> None:
        self.vectors = vectors or {}
        self.documents: tuple[str, ...] = ()
        self.queries: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.documents = tuple(texts)
        return tuple(self.vectors.get(text, (1.0, 0.0)) for text in self.documents)

    def embed_query(self, text: str) -> tuple[float, ...]:
        self.queries.append(text)
        return (1.0, 0.0)


def make_candidate(
    *,
    scope: MemoryScope = USER_SCOPE,
    kind: MemoryKind = MemoryKind.PREFERENCE,
    topic: str = "recall.topic",
    content: str = "eligible-content",
    expires_at: datetime | None = NOW + timedelta(days=30),
    extracted_at: datetime = NOW - timedelta(minutes=2),
    valid_from: datetime = NOW - timedelta(days=1),
    sensitivity: MemorySensitivity = MemorySensitivity.NON_SENSITIVE,
) -> MemoryCandidate:
    return MemoryCandidate(
        scope=scope,
        kind=kind,
        topic=topic,
        content=content,
        confidence=MemoryConfidence(0.9),
        sensitivity=sensitivity,
        provenance=MemoryProvenance(
            source_type=MemorySourceType.USER_DRAFT,
            source_item_digest=SOURCE_DIGEST,
            extractor_identity="recall-component-test",
            extracted_at=extracted_at,
            source_id="draft-1",
        ),
        valid_from=valid_from,
        expires_at=expires_at,
    )


def make_service(
    store: InMemoryMemoryStore,
    clock: ManualClock,
    provider: RecordingEmbeddingProvider,
    ids: tuple[str, ...],
) -> MemoryService:
    generated = iter(ids)
    return MemoryService(
        store,
        DefaultMemoryPolicy(),
        clock=clock,
        id_factory=lambda: next(generated),
        selector=MemorySelector(provider),
    )


def test_filter_before_rank_is_scope_isolated_and_never_embeds_ineligible_records() -> None:
    store = InMemoryMemoryStore()
    clock = ManualClock(NOW)
    provider = RecordingEmbeddingProvider()
    service = make_service(
        store,
        clock,
        provider,
        ("allowed-id", "expired-id", "old-id", "replacement-id"),
    )

    allowed = make_candidate(topic="allowed.topic", content="allowed-content")
    service.confirm(allowed, allowed.digest, scope=USER_SCOPE)
    expiring = make_candidate(
        topic="expired.topic",
        content="expired-content",
        expires_at=NOW + timedelta(minutes=1),
    )
    service.confirm(expiring, expiring.digest, scope=USER_SCOPE)
    old = make_candidate(topic="superseded.topic", content="superseded-content")
    old_result = service.confirm(old, old.digest, scope=USER_SCOPE)
    clock.current = NOW + timedelta(seconds=30)
    replacement = make_candidate(
        topic="superseded.topic",
        content="replacement-content",
        extracted_at=NOW + timedelta(seconds=1),
    )
    service.correct(
        USER_SCOPE,
        old_result.record.memory_id,
        replacement,
        candidate_digest=replacement.digest,
    )

    other_service = make_service(store, clock, provider, ("other-scope-id",))
    other = make_candidate(scope=OTHER_SCOPE, topic="other.topic", content="other-scope-content")
    other_service.confirm(other, other.digest, scope=OTHER_SCOPE)
    clock.current = NOW + timedelta(minutes=2)

    request = MemoryRecallRequest(
        USER_SCOPE,
        "which content is eligible?",
        allowed_kinds=frozenset({MemoryKind.PREFERENCE}),
    )
    result = service.recall(request)

    assert isinstance(result, MemoryRecall)
    assert result.candidate_count == 2
    assert provider.documents == ("allowed-content", "replacement-content")
    assert provider.queries == [request.query]
    assert "other-scope-content" not in provider.documents
    assert "expired-content" not in provider.documents
    assert "superseded-content" not in provider.documents


def test_service_rejects_selector_result_that_reintroduces_superseded_record() -> None:
    store = InMemoryMemoryStore()
    clock = ManualClock(NOW)
    provider = RecordingEmbeddingProvider()
    candidate = make_candidate(content="active-content")
    base_service = make_service(store, clock, provider, ("memory-id",))
    confirmed = base_service.confirm(candidate, candidate.digest, scope=USER_SCOPE).record

    class InvalidSelector:
        def select(
            self,
            records: Sequence[object],
            request: MemoryRecallRequest,
        ) -> MemoryRecall:
            assert len(records) == 1
            superseded = replace(confirmed, status=MemoryLifecycleStatus.SUPERSEDED)
            return MemoryRecall(
                request=request,
                matches=(MemoryMatch(superseded, score=1.0),),
                omitted=(),
                candidate_count=1,
                selector_identity="invalid-selector",
            )

    service = MemoryService(
        store,
        DefaultMemoryPolicy(),
        clock=clock,
        selector=InvalidSelector(),
    )

    with pytest.raises(MemoryDependencyError) as error:
        service.recall(MemoryRecallRequest(USER_SCOPE, "query"))

    assert error.value.reason == "invalid_selector_result"


def test_kind_allowlist_is_applied_before_ranking_and_provider_input() -> None:
    store = InMemoryMemoryStore()
    clock = ManualClock(NOW)
    provider = RecordingEmbeddingProvider()
    service = make_service(store, clock, provider, ("preference-id", "fact-id"))
    preference = make_candidate(topic="preference.topic", content="preference-content")
    fact = make_candidate(
        kind=MemoryKind.USER_FACT,
        topic="fact.topic",
        content="fact-content",
    )
    service.confirm(preference, preference.digest, scope=USER_SCOPE)
    service.confirm(fact, fact.digest, scope=USER_SCOPE)

    result = service.recall(
        MemoryRecallRequest(
            USER_SCOPE,
            "fact-content",
            allowed_kinds=frozenset({MemoryKind.PREFERENCE}),
        )
    )

    assert result.candidate_count == 1
    assert result.records[0].content == "preference-content"
    assert provider.documents == ("preference-content",)
    assert "fact-content" not in provider.documents


def test_ranking_uses_scores_and_stable_memory_id_ties() -> None:
    store = InMemoryMemoryStore()
    clock = ManualClock(NOW)
    provider = RecordingEmbeddingProvider(
        {
            "tie-b-content": (1.0, 0.0),
            "tie-a-content": (1.0, 0.0),
            "max-content": (0.8, 0.0),
            "low-content": (0.2, 0.0),
        }
    )
    service = make_service(store, clock, provider, ("b-id", "a-id", "max-id", "low-id"))
    for topic, content in (
        ("tie-b.topic", "tie-b-content"),
        ("tie-a.topic", "tie-a-content"),
        ("max.topic", "max-content"),
        ("low.topic", "low-content"),
    ):
        candidate = make_candidate(topic=topic, content=content)
        service.confirm(candidate, candidate.digest, scope=USER_SCOPE)

    result = service.recall(
        MemoryRecallRequest(USER_SCOPE, "query", min_score=0.5, max_records=2)
    )

    assert [match.memory_id for match in result.matches] == ["a-id", "b-id"]
    assert [match.score for match in result.matches] == [1.0, 1.0]
    assert result.omitted[0].memory_id == "max-id"
    assert result.omitted[0].reason is MemoryMatchReason.MAX_RECORDS
    assert result.omitted[1].memory_id == "low-id"
    assert result.omitted[1].reason is MemoryMatchReason.BELOW_MIN_SCORE
    assert result.selector_identity == "memory-selector-v1:recording-memory-v1"


def test_post_rank_limits_are_atomic_and_harmful_over_retrieval_is_omitted() -> None:
    store = InMemoryMemoryStore()
    clock = ManualClock(NOW)
    provider = RecordingEmbeddingProvider()
    service = make_service(store, clock, provider, ("long-id", "short-id", "third-id"))
    for topic, content in (
        ("long.topic", "12345"),
        ("short.topic", "123"),
        ("third.topic", "third"),
    ):
        candidate = make_candidate(topic=topic, content=content)
        service.confirm(candidate, candidate.digest, scope=USER_SCOPE)

    result = service.recall(
        MemoryRecallRequest(
            USER_SCOPE,
            "query",
            max_records=2,
            max_characters=5,
        )
    )

    assert [match.memory_id for match in result.matches] == ["long-id"]
    assert result.records[0].content == "12345"
    assert result.omitted[0].memory_id == "short-id"
    assert result.omitted[0].reason is MemoryMatchReason.CHARACTER_BUDGET
    assert result.omitted[1].memory_id == "third-id"
    assert result.omitted[1].reason is MemoryMatchReason.CHARACTER_BUDGET


def test_empty_recall_is_an_explicit_successful_no_result() -> None:
    store = InMemoryMemoryStore()
    clock = ManualClock(NOW)
    provider = RecordingEmbeddingProvider()
    service = make_service(store, clock, provider, ())

    result = service.recall(MemoryRecallRequest(USER_SCOPE, "nothing"))

    assert isinstance(result, MemoryRecall)
    assert result.matches == ()
    assert result.omitted == ()
    assert result.candidate_count == 0
    assert result.is_empty
    assert provider.documents == ()
    assert provider.queries == []
