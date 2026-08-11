"""Request-time, provider-neutral selection for long-term memory records."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType

from dqagent.errors import MemoryValidationError
from dqagent.memory import MemoryKind, MemoryRecord, MemoryScope
from dqagent.retrieval import EmbeddingProvider

__all__ = [
    "MemoryMatch",
    "MemoryMatchReason",
    "MemoryRecall",
    "MemoryRecallRequest",
    "MemorySelector",
]

_DEFAULT_MAX_CHARACTERS = 8_000


class MemoryMatchReason(StrEnum):
    """The deterministic post-ranking disposition of one eligible record."""

    SELECTED = "selected"
    BELOW_MIN_SCORE = "below_min_score"
    MAX_RECORDS = "max_records"
    KIND_NOT_ALLOWED = "kind_not_allowed"
    CHARACTER_BUDGET = "character_budget"


@dataclass(frozen=True, slots=True, init=False)
class MemoryRecallRequest:
    """All request-time controls for one exact-scope memory recall."""

    scope: MemoryScope
    query: str
    min_score: float
    max_records: int
    allowed_kinds: frozenset[MemoryKind]
    max_characters: int

    def __init__(
        self,
        scope: MemoryScope,
        query: str,
        *,
        min_score: float = 0.05,
        max_records: int = 5,
        allowed_kinds: frozenset[MemoryKind] = frozenset(MemoryKind),
        max_characters: int = _DEFAULT_MAX_CHARACTERS,
        character_budget: int | None = None,
    ) -> None:
        if character_budget is not None:
            if max_characters != _DEFAULT_MAX_CHARACTERS and max_characters != character_budget:
                raise MemoryValidationError(
                    "memory recall character budget has conflicting values"
                )
            max_characters = character_budget
        if not isinstance(scope, MemoryScope):
            raise MemoryValidationError("memory recall request scope must be a MemoryScope")
        if not isinstance(query, str) or not query.strip():
            raise MemoryValidationError("memory recall query must not be empty")
        if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
            raise MemoryValidationError("memory recall minimum score must be a number")
        try:
            normalized_min_score = float(min_score)
        except OverflowError as error:
            raise MemoryValidationError("memory recall minimum score must be finite") from error
        if not math.isfinite(normalized_min_score):
            raise MemoryValidationError("memory recall minimum score must be finite")
        _validate_non_negative_integer("memory recall maximum records", max_records)
        _validate_non_negative_integer("memory recall character budget", max_characters)
        if not isinstance(allowed_kinds, frozenset) or any(
            not isinstance(kind, MemoryKind) for kind in allowed_kinds
        ):
            raise MemoryValidationError(
                "memory recall allowed kinds must be a frozenset of MemoryKind values"
            )
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "min_score", normalized_min_score)
        object.__setattr__(self, "max_records", max_records)
        object.__setattr__(self, "allowed_kinds", allowed_kinds)
        object.__setattr__(self, "max_characters", max_characters)

    @property
    def character_budget(self) -> int:
        """Alias using the policy vocabulary for the recall character limit."""

        return self.max_characters


@dataclass(frozen=True, slots=True)
class MemoryMatch:
    """One ranked record and its selected or omitted disposition."""

    record: MemoryRecord
    score: float
    rank: int = 1
    reason: MemoryMatchReason = MemoryMatchReason.SELECTED

    def __post_init__(self) -> None:
        if not isinstance(self.record, MemoryRecord):
            raise MemoryValidationError("memory match requires a MemoryRecord")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise MemoryValidationError("memory match score must be a number")
        if not math.isfinite(float(self.score)):
            raise MemoryValidationError("memory match score must be finite")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise MemoryValidationError("memory match rank must be positive")
        if not isinstance(self.reason, MemoryMatchReason):
            raise MemoryValidationError("memory match reason must be a MemoryMatchReason")
        object.__setattr__(self, "score", float(self.score))

    @property
    def selected(self) -> bool:
        return self.reason is MemoryMatchReason.SELECTED

    @property
    def omitted(self) -> bool:
        return not self.selected

    @property
    def memory_id(self) -> str:
        return self.record.memory_id

    @property
    def omission_reason(self) -> MemoryMatchReason | None:
        return None if self.selected else self.reason


@dataclass(frozen=True, slots=True)
class MemoryRecall:
    """The complete ranked projection for one memory recall request."""

    request: MemoryRecallRequest
    matches: tuple[MemoryMatch, ...]
    omitted: tuple[MemoryMatch, ...]
    candidate_count: int
    selector_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.request, MemoryRecallRequest):
            raise MemoryValidationError("memory recall requires a MemoryRecallRequest")
        if not isinstance(self.matches, tuple) or not all(
            isinstance(match, MemoryMatch) and match.selected for match in self.matches
        ):
            raise MemoryValidationError("memory recall matches must be selected MemoryMatch values")
        if not isinstance(self.omitted, tuple) or not all(
            isinstance(match, MemoryMatch) and match.omitted for match in self.omitted
        ):
            raise MemoryValidationError("memory recall omitted values must be omitted MemoryMatch")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count < 0
        ):
            raise MemoryValidationError("memory recall candidate count must be non-negative")
        if self.candidate_count != len(self.matches) + len(self.omitted):
            raise MemoryValidationError(
                "memory recall candidate count must cover selected and omitted matches"
            )
        if not isinstance(self.selector_identity, str) or not self.selector_identity.strip():
            raise MemoryValidationError("memory recall selector identity must not be empty")
        ids = [match.memory_id for match in (*self.matches, *self.omitted)]
        if len(ids) != len(set(ids)):
            raise MemoryValidationError("memory recall match IDs must be unique")
        all_matches = (*self.matches, *self.omitted)
        if any(match.record.scope != self.request.scope for match in all_matches):
            raise MemoryValidationError("memory recall matches must belong to the request scope")

    @property
    def scope(self) -> MemoryScope:
        return self.request.scope

    @property
    def query(self) -> str:
        return self.request.query

    @property
    def selected(self) -> tuple[MemoryMatch, ...]:
        return self.matches

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        return tuple(match.record for match in self.matches)

    @property
    def omitted_matches(self) -> tuple[MemoryMatch, ...]:
        return self.omitted

    @property
    def is_empty(self) -> bool:
        return not self.matches

    @property
    def scores(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                match.memory_id: match.score
                for match in (*self.matches, *self.omitted)
            }
        )

    @property
    def omission_reasons(self) -> Mapping[str, MemoryMatchReason]:
        return MappingProxyType(
            {
                match.memory_id: match.reason
                for match in self.omitted
            }
        )


class MemorySelector:
    """Ranks eligible records with request-time embeddings and no vector persistence."""

    def __init__(self, embeddings: EmbeddingProvider) -> None:
        if not callable(getattr(embeddings, "embed_documents", None)) or not callable(
            getattr(embeddings, "embed_query", None)
        ):
            raise MemoryValidationError(
                "memory selector embeddings must implement the EmbeddingProvider contract"
            )
        identity = getattr(embeddings, "identity", None)
        if not isinstance(identity, str) or not identity.strip():
            raise MemoryValidationError("memory selector embedding identity must not be empty")
        self._embeddings = embeddings

    @property
    def identity(self) -> str:
        return f"memory-selector-v1:{self._embeddings.identity}"

    def select(
        self,
        records: Sequence[MemoryRecord],
        request: MemoryRecallRequest,
    ) -> MemoryRecall:
        if not isinstance(request, MemoryRecallRequest):
            raise MemoryValidationError("memory selector requires a MemoryRecallRequest")
        candidates = tuple(records)
        _validate_records(candidates, request.scope)
        if not candidates:
            return MemoryRecall(request, (), (), 0, self.identity)

        texts = tuple(record.content for record in candidates)
        vectors = self._embeddings.embed_documents(texts)
        if len(vectors) != len(candidates):
            raise MemoryValidationError("memory selector embedding count does not match records")
        query_vector = self._embeddings.embed_query(request.query)
        _validate_vector(query_vector, "memory query embedding")
        scored = tuple(
            sorted(
                (
                    MemoryMatch(
                        record,
                        _dot(query_vector, _validate_vector(vector, "memory record embedding")),
                        rank=rank,
                    )
                    for rank, (record, vector) in enumerate(
                        zip(candidates, vectors, strict=True),
                        start=1,
                    )
                ),
                key=lambda match: (-match.score, match.memory_id),
            )
        )
        ranked = tuple(
            replace(match, rank=rank)
            for rank, match in enumerate(scored, start=1)
        )
        selected: list[MemoryMatch] = []
        omitted: list[MemoryMatch] = []
        used_characters = 0
        for match in ranked:
            reason: MemoryMatchReason | None = None
            if match.score < request.min_score:
                reason = MemoryMatchReason.BELOW_MIN_SCORE
            elif match.record.kind not in request.allowed_kinds:
                reason = MemoryMatchReason.KIND_NOT_ALLOWED
            elif len(selected) >= request.max_records:
                reason = MemoryMatchReason.MAX_RECORDS
            elif used_characters + len(match.record.content) > request.max_characters:
                reason = MemoryMatchReason.CHARACTER_BUDGET

            if reason is None:
                selected.append(match)
                used_characters += len(match.record.content)
            else:
                omitted.append(replace(match, reason=reason))

        return MemoryRecall(
            request=request,
            matches=tuple(selected),
            omitted=tuple(omitted),
            candidate_count=len(ranked),
            selector_identity=self.identity,
        )


def _validate_non_negative_integer(label: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MemoryValidationError(f"{label} must be a non-negative integer")


def _validate_records(records: tuple[MemoryRecord, ...], scope: MemoryScope) -> None:
    if any(not isinstance(record, MemoryRecord) for record in records):
        raise MemoryValidationError("memory selector records must be MemoryRecord values")
    if any(record.scope != scope for record in records):
        raise MemoryValidationError("memory selector records must belong to the request scope")
    ids = [record.memory_id for record in records]
    if len(ids) != len(set(ids)):
        raise MemoryValidationError("memory selector record IDs must be unique")


def _validate_vector(vector: Sequence[float], label: str) -> tuple[float, ...]:
    try:
        values = tuple(vector)
    except TypeError as error:
        raise MemoryValidationError(f"{label} must be a finite non-empty vector") from error
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise MemoryValidationError(f"{label} must be a finite non-empty vector")
    return tuple(float(value) for value in values)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise MemoryValidationError("memory embedding dimensions do not match")
    score = sum(a * b for a, b in zip(left, right, strict=True))
    if not math.isfinite(score):
        raise MemoryValidationError("memory match score must be finite")
    return score
