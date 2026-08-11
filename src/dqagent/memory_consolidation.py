"""Deterministic memory consolidation decisions independent of persistence."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from dqagent.errors import MemoryValidationError
from dqagent.memory import (
    MemoryCandidate,
    MemoryLifecycleStatus,
    MemoryRecord,
    MemoryScope,
)

__all__ = [
    "ConsolidationAction",
    "ConsolidationReason",
    "MemoryConsolidator",
    "MemoryConsolidationDecision",
    "consolidate_memory",
]


class ConsolidationAction(StrEnum):
    ADD = "add"
    REFRESH = "refresh"
    CONFLICT = "conflict"


class ConsolidationReason(StrEnum):
    NEW_PROPOSITION = "new_proposition"
    EXACT_DUPLICATE = "exact_duplicate"
    EXPIRED_PROPOSITION = "expired_proposition"
    TOPIC_CONFLICT = "topic_conflict"


@dataclass(frozen=True, slots=True)
class MemoryConsolidationDecision:
    """A store-neutral decision for one candidate against one exact-scope snapshot."""

    action: ConsolidationAction
    reason: ConsolidationReason
    existing: MemoryRecord | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, ConsolidationAction):
            raise MemoryValidationError(
                "memory consolidation action must be a ConsolidationAction"
            )
        if not isinstance(self.reason, ConsolidationReason):
            raise MemoryValidationError(
                "memory consolidation reason must be a ConsolidationReason"
            )
        if self.action is ConsolidationAction.CONFLICT:
            if self.reason is not ConsolidationReason.TOPIC_CONFLICT:
                raise MemoryValidationError(
                    "memory conflict consolidation requires a topic conflict reason"
                )
            if not isinstance(self.existing, MemoryRecord):
                raise MemoryValidationError(
                    "memory conflict consolidation requires an existing record"
                )
        elif self.action is ConsolidationAction.REFRESH:
            if self.reason is not ConsolidationReason.EXACT_DUPLICATE:
                raise MemoryValidationError(
                    "memory refresh consolidation requires an exact duplicate reason"
                )
            if not isinstance(self.existing, MemoryRecord):
                raise MemoryValidationError(
                    "memory refresh consolidation requires an existing record"
                )
        elif self.existing is not None and not isinstance(self.existing, MemoryRecord):
            raise MemoryValidationError("memory consolidation existing value must be a record")

    @property
    def is_conflict(self) -> bool:
        return self.action is ConsolidationAction.CONFLICT


class MemoryConsolidator:
    """Compares propositions without reading, writing, or knowing a store implementation."""

    def decide(
        self,
        scope: MemoryScope,
        records: Sequence[MemoryRecord],
        candidate: MemoryCandidate,
        *,
        now: datetime,
    ) -> MemoryConsolidationDecision:
        if not isinstance(scope, MemoryScope):
            raise MemoryValidationError("memory consolidation requires a MemoryScope")
        if not isinstance(records, Sequence):
            raise MemoryValidationError("memory consolidation requires a record sequence")
        if not isinstance(candidate, MemoryCandidate):
            raise MemoryValidationError("memory consolidation requires a candidate")
        if candidate.scope != scope:
            raise MemoryValidationError("memory consolidation candidate scope does not match")
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise MemoryValidationError("memory consolidation clock must be timezone-aware")

        matching_key = [
            record
            for record in records
            if not isinstance(record, MemoryRecord)
        ]
        if matching_key:
            raise MemoryValidationError("memory consolidation records must be MemoryRecord values")

        matching_key = [
            record
            for record in records
            if record.status is MemoryLifecycleStatus.ACTIVE
            and record.kind is candidate.kind
            and record.topic == candidate.topic
        ]
        if not matching_key:
            return MemoryConsolidationDecision(
                ConsolidationAction.ADD,
                ConsolidationReason.NEW_PROPOSITION,
            )

        existing = matching_key[0]
        if existing.expires_at is not None and existing.expires_at <= now:
            return MemoryConsolidationDecision(
                ConsolidationAction.ADD,
                ConsolidationReason.EXPIRED_PROPOSITION,
                existing,
            )
        if existing.content == candidate.content:
            return MemoryConsolidationDecision(
                ConsolidationAction.REFRESH,
                ConsolidationReason.EXACT_DUPLICATE,
                existing,
            )
        return MemoryConsolidationDecision(
            ConsolidationAction.CONFLICT,
            ConsolidationReason.TOPIC_CONFLICT,
            existing,
        )


def consolidate_memory(
    scope: MemoryScope,
    records: Sequence[MemoryRecord],
    candidate: MemoryCandidate,
    *,
    now: datetime,
) -> MemoryConsolidationDecision:
    """Convenience entry point for callers that do not need a consolidator instance."""

    return MemoryConsolidator().decide(scope, records, candidate, now=now)
