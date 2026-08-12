"""Model-free application service for explicit long-term memory management."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from secrets import compare_digest
from types import MappingProxyType
from uuid import uuid4

from dqagent.errors import (
    MemoryAdmissionDeniedError,
    MemoryConflictError,
    MemoryCorruptChangeError,
    MemoryDependencyError,
    MemoryDigestMismatchError,
    MemoryNotFoundError,
    MemoryServiceConflictError,
    MemoryServiceError,
    MemoryServiceNotFoundError,
    MemoryTargetStateError,
    MemoryTopicConflictError,
    MemoryValidationError,
)
from dqagent.memory import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionReason,
    MemoryCandidate,
    MemoryConfirmation,
    MemoryForgetReason,
    MemoryKind,
    MemoryLifecycleStatus,
    MemoryPolicy,
    MemoryRecord,
    MemoryScope,
    MemorySensitivity,
    MemoryTombstone,
    RecallEligibilityDecision,
    RecallEligibilityReason,
)
from dqagent.memory_consolidation import (
    ConsolidationAction,
    MemoryConsolidationDecision,
    MemoryConsolidator,
)
from dqagent.memory_policy import _content_admission_reason
from dqagent.memory_recall import (
    MemoryMatch,
    MemoryMatchReason,
    MemoryRecall,
    MemoryRecallRequest,
    MemorySelector,
)
from dqagent.memory_store import (
    AddMemory,
    ExpireMemory,
    ForgetMemory,
    MemoryChange,
    MemoryChangeSet,
    MemoryScopeSnapshot,
    MemoryStore,
    RefreshMemory,
    SupersedeMemory,
)
from dqagent.retrieval import HashingEmbeddingProvider

__all__ = [
    "MemoryConfirmResult",
    "MemoryConfirmationResult",
    "MemoryCorrectionResult",
    "MemoryEventMetadata",
    "MemoryForgetResult",
    "MemoryListResult",
    "MemoryOperation",
    "MemoryOperationMetadata",
    "MemoryOutcome",
    "MemoryPreview",
    "MemoryProposal",
    "MemoryMatch",
    "MemoryMatchReason",
    "MemoryRecall",
    "MemoryRecallRequest",
    "MemorySelector",
    "MemoryService",
    "MemoryShowResult",
    "MemoryWriteOutcome",
]


class MemoryOperation(StrEnum):
    PROPOSE = "propose"
    CONFIRM = "confirm"
    LIST = "list"
    RECALL = "recall"
    SHOW = "show"
    CORRECT = "correct"
    FORGET = "forget"


class MemoryOutcome(StrEnum):
    PROPOSED = "proposed"
    ADDED = "added"
    REFRESHED = "refreshed"
    LISTED = "listed"
    RECALLED = "recalled"
    SHOWN = "shown"
    CORRECTED = "corrected"
    FORGOTTEN = "forgotten"


class MemoryWriteOutcome(StrEnum):
    ADDED = "added"
    REFRESHED = "refreshed"


@dataclass(frozen=True, slots=True)
class MemoryEventMetadata:
    """Content-free fields suitable for an event sink or audit record."""

    operation: MemoryOperation
    outcome: MemoryOutcome
    scope: MemoryScope
    scope_revision: int | None = None
    memory_id: str | None = None
    candidate_digest: str | None = None
    reason: str | None = None
    record_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, MemoryOperation):
            raise MemoryValidationError("memory event operation must be a MemoryOperation")
        if not isinstance(self.outcome, MemoryOutcome):
            raise MemoryValidationError("memory event outcome must be a MemoryOutcome")
        if not isinstance(self.scope, MemoryScope):
            raise MemoryValidationError("memory event scope must be a MemoryScope")
        if self.scope_revision is not None and (
            isinstance(self.scope_revision, bool)
            or not isinstance(self.scope_revision, int)
            or self.scope_revision < 0
        ):
            raise MemoryValidationError("memory event scope revision must be non-negative")
        if self.record_count is not None and (
            isinstance(self.record_count, bool)
            or not isinstance(self.record_count, int)
            or self.record_count < 0
        ):
            raise MemoryValidationError("memory event record count must be non-negative")

    @property
    def event_attributes(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "operation": self.operation.value,
                "outcome": self.outcome.value,
                "scope_kind": self.scope.kind.value,
                "scope_id": self.scope.scope_id,
                "scope_revision": self.scope_revision,
                "memory_id": self.memory_id,
                "candidate_digest": self.candidate_digest,
                "reason": self.reason,
                "record_count": self.record_count,
            }
        )


MemoryOperationMetadata = MemoryEventMetadata


@dataclass(frozen=True, slots=True)
class MemoryProposal:
    """Transient candidate plus the policy decision shown before confirmation."""

    candidate: MemoryCandidate
    decision: AdmissionDecision
    metadata: MemoryEventMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, MemoryCandidate):
            raise MemoryValidationError("memory proposal requires a MemoryCandidate")
        if not isinstance(self.decision, AdmissionDecision):
            raise MemoryValidationError("memory proposal requires an AdmissionDecision")
        if self.metadata.operation is not MemoryOperation.PROPOSE:
            raise MemoryValidationError("memory proposal metadata has the wrong operation")
        if self.metadata.candidate_digest != self.candidate.digest:
            raise MemoryValidationError("memory proposal metadata digest does not match")

    @property
    def candidate_digest(self) -> str:
        return self.candidate.digest

    @property
    def digest(self) -> str:
        return self.candidate_digest

    @property
    def policy_decision(self) -> AdmissionDecision:
        return self.decision

    @property
    def event_attributes(self) -> Mapping[str, object]:
        return self.metadata.event_attributes


MemoryPreview = MemoryProposal


@dataclass(frozen=True, slots=True)
class MemoryConfirmResult:
    record: MemoryRecord
    outcome: MemoryWriteOutcome
    revision: int
    metadata: MemoryEventMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.record, MemoryRecord):
            raise MemoryValidationError("memory confirmation result requires a MemoryRecord")
        if not isinstance(self.outcome, MemoryWriteOutcome):
            raise MemoryValidationError("memory confirmation result has an invalid outcome")
        if self.metadata.operation is not MemoryOperation.CONFIRM:
            raise MemoryValidationError("memory confirmation metadata has the wrong operation")

    @property
    def event_attributes(self) -> Mapping[str, object]:
        return self.metadata.event_attributes


MemoryConfirmationResult = MemoryConfirmResult


@dataclass(frozen=True, slots=True)
class MemoryListResult:
    scope: MemoryScope
    records: tuple[MemoryRecord, ...]
    revision: int
    metadata: MemoryEventMetadata
    tombstones: tuple[MemoryTombstone, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, MemoryScope):
            raise MemoryValidationError("memory list result scope must be a MemoryScope")
        if not isinstance(self.records, tuple) or not all(
            isinstance(record, MemoryRecord) for record in self.records
        ):
            raise MemoryValidationError(
                "memory list result records must be a tuple of MemoryRecord"
            )
        if not isinstance(self.tombstones, tuple) or not all(
            isinstance(tombstone, MemoryTombstone) for tombstone in self.tombstones
        ):
            raise MemoryValidationError(
                "memory list result tombstones must be a tuple of MemoryTombstone"
            )
        if self.metadata.operation is not MemoryOperation.LIST:
            raise MemoryValidationError("memory list metadata has the wrong operation")

    @property
    def active_records(self) -> tuple[MemoryRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.status is MemoryLifecycleStatus.ACTIVE
        )

    @property
    def event_attributes(self) -> Mapping[str, object]:
        return self.metadata.event_attributes


@dataclass(frozen=True, slots=True)
class MemoryShowResult:
    record: MemoryRecord
    metadata: MemoryEventMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.record, MemoryRecord):
            raise MemoryValidationError("memory show result requires a MemoryRecord")
        if self.metadata.operation is not MemoryOperation.SHOW:
            raise MemoryValidationError("memory show metadata has the wrong operation")

    @property
    def event_attributes(self) -> Mapping[str, object]:
        return self.metadata.event_attributes


@dataclass(frozen=True, slots=True)
class MemoryCorrectionResult:
    superseded: MemoryRecord
    replacement: MemoryRecord
    revision: int
    metadata: MemoryEventMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.superseded, MemoryRecord) or not isinstance(
            self.replacement, MemoryRecord
        ):
            raise MemoryValidationError(
                "memory correction result requires superseded and replacement records"
            )
        if self.metadata.operation is not MemoryOperation.CORRECT:
            raise MemoryValidationError("memory correction metadata has the wrong operation")

    @property
    def record(self) -> MemoryRecord:
        return self.replacement

    @property
    def event_attributes(self) -> Mapping[str, object]:
        return self.metadata.event_attributes


@dataclass(frozen=True, slots=True)
class MemoryForgetResult:
    tombstone: MemoryTombstone
    revision: int
    metadata: MemoryEventMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.tombstone, MemoryTombstone):
            raise MemoryValidationError("memory forget result requires a MemoryTombstone")
        if self.metadata.operation is not MemoryOperation.FORGET:
            raise MemoryValidationError("memory forget metadata has the wrong operation")

    @property
    def memory_id(self) -> str:
        return self.tombstone.memory_id

    @property
    def event_attributes(self) -> Mapping[str, object]:
        return self.metadata.event_attributes


class MemoryService:
    """Coordinates explicit memory operations without model or storage policy leakage."""

    def __init__(
        self,
        store: MemoryStore,
        policy: MemoryPolicy,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        id_factory: Callable[[], str] = lambda: str(uuid4()),
        consolidator: MemoryConsolidator | None = None,
        selector: MemorySelector | None = None,
    ) -> None:
        if not callable(getattr(store, "load", None)) or not callable(
            getattr(store, "apply", None)
        ):
            raise MemoryValidationError(
                "memory service store must implement the MemoryStore contract"
            )
        if not callable(getattr(policy, "assess_write", None)) or not callable(
            getattr(policy, "eligible", None)
        ):
            raise MemoryValidationError(
                "memory service policy must implement the MemoryPolicy contract"
            )
        if not callable(clock):
            raise MemoryValidationError("memory service clock must be callable")
        if not callable(id_factory):
            raise MemoryValidationError("memory service ID factory must be callable")
        if consolidator is not None and not callable(getattr(consolidator, "decide", None)):
            raise MemoryValidationError("memory service consolidator must implement decide")
        if selector is not None and not callable(getattr(selector, "select", None)):
            raise MemoryValidationError("memory service selector must implement select")
        self._store = store
        self._policy = policy
        self._clock = clock
        self._id_factory = id_factory
        self._consolidator = consolidator or MemoryConsolidator()
        self._selector = selector or MemorySelector(HashingEmbeddingProvider())

    def propose(self, candidate: MemoryCandidate, *, scope: MemoryScope) -> MemoryProposal:
        """Return a transient preview; this method never calls the store."""

        _require_candidate(candidate)
        _require_scope(scope)
        now = self._now(MemoryOperation.PROPOSE, scope)
        decision = self._assess(candidate, scope=scope, now=now, operation=MemoryOperation.PROPOSE)
        return MemoryProposal(
            candidate=candidate,
            decision=decision,
            metadata=MemoryEventMetadata(
                operation=MemoryOperation.PROPOSE,
                outcome=MemoryOutcome.PROPOSED,
                scope=scope,
                candidate_digest=candidate.digest,
                reason=decision.reason.value,
            ),
        )

    def preview(self, candidate: MemoryCandidate, *, scope: MemoryScope) -> MemoryProposal:
        """Alias for :meth:`propose` using the user-facing preview vocabulary."""

        return self.propose(candidate, scope=scope)

    def confirm(
        self,
        candidate: MemoryCandidate | MemoryProposal,
        candidate_digest: str | None = None,
        *,
        scope: MemoryScope,
        digest: str | None = None,
    ) -> MemoryConfirmResult:
        """Revalidate an exact candidate and commit an add or deterministic refresh."""

        resolved_candidate = _candidate_from_value(candidate)
        _require_scope(scope)
        resolved_digest = _resolve_digest_argument(candidate_digest, digest)
        expected_digest = _require_exact_digest(
            resolved_candidate,
            resolved_digest,
            scope,
            operation=MemoryOperation.CONFIRM,
        )
        now = self._now(MemoryOperation.CONFIRM, scope, candidate_digest=expected_digest)
        decision = self._assess(
            resolved_candidate,
            scope=scope,
            now=now,
            operation=MemoryOperation.CONFIRM,
            candidate_digest=expected_digest,
        )
        _require_admitted(decision, MemoryOperation.CONFIRM, scope, expected_digest)

        snapshot = self._load(MemoryOperation.CONFIRM, scope, candidate_digest=expected_digest)
        consolidation = self._consolidate(snapshot, resolved_candidate, now=now)
        if consolidation.action is ConsolidationAction.CONFLICT:
            existing = _required_existing(consolidation)
            raise MemoryTopicConflictError(
                "memory candidate conflicts with an active topic",
                operation=MemoryOperation.CONFIRM.value,
                scope_kind=scope.kind.value,
                scope_id=scope.scope_id,
                memory_id=existing.memory_id,
                candidate_digest=expected_digest,
                reason=consolidation.reason.value,
            )

        changes: list[MemoryChange] = []
        existing_for_expiry = consolidation.existing
        if (
            consolidation.reason.value == "expired_proposition"
            and existing_for_expiry is not None
        ):
            changes.append(ExpireMemory(self._expired_record(existing_for_expiry, now=now)))

        if consolidation.action is ConsolidationAction.REFRESH:
            current = _required_existing(consolidation)
            refreshed = self._refreshed_record(current, resolved_candidate, now=now)
            changes.append(RefreshMemory(refreshed))
            outcome = MemoryWriteOutcome.REFRESHED
            result_record = refreshed
        else:
            added = self._new_record(resolved_candidate, now=now, scope=scope)
            changes.append(AddMemory(added))
            outcome = MemoryWriteOutcome.ADDED
            result_record = added

        saved = self._apply(
            MemoryOperation.CONFIRM,
            MemoryChangeSet(scope, tuple(changes)),
            expected_revision=snapshot.revision,
            memory_id=result_record.memory_id,
            candidate_digest=expected_digest,
        )
        return MemoryConfirmResult(
            record=result_record,
            outcome=outcome,
            revision=saved.revision,
            metadata=MemoryEventMetadata(
                operation=MemoryOperation.CONFIRM,
                outcome=(
                    MemoryOutcome.REFRESHED
                    if outcome is MemoryWriteOutcome.REFRESHED
                    else MemoryOutcome.ADDED
                ),
                scope=scope,
                scope_revision=saved.revision,
                memory_id=result_record.memory_id,
                candidate_digest=expected_digest,
                reason=consolidation.reason.value,
            ),
        )

    def list(self, scope: MemoryScope) -> MemoryListResult:
        """List the exact scope, preserving lifecycle records for inspection."""

        _require_scope(scope)
        now = self._now(MemoryOperation.LIST, scope)
        snapshot = self._load(MemoryOperation.LIST, scope)
        snapshot = self._materialize_expiry(snapshot, now=now, operation=MemoryOperation.LIST)
        return MemoryListResult(
            scope=scope,
            records=snapshot.records,
            revision=snapshot.revision,
            tombstones=snapshot.tombstones,
            metadata=MemoryEventMetadata(
                operation=MemoryOperation.LIST,
                outcome=MemoryOutcome.LISTED,
                scope=scope,
                scope_revision=snapshot.revision,
                record_count=len(snapshot.records),
            ),
        )

    def recall(self, request: MemoryRecallRequest) -> MemoryRecall:
        """Recall policy-eligible records from one exact scope at request time."""

        if not isinstance(request, MemoryRecallRequest):
            raise MemoryValidationError("memory recall requires a MemoryRecallRequest")
        scope = request.scope
        now = self._now(MemoryOperation.RECALL, scope)
        snapshot = self._load(MemoryOperation.RECALL, scope)
        eligible = self._eligible_for_recall(snapshot, request=request, now=now)
        try:
            result = self._selector.select(eligible, request)
        except MemoryServiceError:
            raise
        except Exception:
            raise self._dependency_error(
                MemoryOperation.RECALL,
                scope,
                reason="selector_failure",
            ) from None
        if (
            not isinstance(result, MemoryRecall)
            or result.request != request
            or result.candidate_count != len(eligible)
            or not _service_recall_result_is_valid(
                result,
                eligible,
                request=request,
            )
        ):
            raise self._dependency_error(
                MemoryOperation.RECALL,
                scope,
                reason="invalid_selector_result",
            )
        return result

    def show(
        self,
        scope: MemoryScope,
        memory_id: str,
        *,
        materialize_expiry: bool = True,
    ) -> MemoryShowResult:
        """Show one exact record, optionally without writing expiry materialization."""

        _require_scope(scope)
        _require_memory_id(memory_id)
        now = self._now(MemoryOperation.SHOW, scope, memory_id=memory_id)
        snapshot = self._load(MemoryOperation.SHOW, scope, memory_id=memory_id)
        if materialize_expiry:
            snapshot = self._materialize_expiry(snapshot, now=now, operation=MemoryOperation.SHOW)
        record = next((item for item in snapshot.records if item.memory_id == memory_id), None)
        if record is None:
            raise MemoryServiceNotFoundError(
                "memory record was not found",
                operation=MemoryOperation.SHOW.value,
                scope_kind=scope.kind.value,
                scope_id=scope.scope_id,
                memory_id=memory_id,
                reason="not_found",
            )
        return MemoryShowResult(
            record=record,
            metadata=MemoryEventMetadata(
                operation=MemoryOperation.SHOW,
                outcome=MemoryOutcome.SHOWN,
                scope=scope,
                scope_revision=snapshot.revision,
                memory_id=memory_id,
            ),
        )

    def correct(
        self,
        scope: MemoryScope,
        memory_id: str,
        candidate: MemoryCandidate | MemoryProposal,
        *,
        candidate_digest: str | None = None,
        digest: str | None = None,
    ) -> MemoryCorrectionResult:
        """Atomically supersede the explicit target and add its corrected proposition."""

        _require_scope(scope)
        _require_memory_id(memory_id)
        resolved_candidate = _candidate_from_value(candidate)
        resolved_digest = _resolve_digest_argument(candidate_digest, digest)
        expected_digest = _require_exact_digest(
            resolved_candidate,
            resolved_digest,
            scope,
            operation=MemoryOperation.CORRECT,
        )
        now = self._now(MemoryOperation.CORRECT, scope, memory_id=memory_id)
        decision = self._assess(
            resolved_candidate,
            scope=scope,
            now=now,
            operation=MemoryOperation.CORRECT,
            memory_id=memory_id,
            candidate_digest=expected_digest,
        )
        _require_admitted(
            decision,
            MemoryOperation.CORRECT,
            scope,
            expected_digest,
            memory_id=memory_id,
        )

        snapshot = self._load(
            MemoryOperation.CORRECT,
            scope,
            memory_id=memory_id,
            candidate_digest=expected_digest,
        )
        target = next((item for item in snapshot.records if item.memory_id == memory_id), None)
        if target is None:
            raise MemoryServiceNotFoundError(
                "memory record was not found",
                operation=MemoryOperation.CORRECT.value,
                scope_kind=scope.kind.value,
                scope_id=scope.scope_id,
                memory_id=memory_id,
                candidate_digest=expected_digest,
                reason="not_found",
            )
        if target.status is not MemoryLifecycleStatus.ACTIVE:
            raise MemoryTargetStateError(
                "memory record is not active",
                operation=MemoryOperation.CORRECT.value,
                scope_kind=scope.kind.value,
                scope_id=scope.scope_id,
                memory_id=memory_id,
                candidate_digest=expected_digest,
                reason="target_not_active",
            )
        if target.expires_at is not None and target.expires_at <= now:
            raise MemoryTargetStateError(
                "memory record has expired",
                operation=MemoryOperation.CORRECT.value,
                scope_kind=scope.kind.value,
                scope_id=scope.scope_id,
                memory_id=memory_id,
                candidate_digest=expected_digest,
                reason="target_expired",
            )
        if (target.kind, target.topic) != (
            resolved_candidate.kind,
            resolved_candidate.topic,
        ):
            raise MemoryTargetStateError(
                "correction candidate does not match the target proposition",
                operation=MemoryOperation.CORRECT.value,
                scope_kind=scope.kind.value,
                scope_id=scope.scope_id,
                memory_id=memory_id,
                candidate_digest=expected_digest,
                reason="target_proposition_mismatch",
            )
        if now <= target.updated_at:
            raise MemoryTargetStateError(
                "memory service clock must follow the target update",
                operation=MemoryOperation.CORRECT.value,
                scope_kind=scope.kind.value,
                scope_id=scope.scope_id,
                memory_id=memory_id,
                candidate_digest=expected_digest,
                reason="clock_not_after_target_update",
            )

        replacement_id = self._new_memory_id(
            scope=scope,
            operation=MemoryOperation.CORRECT,
            memory_id=memory_id,
            candidate_digest=expected_digest,
        )
        if replacement_id in {record.memory_id for record in snapshot.records} or (
            replacement_id in {tombstone.memory_id for tombstone in snapshot.tombstones}
        ):
            raise MemoryServiceConflictError(
                "replacement memory ID already exists",
                operation=MemoryOperation.CORRECT.value,
                scope_kind=scope.kind.value,
                scope_id=scope.scope_id,
                memory_id=memory_id,
                candidate_digest=expected_digest,
                reason="replacement_id_exists",
            )

        superseded = replace(
            target,
            revision=target.revision + 1,
            status=MemoryLifecycleStatus.SUPERSEDED,
            updated_at=now,
        )
        replacement = MemoryRecord.from_candidate(
            resolved_candidate,
            memory_id=replacement_id,
            revision=1,
            confirmation=MemoryConfirmation(expected_digest, now),
            created_at=now,
            updated_at=now,
            supersedes_id=target.memory_id,
        )
        saved = self._apply(
            MemoryOperation.CORRECT,
            MemoryChangeSet(scope, (SupersedeMemory(superseded, replacement),)),
            expected_revision=snapshot.revision,
            memory_id=memory_id,
            candidate_digest=expected_digest,
        )
        return MemoryCorrectionResult(
            superseded=superseded,
            replacement=replacement,
            revision=saved.revision,
            metadata=MemoryEventMetadata(
                operation=MemoryOperation.CORRECT,
                outcome=MemoryOutcome.CORRECTED,
                scope=scope,
                scope_revision=saved.revision,
                memory_id=memory_id,
                candidate_digest=expected_digest,
                reason="superseded_and_replaced",
            ),
        )

    def forget(
        self,
        scope: MemoryScope,
        memory_id: str,
        *,
        reason: MemoryForgetReason = MemoryForgetReason.USER_REQUEST,
    ) -> MemoryForgetResult:
        """Forget one exact-scope record and leave only its content-free tombstone."""

        _require_scope(scope)
        _require_memory_id(memory_id)
        if not isinstance(reason, MemoryForgetReason):
            raise MemoryValidationError("memory forget reason must be a MemoryForgetReason")
        now = self._now(MemoryOperation.FORGET, scope, memory_id=memory_id)
        snapshot = self._load(MemoryOperation.FORGET, scope, memory_id=memory_id)
        record = next((item for item in snapshot.records if item.memory_id == memory_id), None)
        if record is None:
            raise MemoryServiceNotFoundError(
                "memory record was not found",
                operation=MemoryOperation.FORGET.value,
                scope_kind=scope.kind.value,
                scope_id=scope.scope_id,
                memory_id=memory_id,
                reason="not_found",
            )
        tombstone = MemoryTombstone.from_record(
            record,
            forgotten_at=now,
            reason=reason,
        )
        saved = self._apply(
            MemoryOperation.FORGET,
            MemoryChangeSet(scope, (ForgetMemory(tombstone),)),
            expected_revision=snapshot.revision,
            memory_id=memory_id,
        )
        return MemoryForgetResult(
            tombstone=tombstone,
            revision=saved.revision,
            metadata=MemoryEventMetadata(
                operation=MemoryOperation.FORGET,
                outcome=MemoryOutcome.FORGOTTEN,
                scope=scope,
                scope_revision=saved.revision,
                memory_id=memory_id,
                reason=reason.value,
            ),
        )

    def _assess(
        self,
        candidate: MemoryCandidate,
        *,
        scope: MemoryScope,
        now: datetime,
        operation: MemoryOperation,
        memory_id: str | None = None,
        candidate_digest: str | None = None,
    ) -> AdmissionDecision:
        if candidate.sensitivity is MemorySensitivity.SECRET:
            return AdmissionDecision(
                action=AdmissionAction.DENY,
                reason=AdmissionReason.SECRET_CONTENT_NOT_ALLOWED,
                effective_scope=None,
                expires_at=None,
            )
        if candidate.sensitivity is MemorySensitivity.SENSITIVE:
            return AdmissionDecision(
                action=AdmissionAction.DENY,
                reason=AdmissionReason.SENSITIVE_CONTENT_NOT_ALLOWED,
                effective_scope=None,
                expires_at=None,
            )
        content_reason = _content_admission_reason(candidate.content)
        if content_reason is not None:
            return AdmissionDecision(
                action=AdmissionAction.DENY,
                reason=content_reason,
                effective_scope=None,
                expires_at=None,
            )
        try:
            decision = self._policy.assess_write(candidate, scope=scope, now=now)
        except MemoryServiceError:
            raise
        except Exception:
            raise self._dependency_error(
                operation,
                scope,
                memory_id=memory_id,
                candidate_digest=candidate_digest,
                reason="policy_failure",
            ) from None
        if not isinstance(decision, AdmissionDecision):
            raise self._dependency_error(
                operation,
                scope,
                memory_id=memory_id,
                candidate_digest=candidate_digest,
                reason="invalid_policy_decision",
            )

        # These invariants remain service-owned even if a custom policy is too permissive.
        if candidate.scope != scope:
            return AdmissionDecision(
                AdmissionAction.DENY,
                AdmissionReason.SCOPE_MISMATCH,
                None,
                None,
            )
        if candidate.provenance.extracted_at > now:
            return AdmissionDecision(
                AdmissionAction.DENY,
                AdmissionReason.PROVENANCE_IN_FUTURE,
                None,
                None,
            )
        if candidate.expires_at is not None and candidate.expires_at <= now:
            return AdmissionDecision(
                AdmissionAction.DENY,
                AdmissionReason.CANDIDATE_EXPIRED,
                None,
                None,
            )
        if decision.action is not AdmissionAction.DENY and (
            decision.effective_scope != scope or decision.expires_at != candidate.expires_at
        ):
            raise self._dependency_error(
                operation,
                scope,
                memory_id=memory_id,
                candidate_digest=candidate_digest,
                reason="invalid_policy_scope_or_expiry",
            )
        return decision

    def _eligible_for_recall(
        self,
        snapshot: MemoryScopeSnapshot,
        *,
        request: MemoryRecallRequest,
        now: datetime,
    ) -> tuple[MemoryRecord, ...]:
        eligible: list[MemoryRecord] = []
        for record in snapshot.records:
            try:
                decision = self._policy.eligible(
                    record,
                    scope=request.scope,
                    allowed_kinds=request.allowed_kinds,
                    now=now,
                )
            except MemoryServiceError:
                raise
            except Exception:
                raise self._dependency_error(
                    MemoryOperation.RECALL,
                    request.scope,
                    reason="policy_failure",
                ) from None
            if not isinstance(decision, RecallEligibilityDecision):
                raise self._dependency_error(
                    MemoryOperation.RECALL,
                    request.scope,
                    reason="invalid_policy_decision",
                )
            if not decision.eligible or _service_recall_ineligibility(
                record,
                scope=request.scope,
                allowed_kinds=request.allowed_kinds,
                now=now,
            ) is not None:
                continue
            eligible.append(record)
        return tuple(sorted(eligible, key=lambda record: record.memory_id))

    def _consolidate(
        self,
        snapshot: MemoryScopeSnapshot,
        candidate: MemoryCandidate,
        *,
        now: datetime,
    ) -> MemoryConsolidationDecision:
        try:
            decision = self._consolidator.decide(
                snapshot.scope,
                snapshot.records,
                candidate,
                now=now,
            )
        except MemoryServiceError:
            raise
        except Exception:
            raise self._dependency_error(
                MemoryOperation.CONFIRM,
                snapshot.scope,
                candidate_digest=candidate.digest,
                reason="consolidation_failure",
            ) from None
        if not isinstance(decision, MemoryConsolidationDecision):
            raise self._dependency_error(
                MemoryOperation.CONFIRM,
                snapshot.scope,
                candidate_digest=candidate.digest,
                reason="invalid_consolidation_decision",
            )
        return decision

    def _load(
        self,
        operation: MemoryOperation,
        scope: MemoryScope,
        *,
        memory_id: str | None = None,
        candidate_digest: str | None = None,
    ) -> MemoryScopeSnapshot:
        try:
            snapshot = self._store.load(scope)
        except MemoryServiceError:
            raise
        except MemoryNotFoundError:
            raise self._not_found_error(
                operation,
                scope,
                memory_id=memory_id,
                candidate_digest=candidate_digest,
            ) from None
        except Exception:
            raise self._dependency_error(
                operation,
                scope,
                memory_id=memory_id,
                candidate_digest=candidate_digest,
                reason="store_load_failure",
            ) from None
        if not isinstance(snapshot, MemoryScopeSnapshot) or snapshot.scope != scope:
            raise self._dependency_error(
                operation,
                scope,
                memory_id=memory_id,
                candidate_digest=candidate_digest,
                reason="invalid_store_snapshot",
            )
        return snapshot

    def _apply(
        self,
        operation: MemoryOperation,
        change_set: MemoryChangeSet,
        *,
        expected_revision: int,
        memory_id: str | None = None,
        candidate_digest: str | None = None,
    ) -> MemoryScopeSnapshot:
        try:
            saved = self._store.apply(change_set, expected_revision=expected_revision)
        except MemoryServiceError:
            raise
        except MemoryConflictError:
            raise MemoryServiceConflictError(
                "memory scope changed concurrently",
                operation=operation.value,
                scope_kind=change_set.scope.kind.value,
                scope_id=change_set.scope.scope_id,
                memory_id=memory_id,
                candidate_digest=candidate_digest,
                reason="concurrent_conflict",
            ) from None
        except MemoryNotFoundError:
            raise self._not_found_error(
                operation,
                change_set.scope,
                memory_id=memory_id,
                candidate_digest=candidate_digest,
            ) from None
        except (MemoryCorruptChangeError, MemoryValidationError):
            raise
        except Exception:
            raise self._dependency_error(
                operation,
                change_set.scope,
                memory_id=memory_id,
                candidate_digest=candidate_digest,
                reason="store_apply_failure",
            ) from None
        if not isinstance(saved, MemoryScopeSnapshot) or saved.scope != change_set.scope:
            raise self._dependency_error(
                operation,
                change_set.scope,
                memory_id=memory_id,
                candidate_digest=candidate_digest,
                reason="invalid_store_result",
            )
        return saved

    def _materialize_expiry(
        self,
        snapshot: MemoryScopeSnapshot,
        *,
        now: datetime,
        operation: MemoryOperation,
    ) -> MemoryScopeSnapshot:
        due = tuple(
            ExpireMemory(self._expired_record(record, now=now))
            for record in snapshot.records
            if record.status is MemoryLifecycleStatus.ACTIVE
            and record.expires_at is not None
            and record.expires_at <= now
        )
        if not due:
            return snapshot
        return self._apply(
            operation,
            MemoryChangeSet(snapshot.scope, tuple(due)),
            expected_revision=snapshot.revision,
        )

    def _expired_record(self, record: MemoryRecord, *, now: datetime) -> MemoryRecord:
        if now <= record.updated_at:
            raise MemoryValidationError("memory expiry clock must follow the record update")
        return replace(
            record,
            revision=record.revision + 1,
            status=MemoryLifecycleStatus.EXPIRED,
            updated_at=now,
        )

    def _refreshed_record(
        self,
        current: MemoryRecord,
        candidate: MemoryCandidate,
        *,
        now: datetime,
    ) -> MemoryRecord:
        if now <= current.updated_at:
            raise MemoryTargetStateError(
                "memory service clock must follow the current record update",
                operation=MemoryOperation.CONFIRM.value,
                scope_kind=current.scope.kind.value,
                scope_id=current.scope.scope_id,
                memory_id=current.memory_id,
                candidate_digest=candidate.digest,
                reason="clock_not_after_record_update",
            )
        return MemoryRecord.from_candidate(
            candidate,
            memory_id=current.memory_id,
            revision=current.revision + 1,
            confirmation=MemoryConfirmation(candidate.digest, now),
            created_at=current.created_at,
            updated_at=now,
            supersedes_id=current.supersedes_id,
        )

    def _new_record(
        self,
        candidate: MemoryCandidate,
        *,
        now: datetime,
        scope: MemoryScope,
    ) -> MemoryRecord:
        memory_id = self._new_memory_id(
            scope=scope,
            operation=MemoryOperation.CONFIRM,
            candidate_digest=candidate.digest,
        )
        return MemoryRecord.from_candidate(
            candidate,
            memory_id=memory_id,
            revision=1,
            confirmation=MemoryConfirmation(candidate.digest, now),
            created_at=now,
            updated_at=now,
        )

    def _new_memory_id(
        self,
        *,
        scope: MemoryScope,
        operation: MemoryOperation,
        memory_id: str | None = None,
        candidate_digest: str | None = None,
    ) -> str:
        try:
            generated = self._id_factory()
        except Exception:
            raise self._dependency_error(
                operation,
                scope,
                memory_id=memory_id,
                candidate_digest=candidate_digest,
                reason="ID_factory_failure",
            ) from None
        if (
            not isinstance(generated, str)
            or not generated.strip()
            or generated != generated.strip()
        ):
            raise MemoryValidationError("memory ID factory must return a non-empty ID")
        if any(ord(character) < 32 or ord(character) == 127 for character in generated):
            raise MemoryValidationError("memory ID factory must return a printable ID")
        return generated

    def _now(
        self,
        operation: MemoryOperation,
        scope: MemoryScope,
        *,
        memory_id: str | None = None,
        candidate_digest: str | None = None,
    ) -> datetime:
        try:
            now = self._clock()
        except Exception:
            raise self._dependency_error(
                operation,
                scope,
                memory_id=memory_id,
                candidate_digest=candidate_digest,
                reason="clock_failure",
            ) from None
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise MemoryValidationError("memory service clock must be timezone-aware")
        return now

    def _dependency_error(
        self,
        operation: MemoryOperation,
        scope: MemoryScope,
        *,
        memory_id: str | None = None,
        candidate_digest: str | None = None,
        reason: str,
    ) -> MemoryDependencyError:
        return MemoryDependencyError(
            "memory dependency failed",
            operation=operation.value,
            scope_kind=scope.kind.value,
            scope_id=scope.scope_id,
            memory_id=memory_id,
            candidate_digest=candidate_digest,
            reason=reason,
        )

    def _not_found_error(
        self,
        operation: MemoryOperation,
        scope: MemoryScope,
        *,
        memory_id: str | None = None,
        candidate_digest: str | None = None,
    ) -> MemoryServiceNotFoundError:
        return MemoryServiceNotFoundError(
            "memory record was not found",
            operation=operation.value,
            scope_kind=scope.kind.value,
            scope_id=scope.scope_id,
            memory_id=memory_id,
            candidate_digest=candidate_digest,
            reason="not_found",
        )


def _service_recall_result_is_valid(
    result: MemoryRecall,
    eligible: tuple[MemoryRecord, ...],
    *,
    request: MemoryRecallRequest,
) -> bool:
    expected = {record.memory_id: record for record in eligible}
    returned = (*result.matches, *result.omitted)
    if {match.memory_id for match in returned} != set(expected):
        return False
    if any(expected[match.memory_id] != match.record for match in returned):
        return False
    if len(result.matches) > request.max_records:
        return False
    if any(
        match.score < request.min_score or match.record.kind not in request.allowed_kinds
        for match in result.matches
    ):
        return False
    return sum(len(match.record.content) for match in result.matches) <= request.max_characters


def _service_recall_ineligibility(
    record: MemoryRecord,
    *,
    scope: MemoryScope,
    allowed_kinds: frozenset[MemoryKind],
    now: datetime,
) -> RecallEligibilityReason | None:
    """Keep safety-critical recall filters enforced at the service boundary."""

    if not isinstance(record, MemoryRecord):
        return RecallEligibilityReason.NOT_CONFIRMED_DURABLE_RECORD
    if record.scope != scope:
        return RecallEligibilityReason.SCOPE_MISMATCH
    if record.status is MemoryLifecycleStatus.SUPERSEDED:
        return RecallEligibilityReason.SUPERSEDED
    if record.status is MemoryLifecycleStatus.EXPIRED:
        return RecallEligibilityReason.EXPIRED_STATUS
    if record.status is not MemoryLifecycleStatus.ACTIVE:
        return RecallEligibilityReason.EXPIRED_STATUS
    if record.valid_from > now:
        return RecallEligibilityReason.NOT_YET_VALID
    if record.expires_at is not None and record.expires_at <= now:
        return RecallEligibilityReason.EXPIRED
    if record.sensitivity is MemorySensitivity.SECRET:
        return RecallEligibilityReason.SECRET_CONTENT_NOT_ALLOWED
    if record.sensitivity is MemorySensitivity.SENSITIVE:
        return RecallEligibilityReason.SENSITIVE_CONTENT_NOT_ALLOWED
    if record.kind not in allowed_kinds:
        return RecallEligibilityReason.KIND_NOT_ALLOWED
    return None


def _require_candidate(candidate: object) -> None:
    if not isinstance(candidate, MemoryCandidate):
        raise MemoryValidationError("memory service requires a MemoryCandidate")


def _candidate_from_value(candidate: MemoryCandidate | MemoryProposal) -> MemoryCandidate:
    if isinstance(candidate, MemoryProposal):
        return candidate.candidate
    if isinstance(candidate, MemoryCandidate):
        return candidate
    raise MemoryValidationError("memory service requires a MemoryCandidate or MemoryProposal")


def _require_scope(scope: object) -> None:
    if not isinstance(scope, MemoryScope):
        raise MemoryValidationError("memory service scope must be a MemoryScope")


def _require_memory_id(memory_id: object) -> None:
    if not isinstance(memory_id, str) or not memory_id.strip() or memory_id != memory_id.strip():
        raise MemoryValidationError("memory ID must be a non-empty value")
    if len(memory_id) > 256 or any(
        ord(character) < 32 or ord(character) == 127 for character in memory_id
    ):
        raise MemoryValidationError("memory ID must be a printable value")


def _resolve_digest_argument(
    candidate_digest: str | None,
    digest: str | None,
) -> str | None:
    if candidate_digest is not None and digest is not None and candidate_digest != digest:
        raise MemoryValidationError("memory confirmation received two different digests")
    return candidate_digest if candidate_digest is not None else digest


def _require_exact_digest(
    candidate: MemoryCandidate,
    candidate_digest: str | None,
    scope: MemoryScope,
    *,
    operation: MemoryOperation,
) -> str:
    if not isinstance(candidate_digest, str) or not candidate_digest.strip():
        raise MemoryValidationError("memory confirmation requires a candidate digest")
    expected = candidate.digest
    if not compare_digest(expected, candidate_digest):
        raise MemoryDigestMismatchError(
            "memory candidate digest does not match",
            operation=operation.value,
            scope_kind=scope.kind.value,
            scope_id=scope.scope_id,
            candidate_digest=expected,
            reason="digest_mismatch",
        )
    return expected


def _require_admitted(
    decision: AdmissionDecision,
    operation: MemoryOperation,
    scope: MemoryScope,
    candidate_digest: str,
    *,
    memory_id: str | None = None,
) -> None:
    if decision.action is AdmissionAction.DENY:
        raise MemoryAdmissionDeniedError(
            "memory candidate was denied by policy",
            operation=operation.value,
            scope_kind=scope.kind.value,
            scope_id=scope.scope_id,
            memory_id=memory_id,
            candidate_digest=candidate_digest,
            reason=decision.reason.value,
        )


def _required_existing(decision: MemoryConsolidationDecision) -> MemoryRecord:
    if decision.existing is None:
        raise MemoryValidationError("memory consolidation decision lost its existing record")
    return decision.existing
