"""Behavioral tests for explicit model-free memory management."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from dqagent.errors import (
    MemoryAdmissionDeniedError,
    MemoryConflictError,
    MemoryDependencyError,
    MemoryDigestMismatchError,
    MemoryServiceNotFoundError,
    MemoryTopicConflictError,
    MemoryValidationError,
)
from dqagent.memory import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionReason,
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
from dqagent.memory_service import (
    MemoryOutcome,
    MemoryService,
    MemoryWriteOutcome,
)
from dqagent.memory_store import InMemoryMemoryStore, MemoryScopeSnapshot, SqliteMemoryStore

NOW = datetime(2026, 8, 11, 8, 0, tzinfo=UTC)
USER_SCOPE = MemoryScope(MemoryScopeKind.USER, "user-7")
OTHER_SCOPE = MemoryScope(MemoryScopeKind.USER, "user-8")
SOURCE_DIGEST = hashlib.sha256(b"service source").hexdigest()


@dataclass
class ManualClock:
    current: datetime

    def __call__(self) -> datetime:
        return self.current


def make_candidate(
    *,
    scope: MemoryScope = USER_SCOPE,
    topic: str = "response.language",
    content: str = "The user prefers concise Chinese answers.",
    extracted_at: datetime = NOW - timedelta(minutes=3),
    expires_at: datetime | None = NOW + timedelta(days=90),
    sensitivity: MemorySensitivity = MemorySensitivity.NON_SENSITIVE,
) -> MemoryCandidate:
    return MemoryCandidate(
        scope=scope,
        kind=MemoryKind.PREFERENCE,
        topic=topic,
        content=content,
        confidence=MemoryConfidence(0.9),
        sensitivity=sensitivity,
        provenance=MemoryProvenance(
            source_type=MemorySourceType.USER_DRAFT,
            source_item_digest=SOURCE_DIGEST,
            extractor_identity="explicit-service-test",
            extracted_at=extracted_at,
        ),
        valid_from=NOW - timedelta(days=1),
        expires_at=expires_at,
    )


def make_service(
    store: object | None = None,
    *,
    clock: ManualClock | None = None,
    ids: tuple[str, ...] = ("memory-1", "memory-2", "memory-3"),
) -> tuple[MemoryService, ManualClock, object]:
    resolved_store = store or InMemoryMemoryStore()
    resolved_clock = clock or ManualClock(NOW)
    generated = iter(ids)
    service = MemoryService(
        resolved_store,  # type: ignore[arg-type]
        DefaultMemoryPolicy(),
        clock=resolved_clock,
        id_factory=lambda: next(generated),
    )
    return service, resolved_clock, resolved_store


class PermissiveWritePolicy(DefaultMemoryPolicy):
    def assess_write(
        self,
        candidate: MemoryCandidate,
        *,
        scope: MemoryScope,
        now: datetime,
    ) -> AdmissionDecision:
        return AdmissionDecision(
            action=AdmissionAction.REQUIRE_CONFIRMATION,
            reason=AdmissionReason.USER_CONFIRMATION_REQUIRED,
            effective_scope=candidate.scope,
            expires_at=candidate.expires_at,
        )


def test_propose_is_transient_and_returns_policy_decision() -> None:
    service, _, store = make_service()
    candidate = make_candidate()

    preview = service.propose(candidate, scope=USER_SCOPE)

    assert preview.candidate is candidate
    assert preview.candidate_digest == candidate.digest
    assert preview.decision.requires_confirmation
    assert preview.metadata.outcome is MemoryOutcome.PROPOSED
    assert candidate.content not in str(preview.event_attributes)
    assert store.load(USER_SCOPE) == MemoryScopeSnapshot(USER_SCOPE)  # type: ignore[union-attr]


def test_confirm_rechecks_policy_and_rejects_denied_candidate_without_writing() -> None:
    service, _, store = make_service()
    candidate = make_candidate(sensitivity=MemorySensitivity.SENSITIVE)
    preview = service.preview(candidate, scope=USER_SCOPE)

    assert not preview.decision.requires_confirmation
    with pytest.raises(MemoryAdmissionDeniedError) as error:
        service.confirm(candidate, candidate.digest, scope=USER_SCOPE)

    assert error.value.reason == "sensitive_content_not_allowed"
    assert candidate.content not in str(error.value)
    assert store.load(USER_SCOPE) == MemoryScopeSnapshot(USER_SCOPE)  # type: ignore[union-attr]


def test_confirm_requires_the_exact_candidate_digest_and_has_content_free_error_metadata() -> None:
    service, _, store = make_service()
    candidate = make_candidate()
    preview = service.propose(candidate, scope=USER_SCOPE)
    tampered = replace(candidate, content="The user prefers detailed English answers.")

    with pytest.raises(MemoryDigestMismatchError) as error:
        service.confirm(tampered, preview.candidate_digest, scope=USER_SCOPE)

    assert error.value.operation == "confirm"
    assert error.value.candidate_digest == tampered.digest
    assert candidate.content not in str(error.value)
    assert candidate.content not in str(error.value.event_attributes)
    assert store.load(USER_SCOPE) == MemoryScopeSnapshot(USER_SCOPE)  # type: ignore[union-attr]


def test_confirm_rechecks_clock_after_preview_and_rejects_expired_candidate() -> None:
    clock = ManualClock(NOW)
    service, _, store = make_service(clock=clock)
    candidate = make_candidate(expires_at=NOW + timedelta(minutes=1))
    preview = service.propose(candidate, scope=USER_SCOPE)

    clock.current = NOW + timedelta(minutes=1)
    with pytest.raises(MemoryAdmissionDeniedError) as error:
        service.confirm(candidate, preview.digest, scope=USER_SCOPE)

    assert error.value.reason == "candidate_expired"
    assert store.load(USER_SCOPE) == MemoryScopeSnapshot(USER_SCOPE)  # type: ignore[union-attr]


def test_confirm_adds_then_exact_duplicate_refreshes_the_same_record() -> None:
    clock = ManualClock(NOW)
    service, _, store = make_service(clock=clock)
    original = make_candidate()
    first = service.confirm(original, original.digest, scope=USER_SCOPE)
    refreshed_candidate = replace(
        original,
        confidence=MemoryConfidence(0.7),
        provenance=replace(
            original.provenance,
            extracted_at=NOW + timedelta(minutes=1),
        ),
    )
    clock.current = NOW + timedelta(minutes=2)

    refreshed = service.confirm(
        refreshed_candidate,
        refreshed_candidate.digest,
        scope=USER_SCOPE,
    )

    assert first.outcome is MemoryWriteOutcome.ADDED
    assert refreshed.outcome is MemoryWriteOutcome.REFRESHED
    assert refreshed.record.memory_id == first.record.memory_id
    assert refreshed.record.revision == 2
    assert refreshed.record.content == original.content
    assert refreshed.revision == 2
    assert len(store.load(USER_SCOPE).records) == 1  # type: ignore[union-attr]


def test_confirm_rejects_same_topic_with_different_content_without_overwrite() -> None:
    service, _, store = make_service()
    original = make_candidate()
    first = service.confirm(original, original.digest, scope=USER_SCOPE)
    conflicting = make_candidate(content="The user prefers detailed English answers.")

    with pytest.raises(MemoryTopicConflictError) as error:
        service.confirm(conflicting, conflicting.digest, scope=USER_SCOPE)

    assert isinstance(error.value, MemoryConflictError)
    assert error.value.memory_id == first.record.memory_id
    assert store.load(USER_SCOPE).records == (first.record,)  # type: ignore[union-attr]


def test_two_confirmations_from_one_scope_revision_have_one_cas_winner() -> None:
    store = InMemoryMemoryStore()
    clock = ManualClock(NOW)
    candidate_one = make_candidate(content="The user prefers concise Chinese answers.")
    candidate_two = make_candidate(content="The user prefers formal Chinese answers.")
    barrier = Barrier(2)

    def confirm(candidate: MemoryCandidate) -> object:
        service, _, _ = make_service(store, clock=clock, ids=(candidate.content[:8],))
        barrier.wait()
        try:
            return service.confirm(candidate, candidate.digest, scope=USER_SCOPE)
        except MemoryConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(confirm, (candidate_one, candidate_two)))

    assert sum(isinstance(result, MemoryConflictError) for result in results) == 1
    assert len(store.load(USER_SCOPE).records) == 1


def test_correct_uses_explicit_target_and_atomically_supersedes_old_record() -> None:
    clock = ManualClock(NOW)
    service, _, store = make_service(clock=clock)
    original = make_candidate()
    first = service.confirm(original, original.digest, scope=USER_SCOPE)
    corrected = make_candidate(content="The user prefers detailed English answers.")
    clock.current = NOW + timedelta(minutes=1)

    result = service.correct(
        USER_SCOPE,
        first.record.memory_id,
        corrected,
        candidate_digest=corrected.digest,
    )

    records = store.load(USER_SCOPE).records
    assert result.superseded.status is MemoryLifecycleStatus.SUPERSEDED
    assert result.replacement.status is MemoryLifecycleStatus.ACTIVE
    assert result.replacement.supersedes_id == first.record.memory_id
    assert records == (result.replacement, result.superseded) or records == (
        result.superseded,
        result.replacement,
    )
    assert sum(record.status is MemoryLifecycleStatus.ACTIVE for record in records) == 1
    assert result.revision == 2


def test_correct_requires_an_explicit_candidate_digest() -> None:
    clock = ManualClock(NOW)
    service, _, store = make_service(clock=clock)
    original = make_candidate()
    first = service.confirm(original, original.digest, scope=USER_SCOPE)
    corrected = make_candidate(content="The user prefers detailed English answers.")
    clock.current = NOW + timedelta(minutes=1)

    with pytest.raises(MemoryValidationError, match="requires a candidate digest"):
        service.correct(USER_SCOPE, first.record.memory_id, corrected)

    assert store.load(USER_SCOPE).records == (first.record,)  # type: ignore[union-attr]


def test_correct_rejects_a_candidate_that_differs_from_the_preview_digest() -> None:
    clock = ManualClock(NOW)
    service, _, store = make_service(clock=clock)
    original = make_candidate()
    first = service.confirm(original, original.digest, scope=USER_SCOPE)
    previewed = make_candidate(content="The user prefers detailed English answers.")
    preview = service.propose(previewed, scope=USER_SCOPE)
    submitted = make_candidate(content="The user prefers formal English answers.")
    clock.current = NOW + timedelta(minutes=1)

    with pytest.raises(MemoryDigestMismatchError) as error:
        service.correct(
            USER_SCOPE,
            first.record.memory_id,
            submitted,
            candidate_digest=preview.digest,
        )

    assert error.value.operation == "correct"
    assert error.value.candidate_digest == submitted.digest
    assert store.load(USER_SCOPE).records == (first.record,)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("sensitivity", "reason"),
    [
        (MemorySensitivity.SENSITIVE, AdmissionReason.SENSITIVE_CONTENT_NOT_ALLOWED),
        (MemorySensitivity.SECRET, AdmissionReason.SECRET_CONTENT_NOT_ALLOWED),
    ],
)
def test_service_hard_denies_sensitive_writes_even_with_a_permissive_policy(
    sensitivity: MemorySensitivity,
    reason: AdmissionReason,
) -> None:
    store = InMemoryMemoryStore()
    clock = ManualClock(NOW)
    service = MemoryService(
        store,
        PermissiveWritePolicy(),
        clock=clock,
        id_factory=iter(("memory-1", "memory-2")).__next__,
    )
    original = make_candidate()
    first = service.confirm(original, original.digest, scope=USER_SCOPE)
    sensitive = make_candidate(sensitivity=sensitivity)
    baseline = store.load(USER_SCOPE)

    preview = service.propose(sensitive, scope=USER_SCOPE)
    assert preview.decision.action is AdmissionAction.DENY
    assert preview.decision.reason is reason

    with pytest.raises(MemoryAdmissionDeniedError) as confirm_error:
        service.confirm(sensitive, sensitive.digest, scope=USER_SCOPE)
    assert confirm_error.value.reason == reason.value

    with pytest.raises(MemoryAdmissionDeniedError) as correct_error:
        service.correct(
            USER_SCOPE,
            first.record.memory_id,
            sensitive,
            candidate_digest=sensitive.digest,
        )
    assert correct_error.value.reason == reason.value
    assert store.load(USER_SCOPE) == baseline


def test_service_hard_denies_sensitive_like_content_even_with_a_permissive_policy() -> None:
    store = InMemoryMemoryStore()
    service = MemoryService(store, PermissiveWritePolicy(), clock=lambda: NOW)
    candidate = replace(make_candidate(), content="The user has diabetes.")

    preview = service.preview(candidate, scope=USER_SCOPE)

    assert preview.decision.action is AdmissionAction.DENY
    assert preview.decision.reason is AdmissionReason.SENSITIVE_CONTENT_NOT_ALLOWED
    with pytest.raises(MemoryAdmissionDeniedError):
        service.confirm(candidate, candidate.digest, scope=USER_SCOPE)
    assert store.load(USER_SCOPE) == MemoryScopeSnapshot(USER_SCOPE)


class ApplyFailingStore:
    def __init__(self, snapshot: MemoryScopeSnapshot) -> None:
        self.snapshot = snapshot

    def load(self, scope: MemoryScope) -> MemoryScopeSnapshot:
        return self.snapshot

    def apply(self, change_set: object, *, expected_revision: int) -> MemoryScopeSnapshot:
        raise RuntimeError("store apply failure with an accidental payload")


@pytest.mark.parametrize("operation", ["correct", "forget"])
def test_required_mutations_fail_closed_on_store_apply_failure(operation: str) -> None:
    seed_service, _, seed_store = make_service()
    original = make_candidate()
    first = seed_service.confirm(original, original.digest, scope=USER_SCOPE)
    baseline = seed_store.load(USER_SCOPE)  # type: ignore[union-attr]
    failing_store = ApplyFailingStore(baseline)  # type: ignore[arg-type]
    service, _, _ = make_service(
        failing_store,
        clock=ManualClock(NOW + timedelta(minutes=1)),
        ids=("replacement-1", "replacement-2"),
    )
    corrected = make_candidate(content="The user prefers detailed English answers.")
    call = {
        "correct": lambda: service.correct(
            USER_SCOPE,
            first.record.memory_id,
            corrected,
            candidate_digest=corrected.digest,
        ),
        "forget": lambda: service.forget(USER_SCOPE, first.record.memory_id),
    }[operation]

    with pytest.raises(MemoryDependencyError) as error:
        call()

    assert error.value.reason == "store_apply_failure"
    assert "accidental payload" not in str(error.value)
    assert failing_store.load(USER_SCOPE) == baseline


def test_expiry_is_materialized_and_a_new_candidate_can_replace_an_expired_topic() -> None:
    clock = ManualClock(NOW)
    service, _, store = make_service(clock=clock)
    expiring = make_candidate(expires_at=NOW + timedelta(minutes=1))
    first = service.confirm(expiring, expiring.digest, scope=USER_SCOPE)

    clock.current = NOW + timedelta(minutes=1)
    listed = service.list(USER_SCOPE)
    expired = listed.records[0]
    assert expired.status is MemoryLifecycleStatus.EXPIRED
    assert listed.revision == 2

    replacement_candidate = make_candidate(
        content="The user prefers detailed English answers.",
        expires_at=NOW + timedelta(days=30),
        extracted_at=clock.current - timedelta(seconds=1),
    )
    clock.current = NOW + timedelta(minutes=2)
    replacement = service.confirm(
        replacement_candidate,
        replacement_candidate.digest,
        scope=USER_SCOPE,
    )

    assert replacement.record.memory_id != first.record.memory_id
    assert replacement.record.status is MemoryLifecycleStatus.ACTIVE
    assert any(
        record.status is MemoryLifecycleStatus.ACTIVE
        for record in store.load(USER_SCOPE).records
    )


def test_forget_is_exact_scope_id_and_removes_payload_from_public_reads() -> None:
    service, _, store = make_service()
    candidate = make_candidate()
    first = service.confirm(candidate, candidate.digest, scope=USER_SCOPE)

    forgotten = service.forget(USER_SCOPE, first.record.memory_id)

    assert forgotten.memory_id == first.record.memory_id
    assert forgotten.tombstone.scope == USER_SCOPE
    assert service.list(USER_SCOPE).records == ()
    assert service.list(USER_SCOPE).tombstones == (forgotten.tombstone,)
    assert candidate.content not in repr(forgotten.tombstone)
    with pytest.raises(MemoryServiceNotFoundError):
        service.show(USER_SCOPE, first.record.memory_id)
    assert service.list(OTHER_SCOPE).records == ()
    assert store.load(USER_SCOPE).records == ()  # type: ignore[union-attr]


class FailingStore:
    def load(self, scope: MemoryScope) -> MemoryScopeSnapshot:
        raise RuntimeError("store failure with an accidental payload")

    def apply(self, change_set: object, *, expected_revision: int) -> MemoryScopeSnapshot:
        raise RuntimeError("store failure with an accidental payload")


@pytest.mark.parametrize("operation", ["list", "show", "correct", "forget"])
def test_required_operations_fail_closed_on_store_dependency_failure(operation: str) -> None:
    service, _, _ = make_service(FailingStore())
    candidate = make_candidate()
    call = {
        "list": lambda: service.list(USER_SCOPE),
        "show": lambda: service.show(USER_SCOPE, "memory-1"),
        "correct": lambda: service.correct(
            USER_SCOPE,
            "memory-1",
            candidate,
            candidate_digest=candidate.digest,
        ),
        "forget": lambda: service.forget(USER_SCOPE, "memory-1"),
    }[operation]

    with pytest.raises(MemoryDependencyError) as error:
        call()

    assert error.value.reason == "store_load_failure"
    assert "accidental payload" not in str(error.value)


def test_sqlite_service_smoke_round_trips_correction_and_forgetting(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    clock = ManualClock(NOW)
    first_service, _, _ = make_service(SqliteMemoryStore(database), clock=clock)
    original = make_candidate()
    first = first_service.confirm(original, original.digest, scope=USER_SCOPE)

    second_service, _, _ = make_service(
        SqliteMemoryStore(database),
        clock=clock,
        ids=("memory-2", "memory-3"),
    )
    assert second_service.show(USER_SCOPE, first.record.memory_id).record == first.record

    corrected = make_candidate(content="The user prefers detailed English answers.")
    clock.current = NOW + timedelta(minutes=1)
    correction = second_service.correct(
        USER_SCOPE,
        first.record.memory_id,
        corrected,
        candidate_digest=corrected.digest,
    )
    assert second_service.show(USER_SCOPE, correction.replacement.memory_id).record == (
        correction.replacement
    )

    forgotten = second_service.forget(USER_SCOPE, correction.replacement.memory_id)
    assert second_service.list(USER_SCOPE).records == (correction.superseded,)
    assert second_service.list(USER_SCOPE).tombstones == (forgotten.tombstone,)
