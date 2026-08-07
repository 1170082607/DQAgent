import hashlib
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from dqagent.errors import MemoryError, MemoryValidationError
from dqagent.memory import (
    MEMORY_SCHEMA_VERSION,
    MemoryCandidate,
    MemoryConfidence,
    MemoryConfirmation,
    MemoryForgetReason,
    MemoryKind,
    MemoryLifecycleStatus,
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
    MemoryScopeKind,
    MemorySensitivity,
    MemorySourceType,
    MemoryTombstone,
)

NOW = datetime(2026, 8, 7, 9, 0, tzinfo=UTC)
SOURCE_DIGEST = hashlib.sha256(b"committed source item").hexdigest()


def make_provenance() -> MemoryProvenance:
    return MemoryProvenance(
        source_type=MemorySourceType.COMMITTED_SESSION_TURN,
        source_item_digest=SOURCE_DIGEST,
        extractor_identity="explicit-v1",
        extracted_at=NOW,
        source_id="session-42",
        source_revision=3,
        run_id="run-9",
    )


def make_candidate() -> MemoryCandidate:
    return MemoryCandidate(
        scope=MemoryScope(MemoryScopeKind.USER, "user-7"),
        kind=MemoryKind.PREFERENCE,
        topic="response.language",
        content="The user prefers concise Chinese answers.",
        confidence=MemoryConfidence(0.85),
        sensitivity=MemorySensitivity.NON_SENSITIVE,
        provenance=make_provenance(),
        valid_from=NOW,
        expires_at=NOW + timedelta(days=90),
    )


def make_record() -> MemoryRecord:
    candidate = make_candidate()
    confirmed_at = NOW + timedelta(minutes=1)
    return MemoryRecord.from_candidate(
        candidate,
        memory_id="memory-1",
        revision=1,
        confirmation=MemoryConfirmation(candidate.digest, confirmed_at),
        created_at=confirmed_at,
    )


def test_scope_accepts_only_explicit_user_or_project_identity() -> None:
    assert MemoryScope(MemoryScopeKind.USER, "user-1").kind is MemoryScopeKind.USER
    assert MemoryScope(MemoryScopeKind.PROJECT, "project-1").kind is MemoryScopeKind.PROJECT

    with pytest.raises(ValueError):
        MemoryScopeKind("session")
    with pytest.raises(MemoryValidationError, match="scope kind"):
        MemoryScope("session", "session-1")  # type: ignore[arg-type]
    with pytest.raises(MemoryValidationError, match="scope ID must not be empty"):
        MemoryScope(MemoryScopeKind.USER, "  ")


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), -0.01, 1.01, True],
)
def test_confidence_rejects_non_finite_unbounded_or_boolean_values(value: object) -> None:
    with pytest.raises(MemoryValidationError, match="confidence"):
        MemoryConfidence(value)  # type: ignore[arg-type]


def test_confidence_is_distinct_from_confirmation_and_truth() -> None:
    candidate = make_candidate()

    assert candidate.confidence == MemoryConfidence(0.85)
    assert not hasattr(candidate, "confirmation")
    assert not hasattr(candidate, "truth")

    record = make_record()
    assert record.confidence == candidate.confidence
    assert isinstance(record.confirmation, MemoryConfirmation)


def test_candidate_digest_is_canonical_and_covers_preview_fields() -> None:
    candidate = make_candidate()
    assert candidate.digest == hashlib.sha256(candidate.canonical_json.encode()).hexdigest()
    assert candidate.digest == make_candidate().digest

    different_digest = hashlib.sha256(b"different source item").hexdigest()
    variants = (
        replace(candidate, scope=MemoryScope(MemoryScopeKind.USER, "user-8")),
        replace(candidate, scope=MemoryScope(MemoryScopeKind.PROJECT, "project-1")),
        replace(candidate, kind=MemoryKind.EXPERIENCE),
        replace(candidate, topic="response.format"),
        replace(candidate, content="The user prefers detailed Chinese answers."),
        replace(candidate, confidence=MemoryConfidence(0.5)),
        replace(candidate, sensitivity=MemorySensitivity.SENSITIVE),
        replace(
            candidate,
            provenance=MemoryProvenance(
                source_type=MemorySourceType.USER_DRAFT,
                source_item_digest=SOURCE_DIGEST,
                extractor_identity="explicit-v1",
                extracted_at=NOW,
                source_id="draft-1",
                run_id="run-9",
            ),
        ),
        replace(
            candidate,
            provenance=replace(candidate.provenance, source_item_digest=different_digest),
        ),
        replace(
            candidate,
            provenance=replace(candidate.provenance, extractor_identity="other-v1"),
        ),
        replace(
            candidate,
            provenance=replace(candidate.provenance, extracted_at=NOW - timedelta(seconds=1)),
        ),
        replace(
            candidate,
            provenance=replace(candidate.provenance, source_id="session-43"),
        ),
        replace(
            candidate,
            provenance=replace(candidate.provenance, source_revision=4),
        ),
        replace(candidate, provenance=replace(candidate.provenance, run_id="run-10")),
        replace(candidate, valid_from=NOW - timedelta(days=1)),
        replace(candidate, expires_at=NOW + timedelta(days=30)),
    )

    assert all(variant.digest != candidate.digest for variant in variants)
    assert f'"schema_version":{MEMORY_SCHEMA_VERSION}' in candidate.canonical_json


def test_candidate_digest_normalizes_equivalent_timezone_offsets() -> None:
    candidate = make_candidate()
    equivalent = candidate.valid_from.astimezone(timezone(timedelta(hours=8)))

    assert replace(candidate, valid_from=equivalent).digest == candidate.digest


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("topic", "Response Language", "topic"),
        ("topic", "", "topic"),
        ("content", "  ", "content must not be empty"),
        ("schema_version", 0, "schema version"),
        ("schema_version", 2, "schema version"),
    ],
)
def test_candidate_rejects_invalid_topic_content_and_schema(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(MemoryValidationError, match=message):
        replace(make_candidate(), **{field: value})


def test_provenance_requires_controlled_source_and_committed_revision() -> None:
    provenance = make_provenance()

    with pytest.raises(MemoryValidationError, match="source type"):
        replace(provenance, source_type="session")
    with pytest.raises(MemoryValidationError, match="requires a source ID and revision"):
        replace(provenance, source_revision=None)
    with pytest.raises(MemoryValidationError, match="positive integer"):
        replace(provenance, source_revision=0)
    with pytest.raises(MemoryValidationError, match="SHA-256"):
        replace(provenance, source_item_digest="not-a-digest")


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: replace(make_candidate(), kind="preference"), "memory kind"),
        (lambda: replace(make_candidate(), sensitivity="secret"), "memory sensitivity"),
        (lambda: replace(make_record(), status="active"), "lifecycle status"),
        (
            lambda: replace(make_record().confirmation, candidate_digest="A" * 64),
            "SHA-256",
        ),
        (
            lambda: replace(
                MemoryTombstone.from_record(
                    make_record(),
                    forgotten_at=NOW + timedelta(minutes=2),
                    reason=MemoryForgetReason.USER_REQUEST,
                ),
                reason="user_request",
            ),
            "forget reason",
        ),
    ],
)
def test_controlled_memory_values_reject_raw_free_strings(
    factory: Callable[[], object], message: str
) -> None:
    with pytest.raises(MemoryValidationError, match=message):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: replace(make_provenance(), extracted_at=datetime.now()),
        lambda: replace(make_candidate(), valid_from=datetime.now()),
        lambda: replace(make_candidate(), expires_at=datetime.now()),
        lambda: MemoryConfirmation(make_candidate().digest, datetime.now()),
        lambda: replace(make_record(), created_at=datetime.now()),
        lambda: MemoryTombstone(
            "memory-1",
            1,
            MemoryScope(MemoryScopeKind.USER, "user-1"),
            datetime.now(),
            MemoryForgetReason.USER_REQUEST,
        ),
        lambda: MemoryTombstone.from_record(
            make_record(),
            forgotten_at=datetime.now(),
            reason=MemoryForgetReason.USER_REQUEST,
        ),
    ],
)
def test_memory_values_reject_naive_datetimes(factory: Callable[[], object]) -> None:
    with pytest.raises(MemoryValidationError, match="timezone-aware"):
        factory()


def test_validity_and_record_audit_order_are_structural_invariants() -> None:
    candidate = make_candidate()
    with pytest.raises(MemoryValidationError, match="must follow"):
        replace(candidate, expires_at=candidate.valid_from)

    record = make_record()
    with pytest.raises(MemoryValidationError, match="must not follow"):
        replace(record, updated_at=record.created_at - timedelta(seconds=1))
    with pytest.raises(MemoryValidationError, match="require an elapsed expiry"):
        replace(record, status=MemoryLifecycleStatus.EXPIRED)


def test_record_requires_exact_confirmation_and_has_no_pending_state() -> None:
    candidate = make_candidate()
    confirmed_at = NOW + timedelta(minutes=1)

    with pytest.raises(MemoryValidationError, match="does not match"):
        MemoryRecord.from_candidate(
            candidate,
            memory_id="memory-1",
            revision=1,
            confirmation=MemoryConfirmation("0" * 64, confirmed_at),
            created_at=confirmed_at,
        )
    with pytest.raises(ValueError):
        MemoryLifecycleStatus("pending")
    with pytest.raises(MemoryValidationError, match="positive integer"):
        replace(make_record(), revision=0)
    with pytest.raises(MemoryValidationError, match="memory ID must not be empty"):
        replace(make_record(), memory_id=" ")


def test_record_rejects_payload_changed_after_confirmation() -> None:
    with pytest.raises(MemoryValidationError, match="does not match"):
        replace(make_record(), content="Different content")


def test_candidate_and_record_are_not_interchangeable() -> None:
    candidate = make_candidate()
    record = make_record()

    with pytest.raises(TypeError, match="requires a MemoryCandidate"):
        MemoryRecord.from_candidate(  # type: ignore[arg-type]
            record,
            memory_id="memory-2",
            revision=1,
            confirmation=record.confirmation,
            created_at=record.created_at,
        )
    with pytest.raises(TypeError, match="requires a MemoryRecord"):
        MemoryTombstone.from_record(  # type: ignore[arg-type]
            candidate,
            forgotten_at=NOW + timedelta(minutes=2),
            reason=MemoryForgetReason.USER_REQUEST,
        )


def test_tombstone_replaces_forgotten_payload_with_minimal_metadata() -> None:
    record = make_record()
    tombstone = MemoryTombstone.from_record(
        record,
        forgotten_at=record.updated_at + timedelta(minutes=1),
        reason=MemoryForgetReason.USER_REQUEST,
    )

    assert tombstone.memory_id == record.memory_id
    assert tombstone.scope == record.scope
    assert not hasattr(tombstone, "content")
    assert not hasattr(tombstone, "topic")
    assert not hasattr(tombstone, "provenance")
    with pytest.raises(MemoryValidationError, match="before its latest update"):
        MemoryTombstone.from_record(
            record,
            forgotten_at=record.updated_at - timedelta(seconds=1),
            reason=MemoryForgetReason.USER_REQUEST,
        )


def test_memory_values_are_immutable_and_validation_is_a_memory_error() -> None:
    candidate = make_candidate()

    with pytest.raises(FrozenInstanceError):
        candidate.content = "changed"  # type: ignore[misc]
    with pytest.raises(MemoryError):
        MemoryScope(MemoryScopeKind.USER, "")
