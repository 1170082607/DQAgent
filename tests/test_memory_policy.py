import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from dqagent.errors import MemoryValidationError
from dqagent.memory import (
    MemoryCandidate,
    MemoryConfidence,
    MemoryConfirmation,
    MemoryKind,
    MemoryLifecycleStatus,
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
    MemoryScopeKind,
    MemorySensitivity,
    MemorySourceType,
)
from dqagent.memory_policy import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionReason,
    DefaultMemoryPolicy,
    MemoryPolicy,
    RecallEligibilityAction,
    RecallEligibilityDecision,
    RecallEligibilityReason,
)

NOW = datetime(2026, 8, 7, 9, 0, tzinfo=UTC)
USER_SCOPE = MemoryScope(MemoryScopeKind.USER, "user-7")
OTHER_USER_SCOPE = MemoryScope(MemoryScopeKind.USER, "user-8")
PROJECT_SCOPE = MemoryScope(MemoryScopeKind.PROJECT, "project-1")
ALL_KINDS = frozenset(MemoryKind)
SOURCE_DIGEST = hashlib.sha256(b"committed source item").hexdigest()


def make_provenance(
    *,
    source_type: MemorySourceType = MemorySourceType.COMMITTED_SESSION_TURN,
    extracted_at: datetime = NOW - timedelta(minutes=2),
) -> MemoryProvenance:
    return MemoryProvenance(
        source_type=source_type,
        source_item_digest=SOURCE_DIGEST,
        extractor_identity="explicit-v1",
        extracted_at=extracted_at,
        source_id=(
            "session-42"
            if source_type is MemorySourceType.COMMITTED_SESSION_TURN
            else "draft-1"
        ),
        source_revision=3 if source_type is MemorySourceType.COMMITTED_SESSION_TURN else None,
        run_id="run-9",
    )


def make_candidate(
    *,
    scope: MemoryScope = USER_SCOPE,
    kind: MemoryKind = MemoryKind.PREFERENCE,
    confidence: float = 0.85,
    sensitivity: MemorySensitivity = MemorySensitivity.NON_SENSITIVE,
    provenance: MemoryProvenance | None = None,
    valid_from: datetime = NOW - timedelta(days=1),
    expires_at: datetime | None = NOW + timedelta(days=90),
) -> MemoryCandidate:
    return MemoryCandidate(
        scope=scope,
        kind=kind,
        topic="response.language",
        content="The user prefers concise Chinese answers.",
        confidence=MemoryConfidence(confidence),
        sensitivity=sensitivity,
        provenance=provenance or make_provenance(),
        valid_from=valid_from,
        expires_at=expires_at,
    )


def make_record(
    candidate: MemoryCandidate | None = None,
    *,
    status: MemoryLifecycleStatus = MemoryLifecycleStatus.ACTIVE,
    updated_at: datetime = NOW - timedelta(seconds=30),
) -> MemoryRecord:
    candidate = candidate or make_candidate()
    confirmed_at = candidate.provenance.extracted_at + timedelta(seconds=30)
    record = MemoryRecord.from_candidate(
        candidate,
        memory_id="memory-1",
        revision=1,
        confirmation=MemoryConfirmation(candidate.digest, confirmed_at),
        created_at=confirmed_at,
        updated_at=updated_at,
    )
    return replace(record, status=status)


@pytest.mark.parametrize(
    ("action", "reason", "requires_confirmation"),
    [
        (AdmissionAction.ALLOW, AdmissionReason.WRITE_ALLOWED, False),
        (
            AdmissionAction.REQUIRE_CONFIRMATION,
            AdmissionReason.USER_CONFIRMATION_REQUIRED,
            True,
        ),
        (AdmissionAction.DENY, AdmissionReason.SCOPE_MISMATCH, False),
        (AdmissionAction.DENY, AdmissionReason.KIND_NOT_ALLOWED, False),
        (
            AdmissionAction.DENY,
            AdmissionReason.SENSITIVE_CONTENT_NOT_ALLOWED,
            False,
        ),
        (AdmissionAction.DENY, AdmissionReason.SECRET_CONTENT_NOT_ALLOWED, False),
        (AdmissionAction.DENY, AdmissionReason.PROVENANCE_IN_FUTURE, False),
        (AdmissionAction.DENY, AdmissionReason.CANDIDATE_EXPIRED, False),
    ],
)
def test_admission_action_and_reason_codes_are_stable(
    action: AdmissionAction,
    reason: AdmissionReason,
    requires_confirmation: bool,
) -> None:
    admitted = action is not AdmissionAction.DENY
    decision = AdmissionDecision(
        action=action,
        reason=reason,
        effective_scope=USER_SCOPE if admitted else None,
        expires_at=NOW + timedelta(days=1) if admitted else None,
    )

    assert decision.action.value in {"allow", "require_confirmation", "deny"}
    assert decision.reason.value == reason
    assert decision.requires_confirmation is requires_confirmation


@pytest.mark.parametrize(
    ("scope", "kind", "confidence", "source_type", "expires_at"),
    [
        (USER_SCOPE, MemoryKind.PREFERENCE, 0.0, MemorySourceType.USER_DRAFT, None),
        (
            USER_SCOPE,
            MemoryKind.USER_FACT,
            0.5,
            MemorySourceType.COMMITTED_SESSION_TURN,
            NOW + timedelta(seconds=1),
        ),
        (
            PROJECT_SCOPE,
            MemoryKind.EXPERIENCE,
            1.0,
            MemorySourceType.COMMITTED_SESSION_TURN,
            NOW + timedelta(days=90),
        ),
    ],
)
def test_default_policy_requires_confirmation_for_every_persistable_candidate(
    scope: MemoryScope,
    kind: MemoryKind,
    confidence: float,
    source_type: MemorySourceType,
    expires_at: datetime | None,
) -> None:
    candidate = make_candidate(
        scope=scope,
        kind=kind,
        confidence=confidence,
        provenance=make_provenance(source_type=source_type),
        expires_at=expires_at,
    )

    decision = DefaultMemoryPolicy().assess_write(candidate, scope=scope, now=NOW)

    assert decision == AdmissionDecision(
        action=AdmissionAction.REQUIRE_CONFIRMATION,
        reason=AdmissionReason.USER_CONFIRMATION_REQUIRED,
        effective_scope=scope,
        expires_at=expires_at,
    )


@pytest.mark.parametrize(
    ("candidate", "scope", "reason"),
    [
        (make_candidate(), OTHER_USER_SCOPE, AdmissionReason.SCOPE_MISMATCH),
        (
            make_candidate(sensitivity=MemorySensitivity.SENSITIVE),
            USER_SCOPE,
            AdmissionReason.SENSITIVE_CONTENT_NOT_ALLOWED,
        ),
        (
            make_candidate(sensitivity=MemorySensitivity.SECRET),
            USER_SCOPE,
            AdmissionReason.SECRET_CONTENT_NOT_ALLOWED,
        ),
        (
            make_candidate(scope=PROJECT_SCOPE, kind=MemoryKind.PREFERENCE),
            PROJECT_SCOPE,
            AdmissionReason.KIND_NOT_ALLOWED,
        ),
        (
            make_candidate(
                provenance=make_provenance(extracted_at=NOW + timedelta(microseconds=1))
            ),
            USER_SCOPE,
            AdmissionReason.PROVENANCE_IN_FUTURE,
        ),
        (
            make_candidate(expires_at=NOW),
            USER_SCOPE,
            AdmissionReason.CANDIDATE_EXPIRED,
        ),
    ],
)
def test_default_policy_denies_with_stable_reason(
    candidate: MemoryCandidate,
    scope: MemoryScope,
    reason: AdmissionReason,
) -> None:
    decision = DefaultMemoryPolicy().assess_write(candidate, scope=scope, now=NOW)

    assert decision == AdmissionDecision(
        action=AdmissionAction.DENY,
        reason=reason,
        effective_scope=None,
        expires_at=None,
    )


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (
            make_candidate(sensitivity=MemorySensitivity.SENSITIVE),
            RecallEligibilityReason.SENSITIVE_CONTENT_NOT_ALLOWED,
        ),
        (
            make_candidate(sensitivity=MemorySensitivity.SECRET),
            RecallEligibilityReason.SECRET_CONTENT_NOT_ALLOWED,
        ),
        (
            make_candidate(kind=MemoryKind.EXPERIENCE),
            RecallEligibilityReason.KIND_NOT_ALLOWED,
        ),
        (
            make_candidate(scope=PROJECT_SCOPE, kind=MemoryKind.PREFERENCE),
            RecallEligibilityReason.KIND_NOT_ALLOWED,
        ),
    ],
)
def test_recall_rejects_records_that_default_policy_would_not_admit(
    candidate: MemoryCandidate,
    reason: RecallEligibilityReason,
) -> None:
    policy = DefaultMemoryPolicy()

    assert policy.assess_write(candidate, scope=candidate.scope, now=NOW).action is (
        AdmissionAction.DENY
    )

    decision = policy.eligible(
        make_record(candidate),
        scope=candidate.scope,
        allowed_kinds=ALL_KINDS,
        now=NOW,
    )

    assert decision == RecallEligibilityDecision(
        action=RecallEligibilityAction.INELIGIBLE,
        reason=reason,
    )


def test_high_confidence_never_replaces_confirmation_or_establishes_truth() -> None:
    candidate = make_candidate(confidence=1.0)

    decision = DefaultMemoryPolicy().assess_write(candidate, scope=USER_SCOPE, now=NOW)

    assert decision.action is AdmissionAction.REQUIRE_CONFIRMATION
    assert not hasattr(candidate, "truth")
    assert not hasattr(candidate, "confirmation")


def test_policy_is_deterministic_for_the_same_input_and_clock() -> None:
    policy = DefaultMemoryPolicy()
    candidate = make_candidate()
    record = make_record(candidate)

    assert policy.assess_write(candidate, scope=USER_SCOPE, now=NOW) == policy.assess_write(
        candidate, scope=USER_SCOPE, now=NOW
    )
    assert policy.eligible(
        record, scope=USER_SCOPE, allowed_kinds=ALL_KINDS, now=NOW
    ) == policy.eligible(record, scope=USER_SCOPE, allowed_kinds=ALL_KINDS, now=NOW)


@pytest.mark.parametrize(
    ("record", "scope", "allowed_kinds"),
    [
        (make_record(), USER_SCOPE, ALL_KINDS),
        (
            make_record(make_candidate(expires_at=NOW + timedelta(microseconds=1))),
            USER_SCOPE,
            frozenset({MemoryKind.PREFERENCE}),
        ),
        (
            make_record(
                make_candidate(
                    scope=PROJECT_SCOPE,
                    kind=MemoryKind.EXPERIENCE,
                    valid_from=NOW,
                    expires_at=None,
                )
            ),
            PROJECT_SCOPE,
            frozenset({MemoryKind.EXPERIENCE}),
        ),
    ],
)
def test_recall_eligible_reasons_are_stable(
    record: MemoryRecord,
    scope: MemoryScope,
    allowed_kinds: frozenset[MemoryKind],
) -> None:
    decision = DefaultMemoryPolicy().eligible(
        record,
        scope=scope,
        allowed_kinds=allowed_kinds,
        now=NOW,
    )

    assert decision == RecallEligibilityDecision(
        action=RecallEligibilityAction.ELIGIBLE,
        reason=RecallEligibilityReason.ELIGIBLE,
    )
    assert decision.eligible


def _expired_status_record() -> MemoryRecord:
    candidate = make_candidate(expires_at=NOW)
    return make_record(
        candidate,
        status=MemoryLifecycleStatus.EXPIRED,
        updated_at=NOW,
    )


@pytest.mark.parametrize(
    ("record", "scope", "allowed_kinds", "reason"),
    [
        (
            make_record(),
            OTHER_USER_SCOPE,
            ALL_KINDS,
            RecallEligibilityReason.SCOPE_MISMATCH,
        ),
        (
            make_candidate(),
            USER_SCOPE,
            ALL_KINDS,
            RecallEligibilityReason.NOT_CONFIRMED_DURABLE_RECORD,
        ),
        (
            make_record(status=MemoryLifecycleStatus.SUPERSEDED),
            USER_SCOPE,
            ALL_KINDS,
            RecallEligibilityReason.SUPERSEDED,
        ),
        (
            _expired_status_record(),
            USER_SCOPE,
            ALL_KINDS,
            RecallEligibilityReason.EXPIRED_STATUS,
        ),
        (
            make_record(
                make_candidate(
                    valid_from=NOW + timedelta(microseconds=1),
                    expires_at=NOW + timedelta(days=1),
                )
            ),
            USER_SCOPE,
            ALL_KINDS,
            RecallEligibilityReason.NOT_YET_VALID,
        ),
        (
            make_record(make_candidate(expires_at=NOW), updated_at=NOW),
            USER_SCOPE,
            ALL_KINDS,
            RecallEligibilityReason.EXPIRED,
        ),
        (
            make_record(),
            USER_SCOPE,
            frozenset({MemoryKind.USER_FACT}),
            RecallEligibilityReason.KIND_NOT_ALLOWED,
        ),
    ],
)
def test_recall_ineligible_reasons_are_stable(
    record: MemoryCandidate | MemoryRecord,
    scope: MemoryScope,
    allowed_kinds: frozenset[MemoryKind],
    reason: RecallEligibilityReason,
) -> None:
    decision = DefaultMemoryPolicy().eligible(
        record,
        scope=scope,
        allowed_kinds=allowed_kinds,
        now=NOW,
    )

    assert decision == RecallEligibilityDecision(
        action=RecallEligibilityAction.INELIGIBLE,
        reason=reason,
    )
    assert not decision.eligible


@pytest.mark.parametrize("method", ["write", "recall"])
def test_policy_rejects_an_implicit_or_naive_clock(method: str) -> None:
    policy = DefaultMemoryPolicy()
    naive_now = NOW.replace(tzinfo=None)

    with pytest.raises(MemoryValidationError, match="clock must be timezone-aware"):
        if method == "write":
            policy.assess_write(make_candidate(), scope=USER_SCOPE, now=naive_now)
        else:
            policy.eligible(
                make_record(),
                scope=USER_SCOPE,
                allowed_kinds=ALL_KINDS,
                now=naive_now,
            )


def test_policy_contract_has_no_clock_or_dependency_owned_by_the_implementation() -> None:
    policy: MemoryPolicy = DefaultMemoryPolicy()

    assert policy.identity == "default-memory-policy-v1"


def test_decisions_reject_inconsistent_machine_codes() -> None:
    with pytest.raises(MemoryValidationError, match="inconsistent"):
        AdmissionDecision(
            action=AdmissionAction.ALLOW,
            reason=AdmissionReason.USER_CONFIRMATION_REQUIRED,
            effective_scope=USER_SCOPE,
            expires_at=None,
        )
    with pytest.raises(MemoryValidationError, match="inconsistent"):
        RecallEligibilityDecision(
            action=RecallEligibilityAction.ELIGIBLE,
            reason=RecallEligibilityReason.EXPIRED,
        )


def test_decisions_reject_raw_or_incomplete_contract_values() -> None:
    with pytest.raises(MemoryValidationError, match="AdmissionAction"):
        AdmissionDecision(  # type: ignore[arg-type]
            action="allow",
            reason=AdmissionReason.WRITE_ALLOWED,
            effective_scope=USER_SCOPE,
            expires_at=None,
        )
    with pytest.raises(MemoryValidationError, match="AdmissionReason"):
        AdmissionDecision(  # type: ignore[arg-type]
            action=AdmissionAction.ALLOW,
            reason="write_allowed",
            effective_scope=USER_SCOPE,
            expires_at=None,
        )
    with pytest.raises(MemoryValidationError, match="effective scope"):
        AdmissionDecision(
            action=AdmissionAction.ALLOW,
            reason=AdmissionReason.WRITE_ALLOWED,
            effective_scope=None,
            expires_at=None,
        )
    with pytest.raises(MemoryValidationError, match="no effective scope"):
        AdmissionDecision(
            action=AdmissionAction.DENY,
            reason=AdmissionReason.SCOPE_MISMATCH,
            effective_scope=USER_SCOPE,
            expires_at=None,
        )
    with pytest.raises(MemoryValidationError, match="RecallEligibilityAction"):
        RecallEligibilityDecision(  # type: ignore[arg-type]
            action="eligible",
            reason=RecallEligibilityReason.ELIGIBLE,
        )
    with pytest.raises(MemoryValidationError, match="RecallEligibilityReason"):
        RecallEligibilityDecision(  # type: ignore[arg-type]
            action=RecallEligibilityAction.ELIGIBLE,
            reason="eligible",
        )


def test_policy_fails_closed_for_structurally_invalid_inputs() -> None:
    policy = DefaultMemoryPolicy()

    with pytest.raises(TypeError, match="MemoryCandidate"):
        policy.assess_write(make_record(), scope=USER_SCOPE, now=NOW)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="candidate or record"):
        policy.eligible(  # type: ignore[arg-type]
            object(), scope=USER_SCOPE, allowed_kinds=ALL_KINDS, now=NOW
        )
    with pytest.raises(MemoryValidationError, match="MemoryScope"):
        policy.assess_write(  # type: ignore[arg-type]
            make_candidate(), scope="user-7", now=NOW
        )
    with pytest.raises(MemoryValidationError, match="frozenset"):
        policy.eligible(  # type: ignore[arg-type]
            make_record(), scope=USER_SCOPE, allowed_kinds={MemoryKind.PREFERENCE}, now=NOW
        )
