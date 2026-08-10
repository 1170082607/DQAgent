"""Deterministic admission and pre-ranking eligibility for long-term memory."""

from datetime import datetime

from dqagent.errors import MemoryValidationError
from dqagent.memory import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionReason,
    MemoryCandidate,
    MemoryKind,
    MemoryLifecycleStatus,
    MemoryPolicy,
    MemoryRecord,
    MemoryScope,
    MemoryScopeKind,
    MemorySensitivity,
    RecallEligibilityAction,
    RecallEligibilityDecision,
    RecallEligibilityReason,
)

__all__ = [
    "AdmissionAction",
    "AdmissionDecision",
    "AdmissionReason",
    "DefaultMemoryPolicy",
    "MemoryPolicy",
    "RecallEligibilityAction",
    "RecallEligibilityDecision",
    "RecallEligibilityReason",
]


class DefaultMemoryPolicy:
    """Conservative v1 policy with no external dependencies or implicit clock."""

    identity = "default-memory-policy-v1"

    def assess_write(
        self,
        candidate: MemoryCandidate,
        *,
        scope: MemoryScope,
        now: datetime,
    ) -> AdmissionDecision:
        if not isinstance(candidate, MemoryCandidate):
            raise TypeError("memory write assessment requires a MemoryCandidate")
        _validate_scope(scope)
        _validate_now(now)

        if candidate.scope != scope:
            return _deny_admission(AdmissionReason.SCOPE_MISMATCH)
        if candidate.sensitivity is MemorySensitivity.SECRET:
            return _deny_admission(AdmissionReason.SECRET_CONTENT_NOT_ALLOWED)
        if candidate.sensitivity is MemorySensitivity.SENSITIVE:
            return _deny_admission(AdmissionReason.SENSITIVE_CONTENT_NOT_ALLOWED)
        if not _kind_allowed_in_scope(candidate.kind, scope.kind):
            return _deny_admission(AdmissionReason.KIND_NOT_ALLOWED)
        if candidate.provenance.extracted_at > now:
            return _deny_admission(AdmissionReason.PROVENANCE_IN_FUTURE)
        if candidate.expires_at is not None and candidate.expires_at <= now:
            return _deny_admission(AdmissionReason.CANDIDATE_EXPIRED)

        # Extractor confidence is evidence only; every valid value follows the same consent path.
        return AdmissionDecision(
            action=AdmissionAction.REQUIRE_CONFIRMATION,
            reason=AdmissionReason.USER_CONFIRMATION_REQUIRED,
            effective_scope=candidate.scope,
            expires_at=candidate.expires_at,
        )

    def eligible(
        self,
        record: MemoryCandidate | MemoryRecord,
        *,
        scope: MemoryScope,
        allowed_kinds: frozenset[MemoryKind],
        now: datetime,
    ) -> RecallEligibilityDecision:
        if not isinstance(record, (MemoryCandidate, MemoryRecord)):
            raise TypeError("memory recall eligibility requires a candidate or record")
        _validate_scope(scope)
        _validate_allowed_kinds(allowed_kinds)
        _validate_now(now)

        if record.scope != scope:
            return _ineligible(RecallEligibilityReason.SCOPE_MISMATCH)
        if not isinstance(record, MemoryRecord):
            return _ineligible(RecallEligibilityReason.NOT_CONFIRMED_DURABLE_RECORD)
        if record.status is MemoryLifecycleStatus.SUPERSEDED:
            return _ineligible(RecallEligibilityReason.SUPERSEDED)
        if record.status is MemoryLifecycleStatus.EXPIRED:
            return _ineligible(RecallEligibilityReason.EXPIRED_STATUS)
        if record.valid_from > now:
            return _ineligible(RecallEligibilityReason.NOT_YET_VALID)
        if record.expires_at is not None and record.expires_at <= now:
            return _ineligible(RecallEligibilityReason.EXPIRED)
        if record.sensitivity is MemorySensitivity.SECRET:
            return _ineligible(RecallEligibilityReason.SECRET_CONTENT_NOT_ALLOWED)
        if record.sensitivity is MemorySensitivity.SENSITIVE:
            return _ineligible(RecallEligibilityReason.SENSITIVE_CONTENT_NOT_ALLOWED)
        if not _kind_allowed_in_scope(record.kind, scope.kind) or record.kind not in allowed_kinds:
            return _ineligible(RecallEligibilityReason.KIND_NOT_ALLOWED)
        return RecallEligibilityDecision(
            action=RecallEligibilityAction.ELIGIBLE,
            reason=RecallEligibilityReason.ELIGIBLE,
        )


def _kind_allowed_in_scope(kind: MemoryKind, scope_kind: MemoryScopeKind) -> bool:
    if scope_kind is MemoryScopeKind.USER:
        return kind in {MemoryKind.PREFERENCE, MemoryKind.USER_FACT}
    return kind is MemoryKind.EXPERIENCE


def _deny_admission(reason: AdmissionReason) -> AdmissionDecision:
    return AdmissionDecision(
        action=AdmissionAction.DENY,
        reason=reason,
        effective_scope=None,
        expires_at=None,
    )


def _ineligible(reason: RecallEligibilityReason) -> RecallEligibilityDecision:
    return RecallEligibilityDecision(
        action=RecallEligibilityAction.INELIGIBLE,
        reason=reason,
    )


def _validate_scope(scope: MemoryScope) -> None:
    if not isinstance(scope, MemoryScope):
        raise MemoryValidationError("memory policy scope must be a MemoryScope")


def _validate_allowed_kinds(allowed_kinds: frozenset[MemoryKind]) -> None:
    if not isinstance(allowed_kinds, frozenset) or any(
        not isinstance(kind, MemoryKind) for kind in allowed_kinds
    ):
        raise MemoryValidationError(
            "memory recall allowed kinds must be a frozenset of MemoryKind values"
        )


def _validate_now(now: datetime, *, label: str = "memory policy clock") -> None:
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise MemoryValidationError(f"{label} must be timezone-aware")
