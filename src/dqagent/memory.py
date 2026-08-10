"""Immutable domain values for policy-governed long-term memory."""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from dqagent.errors import MemoryValidationError

MEMORY_SCHEMA_VERSION = 1
_MAX_ID_CHARACTERS = 256
_MAX_CONTENT_CHARACTERS = 4_000
_TOPIC_PATTERN = re.compile(r"[a-z0-9][a-z0-9._/-]{0,127}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class MemoryScopeKind(StrEnum):
    USER = "user"
    PROJECT = "project"


class MemoryKind(StrEnum):
    PREFERENCE = "preference"
    USER_FACT = "user_fact"
    EXPERIENCE = "experience"


class MemorySensitivity(StrEnum):
    NON_SENSITIVE = "non_sensitive"
    SENSITIVE = "sensitive"
    SECRET = "secret"


class MemoryLifecycleStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


class MemorySourceType(StrEnum):
    USER_DRAFT = "user_draft"
    COMMITTED_SESSION_TURN = "committed_session_turn"


class MemoryForgetReason(StrEnum):
    USER_REQUEST = "user_request"
    RETENTION = "retention"


@dataclass(frozen=True, slots=True)
class MemoryScope:
    """An application-supplied owner boundary; a session is never a memory scope."""

    kind: MemoryScopeKind
    scope_id: str

    def __post_init__(self) -> None:
        _require_instance("memory scope kind", self.kind, MemoryScopeKind)
        _validate_id("memory scope ID", self.scope_id)


@dataclass(frozen=True, slots=True)
class MemoryConfidence:
    """Extractor certainty only; it is neither truth nor user confirmation."""

    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise MemoryValidationError("memory confidence must be a number")
        try:
            normalized = float(self.value)
        except OverflowError as error:
            raise MemoryValidationError(
                "memory confidence must be finite and between zero and one"
            ) from error
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise MemoryValidationError("memory confidence must be finite and between zero and one")
        object.__setattr__(self, "value", 0.0 if normalized == 0.0 else normalized)


@dataclass(frozen=True, slots=True)
class MemoryProvenance:
    """A compact source reference, not copied transcript content."""

    source_type: MemorySourceType
    source_item_digest: str
    extractor_identity: str
    extracted_at: datetime
    source_id: str | None = None
    source_revision: int | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        _require_instance("memory source type", self.source_type, MemorySourceType)
        _validate_sha256("source item digest", self.source_item_digest)
        _validate_text("extractor identity", self.extractor_identity, _MAX_ID_CHARACTERS)
        _validate_aware_datetime("extraction timestamp", self.extracted_at)
        if self.source_id is not None:
            _validate_id("memory source ID", self.source_id)
        if self.run_id is not None:
            _validate_id("memory source run ID", self.run_id)
        if self.source_revision is not None:
            _validate_positive_integer("memory source revision", self.source_revision)
        if self.source_type is MemorySourceType.COMMITTED_SESSION_TURN:
            if self.source_id is None or self.source_revision is None:
                raise MemoryValidationError(
                    "committed session provenance requires a source ID and revision"
                )
        elif self.source_revision is not None:
            raise MemoryValidationError("user draft provenance cannot contain a session revision")


@dataclass(frozen=True, slots=True)
class MemoryConfirmation:
    """Explicit confirmation bound to the exact candidate shown to the user."""

    candidate_digest: str
    confirmed_at: datetime

    def __post_init__(self) -> None:
        _validate_sha256("candidate digest", self.candidate_digest)
        _validate_aware_datetime("confirmation timestamp", self.confirmed_at)


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """A transient proposal with no durable identity, revision, or lifecycle state."""

    scope: MemoryScope
    kind: MemoryKind
    topic: str
    content: str
    confidence: MemoryConfidence
    sensitivity: MemorySensitivity
    provenance: MemoryProvenance
    valid_from: datetime
    expires_at: datetime | None = None
    schema_version: int = MEMORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        _require_instance("memory candidate scope", self.scope, MemoryScope)
        _require_instance("memory kind", self.kind, MemoryKind)
        _validate_topic(self.topic)
        _validate_text("memory content", self.content, _MAX_CONTENT_CHARACTERS)
        _require_instance("memory confidence", self.confidence, MemoryConfidence)
        _require_instance("memory sensitivity", self.sensitivity, MemorySensitivity)
        _require_instance("memory provenance", self.provenance, MemoryProvenance)
        _validate_validity(self.valid_from, self.expires_at)

    @property
    def canonical_json(self) -> str:
        """Return the versioned representation used for confirmation binding."""

        payload = {
            "schema_version": self.schema_version,
            "scope": {"kind": self.scope.kind.value, "scope_id": self.scope.scope_id},
            "kind": self.kind.value,
            "topic": self.topic,
            "content": self.content,
            "confidence": self.confidence.value,
            "sensitivity": self.sensitivity.value,
            "provenance": {
                "source_type": self.provenance.source_type.value,
                "source_item_digest": self.provenance.source_item_digest,
                "extractor_identity": self.provenance.extractor_identity,
                "extracted_at": _canonical_datetime(self.provenance.extracted_at),
                "source_id": self.provenance.source_id,
                "source_revision": self.provenance.source_revision,
                "run_id": self.provenance.run_id,
            },
            "valid_from": _canonical_datetime(self.valid_from),
            "expires_at": (
                _canonical_datetime(self.expires_at) if self.expires_at is not None else None
            ),
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """A confirmed durable proposition; pending and forgotten are not record states."""

    memory_id: str
    revision: int
    scope: MemoryScope
    kind: MemoryKind
    topic: str
    content: str
    confidence: MemoryConfidence
    sensitivity: MemorySensitivity
    provenance: MemoryProvenance
    confirmation: MemoryConfirmation
    status: MemoryLifecycleStatus
    valid_from: datetime
    expires_at: datetime | None
    supersedes_id: str | None
    created_at: datetime
    updated_at: datetime
    schema_version: int = MEMORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        _validate_id("memory ID", self.memory_id)
        _validate_positive_integer("memory revision", self.revision)
        _require_instance("memory record scope", self.scope, MemoryScope)
        _require_instance("memory kind", self.kind, MemoryKind)
        _validate_topic(self.topic)
        _validate_text("memory content", self.content, _MAX_CONTENT_CHARACTERS)
        _require_instance("memory confidence", self.confidence, MemoryConfidence)
        _require_instance("memory sensitivity", self.sensitivity, MemorySensitivity)
        _require_instance("memory provenance", self.provenance, MemoryProvenance)
        _require_instance("memory confirmation", self.confirmation, MemoryConfirmation)
        _require_instance("memory lifecycle status", self.status, MemoryLifecycleStatus)
        _validate_validity(self.valid_from, self.expires_at)
        if self.supersedes_id is not None:
            _validate_id("superseded memory ID", self.supersedes_id)
            if self.supersedes_id == self.memory_id:
                raise MemoryValidationError("a memory record cannot supersede itself")
        _validate_aware_datetime("memory creation timestamp", self.created_at)
        _validate_aware_datetime("memory update timestamp", self.updated_at)
        if self.created_at > self.updated_at:
            raise MemoryValidationError("memory creation timestamp must not follow its update")
        if self.provenance.extracted_at > self.confirmation.confirmed_at:
            raise MemoryValidationError("memory confirmation cannot precede extraction")
        if self.confirmation.confirmed_at > self.created_at:
            raise MemoryValidationError("memory creation cannot precede confirmation")
        if self.confirmation.candidate_digest != self._candidate_digest():
            raise MemoryValidationError("memory confirmation digest does not match the record")
        if self.status is MemoryLifecycleStatus.EXPIRED and (
            self.expires_at is None or self.expires_at > self.updated_at
        ):
            raise MemoryValidationError(
                "expired memory records require an elapsed expiry timestamp"
            )

    @classmethod
    def from_candidate(
        cls,
        candidate: MemoryCandidate,
        *,
        memory_id: str,
        revision: int,
        confirmation: MemoryConfirmation,
        created_at: datetime,
        updated_at: datetime | None = None,
        supersedes_id: str | None = None,
    ) -> "MemoryRecord":
        if not isinstance(candidate, MemoryCandidate):
            raise TypeError("memory record creation requires a MemoryCandidate")
        if not isinstance(confirmation, MemoryConfirmation):
            raise TypeError("memory record creation requires a MemoryConfirmation")
        return cls(
            memory_id=memory_id,
            revision=revision,
            scope=candidate.scope,
            kind=candidate.kind,
            topic=candidate.topic,
            content=candidate.content,
            confidence=candidate.confidence,
            sensitivity=candidate.sensitivity,
            provenance=candidate.provenance,
            confirmation=confirmation,
            status=MemoryLifecycleStatus.ACTIVE,
            valid_from=candidate.valid_from,
            expires_at=candidate.expires_at,
            supersedes_id=supersedes_id,
            created_at=created_at,
            updated_at=updated_at or created_at,
            schema_version=candidate.schema_version,
        )

    def _candidate_digest(self) -> str:
        return MemoryCandidate(
            scope=self.scope,
            kind=self.kind,
            topic=self.topic,
            content=self.content,
            confidence=self.confidence,
            sensitivity=self.sensitivity,
            provenance=self.provenance,
            valid_from=self.valid_from,
            expires_at=self.expires_at,
            schema_version=self.schema_version,
        ).digest


class AdmissionAction(StrEnum):
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DENY = "deny"


class AdmissionReason(StrEnum):
    WRITE_ALLOWED = "write_allowed"
    USER_CONFIRMATION_REQUIRED = "user_confirmation_required"
    SCOPE_MISMATCH = "scope_mismatch"
    KIND_NOT_ALLOWED = "kind_not_allowed"
    SENSITIVE_CONTENT_NOT_ALLOWED = "sensitive_content_not_allowed"
    SECRET_CONTENT_NOT_ALLOWED = "secret_content_not_allowed"
    PROVENANCE_IN_FUTURE = "provenance_in_future"
    CANDIDATE_EXPIRED = "candidate_expired"


_DENY_ADMISSION_REASONS = frozenset(
    {
        AdmissionReason.SCOPE_MISMATCH,
        AdmissionReason.KIND_NOT_ALLOWED,
        AdmissionReason.SENSITIVE_CONTENT_NOT_ALLOWED,
        AdmissionReason.SECRET_CONTENT_NOT_ALLOWED,
        AdmissionReason.PROVENANCE_IN_FUTURE,
        AdmissionReason.CANDIDATE_EXPIRED,
    }
)


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Stable machine-readable result of assessing one transient candidate."""

    action: AdmissionAction
    reason: AdmissionReason
    effective_scope: MemoryScope | None
    expires_at: datetime | None

    def __post_init__(self) -> None:
        if not isinstance(self.action, AdmissionAction):
            raise MemoryValidationError("memory admission action must be an AdmissionAction")
        if not isinstance(self.reason, AdmissionReason):
            raise MemoryValidationError("memory admission reason must be an AdmissionReason")
        if self.action is AdmissionAction.ALLOW:
            valid_reason = self.reason is AdmissionReason.WRITE_ALLOWED
        elif self.action is AdmissionAction.REQUIRE_CONFIRMATION:
            valid_reason = self.reason is AdmissionReason.USER_CONFIRMATION_REQUIRED
        else:
            valid_reason = self.reason in _DENY_ADMISSION_REASONS
        if not valid_reason:
            raise MemoryValidationError("memory admission action and reason are inconsistent")

        if self.action is AdmissionAction.DENY:
            if self.effective_scope is not None or self.expires_at is not None:
                raise MemoryValidationError(
                    "denied memory admission has no effective scope or expiry"
                )
            return
        if not isinstance(self.effective_scope, MemoryScope):
            raise MemoryValidationError("admitted memory requires an effective scope")
        if self.expires_at is not None:
            _validate_aware_datetime("memory admission expiry", self.expires_at)

    @property
    def requires_confirmation(self) -> bool:
        return self.action is AdmissionAction.REQUIRE_CONFIRMATION


class RecallEligibilityAction(StrEnum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


class RecallEligibilityReason(StrEnum):
    ELIGIBLE = "eligible"
    SCOPE_MISMATCH = "scope_mismatch"
    NOT_CONFIRMED_DURABLE_RECORD = "not_confirmed_durable_record"
    SUPERSEDED = "superseded"
    EXPIRED_STATUS = "expired_status"
    NOT_YET_VALID = "not_yet_valid"
    EXPIRED = "expired"
    SENSITIVE_CONTENT_NOT_ALLOWED = "sensitive_content_not_allowed"
    SECRET_CONTENT_NOT_ALLOWED = "secret_content_not_allowed"
    KIND_NOT_ALLOWED = "kind_not_allowed"


@dataclass(frozen=True, slots=True)
class RecallEligibilityDecision:
    """Stable pre-ranking eligibility result for one candidate or durable record."""

    action: RecallEligibilityAction
    reason: RecallEligibilityReason

    def __post_init__(self) -> None:
        if not isinstance(self.action, RecallEligibilityAction):
            raise MemoryValidationError(
                "memory recall eligibility action must be a RecallEligibilityAction"
            )
        if not isinstance(self.reason, RecallEligibilityReason):
            raise MemoryValidationError(
                "memory recall eligibility reason must be a RecallEligibilityReason"
            )
        is_eligible = self.action is RecallEligibilityAction.ELIGIBLE
        if is_eligible != (self.reason is RecallEligibilityReason.ELIGIBLE):
            raise MemoryValidationError(
                "memory recall eligibility action and reason are inconsistent"
            )

    @property
    def eligible(self) -> bool:
        return self.action is RecallEligibilityAction.ELIGIBLE


class MemoryPolicy(Protocol):
    """Provider-neutral authority for memory admission and pre-ranking eligibility."""

    identity: str

    def assess_write(
        self,
        candidate: MemoryCandidate,
        *,
        scope: MemoryScope,
        now: datetime,
    ) -> AdmissionDecision: ...

    def eligible(
        self,
        record: MemoryCandidate | MemoryRecord,
        *,
        scope: MemoryScope,
        allowed_kinds: frozenset[MemoryKind],
        now: datetime,
    ) -> RecallEligibilityDecision: ...


@dataclass(frozen=True, slots=True)
class MemoryTombstone:
    """Minimal content-free evidence that a durable memory was forgotten."""

    memory_id: str
    revision: int
    scope: MemoryScope
    forgotten_at: datetime
    reason: MemoryForgetReason
    schema_version: int = MEMORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        _validate_id("forgotten memory ID", self.memory_id)
        _validate_positive_integer("forgotten memory revision", self.revision)
        _require_instance("forgotten memory scope", self.scope, MemoryScope)
        _validate_aware_datetime("memory forgotten timestamp", self.forgotten_at)
        _require_instance("memory forget reason", self.reason, MemoryForgetReason)

    @classmethod
    def from_record(
        cls,
        record: MemoryRecord,
        *,
        forgotten_at: datetime,
        reason: MemoryForgetReason,
    ) -> "MemoryTombstone":
        if not isinstance(record, MemoryRecord):
            raise TypeError("memory forgetting requires a MemoryRecord")
        _validate_aware_datetime("memory forgotten timestamp", forgotten_at)
        if forgotten_at < record.updated_at:
            raise MemoryValidationError("memory cannot be forgotten before its latest update")
        return cls(
            memory_id=record.memory_id,
            revision=record.revision,
            scope=record.scope,
            forgotten_at=forgotten_at,
            reason=reason,
            schema_version=record.schema_version,
        )


def _validate_schema_version(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != MEMORY_SCHEMA_VERSION:
        raise MemoryValidationError(f"memory schema version must be {MEMORY_SCHEMA_VERSION}")


def _validate_positive_integer(label: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MemoryValidationError(f"{label} must be a positive integer")


def _validate_id(label: str, value: str) -> None:
    _validate_text(label, value, _MAX_ID_CHARACTERS)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise MemoryValidationError(f"{label} must not contain control characters")


def _validate_topic(value: str) -> None:
    if not isinstance(value, str) or _TOPIC_PATTERN.fullmatch(value) is None:
        raise MemoryValidationError(
            "memory topic must be a lowercase semantic key using letters, digits, "
            "'.', '/', '_' or '-'"
        )


def _validate_text(label: str, value: str, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError(f"{label} must not be empty")
    if value != value.strip():
        raise MemoryValidationError(f"{label} must not contain surrounding whitespace")
    if len(value) > maximum:
        raise MemoryValidationError(f"{label} must not exceed {maximum} characters")
    if "\x00" in value:
        raise MemoryValidationError(f"{label} must not contain NUL characters")


def _validate_sha256(label: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise MemoryValidationError(f"{label} must be a lowercase SHA-256 digest")


def _validate_aware_datetime(label: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise MemoryValidationError(f"{label} must be timezone-aware")


def _validate_validity(valid_from: datetime, expires_at: datetime | None) -> None:
    _validate_aware_datetime("memory valid-from timestamp", valid_from)
    if expires_at is not None:
        _validate_aware_datetime("memory expiry timestamp", expires_at)
        if expires_at <= valid_from:
            raise MemoryValidationError(
                "memory expiry timestamp must follow its valid-from timestamp"
            )


def _canonical_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _require_instance(label: str, value: object, expected: type[object]) -> None:
    if not isinstance(value, expected):
        raise MemoryValidationError(f"{label} must be a {expected.__name__}")
