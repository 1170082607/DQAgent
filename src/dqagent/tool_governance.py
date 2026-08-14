"""Immutable prepared-action contracts and fail-closed governance decisions.

T3 stops at the authorization boundary.  It prepares effect identity, checks
non-overridable technical guards, and evaluates a small tri-state policy.  It
does not request approval, run hooks, invoke an executor, or start a process.
"""

from __future__ import annotations

import hashlib
import json
import math
import ntpath
import posixpath
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import InitVar, dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias

from dqagent.subprocesses import (
    IsolationCapability,
    normalize_isolation_capabilities,
)
from dqagent.workspace import (
    ResolvedWorkspacePath,
    Sanitizer,
    Workspace,
    WorkspaceAccessError,
    WorkspaceError,
    WorkspacePurpose,
    WorkspaceReason,
)

__all__ = [
    "ACTION_CANONICAL_VERSION",
    "CANONICAL_ACTION_VERSION",
    "ActionKind",
    "ActionPolicy",
    "ActionRecord",
    "DefaultActionPolicy",
    "EffectKind",
    "EffectPrecondition",
    "EffectPreconditions",
    "EffectState",
    "EffectiveLimits",
    "GovernanceDecision",
    "GuardContext",
    "GuardName",
    "GuardResult",
    "HARD_GUARD_ORDER",
    "PolicyDecision",
    "PolicyOutcome",
    "PreparedAction",
    "build_action_record",
    "evaluate_action",
    "evaluate_guards",
    "govern_action",
]


CANONICAL_ACTION_VERSION: Final[int] = 1
ACTION_CANONICAL_VERSION: Final[int] = CANONICAL_ACTION_VERSION
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_MAX_IDENTITY_CHARACTERS = 128
_MAX_DISPLAY_CHARACTERS = 4_096
_MAX_RECORD_DIAGNOSTICS = 8
_MAX_RECORD_TEXT_CHARACTERS = 512
_MAX_RECORD_GUARDS = 7
_MAX_REASON_CHARACTERS = 160
_DEFAULT_INPUT_CHARACTERS = 64_000
_DEFAULT_OUTPUT_CHARACTERS = 32_000
_DEFAULT_DURATION_SECONDS = 30.0
_DEFAULT_ARGV_ITEMS = 128
_DEFAULT_ARGV_CHARACTERS = 32_000
_LIMIT_FIELDS: Final[tuple[str, ...]] = (
    "max_input_characters",
    "max_output_characters",
    "max_duration_seconds",
    "max_argv_items",
    "max_argv_characters",
)
_NON_CANONICAL_FIELD_MARKERS: Final[tuple[str, ...]] = (
    "absolute",
    "credential",
    "display",
    "host_path",
    "password",
    "private_key",
    "repr",
    "root",
    "secret",
    "token",
)
_OBVIOUS_SECRET_VALUE_PATTERN = re.compile(
    r"(?ix)"
    r"(?<![a-z0-9])(?:"
    r"topsecret"
    r"|(?:secret|password|passwd|token|credential)(?:[-_][a-z0-9]+)+"
    r"|(?:api[-_]?key|access[-_]?token|auth[-_]?token|client[-_]?secret|"
    r"private[-_]?key|secret|password|passwd|token|credential)\s*[:=]\s*\S+"
    r")(?![a-z0-9])"
)
_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9:])(?:[A-Za-z]:[\\/]|\\\\)(?:[^\\/\s]+[\\/])*[^\\/\s]*"
    r"|(?<![A-Za-z0-9:])/(?:[^/\s]+/)*[^/\s]+"
)


CanonicalJsonValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["CanonicalJsonValue", ...]
    | Mapping[str, "CanonicalJsonValue"]
)


class ActionKind(StrEnum):
    """The provider-neutral operation being prepared."""

    READ = "read"
    SEARCH = "search"
    PATCH = "patch"
    COMMAND = "command"

    WORKSPACE_READ = "read"
    WORKSPACE_SEARCH = "search"
    WORKSPACE_PATCH = "patch"
    WORKSPACE_COMMAND = "command"


class EffectKind(StrEnum):
    """The effect class used by hard guards and the default policy."""

    NONE = "none"
    READ_ONLY = "none"
    WORKSPACE_MUTATION = "workspace_mutation"
    MUTATION = "workspace_mutation"
    PROCESS_EXECUTION = "process_execution"
    PROCESS = "process_execution"


class EffectState(StrEnum):
    """Observed effect state retained by a bounded action record."""

    NONE = "none"
    PARTIAL = "partial"
    COMPLETE = "complete"
    UNKNOWN = "unknown"


class GuardName(StrEnum):
    """Stable names for the fixed non-overridable guard sequence."""

    MAX_GOVERNED_CALLS = "max_governed_calls"
    WORKSPACE_IDENTITY = "workspace_identity"
    CURRENT_CONTAINMENT = "current_containment"
    PROTECTED_SECRET = "protected_secret"
    LIMITS = "limits"
    PRECONDITIONS = "preconditions"
    CAPABILITIES = "capabilities"


HARD_GUARD_ORDER: Final[tuple[GuardName, ...]] = (
    GuardName.MAX_GOVERNED_CALLS,
    GuardName.WORKSPACE_IDENTITY,
    GuardName.CURRENT_CONTAINMENT,
    GuardName.PROTECTED_SECRET,
    GuardName.LIMITS,
    GuardName.PRECONDITIONS,
    GuardName.CAPABILITIES,
)


class PolicyOutcome(StrEnum):
    """The only policy outcomes available at the T3 boundary."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


def _enum_value(value: object) -> str:
    if isinstance(value, StrEnum):
        return str(value.value)
    if isinstance(value, str):
        return value
    raise TypeError("value must be a string or StrEnum")


def _validate_text(
    label: str,
    value: object,
    *,
    maximum: int,
    allow_controls: bool = False,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    if len(value) > maximum:
        raise ValueError(f"{label} must not exceed {maximum} characters")
    if "\x00" in value:
        raise ValueError(f"{label} must not contain NUL characters")
    if not allow_controls and any(ord(character) < 32 for character in value):
        raise ValueError(f"{label} must not contain control characters")
    return value


def _validate_identity(label: str, value: object) -> str:
    identity = _validate_text(label, value, maximum=_MAX_IDENTITY_CHARACTERS)
    if "/" in identity or "\\" in identity or ":" in identity:
        raise ValueError(f"{label} must be an opaque identity, not a path")
    return identity


def _validate_positive_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_non_negative_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _validate_positive_float(label: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return 0.0 if normalized == 0.0 else normalized


def _validate_sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _normalize_secret_values(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("secret values must be an iterable of strings")
    try:
        candidates = tuple(values)
    except TypeError as error:
        raise TypeError("secret values must be an iterable of strings") from error
    normalized = tuple(
        _validate_text(
            f"secret value {index}",
            value,
            maximum=_MAX_DISPLAY_CHARACTERS,
        )
        for index, value in enumerate(candidates)
    )
    return tuple(sorted(set(normalized), key=lambda value: (-len(value), value)))


def _contains_absolute_path(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if ntpath.isabs(stripped) or posixpath.isabs(stripped):
        return True
    return _ABSOLUTE_PATH_PATTERN.search(value) is not None


def _validate_effect_identity_text(
    label: str,
    value: str,
    secret_values: tuple[str, ...],
    *,
    detect_obvious_secrets: bool = True,
) -> None:
    if _contains_absolute_path(value):
        raise ValueError(f"{label} must not contain an absolute host path")
    if any(secret in value for secret in secret_values):
        raise ValueError(f"{label} must be secret-free")
    if detect_obvious_secrets and _OBVIOUS_SECRET_VALUE_PATTERN.search(value):
        raise ValueError(f"{label} must be secret-free")


def _validate_effect_identity_value(
    value: CanonicalJsonValue,
    *,
    label: str,
    secret_values: tuple[str, ...],
) -> None:
    if isinstance(value, str):
        _validate_effect_identity_text(label, value, secret_values)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_effect_identity_text(f"{label} field name", key, secret_values)
            _validate_effect_identity_value(
                item,
                label=f"{label}.{key}",
                secret_values=secret_values,
            )
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_effect_identity_value(
                item,
                label=f"{label}[{index}]",
                secret_values=secret_values,
            )


def _normalize_logical_path(value: object, label: str) -> PurePosixPath:
    if isinstance(value, PurePosixPath):
        raw = str(value)
    elif isinstance(value, str):
        raw = value
    else:
        raise TypeError(f"{label} must be text or PurePosixPath")
    if not raw:
        raise ValueError(f"{label} must not be empty")
    if "\x00" in raw or "\\" in raw:
        raise ValueError(f"{label} must be a logical POSIX path")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw) or ":" in raw:
        raise ValueError(f"{label} must not be absolute")
    if raw == ".":
        return PurePosixPath(".")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} must not contain ambiguous or parent components")
    return PurePosixPath(*parts)


def _freeze_json(value: object, *, label: str) -> CanonicalJsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain non-finite numbers")
        return 0.0 if value == 0.0 else value
    if isinstance(value, (PurePosixPath,)):
        return str(_normalize_logical_path(value, label))
    if isinstance(value, Mapping):
        frozen: dict[str, CanonicalJsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{label} mapping keys must be non-empty text")
            lowered = key.casefold()
            if any(marker in lowered for marker in _NON_CANONICAL_FIELD_MARKERS):
                raise ValueError(
                    f"{label} must use secret-free, logical effect identity fields"
                )
            _validate_text(f"{label} field name", key, maximum=128)
            frozen[key] = _freeze_json(item, label=f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item, label=label) for item in value)
    raise TypeError(f"{label} must contain JSON-compatible values, not Python representations")


def _thaw_json(value: CanonicalJsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class EffectPrecondition:
    """A content-free expected state for one logical effect target."""

    logical_path: PurePosixPath
    expected_kind: str | None = None
    expected_sha256: str | None = None
    must_exist: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "logical_path",
            _normalize_logical_path(self.logical_path, "precondition logical path"),
        )
        if self.expected_kind is not None:
            kind = _enum_value(self.expected_kind)
            _validate_text("precondition expected kind", kind, maximum=64)
            object.__setattr__(self, "expected_kind", kind)
        if self.expected_sha256 is not None:
            object.__setattr__(
                self,
                "expected_sha256",
                _validate_sha256("precondition expected digest", self.expected_sha256),
            )
        if self.must_exist is not None and not isinstance(self.must_exist, bool):
            raise TypeError("precondition must_exist must be a boolean or None")
        if self.must_exist is False and self.expected_sha256 is not None:
            raise ValueError("a missing precondition cannot require a content digest")


@dataclass(frozen=True, slots=True)
class EffectPreconditions:
    """An immutable, deterministic collection of effect preconditions."""

    items: tuple[EffectPrecondition, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.items, (str, bytes)):
            raise TypeError("effect preconditions must be EffectPrecondition values")
        try:
            values = tuple(self.items)
        except TypeError as error:
            raise TypeError("effect preconditions must be iterable") from error
        if any(not isinstance(item, EffectPrecondition) for item in values):
            raise TypeError("effect preconditions must contain EffectPrecondition values")
        if len({item.logical_path for item in values}) != len(values):
            raise ValueError("effect precondition logical paths must be unique")
        object.__setattr__(
            self,
            "items",
            tuple(sorted(values, key=lambda item: str(item.logical_path))),
        )

    @property
    def values(self) -> tuple[EffectPrecondition, ...]:
        return self.items


@dataclass(frozen=True, slots=True)
class EffectiveLimits:
    """Per-action limits after model arguments have been tightened."""

    max_input_characters: int = _DEFAULT_INPUT_CHARACTERS
    max_output_characters: int = _DEFAULT_OUTPUT_CHARACTERS
    max_duration_seconds: float = _DEFAULT_DURATION_SECONDS
    max_argv_items: int = _DEFAULT_ARGV_ITEMS
    max_argv_characters: int = _DEFAULT_ARGV_CHARACTERS

    def __post_init__(self) -> None:
        for name in (
            "max_input_characters",
            "max_output_characters",
            "max_argv_items",
            "max_argv_characters",
        ):
            _validate_positive_int(name, getattr(self, name))
        object.__setattr__(
            self,
            "max_duration_seconds",
            _validate_positive_float("max_duration_seconds", self.max_duration_seconds),
        )


@dataclass(frozen=True, slots=True)
class PreparedAction:
    """Immutable effect identity produced after bounded validation.

    ``display_text`` is intentionally not part of the canonical payload.  It
    is untrusted presentation input and must be sanitized before recording or
    showing it.  Absolute roots and secret values are not accepted as action
    identity fields; callers provide logical paths and secret-free identities.
    """

    action_kind: ActionKind
    effect_kind: EffectKind
    workspace_id: str
    logical_targets: tuple[PurePosixPath, ...] = ()
    cwd: PurePosixPath | None = None
    argv: tuple[str, ...] = ()
    executable_identity: str | None = None
    environment_identity: tuple[str, ...] = ()
    normalized_fields: Mapping[str, CanonicalJsonValue] = field(default_factory=dict)
    preconditions: EffectPreconditions = field(default_factory=EffectPreconditions)
    required_capabilities: frozenset[IsolationCapability] = frozenset()
    limits: EffectiveLimits = field(default_factory=EffectiveLimits)
    display_text: str = ""
    secret_values: InitVar[Iterable[str]] = field(kw_only=True)

    def __post_init__(self, secret_values: Iterable[str]) -> None:
        if not isinstance(self.action_kind, ActionKind):
            raise TypeError("action kind must be an ActionKind")
        if not isinstance(self.effect_kind, EffectKind):
            raise TypeError("effect kind must be an EffectKind")
        expected_effect = {
            ActionKind.READ: EffectKind.NONE,
            ActionKind.SEARCH: EffectKind.NONE,
            ActionKind.PATCH: EffectKind.WORKSPACE_MUTATION,
            ActionKind.COMMAND: EffectKind.PROCESS_EXECUTION,
        }[self.action_kind]
        if self.effect_kind is not expected_effect:
            raise ValueError("action kind and effect kind are inconsistent")
        _validate_identity("workspace ID", self.workspace_id)
        normalized_secret_values = _normalize_secret_values(secret_values)
        _validate_effect_identity_text(
            "workspace ID",
            self.workspace_id,
            normalized_secret_values,
            detect_obvious_secrets=False,
        )

        if isinstance(self.logical_targets, (str, bytes)):
            raise TypeError("logical targets must be an iterable of logical paths")
        try:
            targets = tuple(
                _normalize_logical_path(item, "logical target") for item in self.logical_targets
            )
        except TypeError as error:
            raise TypeError("logical targets must be an iterable of logical paths") from error
        if (
            self.action_kind in {ActionKind.READ, ActionKind.SEARCH, ActionKind.PATCH}
            and not targets
        ):
            raise ValueError("workspace actions require at least one logical target")
        object.__setattr__(self, "logical_targets", targets)

        if self.cwd is not None:
            object.__setattr__(self, "cwd", _normalize_logical_path(self.cwd, "action cwd"))
        elif self.action_kind is ActionKind.COMMAND:
            object.__setattr__(self, "cwd", PurePosixPath("."))

        if isinstance(self.argv, (str, bytes)):
            raise TypeError("argv must be an iterable of argument strings")
        try:
            arguments = tuple(self.argv)
        except TypeError as error:
            raise TypeError("argv must be an iterable of argument strings") from error
        if any(not isinstance(argument, str) for argument in arguments):
            raise TypeError("argv must contain only strings")
        if self.action_kind is ActionKind.COMMAND and not arguments:
            raise ValueError("command actions require a direct argv")
        if arguments and not arguments[0]:
            raise ValueError("argv executable must not be empty")
        for index, argument in enumerate(arguments):
            _validate_effect_identity_text(
                f"argv[{index}]",
                argument,
                normalized_secret_values,
            )
        object.__setattr__(self, "argv", arguments)

        if self.executable_identity is not None:
            object.__setattr__(
                self,
                "executable_identity",
                _validate_identity("executable identity", self.executable_identity),
            )
            _validate_effect_identity_text(
                "executable identity",
                self.executable_identity,
                normalized_secret_values,
            )
        if isinstance(self.environment_identity, (str, bytes)):
            raise TypeError("environment identity must contain variable names")
        try:
            environment = tuple(self.environment_identity)
        except TypeError as error:
            raise TypeError("environment identity must contain variable names") from error
        for name in environment:
            if not isinstance(name, str) or not name or "=" in name or "\x00" in name:
                raise ValueError("environment identity must contain names, not values")
            _validate_effect_identity_text(
                "environment identity",
                name,
                normalized_secret_values,
                detect_obvious_secrets=False,
            )
        object.__setattr__(self, "environment_identity", tuple(sorted(set(environment))))

        if not isinstance(self.normalized_fields, Mapping):
            raise TypeError("normalized fields must be a mapping")
        frozen_fields = _freeze_json(self.normalized_fields, label="normalized fields")
        if not isinstance(frozen_fields, Mapping):
            raise TypeError("normalized fields must be a mapping")
        _validate_effect_identity_value(
            frozen_fields,
            label="normalized fields",
            secret_values=normalized_secret_values,
        )
        object.__setattr__(
            self,
            "normalized_fields",
            frozen_fields,
        )

        if not isinstance(self.preconditions, EffectPreconditions):
            raise TypeError("preconditions must be EffectPreconditions")
        object.__setattr__(
            self,
            "required_capabilities",
            normalize_isolation_capabilities(self.required_capabilities),
        )
        if not isinstance(self.limits, EffectiveLimits):
            raise TypeError("action limits must be EffectiveLimits")
        if not isinstance(self.display_text, str):
            raise TypeError("display text must be text")
        if len(self.display_text) > _MAX_DISPLAY_CHARACTERS or "\x00" in self.display_text:
            raise ValueError("display text exceeds its bound or contains NUL")

    @property
    def effect_fields(self) -> Mapping[str, CanonicalJsonValue]:
        """Alias emphasizing that normalized fields determine the effect."""

        return self.normalized_fields

    @property
    def canonical_payload(self) -> Mapping[str, object]:
        """Return the versioned content-free identity payload."""

        return {
            "action_kind": self.action_kind.value,
            "argv": list(self.argv),
            "canonical_version": CANONICAL_ACTION_VERSION,
            "cwd": str(self.cwd) if self.cwd is not None else None,
            "effect_kind": self.effect_kind.value,
            "effective_limits": {
                field_name: getattr(self.limits, field_name) for field_name in _LIMIT_FIELDS
            },
            "environment_identity": list(self.environment_identity),
            "executable_identity": self.executable_identity,
            "logical_targets": [str(path) for path in self.logical_targets],
            "normalized_fields": _thaw_json(self.normalized_fields),
            "preconditions": [
                {
                    "expected_kind": item.expected_kind,
                    "expected_sha256": item.expected_sha256,
                    "logical_path": str(item.logical_path),
                    "must_exist": item.must_exist,
                }
                for item in self.preconditions.items
            ],
            "required_capabilities": sorted(
                capability.value for capability in self.required_capabilities
            ),
            "workspace_id": self.workspace_id,
        }

    @property
    def canonical_json(self) -> str:
        """Return sorted-key UTF-8 JSON without Python representation fallbacks."""

        return json.dumps(
            self.canonical_payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return self.canonical_json.encode("utf-8")

    @property
    def canonical_digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def digest(self) -> str:
        """Short alias used by exact-action consumers."""

        return self.canonical_digest


@dataclass(frozen=True, slots=True)
class GuardContext:
    """Trusted current state supplied by application composition."""

    workspace: Workspace
    configured_limits: EffectiveLimits = field(default_factory=EffectiveLimits)
    available_capabilities: frozenset[IsolationCapability] = frozenset()
    governed_call_count: int = 0
    max_governed_calls: int = 1
    current_preconditions: EffectPreconditions | None = None
    backend_identity: str = "unavailable"

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Workspace):
            raise TypeError("guard context requires a Workspace")
        if not isinstance(self.configured_limits, EffectiveLimits):
            raise TypeError("guard context limits must be EffectiveLimits")
        object.__setattr__(
            self,
            "available_capabilities",
            normalize_isolation_capabilities(self.available_capabilities),
        )
        _validate_non_negative_int("governed call count", self.governed_call_count)
        _validate_positive_int("max governed calls", self.max_governed_calls)
        if self.current_preconditions is not None and not isinstance(
            self.current_preconditions, EffectPreconditions
        ):
            raise TypeError("current preconditions must be EffectPreconditions or None")
        object.__setattr__(
            self,
            "backend_identity",
            _validate_identity("backend identity", self.backend_identity),
        )


@dataclass(frozen=True, slots=True)
class GuardResult:
    """One result from the fixed non-overridable hard-guard sequence."""

    name: GuardName
    passed: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, GuardName):
            raise TypeError("guard result name must be a GuardName")
        if not isinstance(self.passed, bool):
            raise TypeError("guard result passed must be a boolean")
        _validate_text("guard result reason", self.reason, maximum=_MAX_REASON_CHARACTERS)

    @property
    def allowed(self) -> bool:
        return self.passed

    @property
    def non_overridable(self) -> bool:
        return True

    @property
    def overridable(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """A stable tri-state response from an application-owned action policy."""

    outcome: PolicyOutcome
    reason: str
    policy_identity: str = "unknown-policy"

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, PolicyOutcome):
            raise TypeError("policy outcome must be a PolicyOutcome")
        _validate_text("policy reason", self.reason, maximum=_MAX_REASON_CHARACTERS)
        object.__setattr__(
            self,
            "policy_identity",
            _validate_identity("policy identity", self.policy_identity),
        )

    @property
    def allowed(self) -> bool:
        return self.outcome is PolicyOutcome.ALLOW

    @property
    def requires_approval(self) -> bool:
        return self.outcome is PolicyOutcome.REQUIRE_APPROVAL


class ActionPolicy(Protocol):
    """Minimal provider-neutral policy contract supplied by trusted composition."""

    @property
    def identity(self) -> str:
        """Stable identity for the composed policy implementation."""

    def decide(self, action: PreparedAction) -> PolicyDecision:
        """Return allow, deny, or require approval for a guarded action."""


@dataclass(frozen=True, slots=True)
class DefaultActionPolicy:
    """Conservative policy for the initial coding actions."""

    identity: str = "default-action-policy-v1"

    def __post_init__(self) -> None:
        _validate_identity("policy identity", self.identity)

    def decide(self, action: PreparedAction) -> PolicyDecision:
        if not isinstance(action, PreparedAction):
            raise TypeError("action policy requires a PreparedAction")
        if action.effect_kind is EffectKind.NONE:
            return PolicyDecision(
                PolicyOutcome.ALLOW,
                "read_only_action",
                self.identity,
            )
        if action.effect_kind is EffectKind.WORKSPACE_MUTATION:
            return PolicyDecision(
                PolicyOutcome.REQUIRE_APPROVAL,
                "workspace_mutation_requires_approval",
                self.identity,
            )
        if action.effect_kind is EffectKind.PROCESS_EXECUTION:
            return PolicyDecision(
                PolicyOutcome.REQUIRE_APPROVAL,
                "process_execution_requires_approval",
                self.identity,
            )
        return PolicyDecision(PolicyOutcome.DENY, "unknown_effect_kind", self.identity)


@dataclass(frozen=True, slots=True)
class GovernanceDecision:
    """The T3 result before approval, hooks, or execution."""

    action: PreparedAction
    guard_results: tuple[GuardResult, ...]
    policy_decision: PolicyDecision
    policy_evaluated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.action, PreparedAction):
            raise TypeError("governance decision action must be a PreparedAction")
        if not isinstance(self.guard_results, tuple) or any(
            not isinstance(result, GuardResult) for result in self.guard_results
        ):
            raise TypeError("governance guard results must be a tuple of GuardResult")
        if len(self.guard_results) > len(HARD_GUARD_ORDER):
            raise ValueError("governance guard results exceed the fixed guard bound")
        if not isinstance(self.policy_decision, PolicyDecision):
            raise TypeError("governance policy decision must be a PolicyDecision")
        if not isinstance(self.policy_evaluated, bool):
            raise TypeError("policy_evaluated must be a boolean")
        if (
            not all(result.passed for result in self.guard_results)
            and self.policy_decision.outcome is not PolicyOutcome.DENY
        ):
            raise ValueError("a hard guard failure cannot be overridden by policy")

    @property
    def guards_passed(self) -> bool:
        return len(self.guard_results) == len(HARD_GUARD_ORDER) and all(
            result.passed for result in self.guard_results
        )

    @property
    def admitted(self) -> bool:
        return self.guards_passed and self.policy_decision.allowed

    @property
    def requires_approval(self) -> bool:
        return self.guards_passed and self.policy_decision.requires_approval


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """Bounded, immutable, sanitized evidence for one governance attempt."""

    canonical_version: int
    action_digest: str
    action_kind: ActionKind
    effect_kind: EffectKind
    workspace_id: str
    guard_results: tuple[GuardResult, ...]
    policy_decision: PolicyDecision
    executor_attempts: int = 0
    effect_state: EffectState = EffectState.NONE
    backend_identity: str = "unavailable"
    backend_capabilities: tuple[IsolationCapability, ...] = ()
    sanitized_action_display: str = ""
    sanitized_diagnostics: tuple[str, ...] = ()
    action_display_truncated: bool = False
    diagnostics_truncated: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.canonical_version, bool)
            or not isinstance(self.canonical_version, int)
            or self.canonical_version != CANONICAL_ACTION_VERSION
        ):
            raise ValueError("action record canonical version is unsupported")
        _validate_sha256("action record digest", self.action_digest)
        if not isinstance(self.action_kind, ActionKind):
            raise TypeError("action record action kind must be an ActionKind")
        if not isinstance(self.effect_kind, EffectKind):
            raise TypeError("action record effect kind must be an EffectKind")
        _validate_identity("action record workspace ID", self.workspace_id)
        if not isinstance(self.guard_results, tuple) or any(
            not isinstance(result, GuardResult) for result in self.guard_results
        ):
            raise TypeError("action record guard results must be a tuple of GuardResult")
        if len(self.guard_results) > _MAX_RECORD_GUARDS:
            raise ValueError("action record guard results exceed the record bound")
        if not isinstance(self.policy_decision, PolicyDecision):
            raise TypeError("action record policy decision must be a PolicyDecision")
        if (
            isinstance(self.executor_attempts, bool)
            or not isinstance(self.executor_attempts, int)
            or self.executor_attempts not in {0, 1}
        ):
            raise ValueError("executor attempts must be zero or one")
        if not isinstance(self.effect_state, EffectState):
            raise TypeError("action record effect state must be an EffectState")
        if self.executor_attempts == 0 and self.effect_state is not EffectState.NONE:
            raise ValueError("an unattempted action must have none effect state")
        _validate_identity("action record backend identity", self.backend_identity)
        object.__setattr__(
            self,
            "backend_capabilities",
            tuple(
                sorted(
                    normalize_isolation_capabilities(self.backend_capabilities),
                    key=lambda capability: capability.value,
                )
            ),
        )
        _validate_text(
            "sanitized action display",
            self.sanitized_action_display or "_",
            maximum=_MAX_RECORD_TEXT_CHARACTERS,
            allow_controls=True,
        )
        if not isinstance(self.sanitized_diagnostics, tuple):
            raise TypeError("sanitized diagnostics must be a tuple of strings")
        diagnostics = self.sanitized_diagnostics
        if len(diagnostics) > _MAX_RECORD_DIAGNOSTICS:
            raise ValueError("sanitized diagnostics exceed the record bound")
        object.__setattr__(self, "sanitized_diagnostics", diagnostics)
        for diagnostic in diagnostics:
            _validate_text(
                "sanitized diagnostic",
                diagnostic,
                maximum=_MAX_RECORD_TEXT_CHARACTERS,
                allow_controls=True,
            )
        if not isinstance(self.action_display_truncated, bool) or not isinstance(
            self.diagnostics_truncated, bool
        ):
            raise TypeError("record truncation flags must be booleans")

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Compatibility alias for the already-sanitized diagnostics."""

        return self.sanitized_diagnostics
    @classmethod
    def from_governance(
        cls,
        decision: GovernanceDecision,
        *,
        workspace: Workspace,
        secret_values: Iterable[str] = (),
        backend_identity: str | None = None,
        backend_capabilities: Iterable[IsolationCapability] = (),
        executor_attempts: int = 0,
        effect_state: EffectState = EffectState.NONE,
        diagnostics: Iterable[str] = (),
        max_text_characters: int = _MAX_RECORD_TEXT_CHARACTERS,
    ) -> ActionRecord:
        """Build a bounded record while keeping raw display data out of it."""

        if not isinstance(decision, GovernanceDecision):
            raise TypeError("action record requires a GovernanceDecision")
        if not isinstance(workspace, Workspace):
            raise TypeError("action record requires a Workspace")
        _validate_positive_int("record text limit", max_text_characters)
        max_text_characters = min(max_text_characters, _MAX_RECORD_TEXT_CHARACTERS)

        sanitizer: Sanitizer | None
        try:
            sanitizer = workspace.sanitizer(secrets=secret_values)
        except Exception:
            sanitizer = None

        def sanitize(value: object) -> tuple[str, bool]:
            if not isinstance(value, str):
                return "[unavailable]", True
            if sanitizer is None:
                return "[unavailable]", True
            try:
                rendered = sanitizer.sanitize_with_evidence(
                    value,
                    max_characters=max_text_characters,
                )
            except Exception:
                return "[unavailable]", True
            return rendered.text, rendered.truncated

        display, display_truncated = sanitize(decision.action.display_text)
        bounded_diagnostics, diagnostics_truncated = _collect_bounded_diagnostics(diagnostics)
        rendered_diagnostics: list[str] = []
        for diagnostic in bounded_diagnostics:
            rendered, truncated = sanitize(diagnostic)
            rendered_diagnostics.append(rendered)
            diagnostics_truncated = diagnostics_truncated or truncated

        rendered_guards = tuple(
            GuardResult(result.name, result.passed, sanitize(result.reason)[0])
            for result in decision.guard_results
        )
        rendered_policy = PolicyDecision(
            decision.policy_decision.outcome,
            sanitize(decision.policy_decision.reason)[0],
            sanitize(decision.policy_decision.policy_identity)[0],
        )
        try:
            capabilities = normalize_isolation_capabilities(backend_capabilities)
        except Exception:
            capabilities = frozenset()
            rendered_diagnostics.append("[backend_capabilities_unavailable]")
            diagnostics_truncated = True

        return cls(
            canonical_version=CANONICAL_ACTION_VERSION,
            action_digest=decision.action.canonical_digest,
            action_kind=decision.action.action_kind,
            effect_kind=decision.action.effect_kind,
            workspace_id=decision.action.workspace_id,
            guard_results=rendered_guards,
            policy_decision=rendered_policy,
            executor_attempts=executor_attempts,
            effect_state=effect_state,
            backend_identity=sanitize(backend_identity or "unavailable")[0],
            backend_capabilities=tuple(capabilities),
            sanitized_action_display=display,
            sanitized_diagnostics=tuple(rendered_diagnostics[:_MAX_RECORD_DIAGNOSTICS]),
            action_display_truncated=display_truncated,
            diagnostics_truncated=diagnostics_truncated
            or len(rendered_diagnostics) > _MAX_RECORD_DIAGNOSTICS,
        )


def _collect_bounded_diagnostics(
    diagnostics: Iterable[str],
) -> tuple[tuple[object, ...], bool]:
    try:
        iterator = iter(diagnostics)
    except Exception:
        return ("[unavailable]",), True

    values: list[object] = []
    for _ in range(_MAX_RECORD_DIAGNOSTICS + 1):
        try:
            values.append(next(iterator))
        except StopIteration:
            return tuple(values), False
        except Exception:
            retained = values[: max(0, _MAX_RECORD_DIAGNOSTICS - 1)]
            retained.append("[unavailable]")
            return tuple(retained), True
    return tuple(values[:_MAX_RECORD_DIAGNOSTICS]), True


def _action_paths(action: PreparedAction) -> tuple[tuple[PurePosixPath, WorkspacePurpose], ...]:
    purpose = {
        ActionKind.READ: WorkspacePurpose.READ,
        ActionKind.SEARCH: WorkspacePurpose.SEARCH,
        ActionKind.PATCH: WorkspacePurpose.PATCH,
        ActionKind.COMMAND: WorkspacePurpose.READ,
    }[action.action_kind]
    paths = [(path, purpose) for path in action.logical_targets]
    paths.extend(
        (precondition.logical_path, purpose)
        for precondition in action.preconditions.items
    )
    if action.cwd is not None:
        paths.append((action.cwd, WorkspacePurpose.COMMAND_CWD))
    return tuple(paths)


def _passed(name: GuardName, reason: str) -> GuardResult:
    return GuardResult(name, True, reason)


def _failed(name: GuardName, reason: str) -> GuardResult:
    return GuardResult(name, False, reason)


def _guard_max_governed_calls(action: PreparedAction, context: GuardContext) -> GuardResult:
    del action
    try:
        count = context.governed_call_count
        maximum = context.max_governed_calls
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or count < 0
            or maximum <= 0
        ):
            return _failed(GuardName.MAX_GOVERNED_CALLS, "malformed_call_capacity")
        if count >= maximum:
            return _failed(GuardName.MAX_GOVERNED_CALLS, "max_governed_calls_exhausted")
        return _passed(GuardName.MAX_GOVERNED_CALLS, "capacity_available")
    except Exception:
        return _failed(GuardName.MAX_GOVERNED_CALLS, "capacity_dependency_failure")


def _guard_workspace_identity(action: PreparedAction, context: GuardContext) -> GuardResult:
    try:
        workspace_id = context.workspace.scope.workspace_id
        if not isinstance(workspace_id, str):
            return _failed(GuardName.WORKSPACE_IDENTITY, "malformed_workspace_identity")
        if action.workspace_id != workspace_id:
            return _failed(GuardName.WORKSPACE_IDENTITY, "workspace_identity_mismatch")
        return _passed(GuardName.WORKSPACE_IDENTITY, "workspace_identity_matches")
    except Exception:
        return _failed(GuardName.WORKSPACE_IDENTITY, "workspace_identity_dependency_failure")


def _guard_current_containment(action: PreparedAction, context: GuardContext) -> GuardResult:
    try:
        for logical_path, purpose in _action_paths(action):
            try:
                if logical_path == PurePosixPath("."):
                    if purpose is not WorkspacePurpose.COMMAND_CWD:
                        return _failed(GuardName.CURRENT_CONTAINMENT, "root_target_not_allowed")
                    resolved = context.workspace.resolve_root()
                else:
                    resolved = context.workspace.resolve(
                        logical_path,
                        purpose=purpose,
                    )
            except WorkspaceAccessError as error:
                if str(error.reason_code) in {
                    WorkspaceReason.PROTECTED.value,
                    WorkspaceReason.SECRET.value,
                }:
                    continue
                return _failed(
                    GuardName.CURRENT_CONTAINMENT,
                    "current_containment_denied",
                )
            except WorkspaceError:
                return _failed(GuardName.CURRENT_CONTAINMENT, "current_containment_denied")
            if not isinstance(resolved, ResolvedWorkspacePath):
                return _failed(GuardName.CURRENT_CONTAINMENT, "malformed_containment_response")
        return _passed(GuardName.CURRENT_CONTAINMENT, "current_containment_verified")
    except Exception:
        return _failed(GuardName.CURRENT_CONTAINMENT, "containment_dependency_failure")


def _guard_protected_secret(action: PreparedAction, context: GuardContext) -> GuardResult:
    try:
        for logical_path, _purpose in _action_paths(action):
            protected = context.workspace.is_protected(logical_path)
            secret = context.workspace.is_secret(logical_path)
            if not isinstance(protected, bool) or not isinstance(secret, bool):
                return _failed(GuardName.PROTECTED_SECRET, "malformed_protection_response")
            if protected:
                return _failed(GuardName.PROTECTED_SECRET, "protected_resource_denied")
            if secret:
                return _failed(GuardName.PROTECTED_SECRET, "secret_resource_denied")
        return _passed(GuardName.PROTECTED_SECRET, "protected_and_secret_checks_passed")
    except Exception:
        return _failed(GuardName.PROTECTED_SECRET, "protection_dependency_failure")


def _guard_limits(action: PreparedAction, context: GuardContext) -> GuardResult:
    try:
        for field_name in _LIMIT_FIELDS:
            action_value = getattr(action.limits, field_name)
            configured_value = getattr(context.configured_limits, field_name)
            if isinstance(action_value, bool) or isinstance(configured_value, bool):
                return _failed(GuardName.LIMITS, "malformed_limit_value")
            if not isinstance(action_value, (int, float)) or not isinstance(
                configured_value, (int, float)
            ):
                return _failed(GuardName.LIMITS, "malformed_limit_value")
            if action_value > configured_value:
                return _failed(GuardName.LIMITS, f"{field_name}_exceeds_configured")
        return _passed(GuardName.LIMITS, "effective_limits_within_configured_limits")
    except Exception:
        return _failed(GuardName.LIMITS, "limits_dependency_failure")


def _guard_preconditions(action: PreparedAction, context: GuardContext) -> GuardResult:
    try:
        if not action.preconditions.items:
            return _passed(GuardName.PRECONDITIONS, "no_preconditions_required")
        current = context.current_preconditions
        if current is None:
            return _failed(GuardName.PRECONDITIONS, "preconditions_unavailable")
        if not isinstance(current, EffectPreconditions):
            return _failed(GuardName.PRECONDITIONS, "malformed_precondition_response")
        if current != action.preconditions:
            return _failed(GuardName.PRECONDITIONS, "precondition_conflict")
        return _passed(GuardName.PRECONDITIONS, "preconditions_match")
    except Exception:
        return _failed(GuardName.PRECONDITIONS, "precondition_dependency_failure")


def _guard_capabilities(action: PreparedAction, context: GuardContext) -> GuardResult:
    try:
        available = context.available_capabilities
        if not isinstance(available, frozenset) or any(
            not isinstance(capability, IsolationCapability) for capability in available
        ):
            return _failed(GuardName.CAPABILITIES, "malformed_capability_response")
        missing = action.required_capabilities.difference(available)
        if missing:
            return _failed(GuardName.CAPABILITIES, "required_capability_missing")
        return _passed(GuardName.CAPABILITIES, "required_capabilities_present")
    except Exception:
        return _failed(GuardName.CAPABILITIES, "capability_dependency_failure")


_GUARD_FUNCTIONS: Final[tuple[Callable[[PreparedAction, GuardContext], GuardResult], ...]] = (
    _guard_max_governed_calls,
    _guard_workspace_identity,
    _guard_current_containment,
    _guard_protected_secret,
    _guard_limits,
    _guard_preconditions,
    _guard_capabilities,
)


def evaluate_guards(
    action: PreparedAction,
    context: GuardContext,
) -> tuple[GuardResult, ...]:
    """Run hard guards in order and stop at the first fail-closed result."""

    if not isinstance(action, PreparedAction):
        raise TypeError("guard evaluation requires a PreparedAction")
    if not isinstance(context, GuardContext):
        raise TypeError("guard evaluation requires a GuardContext")
    results: list[GuardResult] = []
    for expected_name, guard in zip(HARD_GUARD_ORDER, _GUARD_FUNCTIONS, strict=True):
        try:
            result = guard(action, context)
        except Exception:
            result = _failed(expected_name, "guard_dependency_failure")
        if not isinstance(result, GuardResult) or result.name is not expected_name:
            result = _failed(expected_name, "malformed_guard_response")
        results.append(result)
        if not result.passed:
            break
    return tuple(results)


def _policy_failure(reason: str) -> PolicyDecision:
    return PolicyDecision(PolicyOutcome.DENY, reason, "policy-failure")


def _evaluate_policy(policy: ActionPolicy, action: PreparedAction) -> PolicyDecision:
    try:
        identity = policy.identity
        if not isinstance(identity, str) or not identity.strip():
            return _policy_failure("policy_dependency_failure")
        identity = _validate_identity("policy identity", identity)
        decide = policy.decide
        if not callable(decide):
            return _policy_failure("policy_dependency_failure")
        response = decide(action)
        if not isinstance(response, PolicyDecision):
            return _policy_failure("malformed_policy_response")
        return response
    except Exception:
        return _policy_failure("policy_dependency_failure")


def evaluate_action(
    action: PreparedAction,
    context: GuardContext,
    policy: ActionPolicy | None = None,
) -> GovernanceDecision:
    """Evaluate hard guards and policy without approval or effect execution."""

    guard_results = evaluate_guards(action, context)
    if not guard_results or not all(result.passed for result in guard_results):
        return GovernanceDecision(
            action=action,
            guard_results=guard_results,
            policy_decision=PolicyDecision(
                PolicyOutcome.DENY,
                "hard_guard_failed",
                "hard-guards",
            ),
            policy_evaluated=False,
        )
    if policy is None:
        selected_policy: ActionPolicy = DefaultActionPolicy()
    else:
        selected_policy = policy
    policy_decision = _evaluate_policy(selected_policy, action)
    return GovernanceDecision(
        action=action,
        guard_results=guard_results,
        policy_decision=policy_decision,
        policy_evaluated=True,
    )


def govern_action(
    action: PreparedAction,
    context: GuardContext,
    policy: ActionPolicy | None = None,
) -> GovernanceDecision:
    """Alias for the fixed T3 governance boundary."""

    return evaluate_action(action, context, policy)


def build_action_record(
    decision: GovernanceDecision,
    *,
    workspace: Workspace,
    secret_values: Iterable[str] = (),
    backend_identity: str | None = None,
    backend_capabilities: Iterable[IsolationCapability] = (),
    executor_attempts: int = 0,
    effect_state: EffectState = EffectState.NONE,
    diagnostics: Iterable[str] = (),
    max_text_characters: int = _MAX_RECORD_TEXT_CHARACTERS,
) -> ActionRecord:
    """Functional wrapper for the record construction boundary."""

    return ActionRecord.from_governance(
        decision,
        workspace=workspace,
        secret_values=secret_values,
        backend_identity=backend_identity,
        backend_capabilities=backend_capabilities,
        executor_attempts=executor_attempts,
        effect_state=effect_state,
        diagnostics=diagnostics,
        max_text_characters=max_text_characters,
    )
