"""Immutable prepared-action, approval, and hook governance contracts.

T3 prepares effect identity, checks non-overridable technical guards, and
evaluates a small tri-state policy.  T4 adds exact foreground approval and
trusted synchronous hook contracts, but still does not invoke an executor or
start a process.
"""

from __future__ import annotations

import hashlib
import json
import math
import ntpath
import posixpath
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import InitVar, dataclass, field, replace
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias, cast

from dqagent.errors import RunCancelledError, RunDeadlineExceededError
from dqagent.execution import RunContext
from dqagent.subprocesses import (
    IsolationCapability,
    normalize_isolation_capabilities,
)
from dqagent.workspace import (
    ResolvedWorkspacePath,
    Sanitizer,
    Workspace,
    WorkspaceAccessError,
    WorkspaceDriftError,
    WorkspaceError,
    WorkspacePurpose,
    WorkspaceReason,
)

__all__ = [
    "ACTION_CANONICAL_VERSION",
    "ActionHookInput",
    "CANONICAL_ACTION_VERSION",
    "ActionKind",
    "ActionExecutionResult",
    "ActionPolicy",
    "ActionRecord",
    "ApprovalDecision",
    "ApprovalOutcome",
    "ApprovalProvider",
    "ApprovalRequest",
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
    "HookMode",
    "HookOutcome",
    "HookRunResult",
    "HookStage",
    "HookResult",
    "NonInteractiveApprovalProvider",
    "PostActionHook",
    "PostActionHookInput",
    "PreActionHook",
    "PreActionHookInput",
    "PreActionHookSpec",
    "PolicyDecision",
    "PolicyOutcome",
    "PreparedAction",
    "ScriptedApprovalProvider",
    "authorize_action",
    "build_action_record",
    "build_approval_request",
    "build_post_action_hook_input",
    "build_pre_action_hook_input",
    "evaluate_action",
    "evaluate_guards",
    "govern_action",
    "obtain_approval",
    "request_approval",
    "run_post_hooks",
    "run_pre_hooks",
]


CANONICAL_ACTION_VERSION: Final[int] = 1
ACTION_CANONICAL_VERSION: Final[int] = CANONICAL_ACTION_VERSION
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_MAX_IDENTITY_CHARACTERS = 128
_MAX_DISPLAY_CHARACTERS = 4_096
_MAX_RECORD_DIAGNOSTICS = 8
_MAX_RECORD_TEXT_CHARACTERS = 512
_MAX_RECORD_GUARDS = 7
_MAX_RECORD_HOOK_RESULTS = 33
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


@dataclass(frozen=True, slots=True)
class ActionExecutionResult:
    """Bounded executor output separated from the model-visible observation."""

    output: str
    effect_state: EffectState = EffectState.COMPLETE
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.output, str) or not self.output.strip():
            raise ValueError("action execution output must be non-empty text")
        if not isinstance(self.effect_state, EffectState):
            raise TypeError("action execution effect state must be an EffectState")
        if not isinstance(self.diagnostics, tuple):
            raise TypeError("action execution diagnostics must be a tuple of strings")
        if len(self.diagnostics) > _MAX_RECORD_DIAGNOSTICS:
            raise ValueError("action execution diagnostics exceed the record bound")
        if any(not isinstance(item, str) for item in self.diagnostics):
            raise TypeError("action execution diagnostics must contain strings")


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
    approval_outcome: ApprovalOutcome | None = None
    approval_reason: str | None = None
    pre_hook_results: tuple[HookResult, ...] = ()
    post_hook_results: tuple[HookResult, ...] = ()
    observation_failure: bool = False

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
        if self.approval_outcome is not None and not isinstance(
            self.approval_outcome, ApprovalOutcome
        ):
            raise TypeError("approval outcome must be an ApprovalOutcome or None")
        if self.approval_reason is not None:
            _validate_text(
                "approval reason",
                self.approval_reason,
                maximum=_MAX_RECORD_TEXT_CHARACTERS,
                allow_controls=True,
            )
        for label, values in (
            ("pre-hook results", self.pre_hook_results),
            ("post-hook results", self.post_hook_results),
        ):
            if not isinstance(values, tuple) or len(values) > _MAX_RECORD_HOOK_RESULTS:
                raise ValueError(f"{label} exceed the record bound")
            if any(not isinstance(value, HookResult) for value in values):
                raise TypeError(f"{label} must contain HookResult values")
        if not isinstance(self.observation_failure, bool):
            raise TypeError("observation failure must be a boolean")

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
        approval_decision: ApprovalDecision | None = None,
        pre_hook_results: tuple[HookResult, ...] = (),
        post_hook_results: tuple[HookResult, ...] = (),
        observation_failure: bool = False,
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
        approval_outcome: ApprovalOutcome | None = None
        approval_reason: str | None = None
        if approval_decision is not None:
            if not isinstance(approval_decision, ApprovalDecision):
                raise TypeError("approval decision must be an ApprovalDecision")
            approval_outcome = approval_decision.outcome
            approval_reason = sanitize(approval_decision.reason)[0]
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
            approval_outcome=approval_outcome,
            approval_reason=approval_reason,
            pre_hook_results=pre_hook_results,
            post_hook_results=post_hook_results,
            observation_failure=observation_failure,
        )


class ApprovalOutcome(StrEnum):
    """Outcomes at the exact foreground approval boundary."""

    APPROVE = "approve"
    APPROVED = "approve"
    REJECT = "reject"
    UNAVAILABLE = "unavailable"
    MALFORMED = "malformed"
    IDENTITY_MISMATCH = "identity_mismatch"
    DRIFT = "drift"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class HookStage(StrEnum):
    """The two synchronous hook stages owned by governance."""

    PRE = "pre"
    POST = "post"


class HookMode(StrEnum):
    """Whether a pre-hook failure is allowed to leave the action runnable."""

    REQUIRED = "required"
    OPTIONAL = "optional"


class HookOutcome(StrEnum):
    """A bounded hook result; it is separate from the action effect state."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    MALFORMED = "malformed"
    CANCELLED = "cancelled"


_MAX_T4_SECRET_VALUES = 64
_MAX_APPROVAL_TEXT_CHARACTERS = 512
_MAX_APPROVAL_PRECONDITIONS = 32
_MAX_APPROVAL_CAPABILITIES = len(IsolationCapability)
_MAX_SCRIPTED_APPROVAL_RESPONSES = 64
_MAX_HOOKS = 32
_MAX_HOOK_DIAGNOSTICS = 4
_MAX_HOOK_TEXT_CHARACTERS = 512


def _normalize_t4_secret_values(values: Iterable[str]) -> tuple[str, ...]:
    """Bound secret-context consumption before passing it to a sanitizer."""

    if isinstance(values, (str, bytes)):
        raise TypeError("secret values must be an iterable of strings")
    try:
        iterator = iter(values)
    except TypeError as error:
        raise TypeError("secret values must be an iterable of strings") from error
    bounded: list[str] = []
    for _ in range(_MAX_T4_SECRET_VALUES + 1):
        try:
            bounded.append(next(iterator))
        except StopIteration:
            break
    if len(bounded) > _MAX_T4_SECRET_VALUES:
        raise ValueError("secret values exceed the approval and hook bound")
    return _normalize_secret_values(tuple(bounded))


def _validate_sanitized_text(
    label: str,
    value: object,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    """Validate outward text after redaction, including terminal-control safety."""

    if allow_empty and value == "":
        return ""
    text = _validate_text(label, value, maximum=maximum, allow_controls=True)
    if any(ord(character) < 32 and character not in "\t\r\n" for character in text):
        raise ValueError(f"{label} must not contain terminal control characters")
    return text


def _safe_sanitize_t4_text(
    workspace: Workspace,
    value: object,
    *,
    secret_values: tuple[str, ...],
    maximum: int,
    fallback: str = "[unavailable]",
) -> tuple[str, bool]:
    """Sanitize before exposing a bounded approval or hook string."""

    if not isinstance(value, str):
        return fallback, True
    try:
        rendered = workspace.sanitizer(secrets=secret_values).sanitize_with_evidence(
            value,
            max_characters=maximum,
        )
        text = _validate_sanitized_text(
            "sanitized text",
            rendered.text,
            maximum=maximum,
            allow_empty=True,
        )
        if _contains_absolute_path(text):
            return fallback, True
        if text:
            return text, False
        return fallback, rendered.truncated or bool(fallback)
    except Exception:
        return fallback, True


def _workspace_scope_binding(workspace: Workspace) -> str:
    """Return a non-sensitive binding for one immutable workspace scope."""

    scope = workspace.scope
    payload = {
        "workspace_id": scope.workspace_id,
        "root": str(scope.root),
        "limits": {
            field_name: getattr(scope.limits, field_name)
            for field_name in (
                "max_logical_path_characters",
                "max_logical_path_segments",
                "max_snapshot_entries",
                "max_snapshot_bytes",
                "max_snapshot_elapsed_seconds",
                "max_rendered_diff_characters",
                "max_snapshot_file_bytes",
            )
        },
        "protected_paths": [str(path) for path in scope.protected_paths],
        "secret_paths": [str(path) for path in scope.secret_paths],
        "volatile_paths": [str(path) for path in scope.volatile_paths],
        "ignored_paths": [str(path) for path in scope.ignored_paths],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _workspace_scopes_match(left: Workspace, right: Workspace) -> bool:
    return left.scope == right.scope


def _sanitize_approval_request_for_provider(
    request: ApprovalRequest,
    *,
    workspace: Workspace,
    secret_values: tuple[str, ...],
) -> ApprovalRequest | None:
    """Re-project even direct-constructed requests before crossing to a provider."""

    policy_reason, reason_failed = _safe_sanitize_t4_text(
        workspace,
        request.policy_reason,
        secret_values=secret_values,
        maximum=_MAX_APPROVAL_TEXT_CHARACTERS,
    )
    action_display, display_failed = _safe_sanitize_t4_text(
        workspace,
        request.sanitized_action_display,
        secret_values=secret_values,
        maximum=_MAX_APPROVAL_TEXT_CHARACTERS,
    )
    if reason_failed or display_failed:
        return None
    if _contains_absolute_path(policy_reason) or _contains_absolute_path(action_display):
        return None
    for identity in (
        request.run_id,
        request.workspace_id,
        request.policy_identity,
        request.backend_identity,
    ):
        if _contains_absolute_path(identity) or any(
            secret in identity for secret in secret_values
        ):
            return None
    try:
        return replace(
            request,
            policy_reason=policy_reason,
            sanitized_action_display=action_display,
            sanitization_failed=False,
        )
    except Exception:
        return None


def _normalize_bounded_preconditions(
    value: EffectPreconditions,
    *,
    label: str,
) -> EffectPreconditions:
    if not isinstance(value, EffectPreconditions):
        raise TypeError(f"{label} must be EffectPreconditions")
    if len(value.items) > _MAX_APPROVAL_PRECONDITIONS:
        raise ValueError(f"{label} exceed the approval bound")
    for item in value.items:
        if len(str(item.logical_path)) > _MAX_APPROVAL_TEXT_CHARACTERS:
            raise ValueError(f"{label} logical path exceeds the approval bound")
    return value


def _normalize_bounded_capabilities(
    value: Iterable[IsolationCapability],
    *,
    label: str,
) -> tuple[IsolationCapability, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must contain IsolationCapability values")
    try:
        iterator = iter(value)
    except Exception as error:
        raise TypeError(f"{label} must contain IsolationCapability values") from error
    normalized: set[IsolationCapability] = set()
    for _ in range(_MAX_APPROVAL_CAPABILITIES + 1):
        try:
            capability = next(iterator)
        except StopIteration:
            return tuple(sorted(normalized, key=lambda item: item.value))
        except Exception as error:
            raise TypeError(f"{label} must contain IsolationCapability values") from error
        if not isinstance(capability, IsolationCapability):
            raise TypeError(f"{label} must contain IsolationCapability values")
        normalized.add(capability)
    raise ValueError(f"{label} exceed the approval bound")


def _validate_optional_identity(label: str, value: object) -> str | None:
    if value is None:
        return None
    return _validate_identity(label, value)


def _validate_optional_sanitized_text(
    label: str,
    value: object,
    *,
    maximum: int,
) -> str:
    if value is None:
        return ""
    return _validate_sanitized_text(label, value, maximum=maximum, allow_empty=True)


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """Exact, bounded, sanitized data shown to one foreground approver."""

    run_id: str
    workspace_id: str
    action_kind: ActionKind
    effect_kind: EffectKind
    action_digest: str
    policy_reason: str
    sanitized_action_display: str
    policy_identity: str = "unknown-policy"
    preconditions: EffectPreconditions = field(default_factory=EffectPreconditions)
    required_capabilities: tuple[IsolationCapability, ...] = ()
    available_capabilities: tuple[IsolationCapability, ...] = ()
    backend_identity: str = "unavailable"
    canonical_version: int = CANONICAL_ACTION_VERSION
    sanitization_failed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_version, int) or isinstance(
            self.canonical_version, bool
        ):
            raise TypeError("approval request canonical version must be an integer")
        if self.canonical_version != CANONICAL_ACTION_VERSION:
            raise ValueError("approval request canonical version is unsupported")
        object.__setattr__(self, "run_id", _validate_identity("approval run ID", self.run_id))
        object.__setattr__(
            self,
            "workspace_id",
            _validate_identity("approval workspace ID", self.workspace_id),
        )
        if not isinstance(self.action_kind, ActionKind):
            raise TypeError("approval request action kind must be an ActionKind")
        if not isinstance(self.effect_kind, EffectKind):
            raise TypeError("approval request effect kind must be an EffectKind")
        expected_effect = {
            ActionKind.READ: EffectKind.NONE,
            ActionKind.SEARCH: EffectKind.NONE,
            ActionKind.PATCH: EffectKind.WORKSPACE_MUTATION,
            ActionKind.COMMAND: EffectKind.PROCESS_EXECUTION,
        }[self.action_kind]
        if self.effect_kind is not expected_effect:
            raise ValueError("approval action and effect kinds are inconsistent")
        object.__setattr__(
            self,
            "action_digest",
            _validate_sha256("approval action digest", self.action_digest),
        )
        object.__setattr__(
            self,
            "policy_reason",
            _validate_sanitized_text(
                "approval policy reason",
                self.policy_reason,
                maximum=_MAX_APPROVAL_TEXT_CHARACTERS,
            ),
        )
        object.__setattr__(
            self,
            "sanitized_action_display",
            _validate_sanitized_text(
                "approval action display",
                self.sanitized_action_display,
                maximum=_MAX_APPROVAL_TEXT_CHARACTERS,
            ),
        )
        object.__setattr__(
            self,
            "policy_identity",
            _validate_identity("approval policy identity", self.policy_identity),
        )
        object.__setattr__(
            self,
            "preconditions",
            _normalize_bounded_preconditions(self.preconditions, label="approval preconditions"),
        )
        object.__setattr__(
            self,
            "required_capabilities",
            _normalize_bounded_capabilities(
                self.required_capabilities,
                label="approval required capabilities",
            ),
        )
        object.__setattr__(
            self,
            "available_capabilities",
            _normalize_bounded_capabilities(
                self.available_capabilities,
                label="approval available capabilities",
            ),
        )
        object.__setattr__(
            self,
            "backend_identity",
            _validate_identity("approval backend identity", self.backend_identity),
        )
        if not isinstance(self.sanitization_failed, bool):
            raise TypeError("approval request sanitization flag must be a boolean")

    @classmethod
    def from_governance(
        cls,
        decision: GovernanceDecision,
        context: RunContext,
        guard_context: GuardContext,
        *,
        secret_values: Iterable[str] = (),
        max_text_characters: int = _MAX_APPROVAL_TEXT_CHARACTERS,
    ) -> ApprovalRequest:
        """Project a guarded decision into a provider-safe approval request."""

        if not isinstance(decision, GovernanceDecision):
            raise TypeError("approval request requires a GovernanceDecision")
        if not isinstance(context, RunContext):
            raise TypeError("approval request requires a RunContext")
        if not isinstance(guard_context, GuardContext):
            raise TypeError("approval request requires a GuardContext")
        _validate_positive_int("approval text limit", max_text_characters)
        bounded_limit = min(max_text_characters, _MAX_APPROVAL_TEXT_CHARACTERS)
        context.check_active()
        secrets = _normalize_t4_secret_values(secret_values)
        workspace = guard_context.workspace
        display, display_failed = _safe_sanitize_t4_text(
            workspace,
            decision.action.display_text or "[action display unavailable]",
            secret_values=secrets,
            maximum=bounded_limit,
            fallback="[action display unavailable]",
        )
        reason, reason_failed = _safe_sanitize_t4_text(
            workspace,
            decision.policy_decision.reason,
            secret_values=secrets,
            maximum=bounded_limit,
        )
        policy_identity = decision.policy_decision.policy_identity
        try:
            policy_identity = _validate_identity("approval policy identity", policy_identity)
        except Exception:
            policy_identity = "unknown-policy"
            reason_failed = True
        backend_identity = guard_context.backend_identity
        try:
            backend_identity = _validate_identity("approval backend identity", backend_identity)
        except Exception:
            backend_identity = "unavailable"
            reason_failed = True
        return cls(
            run_id=context.run_id,
            workspace_id=decision.action.workspace_id,
            action_kind=decision.action.action_kind,
            effect_kind=decision.action.effect_kind,
            action_digest=decision.action.canonical_digest,
            policy_reason=reason,
            sanitized_action_display=display,
            policy_identity=policy_identity,
            preconditions=decision.action.preconditions,
            required_capabilities=tuple(decision.action.required_capabilities),
            available_capabilities=tuple(guard_context.available_capabilities),
            backend_identity=backend_identity,
            sanitization_failed=display_failed or reason_failed,
        )

    @property
    def action_display(self) -> str:
        """Compatibility alias for callers that use the shorter display name."""

        return self.sanitized_action_display

    @property
    def digest(self) -> str:
        return self.action_digest


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """Immutable provider response or fail-closed boundary classification."""

    outcome: ApprovalOutcome
    reason: str = "approval_unspecified"
    run_id: str | None = None
    workspace_id: str | None = None
    action_digest: str | None = None
    preconditions: EffectPreconditions | None = None
    required_capabilities: tuple[IsolationCapability, ...] | None = None
    available_capabilities: tuple[IsolationCapability, ...] | None = None
    backend_identity: str | None = None
    provider_identity: str = "unknown-approval-provider"
    sanitized_message: str = ""
    sanitization_failed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, ApprovalOutcome):
            raise TypeError("approval outcome must be an ApprovalOutcome")
        object.__setattr__(
            self,
            "reason",
            _validate_sanitized_text(
                "approval reason",
                self.reason,
                maximum=_MAX_APPROVAL_TEXT_CHARACTERS,
            ),
        )
        object.__setattr__(
            self,
            "run_id",
            _validate_optional_identity("approval run ID", self.run_id),
        )
        object.__setattr__(
            self,
            "workspace_id",
            _validate_optional_identity("approval workspace ID", self.workspace_id),
        )
        if self.action_digest is not None:
            object.__setattr__(
                self,
                "action_digest",
                _validate_sha256("approval action digest", self.action_digest),
            )
        if self.preconditions is not None:
            object.__setattr__(
                self,
                "preconditions",
                _normalize_bounded_preconditions(
                    self.preconditions,
                    label="approval decision preconditions",
                ),
            )
        if self.required_capabilities is not None:
            object.__setattr__(
                self,
                "required_capabilities",
                _normalize_bounded_capabilities(
                    self.required_capabilities,
                    label="approval decision required capabilities",
                ),
            )
        if self.available_capabilities is not None:
            object.__setattr__(
                self,
                "available_capabilities",
                _normalize_bounded_capabilities(
                    self.available_capabilities,
                    label="approval decision available capabilities",
                ),
            )
        object.__setattr__(
            self,
            "backend_identity",
            _validate_optional_identity("approval backend identity", self.backend_identity),
        )
        object.__setattr__(
            self,
            "provider_identity",
            _validate_identity("approval provider identity", self.provider_identity),
        )
        object.__setattr__(
            self,
            "sanitized_message",
            _validate_optional_sanitized_text(
                "approval message",
                self.sanitized_message,
                maximum=_MAX_APPROVAL_TEXT_CHARACTERS,
            ),
        )
        if not isinstance(self.sanitization_failed, bool):
            raise TypeError("approval decision sanitization flag must be a boolean")

    @classmethod
    def approve_for(
        cls,
        request: ApprovalRequest,
        *,
        reason: str = "approval_granted",
        provider_identity: str = "scripted-approval-provider",
    ) -> ApprovalDecision:
        """Create an exact approval bound to every effect-relevant request field."""

        if not isinstance(request, ApprovalRequest):
            raise TypeError("approval decision requires an ApprovalRequest")
        return cls(
            outcome=ApprovalOutcome.APPROVE,
            reason=reason,
            run_id=request.run_id,
            workspace_id=request.workspace_id,
            action_digest=request.action_digest,
            preconditions=request.preconditions,
            required_capabilities=request.required_capabilities,
            available_capabilities=request.available_capabilities,
            backend_identity=request.backend_identity,
            provider_identity=provider_identity,
        )

    @property
    def approved(self) -> bool:
        return self.outcome is ApprovalOutcome.APPROVE

    @property
    def accepted(self) -> bool:
        return self.approved


class ApprovalProvider(Protocol):
    """Foreground provider contract with no workspace or executor capability."""

    @property
    def identity(self) -> str:
        """Stable provider identity used in bounded diagnostics."""

    @property
    def supports_deadline(self) -> bool:
        """Whether this provider can enforce and report its own deadline."""

    def request_approval(
        self,
        request: ApprovalRequest,
        context: RunContext,
    ) -> ApprovalDecision:
        """Return one bounded decision for this exact foreground request."""


@dataclass(frozen=True, slots=True)
class NonInteractiveApprovalProvider:
    """Fail closed when no real foreground interaction is configured."""

    identity: str = "non-interactive-approval-provider"

    def __post_init__(self) -> None:
        _validate_identity("approval provider identity", self.identity)

    @property
    def supports_deadline(self) -> bool:
        return False

    def request_approval(
        self,
        request: ApprovalRequest,
        context: RunContext,
    ) -> ApprovalDecision:
        if not isinstance(request, ApprovalRequest):
            raise TypeError("approval provider requires an ApprovalRequest")
        if not isinstance(context, RunContext):
            raise TypeError("approval provider requires a RunContext")
        context.check_active()
        return ApprovalDecision(
            ApprovalOutcome.UNAVAILABLE,
            "non_interactive_provider_unavailable",
            provider_identity=self.identity,
        )

    def decide(self, request: ApprovalRequest, context: RunContext) -> ApprovalDecision:
        """Compatibility alias for policy-shaped test doubles."""

        return self.request_approval(request, context)


class ScriptedApprovalProvider:
    """Finite, one-response-per-request provider for deterministic tests."""

    def __init__(
        self,
        decisions: Iterable[
            ApprovalDecision
            | ApprovalOutcome
            | Callable[[ApprovalRequest, RunContext], ApprovalDecision | ApprovalOutcome]
        ] = (),
        *,
        identity: str = "scripted-approval-provider",
        supports_deadline: bool = True,
    ) -> None:
        self._identity = _validate_identity("approval provider identity", identity)
        if not isinstance(supports_deadline, bool):
            raise TypeError("approval deadline support must be a boolean")
        self._supports_deadline = supports_deadline
        if isinstance(decisions, (str, bytes)):
            raise TypeError("scripted approval decisions must be iterable")
        try:
            iterator = iter(decisions)
        except TypeError as error:
            raise TypeError("scripted approval decisions must be iterable") from error
        values: list[object] = []
        for _ in range(_MAX_SCRIPTED_APPROVAL_RESPONSES + 1):
            try:
                values.append(next(iterator))
            except StopIteration:
                break
        if len(values) > _MAX_SCRIPTED_APPROVAL_RESPONSES:
            raise ValueError("scripted approval decisions exceed the bound")
        self._decisions = tuple(values)
        self._index = 0
        self._calls = 0

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def supports_deadline(self) -> bool:
        return self._supports_deadline

    @property
    def calls(self) -> int:
        return self._calls

    def request_approval(
        self,
        request: ApprovalRequest,
        context: RunContext,
    ) -> ApprovalDecision | object:
        if not isinstance(request, ApprovalRequest):
            raise TypeError("approval provider requires an ApprovalRequest")
        if not isinstance(context, RunContext):
            raise TypeError("approval provider requires a RunContext")
        context.check_active()
        self._calls += 1
        if self._index >= len(self._decisions):
            return ApprovalDecision(
                ApprovalOutcome.UNAVAILABLE,
                "scripted_provider_exhausted",
                provider_identity=self.identity,
            )
        response = self._decisions[self._index]
        self._index += 1
        if callable(response):
            response = response(request, context)
        if isinstance(response, ApprovalOutcome):
            if response is ApprovalOutcome.APPROVE:
                return ApprovalDecision.approve_for(
                    request,
                    provider_identity=self.identity,
                )
            if response is ApprovalOutcome.TIMED_OUT and not self.supports_deadline:
                return ApprovalDecision(
                    ApprovalOutcome.UNAVAILABLE,
                    "approval_timeout_unsupported",
                    provider_identity=self.identity,
                )
            return ApprovalDecision(
                response,
                f"scripted_{response.value}",
                provider_identity=self.identity,
            )
        if (
            isinstance(response, ApprovalDecision)
            and response.outcome is ApprovalOutcome.TIMED_OUT
            and not self.supports_deadline
        ):
            return ApprovalDecision(
                ApprovalOutcome.UNAVAILABLE,
                "approval_timeout_unsupported",
                provider_identity=self.identity,
            )
        return response

    def decide(
        self,
        request: ApprovalRequest,
        context: RunContext,
    ) -> ApprovalDecision | object:
        return self.request_approval(request, context)


def _approval_provider_identity(provider: object) -> str | None:
    try:
        identity = cast(ApprovalProvider, provider).identity
        return _validate_identity("approval provider identity", identity)
    except Exception:
        return None


def _approval_provider_supports_deadline(provider: object) -> bool:
    try:
        value = cast(ApprovalProvider, provider).supports_deadline
    except Exception:
        return False
    return value if isinstance(value, bool) else False


def _sanitize_approval_decision(
    response: object,
    *,
    request: ApprovalRequest,
    workspace: Workspace,
    secret_values: tuple[str, ...],
    provider_identity: str,
) -> ApprovalDecision:
    if not isinstance(response, ApprovalDecision):
        return ApprovalDecision(
            ApprovalOutcome.MALFORMED,
            "malformed_approval_response",
            provider_identity=provider_identity,
        )
    reason, reason_failed = _safe_sanitize_t4_text(
        workspace,
        response.reason,
        secret_values=secret_values,
        maximum=_MAX_APPROVAL_TEXT_CHARACTERS,
    )
    message, message_failed = _safe_sanitize_t4_text(
        workspace,
        response.sanitized_message,
        secret_values=secret_values,
        maximum=_MAX_APPROVAL_TEXT_CHARACTERS,
        fallback="",
    )
    binding_failures = False

    def sanitize_identity(value: str | None, label: str) -> str | None:
        nonlocal binding_failures
        if value is None:
            return None
        rendered, failed = _safe_sanitize_t4_text(
            workspace,
            value,
            secret_values=secret_values,
            maximum=_MAX_IDENTITY_CHARACTERS,
        )
        try:
            identity = _validate_identity(label, rendered)
        except Exception:
            binding_failures = True
            return None
        binding_failures = binding_failures or failed
        return identity

    response_run_id = sanitize_identity(response.run_id, "approval response run ID")
    response_workspace_id = sanitize_identity(
        response.workspace_id,
        "approval response workspace ID",
    )
    response_backend_identity = sanitize_identity(
        response.backend_identity,
        "approval response backend identity",
    )
    preserve_binding = response.outcome is ApprovalOutcome.APPROVE
    try:
        result = ApprovalDecision(
            outcome=response.outcome,
            reason=reason,
            run_id=response_run_id if preserve_binding else None,
            workspace_id=response_workspace_id if preserve_binding else None,
            action_digest=response.action_digest,
            preconditions=response.preconditions,
            required_capabilities=response.required_capabilities,
            available_capabilities=response.available_capabilities,
            backend_identity=response_backend_identity if preserve_binding else None,
            provider_identity=provider_identity,
            sanitized_message=message,
            sanitization_failed=(
                response.sanitization_failed
                or reason_failed
                or message_failed
                or binding_failures
            ),
        )
    except Exception:
        return ApprovalDecision(
            ApprovalOutcome.MALFORMED,
            "malformed_approval_response",
            provider_identity=provider_identity,
        )
    if result.sanitization_failed:
        return ApprovalDecision(
            ApprovalOutcome.MALFORMED,
            "approval_response_sanitization_failed",
            provider_identity=provider_identity,
        )
    return result


def _approval_identity_mismatch(
    request: ApprovalRequest,
    response: ApprovalDecision,
    *,
    reason: str,
) -> ApprovalDecision:
    return ApprovalDecision(
        ApprovalOutcome.IDENTITY_MISMATCH,
        reason,
        run_id=request.run_id,
        workspace_id=request.workspace_id,
        action_digest=request.action_digest,
        preconditions=request.preconditions,
        required_capabilities=request.required_capabilities,
        available_capabilities=request.available_capabilities,
        backend_identity=request.backend_identity,
        provider_identity=response.provider_identity,
    )


def _validate_approval_binding(
    request: ApprovalRequest,
    response: ApprovalDecision,
    context: RunContext,
    *,
    action: PreparedAction | None,
    guard_context: GuardContext | None,
) -> ApprovalDecision:
    """Validate approval identity and, when supplied, fresh execution guards."""

    def check() -> ApprovalDecision:
        if request.run_id != context.run_id:
            return _approval_identity_mismatch(
                request,
                response,
                reason="approval_run_identity_mismatch",
            )
        if response.outcome is not ApprovalOutcome.APPROVE:
            return response
        if (
            response.run_id != request.run_id
            or response.workspace_id != request.workspace_id
            or response.action_digest != request.action_digest
            or response.preconditions != request.preconditions
            or response.required_capabilities != request.required_capabilities
            or response.available_capabilities != request.available_capabilities
            or response.backend_identity != request.backend_identity
        ):
            return _approval_identity_mismatch(
                request,
                response,
                reason="approval_identity_mismatch",
            )
        if action is None and guard_context is None:
            return response
        if not isinstance(action, PreparedAction) or not isinstance(guard_context, GuardContext):
            return ApprovalDecision(
                ApprovalOutcome.MALFORMED,
                "approval_revalidation_context_missing",
                provider_identity=response.provider_identity,
            )
        if (
            action.workspace_id != request.workspace_id
            or action.canonical_digest != request.action_digest
            or action.preconditions != request.preconditions
            or tuple(sorted(action.required_capabilities, key=lambda item: item.value))
            != request.required_capabilities
        ):
            return _approval_identity_mismatch(
                request,
                response,
                reason="approved_action_identity_mismatch",
            )
        try:
            current_workspace_id = guard_context.workspace.scope.workspace_id
            current_capabilities = _normalize_bounded_capabilities(
                guard_context.available_capabilities,
                label="current approval capabilities",
            )
            current_backend_identity = _validate_identity(
                "current approval backend identity",
                guard_context.backend_identity,
            )
        except Exception:
            return ApprovalDecision(
                ApprovalOutcome.DRIFT,
                "approval_revalidation_unavailable",
                provider_identity=response.provider_identity,
            )
        if (
            current_workspace_id != request.workspace_id
            or current_capabilities != request.available_capabilities
            or current_backend_identity != request.backend_identity
        ):
            return ApprovalDecision(
                ApprovalOutcome.DRIFT,
                "approval_capability_or_scope_drift",
                provider_identity=response.provider_identity,
            )
        guard_results = evaluate_guards(action, guard_context)
        if len(guard_results) != len(HARD_GUARD_ORDER) or not all(
            result.passed for result in guard_results
        ):
            failed = next(
                (result for result in guard_results if not result.passed),
                None,
            )
            reason = (
                f"approval_{failed.name.value}_drift"
                if failed is not None
                else "approval_revalidation_drift"
            )
            return ApprovalDecision(
                ApprovalOutcome.DRIFT,
                reason,
                provider_identity=response.provider_identity,
            )
        return response

    return context.run_if_active(check)


def obtain_approval(
    request: ApprovalRequest,
    *,
    provider: ApprovalProvider,
    context: RunContext,
    workspace: Workspace,
    secret_values: Iterable[str] = (),
    action: PreparedAction | None = None,
    guard_context: GuardContext | None = None,
) -> ApprovalDecision:
    """Request and validate one exact approval without invoking an executor."""

    if not isinstance(request, ApprovalRequest):
        raise TypeError("approval requires an ApprovalRequest")
    if not isinstance(context, RunContext):
        raise TypeError("approval requires a RunContext")
    if not isinstance(workspace, Workspace):
        raise TypeError("approval requires a Workspace")
    if (action is None) != (guard_context is None):
        raise TypeError("action and guard_context must be supplied together")
    if action is not None and not isinstance(action, PreparedAction):
        raise TypeError("approval action must be a PreparedAction")
    if guard_context is not None and not isinstance(guard_context, GuardContext):
        raise TypeError("approval guard_context must be a GuardContext")
    context.check_active()
    if guard_context is not None and not _workspace_scopes_match(
        workspace,
        guard_context.workspace,
    ):
        return ApprovalDecision(
            ApprovalOutcome.DRIFT,
            "approval_workspace_binding_drift",
        )
    secrets = _normalize_t4_secret_values(secret_values)
    if request.sanitization_failed:
        return ApprovalDecision(
            ApprovalOutcome.MALFORMED,
            "approval_request_sanitization_failed",
        )
    if request.run_id != context.run_id:
        return ApprovalDecision(
            ApprovalOutcome.IDENTITY_MISMATCH,
            "approval_run_identity_mismatch",
        )
    provider_identity = _approval_provider_identity(provider)
    if provider_identity is None:
        return ApprovalDecision(
            ApprovalOutcome.UNAVAILABLE,
            "approval_provider_unavailable",
        )
    provider_request = _sanitize_approval_request_for_provider(
        request,
        workspace=workspace,
        secret_values=secrets,
    )
    if provider_request is None:
        return ApprovalDecision(
            ApprovalOutcome.MALFORMED,
            "approval_request_sanitization_failed",
            provider_identity=provider_identity,
        )
    supports_deadline = _approval_provider_supports_deadline(provider)
    method = getattr(provider, "request_approval", None)
    if not callable(method):
        method = getattr(provider, "decide", None)
    if not callable(method):
        return ApprovalDecision(
            ApprovalOutcome.UNAVAILABLE,
            "approval_provider_unavailable",
            provider_identity=provider_identity,
        )
    try:
        response = method(provider_request, context)
    except (RunCancelledError, RunDeadlineExceededError):
        raise
    except EOFError:
        response = ApprovalDecision(
            ApprovalOutcome.UNAVAILABLE,
            "approval_eof",
            provider_identity=provider_identity,
        )
    except TimeoutError:
        response = ApprovalDecision(
            ApprovalOutcome.TIMED_OUT if supports_deadline else ApprovalOutcome.UNAVAILABLE,
            "approval_timed_out" if supports_deadline else "approval_timeout_unsupported",
            provider_identity=provider_identity,
        )
    except Exception:
        response = ApprovalDecision(
            ApprovalOutcome.UNAVAILABLE,
            "approval_provider_unavailable",
            provider_identity=provider_identity,
        )
    context.check_active()
    decision = _sanitize_approval_decision(
        response,
        request=provider_request,
        workspace=workspace,
        secret_values=secrets,
        provider_identity=provider_identity,
    )
    if decision.outcome is ApprovalOutcome.TIMED_OUT and not supports_deadline:
        decision = ApprovalDecision(
            ApprovalOutcome.UNAVAILABLE,
            "approval_timeout_unsupported",
            provider_identity=provider_identity,
        )
    return _validate_approval_binding(
        provider_request,
        decision,
        context,
        action=action,
        guard_context=guard_context,
    )


def request_approval(
    request: ApprovalRequest,
    *,
    provider: ApprovalProvider,
    context: RunContext,
    workspace: Workspace,
    secret_values: Iterable[str] = (),
    action: PreparedAction | None = None,
    guard_context: GuardContext | None = None,
) -> ApprovalDecision:
    """Named wrapper for callers that prefer the provider operation vocabulary."""

    return obtain_approval(
        request,
        provider=provider,
        context=context,
        workspace=workspace,
        secret_values=secret_values,
        action=action,
        guard_context=guard_context,
    )


def build_approval_request(
    decision: GovernanceDecision,
    context: RunContext,
    guard_context: GuardContext,
    *,
    secret_values: Iterable[str] = (),
    max_text_characters: int = _MAX_APPROVAL_TEXT_CHARACTERS,
) -> ApprovalRequest:
    """Build a request only for a guarded policy decision requiring approval."""

    if not isinstance(decision, GovernanceDecision):
        raise TypeError("approval request requires a GovernanceDecision")
    if not decision.requires_approval:
        raise ValueError("approval request requires a require_approval policy decision")
    return ApprovalRequest.from_governance(
        decision,
        context,
        guard_context,
        secret_values=secret_values,
        max_text_characters=max_text_characters,
    )


def authorize_action(
    decision: GovernanceDecision,
    *,
    provider: ApprovalProvider,
    context: RunContext,
    guard_context: GuardContext,
    execution_guard_context: GuardContext | None = None,
    secret_values: Iterable[str] = (),
) -> ApprovalDecision:
    """Authorize a required-approval decision and recheck a fresh guard context."""

    secrets = _normalize_t4_secret_values(secret_values)
    request = build_approval_request(
        decision,
        context,
        guard_context,
        secret_values=secrets,
    )
    final_context = execution_guard_context or guard_context
    return obtain_approval(
        request,
        provider=provider,
        context=context,
        workspace=guard_context.workspace,
        secret_values=secrets,
        action=decision.action,
        guard_context=final_context,
    )


@dataclass(frozen=True, slots=True)
class ActionHookInput:
    """Minimal immutable action projection shared by pre and post hooks."""

    request: ApprovalRequest
    policy_outcome: PolicyOutcome
    approval_outcome: ApprovalOutcome | None = None
    approval_reason: str | None = None
    _workspace_binding: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, ApprovalRequest):
            raise TypeError("hook input requires an ApprovalRequest")
        if not isinstance(self.policy_outcome, PolicyOutcome):
            raise TypeError("hook input policy outcome must be a PolicyOutcome")
        if self.approval_outcome is not None and not isinstance(
            self.approval_outcome,
            ApprovalOutcome,
        ):
            raise TypeError("hook input approval outcome must be an ApprovalOutcome or None")
        if self.approval_reason is not None:
            object.__setattr__(
                self,
                "approval_reason",
                _validate_sanitized_text(
                    "hook approval reason",
                    self.approval_reason,
                    maximum=_MAX_HOOK_TEXT_CHARACTERS,
                ),
            )
        if self._workspace_binding is not None:
            object.__setattr__(
                self,
                "_workspace_binding",
                _validate_sha256("hook workspace binding", self._workspace_binding),
            )

    @property
    def run_id(self) -> str:
        return self.request.run_id

    @property
    def workspace_id(self) -> str:
        return self.request.workspace_id

    @property
    def action_digest(self) -> str:
        return self.request.action_digest

    @property
    def sanitized_action_display(self) -> str:
        return self.request.sanitized_action_display

    @property
    def required_capabilities(self) -> tuple[IsolationCapability, ...]:
        return self.request.required_capabilities


@dataclass(frozen=True, slots=True)
class PreActionHookInput(ActionHookInput):
    """Input delivered to one pre-effect hook."""


@dataclass(frozen=True, slots=True)
class PostActionHookInput(ActionHookInput):
    """Input delivered after an effect attempt, without an executor capability."""

    effect_state: EffectState = EffectState.NONE
    executor_attempts: int = 0
    sanitized_effect_diagnostic: str = ""

    def __post_init__(self) -> None:
        ActionHookInput.__post_init__(self)
        if not isinstance(self.effect_state, EffectState):
            raise TypeError("post-hook effect state must be an EffectState")
        if (
            isinstance(self.executor_attempts, bool)
            or not isinstance(self.executor_attempts, int)
            or self.executor_attempts not in {0, 1}
        ):
            raise ValueError("post-hook executor attempts must be zero or one")
        if self.executor_attempts == 0 and self.effect_state is not EffectState.NONE:
            raise ValueError("unattempted post-hook input must have none effect state")
        object.__setattr__(
            self,
            "sanitized_effect_diagnostic",
            _validate_optional_sanitized_text(
                "post-hook effect diagnostic",
                self.sanitized_effect_diagnostic,
                maximum=_MAX_HOOK_TEXT_CHARACTERS,
            ),
        )


@dataclass(frozen=True, slots=True)
class HookResult:
    """Bounded hook output normalized by the governance boundary."""

    outcome: HookOutcome
    reason: str = "hook_completed"
    hook_identity: str = "unknown-hook"
    elapsed_seconds: float = 0.0
    sanitized_diagnostics: tuple[str, ...] = ()
    mode: HookMode | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, HookOutcome):
            raise TypeError("hook outcome must be a HookOutcome")
        object.__setattr__(
            self,
            "reason",
            _validate_sanitized_text(
                "hook reason",
                self.reason,
                maximum=_MAX_HOOK_TEXT_CHARACTERS,
            ),
        )
        object.__setattr__(
            self,
            "hook_identity",
            _validate_identity("hook identity", self.hook_identity),
        )
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0
        ):
            raise ValueError("hook elapsed seconds must be a finite non-negative number")
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        if not isinstance(self.sanitized_diagnostics, tuple):
            raise TypeError("hook diagnostics must be a tuple of strings")
        if len(self.sanitized_diagnostics) > _MAX_HOOK_DIAGNOSTICS:
            raise ValueError("hook diagnostics exceed the bound")
        for diagnostic in self.sanitized_diagnostics:
            _validate_sanitized_text(
                "hook diagnostic",
                diagnostic,
                maximum=_MAX_HOOK_TEXT_CHARACTERS,
            )
        if self.mode is not None and not isinstance(self.mode, HookMode):
            raise TypeError("hook mode must be a HookMode or None")

    @property
    def hook_id(self) -> str:
        return self.hook_identity

    @property
    def succeeded(self) -> bool:
        return self.outcome in {HookOutcome.COMPLETED, HookOutcome.SKIPPED}

    @property
    def failed(self) -> bool:
        return not self.succeeded


@dataclass(frozen=True, slots=True)
class HookRunResult:
    """Ordered hook evidence and whether a pre-stage permits the effect."""

    stage: HookStage
    results: tuple[HookResult, ...]
    proceeded: bool

    def __post_init__(self) -> None:
        if not isinstance(self.stage, HookStage):
            raise TypeError("hook run stage must be a HookStage")
        if not isinstance(self.results, tuple) or any(
            not isinstance(result, HookResult) for result in self.results
        ):
            raise TypeError("hook run results must be a tuple of HookResult")
        if len(self.results) > _MAX_HOOKS + 1:
            raise ValueError("hook run results exceed the bound")
        if not isinstance(self.proceeded, bool):
            raise TypeError("hook run proceeded must be a boolean")

    @property
    def allowed(self) -> bool:
        return self.proceeded

    @property
    def can_execute(self) -> bool:
        return self.proceeded

    @property
    def blocked(self) -> bool:
        return self.stage is HookStage.PRE and not self.proceeded

    @property
    def failures(self) -> tuple[HookResult, ...]:
        return tuple(result for result in self.results if result.failed)


class PreActionHook(Protocol):
    """Trusted synchronous hook with no workspace or subprocess capability."""

    @property
    def identity(self) -> str:
        """Stable hook identity."""

    def __call__(self, input: PreActionHookInput) -> HookResult:
        """Observe a sanitized pre-effect input."""


class PostActionHook(Protocol):
    """Trusted synchronous post-effect hook."""

    @property
    def identity(self) -> str:
        """Stable hook identity."""

    def __call__(self, input: PostActionHookInput) -> HookResult:
        """Observe a sanitized post-effect input."""


@dataclass(frozen=True, slots=True)
class PreActionHookSpec:
    """Trusted composition binding for one ordered pre-hook."""

    hook: PreActionHook
    mode: HookMode = HookMode.REQUIRED

    def __post_init__(self) -> None:
        if not isinstance(self.mode, HookMode):
            raise TypeError("pre-hook mode must be a HookMode")


def _sanitize_hook_decision_reason(
    decision: ApprovalDecision,
    *,
    workspace: Workspace,
    secret_values: tuple[str, ...],
) -> ApprovalDecision:
    return _sanitize_approval_decision(
        decision,
        request=ApprovalRequest(
            run_id=decision.run_id or "hook-run",
            workspace_id=decision.workspace_id or "hook-workspace",
            action_kind=ActionKind.READ,
            effect_kind=EffectKind.NONE,
            action_digest=decision.action_digest or ("0" * 64),
            policy_reason="hook",
            sanitized_action_display="hook",
        ),
        workspace=workspace,
        secret_values=secret_values,
        provider_identity=decision.provider_identity,
    )


def build_pre_action_hook_input(
    decision: GovernanceDecision,
    context: RunContext,
    guard_context: GuardContext,
    *,
    approval_decision: ApprovalDecision | None = None,
    secret_values: Iterable[str] = (),
) -> PreActionHookInput:
    """Build a sanitized pre-hook projection without exposing action capabilities."""

    secrets = _normalize_t4_secret_values(secret_values)
    request = ApprovalRequest.from_governance(
        decision,
        context,
        guard_context,
        secret_values=secrets,
    )
    approval_reason: str | None = None
    approval_outcome: ApprovalOutcome | None = None
    if approval_decision is not None:
        if not isinstance(approval_decision, ApprovalDecision):
            raise TypeError("hook approval decision must be an ApprovalDecision")
        safe_decision = _sanitize_hook_decision_reason(
            approval_decision,
            workspace=guard_context.workspace,
            secret_values=secrets,
        )
        approval_outcome = safe_decision.outcome
        approval_reason = safe_decision.reason
    return PreActionHookInput(
        request=request,
        policy_outcome=decision.policy_decision.outcome,
        approval_outcome=approval_outcome,
        approval_reason=approval_reason,
        _workspace_binding=_workspace_scope_binding(guard_context.workspace),
    )


def build_post_action_hook_input(
    pre_input: PreActionHookInput,
    *,
    effect_state: EffectState,
    executor_attempts: int,
    workspace: Workspace,
    secret_values: Iterable[str] = (),
    effect_diagnostic: str = "",
) -> PostActionHookInput:
    """Build the post-effect projection while preserving the supplied effect state."""

    if not isinstance(pre_input, PreActionHookInput):
        raise TypeError("post-hook input requires a PreActionHookInput")
    if pre_input._workspace_binding != _workspace_scope_binding(workspace):
        raise WorkspaceDriftError(
            "hook workspace binding drift",
            reason_code=WorkspaceReason.DRIFT,
            workspace_id=pre_input.request.workspace_id,
        )
    secrets = _normalize_t4_secret_values(secret_values)
    diagnostic, _ = _safe_sanitize_t4_text(
        workspace,
        effect_diagnostic,
        secret_values=secrets,
        maximum=_MAX_HOOK_TEXT_CHARACTERS,
        fallback="",
    )
    return PostActionHookInput(
        request=pre_input.request,
        policy_outcome=pre_input.policy_outcome,
        approval_outcome=pre_input.approval_outcome,
        approval_reason=pre_input.approval_reason,
        effect_state=effect_state,
        executor_attempts=executor_attempts,
        sanitized_effect_diagnostic=diagnostic,
        _workspace_binding=pre_input._workspace_binding,
    )


def _hook_identity(hook: object) -> str:
    identity = getattr(hook, "identity", None)
    if identity is None:
        identity = getattr(hook, "__name__", type(hook).__name__)
    return _validate_identity("hook identity", identity)


def _invoke_hook(hook: object, input_value: PreActionHookInput | PostActionHookInput) -> object:
    callback = getattr(hook, "run", None)
    if not callable(callback):
        callback = hook if callable(hook) else None
    if callback is None:
        raise TypeError("hook is not callable")
    typed_callback = callback
    return typed_callback(input_value)


def _normalize_hook_result(
    raw_result: object,
    *,
    hook_identity: str,
    mode: HookMode | None,
    elapsed_seconds: float,
    workspace: Workspace,
    secret_values: tuple[str, ...],
) -> HookResult:
    if not isinstance(raw_result, HookResult):
        return HookResult(
            HookOutcome.MALFORMED,
            "malformed_hook_response",
            hook_identity,
            elapsed_seconds,
            mode=mode,
        )
    reason, reason_failed = _safe_sanitize_t4_text(
        workspace,
        raw_result.reason,
        secret_values=secret_values,
        maximum=_MAX_HOOK_TEXT_CHARACTERS,
    )
    diagnostics: list[str] = []
    diagnostics_failed = False
    for diagnostic in raw_result.sanitized_diagnostics[:_MAX_HOOK_DIAGNOSTICS]:
        rendered, failed = _safe_sanitize_t4_text(
            workspace,
            diagnostic,
            secret_values=secret_values,
            maximum=_MAX_HOOK_TEXT_CHARACTERS,
        )
        diagnostics.append(rendered)
        diagnostics_failed = diagnostics_failed or failed
    outcome = raw_result.outcome
    if reason_failed or diagnostics_failed:
        outcome = HookOutcome.MALFORMED
        reason = "hook_result_sanitization_failed"
    return HookResult(
        outcome,
        reason,
        hook_identity,
        elapsed_seconds,
        tuple(diagnostics),
        mode,
    )


def _hook_exception_result(
    error: Exception,
    *,
    hook_identity: str,
    mode: HookMode | None,
    elapsed_seconds: float,
    workspace: Workspace,
    secret_values: tuple[str, ...],
) -> HookResult:
    try:
        diagnostic = str(error)
    except Exception:
        diagnostic = type(error).__name__
    rendered, _ = _safe_sanitize_t4_text(
        workspace,
        diagnostic,
        secret_values=secret_values,
        maximum=_MAX_HOOK_TEXT_CHARACTERS,
    )
    return HookResult(
        HookOutcome.FAILED,
        "hook_exception",
        hook_identity,
        elapsed_seconds,
        (rendered,),
        mode,
    )


def _execute_hook(
    hook: object,
    input_value: PreActionHookInput | PostActionHookInput,
    *,
    mode: HookMode | None,
    context: RunContext,
    workspace: Workspace,
    secret_values: tuple[str, ...],
) -> HookResult:
    try:
        identity = _hook_identity(hook)
    except Exception:
        return HookResult(
            HookOutcome.MALFORMED,
            "malformed_hook_identity",
            "malformed-hook",
            mode=mode,
        )
    started = time.perf_counter()
    try:
        raw_result = _invoke_hook(hook, input_value)
    except (RunCancelledError, RunDeadlineExceededError):
        raise
    except Exception as error:
        elapsed = max(0.0, time.perf_counter() - started)
        return _hook_exception_result(
            error,
            hook_identity=identity,
            mode=mode,
            elapsed_seconds=elapsed,
            workspace=workspace,
            secret_values=secret_values,
        )
    elapsed = max(0.0, time.perf_counter() - started)
    return _normalize_hook_result(
        raw_result,
        hook_identity=identity,
        mode=mode,
        elapsed_seconds=elapsed,
        workspace=workspace,
        secret_values=secret_values,
    )


def _pre_hook_spec(item: object) -> PreActionHookSpec:
    if isinstance(item, PreActionHookSpec):
        return item
    if isinstance(item, tuple) and len(item) == 2:
        hook, mode = item
        if not isinstance(mode, HookMode):
            raise TypeError("pre-hook tuple mode must be a HookMode")
        return PreActionHookSpec(cast(PreActionHook, hook), mode)
    return PreActionHookSpec(cast(PreActionHook, item))


def _hook_config_failure(reason: str, *, mode: HookMode | None) -> HookResult:
    return HookResult(HookOutcome.MALFORMED, reason, "hook-configuration", mode=mode)


def _hook_workspace_binding_matches(
    input_value: ActionHookInput,
    workspace: Workspace,
) -> bool:
    return input_value._workspace_binding == _workspace_scope_binding(workspace)


def run_pre_hooks(
    hooks: Iterable[PreActionHook | PreActionHookSpec],
    input_value: PreActionHookInput,
    *,
    context: RunContext,
    workspace: Workspace,
    secret_values: Iterable[str] = (),
) -> HookRunResult:
    """Run ordered pre-hooks; required failure blocks before any effect."""

    if not isinstance(input_value, PreActionHookInput):
        raise TypeError("pre-hooks require a PreActionHookInput")
    if not isinstance(context, RunContext):
        raise TypeError("pre-hooks require a RunContext")
    if not isinstance(workspace, Workspace):
        raise TypeError("pre-hooks require a Workspace")
    if not _hook_workspace_binding_matches(input_value, workspace):
        return HookRunResult(
            HookStage.PRE,
            (_hook_config_failure("hook_workspace_binding_drift", mode=HookMode.REQUIRED),),
            False,
        )
    secrets = _normalize_t4_secret_values(secret_values)
    try:
        iterator = iter(hooks)
    except Exception:
        return HookRunResult(
            HookStage.PRE,
            (_hook_config_failure("malformed_pre_hook_configuration", mode=HookMode.REQUIRED),),
            False,
        )
    results: list[HookResult] = []
    proceeded = True
    for index in range(_MAX_HOOKS + 1):
        try:
            item = next(iterator)
        except StopIteration:
            break
        except Exception:
            results.append(
                _hook_config_failure(
                    "pre_hook_configuration_failure",
                    mode=HookMode.REQUIRED,
                )
            )
            proceeded = False
            break
        if index == _MAX_HOOKS:
            results.append(_hook_config_failure("pre_hook_limit_exceeded", mode=HookMode.REQUIRED))
            proceeded = False
            break
        try:
            spec = _pre_hook_spec(item)
        except Exception:
            results.append(
                _hook_config_failure(
                    "malformed_pre_hook_configuration",
                    mode=HookMode.REQUIRED,
                )
            )
            proceeded = False
            break
        context.check_active()
        result = _execute_hook(
            spec.hook,
            input_value,
            mode=spec.mode,
            context=context,
            workspace=workspace,
            secret_values=secrets,
        )
        context.check_active()
        results.append(result)
        if result.failed and spec.mode is HookMode.REQUIRED:
            proceeded = False
            break
    return HookRunResult(HookStage.PRE, tuple(results), proceeded)


def run_post_hooks(
    hooks: Iterable[PostActionHook],
    input_value: PostActionHookInput,
    *,
    context: RunContext,
    workspace: Workspace,
    secret_values: Iterable[str] = (),
) -> HookRunResult:
    """Run all configured post-hooks in order without changing effect evidence."""

    if not isinstance(input_value, PostActionHookInput):
        raise TypeError("post-hooks require a PostActionHookInput")
    if not isinstance(context, RunContext):
        raise TypeError("post-hooks require a RunContext")
    if not isinstance(workspace, Workspace):
        raise TypeError("post-hooks require a Workspace")
    if not _hook_workspace_binding_matches(input_value, workspace):
        return HookRunResult(
            HookStage.POST,
            (_hook_config_failure("hook_workspace_binding_drift", mode=None),),
            True,
        )
    secrets = _normalize_t4_secret_values(secret_values)
    try:
        iterator = iter(hooks)
    except Exception:
        return HookRunResult(
            HookStage.POST,
            (_hook_config_failure("malformed_post_hook_configuration", mode=None),),
            True,
        )
    results: list[HookResult] = []
    for index in range(_MAX_HOOKS + 1):
        try:
            hook = next(iterator)
        except StopIteration:
            break
        except Exception:
            results.append(_hook_config_failure("post_hook_configuration_failure", mode=None))
            break
        if index == _MAX_HOOKS:
            results.append(_hook_config_failure("post_hook_limit_exceeded", mode=None))
            break
        try:
            context.check_active()
            if isinstance(hook, PreActionHookSpec):
                result = _hook_config_failure("malformed_post_hook_configuration", mode=None)
            else:
                result = _execute_hook(
                    hook,
                    input_value,
                    mode=None,
                    context=context,
                    workspace=workspace,
                    secret_values=secrets,
                )
            context.check_active()
        except (RunCancelledError, RunDeadlineExceededError) as error:
            object.__setattr__(error, "_dqagent_partial_post_hook_results", tuple(results))
            raise
        results.append(result)
    return HookRunResult(HookStage.POST, tuple(results), True)


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


def _guard_workspace_identity(
    action: PreparedAction,
    context: GuardContext,
    *,
    expected_workspace: Workspace | None = None,
) -> GuardResult:
    try:
        workspace_id = context.workspace.scope.workspace_id
        if not isinstance(workspace_id, str):
            return _failed(GuardName.WORKSPACE_IDENTITY, "malformed_workspace_identity")
        if action.workspace_id != workspace_id:
            return _failed(GuardName.WORKSPACE_IDENTITY, "workspace_identity_mismatch")
        if expected_workspace is not None and (
            not isinstance(expected_workspace, Workspace)
            or not _workspace_scopes_match(expected_workspace, context.workspace)
        ):
            return _failed(GuardName.WORKSPACE_IDENTITY, "workspace_scope_binding_mismatch")
        return _passed(GuardName.WORKSPACE_IDENTITY, "workspace_identity_matches")
    except Exception:
        return _failed(GuardName.WORKSPACE_IDENTITY, "workspace_identity_dependency_failure")


def _guard_current_containment(action: PreparedAction, context: GuardContext) -> GuardResult:
    try:
        for logical_path, purpose in _action_paths(action):
            try:
                if logical_path == PurePosixPath("."):
                    if purpose not in {
                        WorkspacePurpose.COMMAND_CWD,
                        WorkspacePurpose.SEARCH,
                    }:
                        return _failed(GuardName.CURRENT_CONTAINMENT, "root_target_not_allowed")
                    resolved = context.workspace.resolve_root(
                        purpose=(
                            WorkspacePurpose.SEARCH
                            if purpose is WorkspacePurpose.SEARCH
                            else WorkspacePurpose.SNAPSHOT
                        )
                    )
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
                if (
                    str(error.reason_code)
                    in {
                        WorkspaceReason.TARGET_MISSING.value,
                        WorkspaceReason.PARENT_MISSING.value,
                    }
                    and action.action_kind in {ActionKind.READ, ActionKind.SEARCH}
                ):
                    continue
                return _failed(
                    GuardName.CURRENT_CONTAINMENT,
                    "current_containment_denied",
                )
            except WorkspaceError as error:
                if (
                    str(error.reason_code)
                    in {
                        WorkspaceReason.TARGET_MISSING.value,
                        WorkspaceReason.PARENT_MISSING.value,
                    }
                    and action.action_kind in {ActionKind.READ, ActionKind.SEARCH}
                ):
                    continue
                return _failed(GuardName.CURRENT_CONTAINMENT, "current_containment_denied")
            if not isinstance(resolved, ResolvedWorkspacePath):
                return _failed(GuardName.CURRENT_CONTAINMENT, "malformed_containment_response")
            if action.action_kind is ActionKind.PATCH and resolved.followed_link:
                return _failed(GuardName.CURRENT_CONTAINMENT, "patch_link_target_denied")
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
    *,
    expected_workspace: Workspace | None = None,
) -> tuple[GuardResult, ...]:
    """Run hard guards in order and stop at the first fail-closed result."""

    if not isinstance(action, PreparedAction):
        raise TypeError("guard evaluation requires a PreparedAction")
    if not isinstance(context, GuardContext):
        raise TypeError("guard evaluation requires a GuardContext")
    results: list[GuardResult] = []
    for expected_name, guard in zip(HARD_GUARD_ORDER, _GUARD_FUNCTIONS, strict=True):
        try:
            if expected_name is GuardName.WORKSPACE_IDENTITY:
                result = _guard_workspace_identity(
                    action,
                    context,
                    expected_workspace=expected_workspace,
                )
            else:
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
    *,
    expected_workspace: Workspace | None = None,
) -> GovernanceDecision:
    """Evaluate hard guards and policy without approval or effect execution."""

    guard_results = evaluate_guards(
        action,
        context,
        expected_workspace=expected_workspace,
    )
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
    approval_decision: ApprovalDecision | None = None,
    pre_hook_results: tuple[HookResult, ...] = (),
    post_hook_results: tuple[HookResult, ...] = (),
    observation_failure: bool = False,
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
        approval_decision=approval_decision,
        pre_hook_results=pre_hook_results,
        post_hook_results=post_hook_results,
        observation_failure=observation_failure,
    )
