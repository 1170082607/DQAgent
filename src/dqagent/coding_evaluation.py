"""Versioned disposable evaluation for the production coding application.

The v1 substrate deliberately stays above :class:`CodingAgentApplication`.
Cases provide data fixtures and a small trusted composition description; the
runner supplies the real resolver, governed tools, subprocess backend,
context builder, diff observer, and validators.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from dqagent.coding import (
    CodingAgentApplication,
    CodingFailureEvidence,
    CodingRequest,
    CodingRunResult,
)
from dqagent.errors import DQAgentError, ErrorCategory
from dqagent.events import RunEvent, RunEventType
from dqagent.execution import RunContext
from dqagent.llm import LLMClient
from dqagent.models import (
    Completion,
    ConversationItem,
    Message,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolErrorCode,
    ToolOutcome,
    ToolResult,
)
from dqagent.tool_governance import (
    HARD_GUARD_ORDER,
    ActionKind,
    ActionRecord,
    ApprovalDecision,
    ApprovalOutcome,
    DefaultActionPolicy,
    EffectState,
    HookMode,
    HookOutcome,
    HookResult,
    PolicyOutcome,
    PostActionHookInput,
    PreActionHook,
    PreActionHookInput,
    PreActionHookSpec,
    ScriptedApprovalProvider,
)
from dqagent.validators import (
    ValidatorDefinition,
    ValidatorResult,
    ValidatorStatus,
)
from dqagent.workspace import (
    Workspace,
    WorkspaceChange,
    WorkspaceChangeKind,
    WorkspaceDiff,
    WorkspaceObserver,
    WorkspaceScope,
    WorkspaceSnapshot,
)

__all__ = [
    "CODING_EVALUATION_REPORT_SCHEMA_VERSION",
    "CODING_EVALUATION_SCHEMA_VERSION",
    "CodingEvaluationCase",
    "CodingEvaluationCaseResult",
    "CodingEvaluationCheck",
    "CodingEvaluationChangeExpectation",
    "CodingEvaluationComposition",
    "CodingEvaluationContextExpectation",
    "CodingEvaluationDefinitionError",
    "CodingEvaluationDiffExpectation",
    "CodingEvaluationExpected",
    "CodingEvaluationFixture",
    "CodingEvaluationLimits",
    "CodingEvaluationMode",
    "CodingEvaluationReport",
    "CodingEvaluationRunner",
    "CodingEvaluationToolCallExpectation",
    "CodingEvaluationSuite",
    "CodingEvaluationValidatorFixture",
    "CodingEvaluationValidatorExpectation",
    "CodingRepositoryFixture",
    "compute_coding_fixture_digest",
    "load_coding_evaluation_suite",
]


CODING_EVALUATION_SCHEMA_VERSION = 1
CODING_EVALUATION_REPORT_SCHEMA_VERSION = 1

_MAX_CASES = 32
_MAX_FILES = 128
_MAX_FILE_CHARACTERS = 128_000
_MAX_COMPLETIONS = 64
_MAX_APPROVAL_DECISIONS = 64
_MAX_VALIDATORS = 32
_MAX_EXPECTED_CHANGES = 128
_MAX_EXPECTED_GOVERNANCE = 128
_MAX_EXPECTED_TOOL_CALLS = 128
_MAX_EXPECTED_CHECKS = 256
_MAX_REPORT_TEXT = 8_192
_MAX_REPORT_DIFF = 32_000
_MAX_REPORT_VALIDATOR_OUTPUT = 4_096
_MAX_REPORT_LIMITATIONS = 32
_MAX_REPORT_LIMITATION_TEXT = 256
_MAX_REPORT_CHARACTERS = 1_000_000
_MAX_FIXTURE_PATHS = 32
_MAX_FIXTURE_PATH_CHARACTERS = 16_384
_MAX_SECRET_NAMES = 64
_MAX_SECRET_VALUES = 64
_MAX_SECRET_NAMES_CHARACTERS = 4_096
_MAX_SECRET_VALUES_CHARACTERS = 16_384
_MAX_RAW_JSON_ITEMS = _MAX_EXPECTED_CHECKS
_MAX_RAW_JSON_DEPTH = 16
_REQUIRED_CODING_TOOL_NAMES = frozenset(
    {
        "workspace_read",
        "workspace_search",
        "workspace_patch",
        "workspace_command",
    }
)


class CodingEvaluationDefinitionError(DQAgentError):
    """Raised when a coding evaluation suite or fixture is malformed."""

    category = ErrorCategory.CONFIGURATION


class CodingEvaluationMode(StrEnum):
    """Modes supported by the initial credential-free coding substrate."""

    DETERMINISTIC = "deterministic"


def _bounded_text(value: object, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    if len(value) > maximum or "\x00" in value:
        raise ValueError(f"{label} exceeds its bound or contains NUL")
    return value


def _validate_raw_json_case(value: object, label: str = "case", depth: int = 0) -> None:
    if depth > _MAX_RAW_JSON_DEPTH:
        raise ValueError(f"{label} exceeds its nesting bound")
    if type(value) is dict:
        if len(value) > _MAX_RAW_JSON_ITEMS:
            raise ValueError(f"{label} object exceeds its bound")
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} object keys must be strings")
            _validate_raw_json_case(item, f"{label}.{key}", depth + 1)
        return
    if type(value) is list:
        if len(value) > _MAX_RAW_JSON_ITEMS:
            raise ValueError(f"{label} array exceeds its bound")
        for index, item in enumerate(value):
            _validate_raw_json_case(item, f"{label}[{index}]", depth + 1)
        return
    if value is None or type(value) in {bool, float, int, str}:
        return
    raise TypeError(f"{label} must use JSON-native containers")


def _bounded_raw_array(value: object, label: str, maximum: int) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a JSON array")
    if len(value) > maximum:
        raise ValueError(f"{label} exceeds its bound ({maximum})")
    return value


def _bounded_raw_mapping(value: object, label: str, maximum: int) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be a JSON object")
    if len(value) > maximum:
        raise ValueError(f"{label} exceeds its bound ({maximum})")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} object keys must be strings")
    return value


def _relative_path(value: object, label: str, *, allow_root: bool = False) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a relative POSIX path")
    if allow_root and value == ".":
        return PurePosixPath(".")
    if (
        "\x00" in value
        or "\\" in value
        or value.startswith("/")
        or value.startswith("//")
        or (len(value) > 1 and value[1] == ":")
    ):
        raise ValueError(f"{label} must be a relative POSIX path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} contains an ambiguous path component")
    return PurePosixPath(*parts)


def _portable_path_parts(path: str | PurePosixPath) -> tuple[str, ...]:
    normalized = PurePosixPath(path).as_posix()
    if normalized == ".":
        return ()
    return tuple(part.casefold() for part in PurePosixPath(normalized).parts)


def _is_strict_path_prefix(parent: tuple[str, ...], child: tuple[str, ...]) -> bool:
    return len(parent) < len(child) and child[: len(parent)] == parent


def _validate_portable_fixture_identity(
    files: Mapping[str, str],
    skill_roots: Mapping[str, str],
) -> None:
    file_entries = [
        (_portable_path_parts(path), path)
        for path in files
    ]
    for index, (path_parts, path) in enumerate(file_entries):
        for other_parts, other_path in file_entries[index + 1 :]:
            if (
                path_parts == other_parts
                or _is_strict_path_prefix(path_parts, other_parts)
                or _is_strict_path_prefix(other_parts, path_parts)
            ):
                raise ValueError(
                    "portable fixture path collision between "
                    f"{path!r} and {other_path!r}"
                )

    root_entries = [
        (_portable_path_parts(path), path)
        for path in skill_roots.values()
    ]
    for index, (path_parts, path) in enumerate(root_entries):
        for other_parts, other_path in root_entries[index + 1 :]:
            if path_parts == other_parts:
                raise ValueError(
                    "portable fixture path collision between skill roots "
                    f"{path!r} and {other_path!r}"
                )
        for file_parts, file_path in file_entries:
            if path_parts == file_parts or _is_strict_path_prefix(file_parts, path_parts):
                raise ValueError(
                    "portable fixture path collision between file "
                    f"{file_path!r} and skill root {path!r}"
                )


def _normalize_path_tuple(
    values: Iterable[str | PurePosixPath],
    label: str,
    *,
    allow_root: bool = False,
    max_items: int = _MAX_FIXTURE_PATHS,
    aggregate_maximum: int = _MAX_FIXTURE_PATH_CHARACTERS,
) -> tuple[PurePosixPath, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be an iterable")
    normalized: list[PurePosixPath] = []
    seen: set[PurePosixPath] = set()
    try:
        iterator = iter(values)
    except TypeError as error:
        raise TypeError(f"{label} must be an iterable") from error
    aggregate_characters = 0
    for index in range(max_items + 1):
        try:
            value = next(iterator)
        except StopIteration:
            break
        except TypeError as error:
            raise TypeError(f"{label} must be an iterable") from error
        if index >= max_items:
            raise ValueError(f"{label} exceeds its bound")
        path = _relative_path(
            value.as_posix() if isinstance(value, PurePosixPath) else value,
            label,
            allow_root=allow_root,
        )
        aggregate_characters += len(path.as_posix())
        if aggregate_characters > aggregate_maximum:
            raise ValueError(f"{label} character budget exceeded")
        if path not in seen:
            normalized.append(path)
            seen.add(path)
    return tuple(normalized)


def _normalize_string_tuple(
    values: Iterable[str],
    label: str,
    maximum: int,
    *,
    max_items: int,
    aggregate_maximum: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be an iterable")
    try:
        iterator = iter(values)
    except TypeError as error:
        raise TypeError(f"{label} must be an iterable") from error
    normalized: list[str] = []
    seen: set[str] = set()
    aggregate_characters = 0
    for index in range(max_items + 1):
        try:
            value = next(iterator)
        except StopIteration:
            break
        except TypeError as error:
            raise TypeError(f"{label} must be an iterable") from error
        if index >= max_items:
            raise ValueError(f"{label} exceeds item bound")
        text = _bounded_text(value, label, maximum)
        aggregate_characters += len(text)
        if aggregate_characters > aggregate_maximum:
            raise ValueError(f"{label} character budget exceeded")
        if text not in seen:
            normalized.append(text)
            seen.add(text)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class CodingRepositoryFixture:
    """Files and contained skill roots materialized for one case."""

    files: Mapping[str, str]
    skill_roots: Mapping[str, str] = field(default_factory=dict)
    protected_paths: tuple[PurePosixPath, ...] = ()
    secret_paths: tuple[PurePosixPath, ...] = ()
    volatile_paths: tuple[PurePosixPath, ...] = ()
    ignored_paths: tuple[PurePosixPath, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.files, Mapping) or not self.files:
            raise ValueError("coding repository fixture requires files")
        if len(self.files) > _MAX_FILES:
            raise ValueError("coding repository fixture files are unbounded")
        normalized_files: dict[str, str] = {}
        for raw_path, content in self.files.items():
            path = _relative_path(raw_path, "fixture file path")
            normalized_files[path.as_posix()] = _bounded_text(
                content,
                "fixture file content",
                _MAX_FILE_CHARACTERS,
            )
        normalized_roots: dict[str, str] = {}
        if not isinstance(self.skill_roots, Mapping) or len(self.skill_roots) > 16:
            raise ValueError("coding skill roots are malformed or unbounded")
        for key, raw_path in self.skill_roots.items():
            normalized_key = _bounded_text(key, "skill root identity", 64)
            normalized_root = _relative_path(raw_path, "skill root path", allow_root=True)
            normalized_roots[normalized_key] = normalized_root.as_posix()
        _validate_portable_fixture_identity(normalized_files, normalized_roots)
        object.__setattr__(self, "files", dict(sorted(normalized_files.items())))
        object.__setattr__(self, "skill_roots", dict(sorted(normalized_roots.items())))
        for name in (
            "protected_paths",
            "secret_paths",
            "volatile_paths",
            "ignored_paths",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_path_tuple(
                    getattr(self, name),
                    f"fixture {name}",
                    max_items=_MAX_FIXTURE_PATHS,
                    aggregate_maximum=_MAX_FIXTURE_PATH_CHARACTERS,
                ),
            )


@dataclass(frozen=True, slots=True)
class CodingEvaluationValidatorFixture:
    """Trusted direct-argv validator configuration from a reviewed case."""

    validator_id: str
    argv: tuple[str, ...]
    cwd: PurePosixPath = PurePosixPath(".")
    timeout_seconds: float = 5.0
    stdout_limit_bytes: int = 4_096
    stderr_limit_bytes: int = 4_096
    accepted_exit_codes: frozenset[int] = frozenset({0})

    def __post_init__(self) -> None:
        _bounded_text(self.validator_id, "validator fixture ID", 128)
        if not self.argv or len(self.argv) > 32:
            raise ValueError("validator fixture argv is empty or unbounded")
        if any(not isinstance(item, str) or not item or "\x00" in item for item in self.argv):
            raise ValueError("validator fixture argv is malformed")
        object.__setattr__(
            self,
            "cwd",
            _relative_path(self.cwd.as_posix(), "validator fixture cwd", allow_root=True),
        )
        if self.timeout_seconds <= 0 or self.timeout_seconds > 86_400:
            raise ValueError("validator fixture timeout is outside its bound")
        for label, value in (
            ("validator stdout limit", self.stdout_limit_bytes),
            ("validator stderr limit", self.stderr_limit_bytes),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 4 * 1024 * 1024
            ):
                raise ValueError(f"{label} is outside its bound")
        if not self.accepted_exit_codes or len(self.accepted_exit_codes) > 8:
            raise ValueError("validator fixture exit codes are malformed")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.accepted_exit_codes
        ):
            raise ValueError("validator fixture exit codes are malformed")


@dataclass(frozen=True, slots=True)
class CodingEvaluationComposition:
    """Small trusted composition vocabulary; policy remains fixed in v1."""

    policy: str = "default"
    executable_allowlist: Mapping[str, str] = field(default_factory=dict)
    validators: tuple[CodingEvaluationValidatorFixture, ...] = ()
    max_iterations: int = 8
    max_governed_calls: int = 1
    secret_names: tuple[str, ...] = ()
    secret_values: tuple[str, ...] = ()
    hook_fixture: str = "none"

    def __post_init__(self) -> None:
        if self.policy != "default":
            raise ValueError("coding evaluation only supports the default policy fixture")
        if self.hook_fixture not in {
            "none",
            "required_pre_hook_failure",
            "post_hook_failure",
        }:
            raise ValueError("unsupported coding evaluation hook fixture")
        if not isinstance(self.executable_allowlist, Mapping) or len(self.executable_allowlist) > 8:
            raise ValueError("executable fixture is malformed or unbounded")
        normalized_executables: dict[str, str] = {}
        for identity, target in self.executable_allowlist.items():
            _bounded_text(identity, "executable fixture identity", 128)
            if target != "current_python":
                raise ValueError("executable fixture must use current_python")
            normalized_executables[identity] = target
        if len(self.validators) > _MAX_VALIDATORS:
            raise ValueError("coding evaluation validators are unbounded")
        validator_ids = [item.validator_id for item in self.validators]
        if len(validator_ids) != len(set(validator_ids)):
            raise ValueError("coding evaluation validator IDs must be unique")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or not 1 <= self.max_iterations <= 128
        ):
            raise ValueError("coding evaluation max_iterations is outside its bound")
        if (
            isinstance(self.max_governed_calls, bool)
            or not isinstance(self.max_governed_calls, int)
            or not 1 <= self.max_governed_calls <= 128
        ):
            raise ValueError("coding evaluation max_governed_calls is outside its bound")
        object.__setattr__(
            self, "executable_allowlist", dict(sorted(normalized_executables.items()))
        )
        object.__setattr__(
            self,
            "secret_names",
            _normalize_string_tuple(
                self.secret_names,
                "secret names",
                128,
                max_items=_MAX_SECRET_NAMES,
                aggregate_maximum=_MAX_SECRET_NAMES_CHARACTERS,
            ),
        )
        object.__setattr__(
            self,
            "secret_values",
            _normalize_string_tuple(
                self.secret_values,
                "secret values",
                512,
                max_items=_MAX_SECRET_VALUES,
                aggregate_maximum=_MAX_SECRET_VALUES_CHARACTERS,
            ),
        )


@dataclass(frozen=True, slots=True)
class CodingEvaluationFixture:
    """Deterministic substitutions and one purpose-built failure selector."""

    model_completions: tuple[Completion, ...]
    approval_decisions: tuple[ApprovalOutcome, ...]
    failure: str = "none"

    def __post_init__(self) -> None:
        if not self.model_completions or len(self.model_completions) > _MAX_COMPLETIONS:
            raise ValueError("model completion fixture is empty or unbounded")
        if len(self.approval_decisions) > _MAX_APPROVAL_DECISIONS:
            raise ValueError("approval fixture is unbounded")
        allowed = {
            "none",
            "validator_runner_error",
            "validator_runner_empty",
            "cleanup_failure",
            "reject_then_stale_approval",
            "incomplete_observation",
        }
        if self.failure not in allowed:
            raise ValueError("unsupported coding evaluation failure fixture")


@dataclass(frozen=True, slots=True)
class CodingEvaluationChangeExpectation:
    path: PurePosixPath
    kind: WorkspaceChangeKind
    before_sha256: str | None = None
    after_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _relative_path(self.path.as_posix(), "expected diff path", allow_root=False),
        )
        for label, value in (
            ("before SHA-256", self.before_sha256),
            ("after SHA-256", self.after_sha256),
        ):
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{label} is malformed")


@dataclass(frozen=True, slots=True)
class CodingEvaluationDiffExpectation:
    expected_changes: tuple[CodingEvaluationChangeExpectation, ...]
    forbidden_paths: tuple[PurePosixPath, ...]
    target_complete: bool = True
    forbidden_complete: bool = True
    rendered_diff_complete: bool = True

    def __post_init__(self) -> None:
        if len(self.expected_changes) > _MAX_EXPECTED_CHANGES:
            raise ValueError("expected diff changes are unbounded")
        object.__setattr__(
            self,
            "forbidden_paths",
            _normalize_path_tuple(self.forbidden_paths, "forbidden diff paths", allow_root=True),
        )


@dataclass(frozen=True, slots=True)
class CodingEvaluationValidatorExpectation:
    validator_id: str
    status: ValidatorStatus

    def __post_init__(self) -> None:
        _bounded_text(self.validator_id, "expected validator ID", 128)


@dataclass(frozen=True, slots=True)
class CodingEvaluationGovernanceExpectation:
    action_kind: ActionKind
    policy_outcome: PolicyOutcome
    approval_outcome: ApprovalOutcome | None
    effect_state: EffectState
    executor_attempts: int
    guards_passed: bool
    pre_hook_outcomes: tuple[HookOutcome, ...] = ()
    post_hook_outcomes: tuple[HookOutcome, ...] = ()
    diagnostics_contains_all: tuple[str, ...] = ()
    diagnostics_absent_all: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.executor_attempts not in {0, 1}:
            raise ValueError("expected executor attempts must be zero or one")


@dataclass(frozen=True, slots=True)
class CodingEvaluationToolCallExpectation:
    """A bounded provider-neutral call and result expectation."""

    call_id: str
    name: str
    arguments: Mapping[str, object]
    outcome: ToolOutcome
    error_code: ToolErrorCode | None = None

    def __post_init__(self) -> None:
        _bounded_text(self.call_id, "expected tool call ID", 128)
        _bounded_text(self.name, "expected tool name", 128)
        if not isinstance(self.arguments, Mapping):
            raise TypeError("expected tool call arguments must be a mapping")
        if self.outcome is ToolOutcome.ERROR and self.error_code is None:
            raise ValueError("expected tool errors must include an error code")
        if self.outcome is ToolOutcome.SUCCESS and self.error_code is not None:
            raise ValueError("expected successful tools must not include an error code")
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True, slots=True)
class CodingEvaluationContextExpectation:
    """Content-free repository/context predicates for representative cases."""

    selected_instruction_sources: tuple[str, ...] = ()
    selected_skill_key: str | None = None
    selected_skill_body: bool | None = None
    omission_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for label, values in (
            ("selected instruction sources", self.selected_instruction_sources),
            ("context omission reasons", self.omission_reasons),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{label} must contain non-empty strings")
        if self.selected_skill_key is not None and (
            not isinstance(self.selected_skill_key, str) or not self.selected_skill_key.strip()
        ):
            raise ValueError("selected skill key must be non-empty text or None")
        if self.selected_skill_body is not None and not isinstance(self.selected_skill_body, bool):
            raise TypeError("selected skill body expectation must be a boolean or None")


@dataclass(frozen=True, slots=True)
class CodingEvaluationLimits:
    max_model_attempts: int
    max_governed_calls: int
    max_elapsed_seconds: float
    max_rendered_diff_characters: int
    max_validator_output_characters: int

    def __post_init__(self) -> None:
        for label, value in (
            ("max model attempts", self.max_model_attempts),
            ("max governed calls", self.max_governed_calls),
            ("max rendered diff characters", self.max_rendered_diff_characters),
            ("max validator output characters", self.max_validator_output_characters),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if self.max_elapsed_seconds <= 0:
            raise ValueError("max elapsed seconds must be positive")


@dataclass(frozen=True, slots=True)
class CodingEvaluationExpected:
    run_status: str
    answer_non_empty: bool
    answer_exact: str | None
    answer_contains_all: tuple[str, ...]
    answer_absent_all: tuple[str, ...]
    verdict: str | None
    diff: CodingEvaluationDiffExpectation
    validators: tuple[CodingEvaluationValidatorExpectation, ...]
    governance: tuple[CodingEvaluationGovernanceExpectation, ...]
    required_event_types: tuple[RunEventType, ...]
    limits: CodingEvaluationLimits
    context: CodingEvaluationContextExpectation = field(
        default_factory=CodingEvaluationContextExpectation
    )
    error_type: str | None = None
    error_category: str | None = None
    fixture_consumed: bool = True
    tool_calls: tuple[CodingEvaluationToolCallExpectation, ...] = ()

    def __post_init__(self) -> None:
        if self.run_status not in {"completed", "error"}:
            raise ValueError("expected run status must be completed or error")
        if self.run_status == "completed" and self.verdict is None:
            raise ValueError("completed coding cases require an expected verdict")
        if (
            len(self.validators) > _MAX_VALIDATORS
            or len(self.governance) > _MAX_EXPECTED_GOVERNANCE
            or len(self.tool_calls) > _MAX_EXPECTED_TOOL_CALLS
        ):
            raise ValueError("expected validator, governance, or tool trajectory is unbounded")
        tool_call_ids = [item.call_id for item in self.tool_calls]
        if len(tool_call_ids) != len(set(tool_call_ids)):
            raise ValueError("expected tool call IDs must be unique")


@dataclass(frozen=True, slots=True)
class CodingEvaluationCase:
    case_id: str
    modes: frozenset[CodingEvaluationMode]
    request: CodingRequest
    repository: CodingRepositoryFixture
    fixture: CodingEvaluationFixture
    composition: CodingEvaluationComposition
    expected: CodingEvaluationExpected
    fixture_digest: str

    def __post_init__(self) -> None:
        _bounded_text(self.case_id, "coding evaluation case ID", 128)
        if not self.modes:
            raise ValueError("coding evaluation case modes must not be empty")
        if len(self.fixture_digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.fixture_digest
        ):
            raise ValueError("coding evaluation fixture digest is malformed")


@dataclass(frozen=True, slots=True)
class CodingEvaluationSuite:
    suite_id: str
    schema_version: int
    cases: tuple[CodingEvaluationCase, ...]

    def __post_init__(self) -> None:
        _bounded_text(self.suite_id, "coding evaluation suite ID", 128)
        if self.schema_version != CODING_EVALUATION_SCHEMA_VERSION:
            raise ValueError("unsupported coding evaluation schema version")
        if not self.cases or len(self.cases) > _MAX_CASES:
            raise ValueError("coding evaluation cases are empty or unbounded")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("coding evaluation case IDs must be unique")


@dataclass(frozen=True, slots=True)
class CodingEvaluationCheck:
    name: str
    passed: bool
    detail: str

    def __post_init__(self) -> None:
        _bounded_text(self.name, "evaluation check name", 128)
        _bounded_text(self.detail, "evaluation check detail", _MAX_REPORT_TEXT)


@dataclass(frozen=True, slots=True)
class CodingEvaluationCaseResult:
    case_id: str
    passed: bool
    evaluation_passed: bool
    cleanup_passed: bool
    fixture_digest: str
    run_id: str | None
    run_state: str | None
    verdict: str | None
    output: str | None
    checks: tuple[CodingEvaluationCheck, ...]
    tool_calls: tuple[Mapping[str, object], ...] = ()
    changed_paths: tuple[str, ...] = ()
    diff: Mapping[str, object] = field(default_factory=dict)
    validators: tuple[Mapping[str, object], ...] = ()
    governance: tuple[Mapping[str, object], ...] = ()
    context: Mapping[str, object] = field(default_factory=dict)
    observation_limitations: tuple[str, ...] = ()
    error: str | None = None
    cleanup_error: str | None = None

    def __post_init__(self) -> None:
        _bounded_text(self.case_id, "evaluation result case ID", 128)
        if len(self.checks) > _MAX_EXPECTED_CHECKS:
            raise ValueError("evaluation checks are unbounded")
        if not isinstance(self.tool_calls, tuple) or len(self.tool_calls) > (
            _MAX_EXPECTED_TOOL_CALLS
        ):
            raise ValueError("evaluation tool-call evidence is unbounded")
        if not isinstance(self.observation_limitations, tuple) or len(
            self.observation_limitations
        ) > _MAX_REPORT_LIMITATIONS:
            raise ValueError("evaluation observation limitations are unbounded")
        for value in self.observation_limitations:
            _bounded_text(value, "evaluation observation limitation", _MAX_REPORT_LIMITATION_TEXT)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "case_id": self.case_id,
            "passed": self.passed,
            "evaluation_passed": self.evaluation_passed,
            "cleanup_passed": self.cleanup_passed,
            "fixture_digest": self.fixture_digest,
            "run_id": self.run_id,
            "run_state": self.run_state,
            "verdict": self.verdict,
            "output": self.output,
            "changed_paths": list(self.changed_paths),
            "diff": dict(self.diff),
            "validators": [dict(value) for value in self.validators],
            "governance": [dict(value) for value in self.governance],
            "context": dict(self.context),
            "observation_limitations": list(self.observation_limitations),
            "checks": [
                {"name": check.name, "passed": check.passed, "detail": check.detail}
                for check in self.checks
            ],
            "error": self.error,
            "cleanup": {
                "passed": self.cleanup_passed,
                "error": self.cleanup_error,
            },
        }
        if self.tool_calls:
            result["tool_calls"] = [dict(value) for value in self.tool_calls]
        return result


@dataclass(frozen=True, slots=True)
class CodingEvaluationReport:
    suite_id: str
    suite_schema_version: int
    mode: CodingEvaluationMode
    generated_at: datetime
    results: tuple[CodingEvaluationCaseResult, ...]
    skipped_case_ids: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    @property
    def deterministic_fingerprint(self) -> str:
        payload = self.to_dict(include_fingerprint=False)
        results = cast(list[dict[str, object]], payload["results"])
        results.sort(key=lambda value: str(value.get("case_id", "")))
        _drop_volatile(payload)
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        result_dicts = [result.to_dict() for result in self.results]
        summary = {
            "passed": self.passed,
            "passed_cases": sum(result.passed for result in self.results),
            "failed_cases": sum(not result.passed for result in self.results),
            "executed_cases": len(self.results),
            "skipped_cases": len(self.skipped_case_ids),
        }
        report: dict[str, object] = {
            "report_schema_version": CODING_EVALUATION_REPORT_SCHEMA_VERSION,
            "suite_id": self.suite_id,
            "suite_schema_version": self.suite_schema_version,
            "mode": self.mode.value,
            "generated_at": self.generated_at.isoformat(),
            "summary": summary,
            "skipped_case_ids": list(self.skipped_case_ids),
            "results": result_dicts,
        }
        if include_fingerprint:
            report["deterministic_fingerprint"] = self.deterministic_fingerprint
        rendered_length = len(json.dumps(report, ensure_ascii=True, separators=(",", ":")))
        if rendered_length > _MAX_REPORT_CHARACTERS:
            raise ValueError("coding evaluation report exceeds its bound")
        return report


class _ScriptedCodingLLM:
    """Fresh, finite model fixture owned by one case only."""

    def __init__(
        self,
        completions: Sequence[Completion],
        *,
        user_message: str,
        context_expectation: CodingEvaluationContextExpectation,
        require_repository_guidance: bool,
    ) -> None:
        self._completions = tuple(completions)
        self._index = 0
        self._user_message = user_message
        self._context_expectation = context_expectation
        self._require_repository_guidance = require_repository_guidance

    @property
    def consumed_all(self) -> bool:
        return self._index == len(self._completions)

    @property
    def consumed(self) -> int:
        return self._index

    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolDefinition] = (),
        *,
        context: RunContext | None = None,
    ) -> Completion:
        message_values = tuple(
            item.content for item in messages if isinstance(item, Message)
        )
        if self._user_message not in message_values:
            raise RuntimeError("coding evaluation active request missing from model context")
        tool_names = {
            item.name for item in tools if isinstance(item, ToolDefinition)
        }
        if not _REQUIRED_CODING_TOOL_NAMES.issubset(tool_names):
            raise RuntimeError("coding evaluation tool definitions missing from model context")
        rendered_context = "\n".join(message_values)
        if self._require_repository_guidance and "[repository-instruction " not in rendered_context:
            raise RuntimeError("coding evaluation repository guidance missing from model context")
        for source in self._context_expectation.selected_instruction_sources:
            marker = f"source={json.dumps(source, ensure_ascii=True)}"
            if marker not in rendered_context:
                raise RuntimeError(
                    "coding evaluation instruction provenance missing from model context"
                )
        selected_skill = self._context_expectation.selected_skill_key
        if self._context_expectation.selected_skill_body and selected_skill is not None:
            marker = f"key={json.dumps(selected_skill, ensure_ascii=True)}"
            if "[skill-body " not in rendered_context or marker not in rendered_context:
                raise RuntimeError("coding evaluation selected skill missing from model context")
        if context is not None:
            context.check_active()
        if self._index >= len(self._completions):
            raise RuntimeError("coding evaluation model fixture was consumed")
        completion = self._completions[self._index]
        self._index += 1
        return completion


class _RunEventCollector:
    """Retain only the bounded production event objects needed by error evaluation."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


class _FailureValidatorRunner:
    def __init__(self, workspace: Workspace, *, empty: bool) -> None:
        self.workspace = workspace
        self._empty = empty

    def run(
        self,
        definitions: Iterable[ValidatorDefinition],
        context: RunContext | None = None,
    ) -> tuple[ValidatorResult, ...]:
        del definitions, context
        if self._empty:
            return ()
        raise RuntimeError("coding evaluation validator failure fixture")


class _FixtureHook:
    def __init__(self, identity: str, outcome: HookOutcome, reason: str) -> None:
        self._identity = identity
        self._outcome = outcome
        self._reason = reason

    @property
    def identity(self) -> str:
        return self._identity

    def __call__(self, input_value: PreActionHookInput | PostActionHookInput) -> HookResult:
        del input_value
        return HookResult(
            outcome=self._outcome,
            reason=self._reason,
            hook_identity=self._identity,
        )


class _IncompleteObservationObserver:
    """Production observer adapter that exposes one bounded incomplete fixture."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self._delegate = WorkspaceObserver(workspace)

    def capture_baseline(
        self,
        *,
        target_paths: Iterable[str | PurePosixPath] = (),
        cancel: object | None = None,
    ) -> WorkspaceSnapshot:
        return self._delegate.capture_baseline(target_paths=target_paths, cancel=cancel)

    def capture_final(
        self,
        *,
        target_paths: Iterable[str | PurePosixPath] = (),
        cancel: object | None = None,
    ) -> WorkspaceSnapshot:
        snapshot = self._delegate.capture_final(target_paths=target_paths, cancel=cancel)
        return replace(
            snapshot,
            completeness=replace(snapshot.completeness, target_complete=False),
        )

    def diff(
        self,
        baseline: WorkspaceSnapshot,
        final: WorkspaceSnapshot,
        *,
        target_paths: Iterable[str | PurePosixPath] = (),
        forbidden_paths: Iterable[str | PurePosixPath] = (),
    ) -> WorkspaceDiff:
        diff = self._delegate.diff(
            baseline,
            final,
            target_paths=target_paths,
            forbidden_paths=forbidden_paths,
        )
        return replace(
            diff,
            completeness=replace(diff.completeness, target_complete=False),
        )


class _DisposableRepository:
    def __init__(self, case: CodingEvaluationCase) -> None:
        self._case = case
        self._temporary = tempfile.TemporaryDirectory(prefix="dqagent-coding-eval-")
        self.root = Path(self._temporary.name)
        try:
            self._materialize()
            repository = case.repository
            self.workspace = Workspace(
                WorkspaceScope(
                    f"t13-{case.case_id}",
                    self.root,
                    protected_paths=repository.protected_paths,
                    secret_paths=repository.secret_paths,
                    volatile_paths=repository.volatile_paths,
                    ignored_paths=repository.ignored_paths,
                )
            )
        except Exception:
            self._temporary.cleanup()
            raise

    def _materialize(self) -> None:
        repository = self._case.repository
        for raw_path, content in repository.files.items():
            target = self.root.joinpath(*PurePosixPath(raw_path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        for raw_root in repository.skill_roots.values():
            self.root.joinpath(*PurePosixPath(raw_root).parts).mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> tuple[bool, str | None]:
        try:
            self._temporary.cleanup()
        except Exception as error:
            return False, f"temporary repository cleanup failed: {type(error).__name__}"
        if self._case.fixture.failure == "cleanup_failure":
            return False, "cleanup failure fixture reported after repository removal"
        return True, None


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _request_digest_payload(request: CodingRequest) -> Mapping[str, object]:
    return {
        "message": request.user_message,
        "targets": [
            value.as_posix() if isinstance(value, PurePosixPath) else value
            for value in request.target_paths
        ],
        "skills": list(request.skill_keys),
    }


def _expected_digest_payload(expected: CodingEvaluationExpected) -> Mapping[str, object]:
    governance: list[dict[str, object]] = []
    for item in expected.governance:
        value: dict[str, object] = {
            "action_kind": item.action_kind.value,
            "policy_outcome": item.policy_outcome.value,
            "approval_outcome": (
                None if item.approval_outcome is None else item.approval_outcome.value
            ),
            "effect_state": item.effect_state.value,
            "executor_attempts": item.executor_attempts,
            "guards_passed": item.guards_passed,
        }
        if item.pre_hook_outcomes:
            value["pre_hook_outcomes"] = [outcome.value for outcome in item.pre_hook_outcomes]
        if item.post_hook_outcomes:
            value["post_hook_outcomes"] = [outcome.value for outcome in item.post_hook_outcomes]
        if item.diagnostics_contains_all:
            value["diagnostics_contains_all"] = list(item.diagnostics_contains_all)
        if item.diagnostics_absent_all:
            value["diagnostics_absent_all"] = list(item.diagnostics_absent_all)
        governance.append(value)

    payload: dict[str, object] = {
        "run": {
            "status": expected.run_status,
            "error_type": expected.error_type,
            "error_category": expected.error_category,
        },
        "answer": {
            "non_empty": expected.answer_non_empty,
            "exact": expected.answer_exact,
            "contains_all": list(expected.answer_contains_all),
            "absent_all": list(expected.answer_absent_all),
        },
        "verdict": expected.verdict,
        "diff": {
            "expected_changes": [
                {
                    "path": item.path.as_posix(),
                    "kind": item.kind.value,
                    "before_sha256": item.before_sha256,
                    "after_sha256": item.after_sha256,
                }
                for item in expected.diff.expected_changes
            ],
            "forbidden_paths": [path.as_posix() for path in expected.diff.forbidden_paths],
            "target_complete": expected.diff.target_complete,
            "forbidden_complete": expected.diff.forbidden_complete,
            "rendered_diff_complete": expected.diff.rendered_diff_complete,
        },
        "validators": [
            {"id": item.validator_id, "status": item.status.value} for item in expected.validators
        ],
        "governance": governance,
        "required_event_types": [item.value for item in expected.required_event_types],
        "limits": {
            "max_model_attempts": expected.limits.max_model_attempts,
            "max_governed_calls": expected.limits.max_governed_calls,
            "max_elapsed_seconds": expected.limits.max_elapsed_seconds,
            "max_rendered_diff_characters": expected.limits.max_rendered_diff_characters,
            "max_validator_output_characters": expected.limits.max_validator_output_characters,
        },
        "fixture_consumed": expected.fixture_consumed,
    }
    if (
        expected.context.selected_instruction_sources
        or expected.context.selected_skill_key is not None
        or expected.context.selected_skill_body is not None
        or expected.context.omission_reasons
    ):
        payload["context"] = {
            "selected_instruction_sources": list(expected.context.selected_instruction_sources),
            "selected_skill_key": expected.context.selected_skill_key,
            "selected_skill_body": expected.context.selected_skill_body,
            "omission_reasons": list(expected.context.omission_reasons),
        }
    if expected.tool_calls:
        payload["tool_calls"] = [
            {
                "call_id": item.call_id,
                "name": item.name,
                "arguments": dict(item.arguments),
                "outcome": item.outcome.value,
                "error_code": None if item.error_code is None else item.error_code.value,
            }
            for item in expected.tool_calls
        ]
    return payload


def compute_coding_fixture_digest(case_data: Mapping[str, object]) -> str:
    """Hash a bounded JSON-native case definition and trusted deterministic inputs."""

    _validate_raw_json_case(case_data)
    raw_case = case_data
    raw_request = cast(Mapping[str, Any], raw_case.get("request", {}))
    raw_repository = cast(Mapping[str, Any], case_data.get("repository", {}))
    raw_fixture = cast(Mapping[str, Any], case_data.get("fixture", {}))
    raw_composition = cast(Mapping[str, Any], case_data.get("composition", {}))
    raw_targets = _bounded_raw_array(raw_request["targets"], "request targets", 128)
    raw_skills = _bounded_raw_array(raw_request.get("skills", []), "request skills", 32)
    raw_executable_allowlist = _bounded_raw_mapping(
        raw_composition.get("executable_allowlist", {}),
        "executable fixture",
        8,
    )
    raw_validators = _bounded_raw_array(
        raw_composition.get("validators", []),
        "validator fixtures",
        _MAX_VALIDATORS,
    )
    raw_secret_names = _bounded_raw_array(
        raw_composition.get("secret_names", []),
        "secret names",
        _MAX_SECRET_NAMES,
    )
    raw_secret_values = _bounded_raw_array(
        raw_composition.get("secret_values", []),
        "secret values",
        _MAX_SECRET_VALUES,
    )
    raw_files = _bounded_raw_mapping(
        raw_repository.get("files", {}),
        "fixture files",
        _MAX_FILES,
    )
    raw_skill_roots = _bounded_raw_mapping(
        raw_repository.get("skill_roots", {}),
        "skill roots",
        16,
    )
    raw_protected_paths = _bounded_raw_array(
        raw_repository.get("protected_paths", []),
        "protected fixture paths",
        _MAX_FIXTURE_PATHS,
    )
    raw_secret_paths = _bounded_raw_array(
        raw_repository.get("secret_paths", []),
        "secret fixture paths",
        _MAX_FIXTURE_PATHS,
    )
    raw_volatile_paths = _bounded_raw_array(
        raw_repository.get("volatile_paths", []),
        "volatile fixture paths",
        _MAX_FIXTURE_PATHS,
    )
    raw_ignored_paths = _bounded_raw_array(
        raw_repository.get("ignored_paths", []),
        "ignored fixture paths",
        _MAX_FIXTURE_PATHS,
    )
    raw_completions = _bounded_raw_array(
        raw_fixture.get("model_completions", []),
        "model completion fixtures",
        _MAX_COMPLETIONS,
    )
    raw_approval_decisions = _bounded_raw_array(
        raw_fixture.get("approval_decisions", []),
        "approval fixtures",
        _MAX_APPROVAL_DECISIONS,
    )
    request = CodingRequest(
        cast(str, raw_request["message"]),
        tuple(cast(Sequence[str | PurePosixPath], raw_targets)),
        tuple(cast(Sequence[str], raw_skills)),
    )
    expected = _parse_expected(cast(Mapping[str, Any], raw_case["expected"]))
    composition = {
        "policy": raw_composition.get("policy", "default"),
        "executable_allowlist": dict(cast(Mapping[str, str], raw_executable_allowlist)),
        "validators": [
            {
                "id": item["id"],
                "argv": list(item["argv"]),
                "cwd": item.get("cwd", "."),
                "timeout_seconds": float(item.get("timeout_seconds", 5.0)),
                "stdout_limit_bytes": int(item.get("stdout_limit_bytes", 4_096)),
                "stderr_limit_bytes": int(item.get("stderr_limit_bytes", 4_096)),
                "accepted_exit_codes": list(item.get("accepted_exit_codes", [0])),
            }
            for item in cast(list[dict[str, Any]], raw_validators)
        ],
        "max_iterations": int(raw_composition.get("max_iterations", 8)),
        "max_governed_calls": int(raw_composition.get("max_governed_calls", 1)),
        "secret_names": list(raw_secret_names),
        "secret_values": list(raw_secret_values),
    }
    hook_fixture = raw_composition.get("hook_fixture", "none")
    if hook_fixture != "none":
        composition["hook_fixture"] = hook_fixture

    selected = {
        "case": {
            "id": raw_case.get("id"),
            "modes": sorted(cast(Sequence[str], raw_case.get("modes", []))),
        },
        "request": _request_digest_payload(request),
        "repository": {
            "files": dict(cast(Mapping[str, str], raw_files)),
            "skill_roots": dict(cast(Mapping[str, str], raw_skill_roots)),
            "protected_paths": list(raw_protected_paths),
            "secret_paths": list(raw_secret_paths),
            "volatile_paths": list(raw_volatile_paths),
            "ignored_paths": list(raw_ignored_paths),
        },
        "fixture": {
            "model_completions": [
                {
                    "content": item.get("content"),
                    "tool_calls": [
                        {
                            "call_id": call["call_id"],
                            "name": call["name"],
                            "arguments": call["arguments"],
                        }
                        for call in cast(list[dict[str, Any]], item.get("tool_calls", []))
                    ],
                    "response_id": item.get("response_id"),
                    "model": item.get("model"),
                    "usage": item.get("usage"),
                }
                for item in cast(list[dict[str, Any]], raw_completions)
            ],
            "approval_decisions": list(raw_approval_decisions),
            "failure": raw_fixture.get("failure", "none"),
        },
        "composition": composition,
        "expected": _expected_digest_payload(expected),
    }
    return hashlib.sha256(_canonical_json(selected).encode("utf-8")).hexdigest()


def _completion_payload(completion: Completion) -> Mapping[str, object]:
    return {
        "content": completion.content,
        "tool_calls": [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": json.loads(call.arguments),
            }
            for call in completion.tool_calls
        ],
        "response_id": completion.response_id,
        "model": completion.model,
        "usage": (
            None
            if completion.usage is None
            else {
                "input_tokens": completion.usage.input_tokens,
                "output_tokens": completion.usage.output_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
        ),
    }


def _report_tool_calls(
    conversation: Sequence[ConversationItem],
    *,
    workspace: Workspace,
    secret_values: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    results = {
        item.call_id: item
        for item in conversation
        if isinstance(item, ToolResult)
    }
    payload: list[Mapping[str, object]] = []
    for call in (
        item for item in conversation if isinstance(item, ToolCall)
    ):
        raw_arguments = _safe_text(
            call.arguments,
            workspace=workspace,
            secret_values=secret_values,
            maximum=_MAX_REPORT_TEXT,
        )
        try:
            arguments: object = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            arguments = {"parse_error": "invalid_json"}
        result = results.get(call.call_id)
        payload.append(
            {
                "call_id": _safe_text(
                    call.call_id,
                    workspace=workspace,
                    secret_values=secret_values,
                    maximum=128,
                ),
                "name": _safe_text(
                    call.name,
                    workspace=workspace,
                    secret_values=secret_values,
                    maximum=128,
                ),
                "arguments": arguments,
                "outcome": None if result is None else result.outcome.value,
                "error_code": (
                    None
                    if result is None or result.error_code is None
                    else result.error_code.value
                ),
            }
        )
        if len(payload) >= _MAX_EXPECTED_TOOL_CALLS:
            break
    return tuple(payload)


def _case_fixture_digest(case: CodingEvaluationCase) -> str:
    repository = case.repository
    composition = case.composition
    fixture = case.fixture
    composition_payload: dict[str, object] = {
        "policy": composition.policy,
        "executable_allowlist": dict(composition.executable_allowlist),
        "validators": [
            {
                "id": item.validator_id,
                "argv": list(item.argv),
                "cwd": item.cwd.as_posix(),
                "timeout_seconds": item.timeout_seconds,
                "stdout_limit_bytes": item.stdout_limit_bytes,
                "stderr_limit_bytes": item.stderr_limit_bytes,
                "accepted_exit_codes": sorted(item.accepted_exit_codes),
            }
            for item in composition.validators
        ],
        "max_iterations": composition.max_iterations,
        "max_governed_calls": composition.max_governed_calls,
        "secret_names": list(composition.secret_names),
        "secret_values": list(composition.secret_values),
    }
    if composition.hook_fixture != "none":
        composition_payload["hook_fixture"] = composition.hook_fixture
    payload = {
        "id": case.case_id,
        "modes": sorted(item.value for item in case.modes),
        "request": _request_digest_payload(case.request),
        "repository": {
            "files": dict(repository.files),
            "skill_roots": dict(repository.skill_roots),
            "protected_paths": [path.as_posix() for path in repository.protected_paths],
            "secret_paths": [path.as_posix() for path in repository.secret_paths],
            "volatile_paths": [path.as_posix() for path in repository.volatile_paths],
            "ignored_paths": [path.as_posix() for path in repository.ignored_paths],
        },
        "fixture": {
            "model_completions": [_completion_payload(item) for item in fixture.model_completions],
            "approval_decisions": [item.value for item in fixture.approval_decisions],
            "failure": fixture.failure,
        },
        "composition": composition_payload,
        "expected": _expected_digest_payload(case.expected),
    }
    return compute_coding_fixture_digest(payload)


def _parse_completion(data: Mapping[str, Any]) -> Completion:
    usage = None
    if "usage" in data:
        raw_usage = cast(dict[str, int], data["usage"])
        usage = TokenUsage(
            input_tokens=raw_usage["input_tokens"],
            output_tokens=raw_usage["output_tokens"],
            total_tokens=raw_usage["total_tokens"],
        )
    return Completion(
        content=cast(str | None, data.get("content")),
        tool_calls=tuple(
            ToolCall(
                cast(str, item["call_id"]),
                cast(str, item["name"]),
                _canonical_json(item["arguments"]),
            )
            for item in cast(list[dict[str, Any]], data.get("tool_calls", []))
        ),
        response_id=cast(str | None, data.get("response_id")),
        model=cast(str | None, data.get("model")),
        usage=usage,
    )


def _parse_validator_fixture(data: Mapping[str, Any]) -> CodingEvaluationValidatorFixture:
    return CodingEvaluationValidatorFixture(
        validator_id=cast(str, data["id"]),
        argv=tuple(cast(list[str], data["argv"])),
        cwd=PurePosixPath(cast(str, data.get("cwd", "."))),
        timeout_seconds=float(data.get("timeout_seconds", 5.0)),
        stdout_limit_bytes=int(data.get("stdout_limit_bytes", 4_096)),
        stderr_limit_bytes=int(data.get("stderr_limit_bytes", 4_096)),
        accepted_exit_codes=frozenset(cast(list[int], data.get("accepted_exit_codes", [0]))),
    )


def _parse_expected(data: Mapping[str, Any]) -> CodingEvaluationExpected:
    raw_answer = cast(dict[str, Any], data["answer"])
    raw_diff = cast(dict[str, Any], data["diff"])
    raw_context = cast(dict[str, Any], data.get("context", {}))
    changes = tuple(
        CodingEvaluationChangeExpectation(
            path=PurePosixPath(cast(str, item["path"])),
            kind=WorkspaceChangeKind(cast(str, item["kind"])),
            before_sha256=cast(str | None, item.get("before_sha256")),
            after_sha256=cast(str | None, item.get("after_sha256")),
        )
        for item in cast(list[dict[str, Any]], raw_diff["expected_changes"])
    )
    raw_limits = cast(dict[str, Any], data["limits"])
    return CodingEvaluationExpected(
        run_status=cast(str, cast(dict[str, Any], data["run"])["status"]),
        answer_non_empty=cast(bool, raw_answer.get("non_empty", True)),
        answer_exact=cast(str | None, raw_answer.get("exact")),
        answer_contains_all=tuple(cast(list[str], raw_answer.get("contains_all", []))),
        answer_absent_all=tuple(cast(list[str], raw_answer.get("absent_all", []))),
        verdict=cast(str | None, data.get("verdict")),
        diff=CodingEvaluationDiffExpectation(
            expected_changes=changes,
            forbidden_paths=tuple(
                PurePosixPath(value) for value in cast(list[str], raw_diff["forbidden_paths"])
            ),
            target_complete=cast(bool, raw_diff.get("target_complete", True)),
            forbidden_complete=cast(bool, raw_diff.get("forbidden_complete", True)),
            rendered_diff_complete=cast(bool, raw_diff.get("rendered_diff_complete", True)),
        ),
        validators=tuple(
            CodingEvaluationValidatorExpectation(
                validator_id=cast(str, item["id"]),
                status=ValidatorStatus(cast(str, item["status"])),
            )
            for item in cast(list[dict[str, Any]], data["validators"])
        ),
        governance=tuple(
            CodingEvaluationGovernanceExpectation(
                action_kind=ActionKind(cast(str, item["action_kind"])),
                policy_outcome=PolicyOutcome(cast(str, item["policy_outcome"])),
                approval_outcome=(
                    ApprovalOutcome(cast(str, item["approval_outcome"]))
                    if item.get("approval_outcome") is not None
                    else None
                ),
                effect_state=EffectState(cast(str, item["effect_state"])),
                executor_attempts=int(item["executor_attempts"]),
                guards_passed=cast(bool, item["guards_passed"]),
                pre_hook_outcomes=tuple(
                    HookOutcome(value)
                    for value in cast(list[str], item.get("pre_hook_outcomes", []))
                ),
                post_hook_outcomes=tuple(
                    HookOutcome(value)
                    for value in cast(list[str], item.get("post_hook_outcomes", []))
                ),
                diagnostics_contains_all=tuple(
                    cast(list[str], item.get("diagnostics_contains_all", []))
                ),
                diagnostics_absent_all=tuple(
                    cast(list[str], item.get("diagnostics_absent_all", []))
                ),
            )
            for item in cast(list[dict[str, Any]], data["governance"])
        ),
        tool_calls=tuple(
            CodingEvaluationToolCallExpectation(
                call_id=cast(str, item["call_id"]),
                name=cast(str, item["name"]),
                arguments=cast(dict[str, object], item["arguments"]),
                outcome=ToolOutcome(cast(str, item["outcome"])),
                error_code=(
                    ToolErrorCode(cast(str, item["error_code"]))
                    if item.get("error_code") is not None
                    else None
                ),
            )
            for item in cast(list[dict[str, Any]], data.get("tool_calls", []))
        ),
        required_event_types=tuple(
            RunEventType(value) for value in cast(list[str], data["required_event_types"])
        ),
        limits=CodingEvaluationLimits(
            max_model_attempts=int(raw_limits["max_model_attempts"]),
            max_governed_calls=int(raw_limits["max_governed_calls"]),
            max_elapsed_seconds=float(raw_limits["max_elapsed_seconds"]),
            max_rendered_diff_characters=int(raw_limits["max_rendered_diff_characters"]),
            max_validator_output_characters=int(raw_limits["max_validator_output_characters"]),
        ),
        context=CodingEvaluationContextExpectation(
            selected_instruction_sources=tuple(
                cast(list[str], raw_context.get("selected_instruction_sources", []))
            ),
            selected_skill_key=cast(str | None, raw_context.get("selected_skill_key")),
            selected_skill_body=cast(bool | None, raw_context.get("selected_skill_body")),
            omission_reasons=tuple(
                cast(list[str], raw_context.get("omission_reasons", []))
            ),
        ),
        error_type=cast(str | None, cast(dict[str, Any], data["run"]).get("error_type")),
        error_category=cast(str | None, cast(dict[str, Any], data["run"]).get("error_category")),
        fixture_consumed=cast(bool, data.get("fixture_consumed", True)),
    )


def _parse_case(data: Mapping[str, Any]) -> CodingEvaluationCase:
    raw_request = cast(dict[str, Any], data["request"])
    raw_repository = cast(dict[str, Any], data["repository"])
    raw_fixture = cast(dict[str, Any], data["fixture"])
    raw_composition = cast(dict[str, Any], data["composition"])
    repository = CodingRepositoryFixture(
        files=cast(dict[str, str], raw_repository["files"]),
        skill_roots=cast(dict[str, str], raw_repository.get("skill_roots", {})),
        protected_paths=tuple(
            PurePosixPath(value) for value in raw_repository.get("protected_paths", [])
        ),
        secret_paths=tuple(
            PurePosixPath(value) for value in raw_repository.get("secret_paths", [])
        ),
        volatile_paths=tuple(
            PurePosixPath(value) for value in raw_repository.get("volatile_paths", [])
        ),
        ignored_paths=tuple(
            PurePosixPath(value) for value in raw_repository.get("ignored_paths", [])
        ),
    )
    fixture = CodingEvaluationFixture(
        model_completions=tuple(
            _parse_completion(item)
            for item in cast(list[dict[str, Any]], raw_fixture["model_completions"])
        ),
        approval_decisions=tuple(
            ApprovalOutcome(value)
            for value in cast(list[str], raw_fixture.get("approval_decisions", []))
        ),
        failure=cast(str, raw_fixture.get("failure", "none")),
    )
    composition = CodingEvaluationComposition(
        policy=cast(str, raw_composition.get("policy", "default")),
        executable_allowlist=cast(dict[str, str], raw_composition.get("executable_allowlist", {})),
        validators=tuple(
            _parse_validator_fixture(item)
            for item in cast(list[dict[str, Any]], raw_composition.get("validators", []))
        ),
        max_iterations=int(raw_composition.get("max_iterations", 8)),
        max_governed_calls=int(raw_composition.get("max_governed_calls", 1)),
        secret_names=tuple(cast(list[str], raw_composition.get("secret_names", []))),
        secret_values=tuple(cast(list[str], raw_composition.get("secret_values", []))),
        hook_fixture=cast(str, raw_composition.get("hook_fixture", "none")),
    )
    return CodingEvaluationCase(
        case_id=cast(str, data["id"]),
        modes=frozenset(CodingEvaluationMode(value) for value in cast(list[str], data["modes"])),
        request=CodingRequest(
            cast(str, raw_request["message"]),
            tuple(cast(list[str], raw_request["targets"])),
            tuple(cast(list[str], raw_request.get("skills", []))),
        ),
        repository=repository,
        fixture=fixture,
        composition=composition,
        expected=_parse_expected(cast(dict[str, Any], data["expected"])),
        fixture_digest=cast(str, data["fixture_digest"]),
    )


def _validate_case_semantics(case: CodingEvaluationCase, raw_case: Mapping[str, Any]) -> None:
    expected_validator_ids = [item.validator_id for item in case.expected.validators]
    configured_validator_ids = [item.validator_id for item in case.composition.validators]
    if expected_validator_ids != configured_validator_ids:
        raise CodingEvaluationDefinitionError(
            f"case '{case.case_id}' validator expectations must match trusted composition order"
        )
    if len(case.expected.required_event_types) > _MAX_EXPECTED_CHECKS:
        raise CodingEvaluationDefinitionError(
            f"case '{case.case_id}' event requirements are unbounded"
        )
    actual_digest = compute_coding_fixture_digest(raw_case)
    if actual_digest != case.fixture_digest:
        raise CodingEvaluationDefinitionError(f"case '{case.case_id}' fixture digest mismatch")
    expected_paths = {item.path for item in case.expected.diff.expected_changes}
    if len(expected_paths) != len(case.expected.diff.expected_changes):
        raise CodingEvaluationDefinitionError(
            f"case '{case.case_id}' expected diff paths must be unique"
        )
    if case.expected.run_status == "error" and case.expected.verdict is not None:
        raise CodingEvaluationDefinitionError(
            f"case '{case.case_id}' error case cannot require a verdict"
        )


def load_coding_evaluation_suite(path: Path) -> CodingEvaluationSuite:
    """Load a schema-versioned disposable coding evaluation suite."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CodingEvaluationDefinitionError(
            f"cannot load coding evaluation suite '{path}': {error}"
        ) from error
    if not isinstance(raw, dict):
        raise CodingEvaluationDefinitionError("coding evaluation suite must be a JSON object")
    data = cast(dict[str, Any], raw)
    if data.get("schema_version") != CODING_EVALUATION_SCHEMA_VERSION:
        raise CodingEvaluationDefinitionError(
            f"unsupported coding evaluation schema version: {data.get('schema_version')!r}"
        )
    try:
        Draft202012Validator(_CODING_EVALUATION_SCHEMA).validate(data)
    except ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path) or "root"
        raise CodingEvaluationDefinitionError(
            f"invalid coding evaluation suite at {location}: {error.message}"
        ) from error
    try:
        cases = tuple(_parse_case(cast(dict[str, Any], item)) for item in data["cases"])
        suite = CodingEvaluationSuite(
            suite_id=cast(str, data["suite_id"]),
            schema_version=cast(int, data["schema_version"]),
            cases=cases,
        )
        for raw_case, case in zip(cast(list[dict[str, Any]], data["cases"]), cases, strict=True):
            _validate_case_semantics(case, raw_case)
        return suite
    except CodingEvaluationDefinitionError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise CodingEvaluationDefinitionError(
            f"invalid coding evaluation suite: {error}"
        ) from error


def _resolve_fixture_argv(argv: Iterable[str]) -> tuple[str, ...]:
    return tuple(sys.executable if value == "{python}" else value for value in argv)


def _validator_definitions(
    validators: Sequence[CodingEvaluationValidatorFixture],
) -> tuple[ValidatorDefinition, ...]:
    return tuple(
        ValidatorDefinition(
            item.validator_id,
            _resolve_fixture_argv(item.argv),
            cwd=item.cwd,
            timeout_seconds=item.timeout_seconds,
            stdout_limit_bytes=item.stdout_limit_bytes,
            stderr_limit_bytes=item.stderr_limit_bytes,
            accepted_exit_codes=item.accepted_exit_codes,
        )
        for item in validators
    )


def _safe_text(
    value: object,
    *,
    workspace: Workspace | None,
    secret_values: Sequence[str],
    maximum: int,
    fallback: str = "[unavailable]",
) -> str:
    if not isinstance(value, str):
        return fallback
    try:
        if workspace is not None:
            rendered = workspace.sanitize(value, secrets=secret_values, max_characters=maximum)
        else:
            rendered = value
            for secret in sorted(secret_values, key=len, reverse=True):
                if secret:
                    rendered = rendered.replace(secret, "[REDACTED]")
            rendered = rendered[:maximum]
    except Exception:
        return fallback
    return "".join(
        character if ord(character) >= 32 or character in "\t\r\n" else "?"
        for character in rendered
    )


def _safe_observation_limitations(
    values: Iterable[str],
    *,
    diff: Any | None,
    action_records: Sequence[ActionRecord],
    workspace: Workspace | None,
    secret_values: Sequence[str],
) -> tuple[str, ...]:
    raw_values = list(values)
    if diff is not None:
        completeness = diff.completeness
        if not completeness.target_complete:
            raw_values.append("target_observation_incomplete")
        if not completeness.forbidden_complete:
            raw_values.append("forbidden_observation_incomplete")
        if not completeness.rendered_diff_complete:
            raw_values.append("rendered_diff_incomplete")
    if any(record.effect_state.value == "unknown" for record in action_records):
        raw_values.append("workspace_effect_unknown")

    safe_values: list[str] = []
    for value in raw_values[:_MAX_REPORT_LIMITATIONS]:
        rendered = _safe_text(
            value,
            workspace=workspace,
            secret_values=secret_values,
            maximum=_MAX_REPORT_LIMITATION_TEXT,
        )
        if rendered not in safe_values:
            safe_values.append(rendered)
    return tuple(safe_values[:_MAX_REPORT_LIMITATIONS])


def _safe_check(
    name: str,
    passed: bool,
    detail: str,
    *,
    workspace: Workspace | None,
    secret_values: Sequence[str],
) -> CodingEvaluationCheck:
    return CodingEvaluationCheck(
        name,
        bool(passed),
        _safe_text(
            detail, workspace=workspace, secret_values=secret_values, maximum=_MAX_REPORT_TEXT
        ),
    )


def _is_subsequence(required: Sequence[RunEventType], actual: Sequence[RunEventType]) -> bool:
    position = 0
    for event_type in actual:
        if position < len(required) and event_type is required[position]:
            position += 1
    return position == len(required)


def _action_guards_passed(record: ActionRecord) -> bool:
    return len(record.guard_results) == len(HARD_GUARD_ORDER) and all(
        result.passed for result in record.guard_results
    )


def _change_matches(
    actual: WorkspaceChange,
    expected: CodingEvaluationChangeExpectation,
) -> bool:
    return (
        actual.logical_path == expected.path
        and actual.kind is expected.kind
        and (
            expected.before_sha256 is None
            or (actual.before is not None and actual.before.digest == expected.before_sha256)
        )
        and (
            expected.after_sha256 is None
            or (actual.after is not None and actual.after.digest == expected.after_sha256)
        )
    )


def _path_forbidden(path: PurePosixPath, forbidden: Sequence[PurePosixPath]) -> bool:
    return any(
        item == PurePosixPath(".") or path == item or item in path.parents for item in forbidden
    )


def _event_metrics(events: Sequence[RunEvent]) -> tuple[int, float]:
    return (
        sum(event.type is RunEventType.MODEL_REQUEST_STARTED for event in events),
        events[-1].elapsed_seconds if events else 0.0,
    )


def _report_governance(
    records: Sequence[ActionRecord],
    *,
    workspace: Workspace | None,
    secret_values: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "action_kind": record.action_kind.value,
            "effect_kind": record.effect_kind.value,
            "policy_outcome": record.policy_decision.outcome.value,
            "approval_outcome": (
                None if record.approval_outcome is None else record.approval_outcome.value
            ),
            "effect_state": record.effect_state.value,
            "executor_attempts": record.executor_attempts,
            "guards_passed": _action_guards_passed(record),
            "action_digest": record.action_digest,
            "action_display": _safe_text(
                record.sanitized_action_display,
                workspace=workspace,
                secret_values=secret_values,
                maximum=_MAX_REPORT_TEXT,
            ),
            "diagnostics": [
                _safe_text(
                    value,
                    workspace=workspace,
                    secret_values=secret_values,
                    maximum=256,
                )
                for value in record.sanitized_diagnostics[:8]
            ],
            "pre_hooks": [
                {
                    "identity": item.hook_identity,
                    "outcome": item.outcome.value,
                    "mode": None if item.mode is None else item.mode.value,
                }
                for item in record.pre_hook_results
            ],
            "post_hooks": [
                {
                    "identity": item.hook_identity,
                    "outcome": item.outcome.value,
                    "mode": None if item.mode is None else item.mode.value,
                }
                for item in record.post_hook_results
            ],
            "observation_failure": record.observation_failure,
        }
        for record in records
    )


def _report_validators(
    validators: Sequence[ValidatorResult],
    *,
    workspace: Workspace | None,
    secret_values: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "validator_id": _safe_text(
                result.validator_id,
                workspace=workspace,
                secret_values=secret_values,
                maximum=128,
            ),
            "status": result.status.value,
            "returncode": result.returncode,
            "stdout": _safe_text(
                result.stdout,
                workspace=workspace,
                secret_values=secret_values,
                maximum=_MAX_REPORT_VALIDATOR_OUTPUT,
            ),
            "stderr": _safe_text(
                result.stderr,
                workspace=workspace,
                secret_values=secret_values,
                maximum=_MAX_REPORT_VALIDATOR_OUTPUT,
            ),
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
            "cleanup_succeeded": result.cleanup.succeeded,
            "spawned": result.spawned,
            "diagnostics": [
                _safe_text(
                    value,
                    workspace=workspace,
                    secret_values=secret_values,
                    maximum=256,
                )
                for value in result.diagnostics[:8]
            ],
        }
        for result in validators
    )


def _report_context(
    repository_context: Any,
    context_window: Any,
) -> Mapping[str, object]:
    """Retain only the existing content-free context evidence projections."""

    return {
        "repository": dict(repository_context.event_attributes()),
        "window": dict(context_window.event_attributes()),
    }


class CodingEvaluationRunner:
    """Run each case through a fresh production coding application."""

    def __init__(self, mode: CodingEvaluationMode = CodingEvaluationMode.DETERMINISTIC) -> None:
        if not isinstance(mode, CodingEvaluationMode):
            raise ValueError("unsupported coding evaluation mode")
        self._mode = mode

    def run(self, suite: CodingEvaluationSuite) -> CodingEvaluationReport:
        if not isinstance(suite, CodingEvaluationSuite):
            raise TypeError("coding evaluation runner requires a CodingEvaluationSuite")
        results: list[CodingEvaluationCaseResult] = []
        skipped: list[str] = []
        for case in suite.cases:
            if self._mode not in case.modes:
                skipped.append(case.case_id)
                continue
            if _case_fixture_digest(case) != case.fixture_digest:
                raise CodingEvaluationDefinitionError(
                    f"case '{case.case_id}' fixture digest mismatch"
                )
            results.append(self._run_case(case))
        return CodingEvaluationReport(
            suite_id=suite.suite_id,
            suite_schema_version=suite.schema_version,
            mode=self._mode,
            generated_at=datetime.now(UTC),
            results=tuple(results),
            skipped_case_ids=tuple(skipped),
        )

    def _run_case(self, case: CodingEvaluationCase) -> CodingEvaluationCaseResult:
        repository: _DisposableRepository | None = None
        workspace: Workspace | None = None
        secret_values = case.composition.secret_values
        checks: list[CodingEvaluationCheck] = []
        evaluation_passed = False
        run_id: str | None = None
        run_state: str | None = None
        verdict: str | None = None
        output: str | None = None
        error_text: str | None = None
        changed_paths: tuple[str, ...] = ()
        diff_payload: Mapping[str, object] = {}
        validators_payload: tuple[Mapping[str, object], ...] = ()
        governance_payload: tuple[Mapping[str, object], ...] = ()
        tool_calls_payload: tuple[Mapping[str, object], ...] = ()
        context_payload: Mapping[str, object] = {}
        observation_limitations: tuple[str, ...] = ()
        model = _ScriptedCodingLLM(
            case.fixture.model_completions,
            user_message=case.request.user_message,
            context_expectation=case.expected.context,
            require_repository_guidance="AGENTS.md" in case.repository.files,
        )
        event_collector = _RunEventCollector()
        try:
            repository = _DisposableRepository(case)
            workspace = repository.workspace
            application = self._application_for(
                case,
                repository.workspace,
                model,
                event_collector,
            )
            try:
                result = application.run(case.request)
            except Exception as error:
                error_text = _safe_text(
                    str(error),
                    workspace=workspace,
                    secret_values=secret_values,
                    maximum=_MAX_REPORT_TEXT,
                )
                evidence = getattr(error, "coding_failure_evidence", None)
                if isinstance(evidence, CodingFailureEvidence):
                    run_id = getattr(evidence, "run_id", None)
                    evidence_action_records = evidence.action_records
                    evidence_validators = evidence.validator_results
                    evidence_diff = evidence.diff
                    validators_payload = _report_validators(
                        evidence_validators,
                        workspace=workspace,
                        secret_values=secret_values,
                    )
                    governance_payload = _report_governance(
                        evidence_action_records,
                        workspace=workspace,
                        secret_values=secret_values,
                    )
                    if evidence_diff is not None:
                        changed_paths = tuple(
                            path.as_posix() for path in evidence_diff.changed_paths
                        )
                        diff_payload = self._diff_payload(evidence_diff, workspace, secret_values)
                    observation_limitations = _safe_observation_limitations(
                        evidence.observation_limitations,
                        diff=evidence_diff,
                        action_records=evidence_action_records,
                        workspace=workspace,
                        secret_values=secret_values,
                    )
                    if (
                        evidence.repository_context is not None
                        and evidence.context_window is not None
                    ):
                        context_payload = _report_context(
                            evidence.repository_context,
                            evidence.context_window,
                        )
                checks.extend(
                    self._evaluate_error(
                        case,
                        error,
                        evidence if isinstance(evidence, CodingFailureEvidence) else None,
                        event_collector.events,
                        workspace,
                        secret_values,
                    )
                )
            else:
                if not isinstance(result, CodingRunResult):
                    raise TypeError("production coding path returned an invalid result")
                run_id = result.run_id
                run_state = result.state.value
                verdict = result.verdict.value
                output = _safe_text(
                    result.output.content,
                    workspace=workspace,
                    secret_values=secret_values,
                    maximum=_MAX_REPORT_TEXT,
                )
                changed_paths = tuple(path.as_posix() for path in result.diff.changed_paths)
                diff_payload = self._diff_payload(result.diff, workspace, secret_values)
                validators_payload = _report_validators(
                    result.validator_results,
                    workspace=workspace,
                    secret_values=secret_values,
                )
                governance_payload = _report_governance(
                    result.action_records,
                    workspace=workspace,
                    secret_values=secret_values,
                )
                tool_calls_payload = _report_tool_calls(
                    result.agent.conversation,
                    workspace=workspace,
                    secret_values=secret_values,
                )
                context_payload = _report_context(
                    result.repository_context,
                    result.context_window,
                )
                observation_limitations = _safe_observation_limitations(
                    result.observation_limitations,
                    diff=result.diff,
                    action_records=result.action_records,
                    workspace=workspace,
                    secret_values=secret_values,
                )
                checks.extend(self._evaluate_result(case, result, workspace, secret_values))
            checks.append(
                _safe_check(
                    "fixture.model_completions_consumed",
                    model.consumed_all == case.expected.fixture_consumed,
                    (
                        f"expected fixture_consumed={case.expected.fixture_consumed}, "
                        f"consumed={model.consumed}/{len(case.fixture.model_completions)}"
                    ),
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
            evaluation_passed = all(check.passed for check in checks)
        except Exception as error:
            if workspace is None:
                error_text = f"pre_workspace_failure:{type(error).__name__}"
                failure_detail = f"runner failed before completing case: {error_text}"
            else:
                error_text = _safe_text(
                    str(error),
                    workspace=workspace,
                    secret_values=secret_values,
                    maximum=_MAX_REPORT_TEXT,
                )
                failure_detail = (
                    f"runner failed before completing case: {type(error).__name__}: {error}"
                )
            checks.append(
                _safe_check(
                    "runner.failure",
                    False,
                    failure_detail,
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
            evaluation_passed = False
        finally:
            cleanup_passed = False
            cleanup_error: str | None = "temporary repository was not materialized"
            if repository is not None:
                cleanup_passed, cleanup_error = repository.cleanup()
            checks.append(
                _safe_check(
                    "cleanup.completed",
                    cleanup_passed,
                    cleanup_error or "temporary repository cleaned up",
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
        return CodingEvaluationCaseResult(
            case_id=case.case_id,
            passed=evaluation_passed and cleanup_passed,
            evaluation_passed=evaluation_passed,
            cleanup_passed=cleanup_passed,
            fixture_digest=case.fixture_digest,
            run_id=run_id,
            run_state=run_state,
            verdict=verdict,
            output=output,
            checks=tuple(checks),
            tool_calls=tool_calls_payload,
            changed_paths=changed_paths,
            diff=diff_payload,
            validators=validators_payload,
            governance=governance_payload,
            context=context_payload,
            observation_limitations=observation_limitations,
            error=error_text,
            cleanup_error=cleanup_error,
        )

    def _application_for(
        self,
        case: CodingEvaluationCase,
        workspace: Workspace,
        model: LLMClient,
        event_collector: _RunEventCollector,
    ) -> CodingAgentApplication:
        composition = case.composition
        validator_runner: object | None = None
        if case.fixture.failure == "validator_runner_error":
            validator_runner = _FailureValidatorRunner(workspace, empty=False)
        elif case.fixture.failure == "validator_runner_empty":
            validator_runner = _FailureValidatorRunner(workspace, empty=True)
        executable_allowlist = {
            identity: Path(sys.executable) for identity in composition.executable_allowlist
        }
        if case.fixture.failure == "reject_then_stale_approval":
            def stale_approval(request: object, context: object) -> ApprovalDecision:
                del request, context
                return ApprovalDecision(
                    ApprovalOutcome.DRIFT,
                    "fixture_stale_approval",
                    provider_identity="stale-approval-fixture",
                )

            approval_provider: object = ScriptedApprovalProvider(
                (ApprovalOutcome.REJECT, stale_approval)
            )
        else:
            approval_provider = ScriptedApprovalProvider(case.fixture.approval_decisions)
        kwargs: dict[str, object] = {
            "policy": DefaultActionPolicy(),
            "approval_provider": approval_provider,
            "validators": _validator_definitions(composition.validators),
            "executable_allowlist": executable_allowlist,
            "skill_roots": {
                key: workspace.root / PurePosixPath(root)
                for key, root in case.repository.skill_roots.items()
            },
            "secret_names": composition.secret_names,
            "secret_values": composition.secret_values,
            "forbidden_paths": case.expected.diff.forbidden_paths,
            "max_iterations": composition.max_iterations,
            "max_governed_calls": composition.max_governed_calls,
            "event_sinks": (event_collector,),
        }
        if case.fixture.failure == "incomplete_observation":
            kwargs["observer"] = _IncompleteObservationObserver(workspace)
        if composition.hook_fixture == "required_pre_hook_failure":
            kwargs["pre_hooks"] = (
                PreActionHookSpec(
                    cast(
                        PreActionHook,
                        _FixtureHook(
                            "t14-required-pre-hook",
                            HookOutcome.FAILED,
                            "fixture_required_pre_hook_failure",
                        ),
                    ),
                    HookMode.REQUIRED,
                ),
            )
        elif composition.hook_fixture == "post_hook_failure":
            kwargs["post_hooks"] = (
                _FixtureHook(
                    "t14-post-hook",
                    HookOutcome.FAILED,
                    "fixture_post_hook_failure",
                ),
            )
        if validator_runner is not None:
            kwargs["validator_runner"] = validator_runner
        return CodingAgentApplication.create(workspace, model, **kwargs)

    @staticmethod
    def _diff_payload(
        diff: Any,
        workspace: Workspace | None,
        secret_values: Sequence[str],
    ) -> Mapping[str, object]:
        blind_spot_reasons = [
            _safe_text(
                item.reason_code,
                workspace=workspace,
                secret_values=secret_values,
                maximum=_MAX_REPORT_LIMITATION_TEXT,
            )
            for item in diff.completeness.blind_spots[:_MAX_REPORT_LIMITATIONS]
        ]
        if diff.completeness.rendered_diff_omission_reason is not None:
            blind_spot_reasons.append(
                _safe_text(
                    str(diff.completeness.rendered_diff_omission_reason),
                    workspace=workspace,
                    secret_values=secret_values,
                    maximum=_MAX_REPORT_LIMITATION_TEXT,
                )
            )
        return {
            "changed_paths": [change.logical_path.as_posix() for change in diff.changes],
            "change_kinds": [change.kind.value for change in diff.changes],
            "target_complete": diff.completeness.target_complete,
            "forbidden_complete": diff.completeness.forbidden_complete,
            "rendered_diff_complete": diff.completeness.rendered_diff_complete,
            "blind_spot_count": len(diff.completeness.blind_spots),
            "blind_spot_reasons": list(dict.fromkeys(blind_spot_reasons))[
                :_MAX_REPORT_LIMITATIONS
            ],
            "rendered_diff": _safe_text(
                diff.rendered_diff,
                workspace=workspace,
                secret_values=secret_values,
                maximum=_MAX_REPORT_DIFF,
            ),
        }

    def _evaluate_error(
        self,
        case: CodingEvaluationCase,
        error: Exception,
        evidence: CodingFailureEvidence | None,
        events: Sequence[RunEvent],
        workspace: Workspace,
        secret_values: Sequence[str],
    ) -> tuple[CodingEvaluationCheck, ...]:
        expected = case.expected
        checks: list[CodingEvaluationCheck] = [
            _safe_check(
                "run.status",
                expected.run_status == "error",
                f"expected run status {expected.run_status!r}, observed error",
                workspace=workspace,
                secret_values=secret_values,
            )
        ]
        if expected.error_type is not None:
            checks.append(
                _safe_check(
                    "run.error_type",
                    type(error).__name__ == expected.error_type,
                    (
                        f"expected error type {expected.error_type!r}, "
                        f"observed {type(error).__name__!r}"
                    ),
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
        if expected.error_category is not None:
            actual_category = getattr(getattr(error, "category", None), "value", None)
            checks.append(
                _safe_check(
                    "run.error_category",
                    actual_category == expected.error_category,
                    (
                        f"expected error category {expected.error_category!r}, "
                        f"observed {actual_category!r}"
                    ),
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
        checks.extend(
            self._evaluate_tool_calls(
                case,
                None,
                workspace,
                secret_values,
            )
        )
        if evidence is None:
            checks.append(
                _safe_check(
                    "failure.evidence_available",
                    False,
                    "failure evidence is unavailable",
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
            return tuple(checks)

        checks.append(
            _safe_check(
                "failure.evidence_available",
                True,
                "bounded failure evidence is available",
                workspace=workspace,
                secret_values=secret_values,
            )
        )
        if (
            expected.answer_non_empty
            or expected.answer_exact is not None
            or expected.answer_contains_all
            or expected.answer_absent_all
        ):
            checks.append(
                _safe_check(
                    "answer.evidence_available",
                    False,
                    "failure evidence does not contain a final answer",
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
        checks.extend(self._evaluate_diff(case, evidence.diff, workspace, secret_values))
        checks.extend(
            self._evaluate_validators(case, evidence.validator_results, workspace, secret_values)
        )
        checks.extend(
            self._evaluate_governance(case, evidence.action_records, workspace, secret_values)
        )
        checks.extend(self._evaluate_events(case, events, workspace, secret_values))
        checks.extend(
            self._evaluate_limits(
                case,
                events,
                evidence.action_records,
                evidence.diff,
                evidence.validator_results,
                workspace,
                secret_values,
            )
        )
        return tuple(checks)

    def _evaluate_result(
        self,
        case: CodingEvaluationCase,
        result: CodingRunResult,
        workspace: Workspace,
        secret_values: Sequence[str],
    ) -> tuple[CodingEvaluationCheck, ...]:
        expected = case.expected
        checks: list[CodingEvaluationCheck] = []
        checks.append(
            _safe_check(
                "run.status",
                expected.run_status == "completed",
                f"expected run status {expected.run_status!r}, observed completed",
                workspace=workspace,
                secret_values=secret_values,
            )
        )
        checks.extend(
            self._evaluate_tool_calls(
                case,
                result.agent.conversation,
                workspace,
                secret_values,
            )
        )
        output = result.output.content
        checks.append(
            _safe_check(
                "answer.non_empty",
                not expected.answer_non_empty or bool(output.strip()),
                "final answer is non-empty" if output.strip() else "final answer is empty",
                workspace=workspace,
                secret_values=secret_values,
            )
        )
        if expected.answer_exact is not None:
            checks.append(
                _safe_check(
                    "answer.exact",
                    output == expected.answer_exact,
                    f"expected {expected.answer_exact!r}, observed {output!r}",
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
        if expected.answer_contains_all:
            missing = [
                value
                for value in expected.answer_contains_all
                if value.casefold() not in output.casefold()
            ]
            checks.append(
                _safe_check(
                    "answer.contains_all",
                    not missing,
                    "all required fragments found"
                    if not missing
                    else f"missing fragments: {missing}",
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
        if expected.answer_absent_all:
            present = [
                value
                for value in expected.answer_absent_all
                if value.casefold() in output.casefold()
            ]
            checks.append(
                _safe_check(
                    "answer.absent_all",
                    not present,
                    "no forbidden fragments found"
                    if not present
                    else f"forbidden fragments found: {present}",
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
        checks.extend(self._evaluate_diff(case, result.diff, workspace, secret_values))
        checks.extend(
            self._evaluate_validators(case, result.validator_results, workspace, secret_values)
        )
        checks.extend(
            self._evaluate_governance(case, result.action_records, workspace, secret_values)
        )
        checks.extend(
            self._evaluate_context(
                case,
                result.repository_context,
                result.context_window,
                workspace,
                secret_values,
            )
        )
        checks.extend(self._evaluate_events(case, result.events, workspace, secret_values))
        if expected.verdict is not None:
            checks.append(
                _safe_check(
                    "verdict.exact",
                    result.verdict.value == expected.verdict,
                    f"expected verdict {expected.verdict!r}, observed {result.verdict.value!r}",
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
        checks.extend(
            self._evaluate_limits(
                case,
                result.events,
                result.action_records,
                result.diff,
                result.validator_results,
                workspace,
                secret_values,
            )
        )
        return tuple(checks)

    @staticmethod
    def _evaluate_tool_calls(
        case: CodingEvaluationCase,
        conversation: Sequence[ConversationItem] | None,
        workspace: Workspace,
        secret_values: Sequence[str],
    ) -> tuple[CodingEvaluationCheck, ...]:
        expected = case.expected.tool_calls
        if not expected:
            return ()
        if conversation is None:
            return (
                _safe_check(
                    "tools.evidence_available",
                    False,
                    "tool-call trajectory evidence is unavailable",
                    workspace=workspace,
                    secret_values=secret_values,
                ),
            )

        actual_calls = [item for item in conversation if isinstance(item, ToolCall)]
        actual_results = [item for item in conversation if isinstance(item, ToolResult)]
        results_by_call_id = {item.call_id: item for item in actual_results}
        checks = [
            _safe_check(
                "tools.count",
                len(actual_calls) == len(expected) and len(actual_results) == len(expected),
                (
                    f"expected {len(expected)} tool calls/results, observed "
                    f"{len(actual_calls)} calls/{len(actual_results)} results"
                ),
                workspace=workspace,
                secret_values=secret_values,
            )
        ]
        for index, wanted in enumerate(expected):
            if index >= len(actual_calls):
                checks.append(
                    _safe_check(
                        f"tools.call[{index}]",
                        False,
                        f"expected tool call {wanted.call_id!r} is missing",
                        workspace=workspace,
                        secret_values=secret_values,
                    )
                )
                continue
            actual_call = actual_calls[index]
            actual_result = results_by_call_id.get(actual_call.call_id)
            try:
                actual_arguments: object = json.loads(actual_call.arguments)
            except (TypeError, json.JSONDecodeError):
                actual_arguments = None
            observed = (
                None
                if actual_result is None
                else (
                    actual_result.call_id,
                    actual_result.name,
                    actual_result.outcome.value,
                    None
                    if actual_result.error_code is None
                    else actual_result.error_code.value,
                )
            )
            passed = (
                actual_call.call_id == wanted.call_id
                and actual_call.name == wanted.name
                and actual_arguments == dict(wanted.arguments)
                and actual_result is not None
                and actual_result.call_id == wanted.call_id
                and actual_result.name == wanted.name
                and actual_result.outcome is wanted.outcome
                and actual_result.error_code is wanted.error_code
            )
            checks.append(
                _safe_check(
                    f"tools.call[{index}]",
                    passed,
                    (
                        f"expected {wanted.call_id}/{wanted.name} "
                        f"{dict(wanted.arguments)!r} -> "
                        f"{wanted.outcome.value}/"
                        f"{wanted.error_code.value if wanted.error_code else None}, "
                        f"observed {actual_call.call_id}/{actual_call.name} "
                        f"{actual_arguments!r} -> {observed!r}"
                    ),
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
        return tuple(checks)

    def _evaluate_diff(
        self,
        case: CodingEvaluationCase,
        diff: Any | None,
        workspace: Workspace,
        secret_values: Sequence[str],
    ) -> tuple[CodingEvaluationCheck, ...]:
        if diff is None:
            return (
                _safe_check(
                    "diff.evidence_available",
                    False,
                    "workspace diff evidence is unavailable",
                    workspace=workspace,
                    secret_values=secret_values,
                ),
            )
        expected = case.expected.diff
        actual = diff.changes
        checks: list[CodingEvaluationCheck] = [
            _safe_check(
                "diff.exact_changes",
                len(actual) == len(expected.expected_changes)
                and all(
                    _change_matches(item, wanted)
                    for item, wanted in zip(actual, expected.expected_changes, strict=False)
                ),
                f"expected {len(expected.expected_changes)} exact changes, observed {len(actual)}",
                workspace=workspace,
                secret_values=secret_values,
            ),
            _safe_check(
                "diff.forbidden_paths",
                not any(
                    _path_forbidden(item.logical_path, expected.forbidden_paths) for item in actual
                ),
                "no forbidden path changed"
                if not any(
                    _path_forbidden(item.logical_path, expected.forbidden_paths) for item in actual
                )
                else "a forbidden path changed",
                workspace=workspace,
                secret_values=secret_values,
            ),
            _safe_check(
                "diff.target_complete",
                diff.completeness.target_complete == expected.target_complete,
                (
                    f"expected target_complete={expected.target_complete}, "
                    f"observed {diff.completeness.target_complete}"
                ),
                workspace=workspace,
                secret_values=secret_values,
            ),
            _safe_check(
                "diff.forbidden_complete",
                diff.completeness.forbidden_complete == expected.forbidden_complete,
                (
                    f"expected forbidden_complete={expected.forbidden_complete}, "
                    f"observed {diff.completeness.forbidden_complete}"
                ),
                workspace=workspace,
                secret_values=secret_values,
            ),
            _safe_check(
                "diff.rendered_complete",
                diff.completeness.rendered_diff_complete == expected.rendered_diff_complete,
                (
                    f"expected rendered_diff_complete={expected.rendered_diff_complete}, "
                    f"observed {diff.completeness.rendered_diff_complete}"
                ),
                workspace=workspace,
                secret_values=secret_values,
            ),
        ]
        return tuple(checks)

    def _evaluate_validators(
        self,
        case: CodingEvaluationCase,
        actual: Sequence[ValidatorResult],
        workspace: Workspace,
        secret_values: Sequence[str],
    ) -> tuple[CodingEvaluationCheck, ...]:
        expected = case.expected.validators
        observed = [(item.validator_id, item.status.value) for item in actual]
        wanted = [(item.validator_id, item.status.value) for item in expected]
        return (
            _safe_check(
                "validators.exact_outcomes",
                observed == wanted,
                f"expected validator outcomes {wanted!r}, observed {observed!r}",
                workspace=workspace,
                secret_values=secret_values,
            ),
        )

    def _evaluate_governance(
        self,
        case: CodingEvaluationCase,
        actual: Sequence[ActionRecord],
        workspace: Workspace,
        secret_values: Sequence[str],
    ) -> tuple[CodingEvaluationCheck, ...]:
        expected = case.expected.governance
        observed = [
            (
                item.action_kind.value,
                item.policy_decision.outcome.value,
                None if item.approval_outcome is None else item.approval_outcome.value,
                item.effect_state.value,
                item.executor_attempts,
                _action_guards_passed(item),
                tuple(result.outcome.value for result in item.pre_hook_results),
                tuple(result.outcome.value for result in item.post_hook_results),
            )
            for item in actual
        ]
        wanted = [
            (
                item.action_kind.value,
                item.policy_outcome.value,
                None if item.approval_outcome is None else item.approval_outcome.value,
                item.effect_state.value,
                item.executor_attempts,
                item.guards_passed,
                tuple(result.value for result in item.pre_hook_outcomes),
                tuple(result.value for result in item.post_hook_outcomes),
            )
            for item in expected
        ]
        checks: list[CodingEvaluationCheck] = [
            _safe_check(
                "governance.exact_trajectory",
                observed == wanted,
                f"expected governance trajectory {wanted!r}, observed {observed!r}",
                workspace=workspace,
                secret_values=secret_values,
            ),
        ]
        for index, (record, expectation) in enumerate(zip(actual, expected, strict=False)):
            diagnostics = tuple(record.sanitized_diagnostics)
            missing = [
                value
                for value in expectation.diagnostics_contains_all
                if value not in diagnostics
            ]
            present = [
                value
                for value in expectation.diagnostics_absent_all
                if value in diagnostics
            ]
            if expectation.diagnostics_contains_all or expectation.diagnostics_absent_all:
                checks.append(
                    _safe_check(
                        f"governance[{index}].diagnostics",
                        not missing and not present,
                        (
                            "required diagnostics observed and forbidden diagnostics absent"
                            if not missing and not present
                            else f"missing={missing}, forbidden_present={present}"
                        ),
                        workspace=workspace,
                        secret_values=secret_values,
                    )
                )
        return tuple(checks)

    def _evaluate_context(
        self,
        case: CodingEvaluationCase,
        repository_context: Any,
        context_window: Any,
        workspace: Workspace,
        secret_values: Sequence[str],
    ) -> tuple[CodingEvaluationCheck, ...]:
        expected = case.expected.context
        if (
            not expected.selected_instruction_sources
            and expected.selected_skill_key is None
            and expected.selected_skill_body is None
            and not expected.omission_reasons
        ):
            return ()
        actual_sources = tuple(
            item.source.as_posix() for item in repository_context.instructions
        )
        actual_skill_key = (
            repository_context.selected_skill.key
            if repository_context.selected_skill is not None
            else None
        )
        actual_skill_body = repository_context.selected_skill is not None
        actual_omission_reasons = tuple(
            dict.fromkeys(
                getattr(item.reason, "value", str(item.reason))
                for item in repository_context.all_omissions
            )
        )
        checks: list[CodingEvaluationCheck] = []
        if expected.selected_instruction_sources:
            checks.append(
                _safe_check(
                    "context.instruction_sources",
                    actual_sources == expected.selected_instruction_sources,
                    (
                        f"expected instruction sources {expected.selected_instruction_sources!r}, "
                        f"observed {actual_sources!r}"
                    ),
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
        if expected.selected_skill_key is not None:
            checks.append(
                _safe_check(
                    "context.selected_skill_key",
                    actual_skill_key == expected.selected_skill_key,
                    (
                        f"expected selected skill {expected.selected_skill_key!r}, "
                        f"observed {actual_skill_key!r}"
                    ),
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
        if expected.selected_skill_body is not None:
            checks.append(
                _safe_check(
                    "context.selected_skill_body",
                    actual_skill_body == expected.selected_skill_body,
                    (
                        f"expected selected skill body={expected.selected_skill_body}, "
                        f"observed {actual_skill_body}"
                    ),
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
        if expected.omission_reasons:
            missing = [
                reason
                for reason in expected.omission_reasons
                if reason not in actual_omission_reasons
            ]
            checks.append(
                _safe_check(
                    "context.omission_reasons",
                    not missing,
                    (
                        "expected omission reasons observed"
                        if not missing
                        else f"missing omission reasons: {missing}"
                    ),
                    workspace=workspace,
                    secret_values=secret_values,
                )
            )
        del context_window
        return tuple(checks)

    def _evaluate_events(
        self,
        case: CodingEvaluationCase,
        events: Sequence[RunEvent],
        workspace: Workspace,
        secret_values: Sequence[str],
    ) -> tuple[CodingEvaluationCheck, ...]:
        expected = case.expected.required_event_types
        actual = tuple(event.type for event in events)
        return (
            _safe_check(
                "trace.required_event_types",
                _is_subsequence(expected, actual),
                "required production event sequence observed"
                if _is_subsequence(expected, actual)
                else f"actual event sequence: {[item.value for item in actual]}",
                workspace=workspace,
                secret_values=secret_values,
            ),
        )

    def _evaluate_limits(
        self,
        case: CodingEvaluationCase,
        events: Sequence[RunEvent],
        action_records: Sequence[ActionRecord],
        diff: Any | None,
        validator_results: Sequence[ValidatorResult],
        workspace: Workspace,
        secret_values: Sequence[str],
    ) -> tuple[CodingEvaluationCheck, ...]:
        expected = case.expected.limits
        events_available = bool(events)
        attempts, elapsed = _event_metrics(events) if events_available else (0, 0.0)
        validator_evidence_available = bool(validator_results) or not case.composition.validators
        validator_output = sum(len(item.stdout) + len(item.stderr) for item in validator_results)
        return (
            _safe_check(
                "limits.max_model_attempts",
                events_available and attempts <= expected.max_model_attempts,
                (
                    f"observed {attempts}, maximum {expected.max_model_attempts}"
                    if events_available
                    else "model-attempt evidence is unavailable"
                ),
                workspace=workspace,
                secret_values=secret_values,
            ),
            _safe_check(
                "limits.max_governed_calls",
                len(action_records) <= expected.max_governed_calls,
                f"observed {len(action_records)}, maximum {expected.max_governed_calls}",
                workspace=workspace,
                secret_values=secret_values,
            ),
            _safe_check(
                "limits.max_elapsed_seconds",
                events_available and elapsed <= expected.max_elapsed_seconds,
                (
                    f"observed {elapsed:.6f}s, maximum {expected.max_elapsed_seconds:g}s"
                    if events_available
                    else "elapsed-time evidence is unavailable"
                ),
                workspace=workspace,
                secret_values=secret_values,
            ),
            _safe_check(
                "limits.max_rendered_diff_characters",
                diff is not None
                and len(diff.rendered_diff) <= expected.max_rendered_diff_characters,
                (
                    (
                        f"observed {len(diff.rendered_diff)}, "
                        f"maximum {expected.max_rendered_diff_characters}"
                    )
                    if diff is not None
                    else "rendered-diff evidence is unavailable"
                ),
                workspace=workspace,
                secret_values=secret_values,
            ),
            _safe_check(
                "limits.max_validator_output_characters",
                validator_evidence_available
                and validator_output <= expected.max_validator_output_characters,
                (
                    f"observed {validator_output}, "
                    f"maximum {expected.max_validator_output_characters}"
                    if validator_evidence_available
                    else "validator-output evidence is unavailable"
                ),
                workspace=workspace,
                secret_values=secret_values,
            ),
        )


def _drop_volatile(value: object) -> object:
    if isinstance(value, dict):
        for key in ("generated_at", "run_id", "output", "elapsed_seconds", "detail"):
            value.pop(key, None)
        for item in value.values():
            _drop_volatile(item)
    elif isinstance(value, list):
        for item in value:
            _drop_volatile(item)
    return value


_COMPLETION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "content": {"type": "string", "minLength": 1, "maxLength": _MAX_FILE_CHARACTERS},
        "tool_calls": {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["call_id", "name", "arguments"],
                "properties": {
                    "call_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "name": {"type": "string", "minLength": 1, "maxLength": 128},
                    "arguments": {"type": "object"},
                },
            },
        },
        "response_id": {"type": "string", "maxLength": 128},
        "model": {"type": "string", "maxLength": 128},
        "usage": {
            "type": "object",
            "additionalProperties": False,
            "required": ["input_tokens", "output_tokens", "total_tokens"],
            "properties": {
                "input_tokens": {"type": "integer", "minimum": 0},
                "output_tokens": {"type": "integer", "minimum": 0},
                "total_tokens": {"type": "integer", "minimum": 0},
            },
        },
    },
    "anyOf": [{"required": ["content"]}, {"required": ["tool_calls"]}],
}


_CODING_EVALUATION_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "suite_id", "cases"],
    "properties": {
        "schema_version": {"const": CODING_EVALUATION_SCHEMA_VERSION},
        "suite_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "cases": {
            "type": "array",
            "minItems": 1,
            "maxItems": _MAX_CASES,
            "items": {"$ref": "#/$defs/case"},
        },
    },
    "$defs": {
        "path": {
            "type": "string",
            "minLength": 1,
            "maxLength": 512,
            "pattern": r"^[^\\\x00:/][^\\\x00]*$",
        },
        "case": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "id",
                "modes",
                "request",
                "repository",
                "fixture",
                "composition",
                "expected",
                "fixture_digest",
            ],
            "properties": {
                "id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$", "maxLength": 128},
                "modes": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"const": "deterministic"},
                },
                "request": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["message", "targets"],
                    "properties": {
                        "message": {"type": "string", "minLength": 1, "maxLength": 16_000},
                        "targets": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 128,
                            "items": {"$ref": "#/$defs/path"},
                        },
                        "skills": {
                            "type": "array",
                            "maxItems": 32,
                            "items": {"type": "string", "minLength": 1, "maxLength": 128},
                        },
                    },
                },
                "repository": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["files"],
                    "properties": {
                        "files": {
                            "type": "object",
                            "minProperties": 1,
                            "maxProperties": _MAX_FILES,
                            "additionalProperties": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": _MAX_FILE_CHARACTERS,
                            },
                        },
                        "skill_roots": {
                            "type": "object",
                            "maxProperties": 16,
                            "additionalProperties": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 512,
                            },
                        },
                        "protected_paths": {
                            "type": "array",
                            "maxItems": 32,
                            "items": {"$ref": "#/$defs/path"},
                        },
                        "secret_paths": {
                            "type": "array",
                            "maxItems": 32,
                            "items": {"$ref": "#/$defs/path"},
                        },
                        "volatile_paths": {
                            "type": "array",
                            "maxItems": 32,
                            "items": {"$ref": "#/$defs/path"},
                        },
                        "ignored_paths": {
                            "type": "array",
                            "maxItems": 32,
                            "items": {"$ref": "#/$defs/path"},
                        },
                    },
                },
                "fixture": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["model_completions", "approval_decisions", "failure"],
                    "properties": {
                        "model_completions": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": _MAX_COMPLETIONS,
                            "items": {"$ref": "#/$defs/completion"},
                        },
                        "approval_decisions": {
                            "type": "array",
                            "maxItems": _MAX_APPROVAL_DECISIONS,
                            "items": {"enum": [item.value for item in ApprovalOutcome]},
                        },
                        "failure": {
                            "enum": [
                                "none",
                                "validator_runner_error",
                                "validator_runner_empty",
                                "cleanup_failure",
                                "reject_then_stale_approval",
                                "incomplete_observation",
                            ]
                        },
                    },
                },
                "composition": {"$ref": "#/$defs/composition"},
                "expected": {"$ref": "#/$defs/expected"},
                "fixture_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
        },
        "completion": _COMPLETION_SCHEMA,
        "composition": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "policy",
                "executable_allowlist",
                "validators",
                "max_iterations",
                "max_governed_calls",
                "secret_names",
                "secret_values",
            ],
            "properties": {
                "policy": {"const": "default"},
                "executable_allowlist": {
                    "type": "object",
                    "maxProperties": 8,
                    "additionalProperties": {"const": "current_python"},
                },
                "validators": {
                    "type": "array",
                    "maxItems": _MAX_VALIDATORS,
                    "items": {"$ref": "#/$defs/validator"},
                },
                "max_iterations": {"type": "integer", "minimum": 1, "maximum": 128},
                "max_governed_calls": {"type": "integer", "minimum": 1, "maximum": 128},
                "secret_names": {
                    "type": "array",
                    "maxItems": 64,
                    "items": {"type": "string", "minLength": 1, "maxLength": 128},
                },
                "secret_values": {
                    "type": "array",
                    "maxItems": 64,
                    "items": {"type": "string", "minLength": 1, "maxLength": 512},
                },
                "hook_fixture": {
                    "enum": [
                        "none",
                        "required_pre_hook_failure",
                        "post_hook_failure",
                    ]
                },
            },
        },
        "validator": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "argv"],
            "properties": {
                "id": {"type": "string", "minLength": 1, "maxLength": 128},
                "argv": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 4_096,
                        "pattern": "^[^\\x00]*$",
                    },
                },
                "cwd": {"type": "string", "minLength": 1, "maxLength": 512},
                "timeout_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 86_400},
                "stdout_limit_bytes": {"type": "integer", "minimum": 0, "maximum": 4 * 1024 * 1024},
                "stderr_limit_bytes": {"type": "integer", "minimum": 0, "maximum": 4 * 1024 * 1024},
                "accepted_exit_codes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {"type": "integer"},
                },
            },
        },
        "expected": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "run",
                "answer",
                "verdict",
                "diff",
                "validators",
                "governance",
                "required_event_types",
                "limits",
                "fixture_consumed",
            ],
            "properties": {
                "run": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["status"],
                    "properties": {
                        "status": {"enum": ["completed", "error"]},
                        "error_type": {"type": "string", "minLength": 1, "maxLength": 128},
                        "error_category": {"type": "string", "minLength": 1, "maxLength": 64},
                    },
                },
                "answer": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["non_empty", "contains_all", "absent_all"],
                    "properties": {
                        "non_empty": {"type": "boolean"},
                        "exact": {"type": "string", "minLength": 1, "maxLength": _MAX_REPORT_TEXT},
                        "contains_all": {
                            "type": "array",
                            "maxItems": 32,
                            "items": {"type": "string", "minLength": 1, "maxLength": 512},
                        },
                        "absent_all": {
                            "type": "array",
                            "maxItems": 32,
                            "items": {"type": "string", "minLength": 1, "maxLength": 512},
                        },
                    },
                },
                "tool_calls": {
                    "type": "array",
                    "maxItems": _MAX_EXPECTED_TOOL_CALLS,
                    "items": {"$ref": "#/$defs/tool_call_expectation"},
                },
                "verdict": {
                    "type": ["string", "null"],
                    "enum": ["passed", "failed", "indeterminate", "not_validated", None],
                },
                "diff": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "expected_changes",
                        "forbidden_paths",
                        "target_complete",
                        "forbidden_complete",
                        "rendered_diff_complete",
                    ],
                    "properties": {
                        "expected_changes": {
                            "type": "array",
                            "maxItems": _MAX_EXPECTED_CHANGES,
                            "items": {"$ref": "#/$defs/change"},
                        },
                        "forbidden_paths": {
                            "type": "array",
                            "maxItems": 32,
                            "items": {"$ref": "#/$defs/path"},
                        },
                        "target_complete": {"type": "boolean"},
                        "forbidden_complete": {"type": "boolean"},
                        "rendered_diff_complete": {"type": "boolean"},
                    },
                },
                "context": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "selected_instruction_sources": {
                            "type": "array",
                            "maxItems": 32,
                            "items": {"type": "string", "minLength": 1, "maxLength": 512},
                        },
                        "selected_skill_key": {
                            "type": ["string", "null"],
                            "minLength": 1,
                            "maxLength": 128,
                        },
                        "selected_skill_body": {"type": ["boolean", "null"]},
                        "omission_reasons": {
                            "type": "array",
                            "maxItems": 32,
                            "items": {"type": "string", "minLength": 1, "maxLength": 128},
                        },
                    },
                },
                "validators": {
                    "type": "array",
                    "maxItems": _MAX_VALIDATORS,
                    "items": {"$ref": "#/$defs/validator_expectation"},
                },
                "governance": {
                    "type": "array",
                    "maxItems": _MAX_EXPECTED_GOVERNANCE,
                    "items": {"$ref": "#/$defs/governance"},
                },
                "required_event_types": {
                    "type": "array",
                    "maxItems": _MAX_EXPECTED_CHECKS,
                    "items": {"enum": [item.value for item in RunEventType]},
                },
                "limits": {"$ref": "#/$defs/limits"},
                "fixture_consumed": {"type": "boolean"},
            },
        },
        "change": {
            "type": "object",
            "additionalProperties": False,
            "required": ["path", "kind"],
            "properties": {
                "path": {"$ref": "#/$defs/path"},
                "kind": {"enum": [item.value for item in WorkspaceChangeKind]},
                "before_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "after_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
        },
        "tool_call_expectation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["call_id", "name", "arguments", "outcome"],
            "properties": {
                "call_id": {"type": "string", "minLength": 1, "maxLength": 128},
                "name": {"type": "string", "minLength": 1, "maxLength": 128},
                "arguments": {"type": "object"},
                "outcome": {"enum": [item.value for item in ToolOutcome]},
                "error_code": {
                    "type": ["string", "null"],
                    "enum": [item.value for item in ToolErrorCode] + [None],
                },
            },
        },
        "validator_expectation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "status"],
            "properties": {
                "id": {"type": "string", "minLength": 1, "maxLength": 128},
                "status": {"enum": [item.value for item in ValidatorStatus]},
            },
        },
        "governance": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "action_kind",
                "policy_outcome",
                "approval_outcome",
                "effect_state",
                "executor_attempts",
                "guards_passed",
            ],
            "properties": {
                "action_kind": {"enum": [item.value for item in ActionKind]},
                "policy_outcome": {"enum": [item.value for item in PolicyOutcome]},
                "approval_outcome": {
                    "type": ["string", "null"],
                    "enum": [item.value for item in ApprovalOutcome] + [None],
                },
                "effect_state": {"enum": [item.value for item in EffectState]},
                "executor_attempts": {"enum": [0, 1]},
                "guards_passed": {"type": "boolean"},
                "pre_hook_outcomes": {
                    "type": "array",
                    "maxItems": 32,
                    "items": {"enum": [item.value for item in HookOutcome]},
                },
                "post_hook_outcomes": {
                    "type": "array",
                    "maxItems": 32,
                    "items": {"enum": [item.value for item in HookOutcome]},
                },
                "diagnostics_contains_all": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1, "maxLength": 256},
                },
                "diagnostics_absent_all": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1, "maxLength": 256},
                },
            },
        },
        "limits": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "max_model_attempts",
                "max_governed_calls",
                "max_elapsed_seconds",
                "max_rendered_diff_characters",
                "max_validator_output_characters",
            ],
            "properties": {
                "max_model_attempts": {"type": "integer", "minimum": 0, "maximum": 128},
                "max_governed_calls": {"type": "integer", "minimum": 0, "maximum": 128},
                "max_elapsed_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 86_400},
                "max_rendered_diff_characters": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 200_000,
                },
                "max_validator_output_characters": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 8 * 1024 * 1024,
                },
            },
        },
    },
}
