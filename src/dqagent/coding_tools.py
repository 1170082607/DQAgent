"""Governed, bounded workspace read and literal search tools."""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Final, cast

from dqagent.models import ToolDefinition, ToolErrorCode
from dqagent.subprocesses import (
    IsolationCapability,
    LocalSubprocessRunner,
    SubprocessRequest,
    SubprocessResult,
    SubprocessRunner,
    SubprocessStatus,
    build_minimal_environment,
    normalize_isolation_capabilities,
    normalize_secret_values,
)
from dqagent.tool_governance import (
    ActionExecutionResult,
    ActionKind,
    ActionPolicy,
    ApprovalProvider,
    CanonicalJsonValue,
    DefaultActionPolicy,
    EffectiveLimits,
    EffectKind,
    EffectPrecondition,
    EffectPreconditions,
    EffectState,
    GuardContext,
    PostActionHook,
    PreActionHook,
    PreActionHookSpec,
    PreparedAction,
)
from dqagent.tools import (
    ActionExecutionError,
    ActionOutputSanitizer,
    ActionPreparationError,
    ActionTool,
    GuardContextFactory,
    ToolExecutionContext,
    ToolRegistry,
)
from dqagent.workspace import (
    PathKind,
    ResolvedWorkspacePath,
    Sanitizer,
    Workspace,
    WorkspaceError,
    WorkspacePurpose,
    WorkspaceReason,
    WorkspaceWalkEntry,
)

__all__ = [
    "CODING_TOOL_LIMITS",
    "DEFAULT_CODING_TOOL_LIMITS",
    "WORKSPACE_READ_INPUT_SCHEMA",
    "WORKSPACE_READ_SCHEMA",
    "WORKSPACE_SEARCH_INPUT_SCHEMA",
    "WORKSPACE_SEARCH_SCHEMA",
    "WORKSPACE_PATCH_INPUT_SCHEMA",
    "WORKSPACE_PATCH_SCHEMA",
    "WORKSPACE_COMMAND_INPUT_SCHEMA",
    "WORKSPACE_COMMAND_SCHEMA",
    "CommandExecutable",
    "CommandToolLimits",
    "CodingToolLimits",
    "create_coding_tool_registry",
    "create_coding_tools",
    "create_workspace_read_action_tool",
    "create_workspace_read_tool",
    "create_workspace_search_action_tool",
    "create_workspace_search_tool",
    "create_workspace_patch_action_tool",
    "create_workspace_patch_tool",
    "create_workspace_command_action_tool",
    "create_workspace_command_tool",
]


_DEFAULT_MAX_ARGUMENT_BYTES: Final[int] = 64_000
_DEFAULT_MAX_READ_SOURCE_BYTES: Final[int] = 1_000_000
_DEFAULT_MAX_READ_LINE_COUNT: Final[int] = 512
_DEFAULT_MAX_READ_LINE_CHARACTERS: Final[int] = 8_192
_DEFAULT_MAX_READ_ELAPSED_SECONDS: Final[float] = 5.0
_DEFAULT_MAX_READ_OUTPUT_CHARACTERS: Final[int] = 16_000
_DEFAULT_MAX_SEARCH_VISITED_FILES: Final[int] = 1_000
_DEFAULT_MAX_SEARCH_SOURCE_BYTES: Final[int] = 4_000_000
_DEFAULT_MAX_SEARCH_FILE_BYTES: Final[int] = 1_000_000
_DEFAULT_MAX_SEARCH_MATCHES: Final[int] = 256
_DEFAULT_MAX_SEARCH_LINE_CHARACTERS: Final[int] = 8_192
_DEFAULT_MAX_SEARCH_ELAPSED_SECONDS: Final[float] = 5.0
_DEFAULT_MAX_SEARCH_OUTPUT_CHARACTERS: Final[int] = 16_000
_DEFAULT_MAX_QUERY_CHARACTERS: Final[int] = 4_096
_DEFAULT_MAX_GLOB_CHARACTERS: Final[int] = 512
_DEFAULT_MAX_PATCH_CONTENT_BYTES: Final[int] = 1_000_000
_DEFAULT_MAX_PATCH_REPLACEMENTS: Final[int] = 64
_DEFAULT_MAX_PATCH_REPLACEMENT_CHARACTERS: Final[int] = 64_000
_DEFAULT_MAX_PATCH_ELAPSED_SECONDS: Final[float] = 5.0
_DEFAULT_MAX_PATCH_OUTPUT_CHARACTERS: Final[int] = 4_096
_SCHEMA_MAX_PATH_CHARACTERS: Final[int] = 4_096
_SCHEMA_MAX_START_LINE: Final[int] = 10_000_000
_SCHEMA_MAX_LINE_COUNT: Final[int] = 100_000
_SCHEMA_MAX_QUERY_CHARACTERS: Final[int] = 4_096
_SCHEMA_MAX_GLOB_CHARACTERS: Final[int] = 512
_SCHEMA_MAX_MATCHES: Final[int] = 100_000
_SCHEMA_MAX_PATCH_CONTENT_CHARACTERS: Final[int] = 1_000_000
_SCHEMA_MAX_PATCH_REPLACEMENTS: Final[int] = 1_024
_SCHEMA_MAX_PATCH_REPLACEMENT_CHARACTERS: Final[int] = 1_000_000
_SCHEMA_MAX_PATCH_OCCURRENCES: Final[int] = 100_000
_SCHEMA_MAX_COMMAND_ARGV_ITEMS: Final[int] = 128
_SCHEMA_MAX_COMMAND_ARGUMENT_CHARACTERS: Final[int] = 32_000
_SCHEMA_MAX_COMMAND_ARGUMENT_LENGTH: Final[int] = 8_192
_SCHEMA_MAX_COMMAND_TIMEOUT_SECONDS: Final[float] = 86_400.0
_PATCH_WILDCARD_CHARACTERS: Final[frozenset[str]] = frozenset("*?[]")
_READ_MARKER: Final[str] = "...[line-limit]"
_OUTPUT_MARKER: Final[str] = "...[output-limit]"
_MIN_READ_OUTPUT_CHARACTERS: Final[int] = len(
    '{"output_limit":true,"status":"output_limit","tool":"workspace_read"}'
)
_MIN_SEARCH_OUTPUT_CHARACTERS: Final[int] = len(
    '{"output_limit":true,"status":"output_limit","tool":"workspace_search"}'
)
_MIN_PATCH_OUTPUT_CHARACTERS: Final[int] = len(
    '{"output_limit":true,"status":"output_limit","tool":"workspace_patch"}'
)
_CHUNK_BYTES: Final[int] = 64 * 1024
_OMISSION_KEYS: Final[tuple[str, ...]] = (
    "protected",
    "secret",
    "link",
    "binary",
    "invalid_text",
    "oversized",
    "missing",
    "denied",
    "drift",
    "unsupported",
    "filesystem",
    "visited_files_limit",
)
_CODING_FIXED_SERIALIZED_LITERALS: Final[frozenset[str]] = frozenset(
    {
        "line_count",
        "elapsed_limit",
        "line_limit",
        "output_limit",
        "returned_lines",
        "source_bytes",
        "source_limit",
        "start_line",
        "status",
        "tool",
        "path",
        "reason",
        "case_sensitive",
        "glob",
        "line_projection_limit",
        "match_limit",
        "matches",
        "omissions",
        "visited_files_limit",
        "visited_files",
        "query",
        *_OMISSION_KEYS,
        "workspace_read",
        "workspace_search",
        "workspace_patch",
        "operation",
        "create",
        "update",
        "delete",
        "content",
        "expected_sha256",
        "replacements",
        "old",
        "new",
        "expected_occurrences",
        "bytes",
        "bytes_written",
        "bytes_deleted",
        "replacement_count",
        "effect_state",
        "none",
        "partial",
        "unknown",
        "created",
        "updated",
        "deleted",
        "patch_link_target_denied",
        "patch_precondition_conflict",
        "patch_target_race",
        "patch_temp_create_failed",
        "patch_temp_write_failed",
        "patch_temp_cleanup_failed",
        "patch_atomic_replace_failed",
        "patch_delete_failed",
        "patch_create_open_failed",
        "patch_create_write_failed",
        "patch_create_close_failed",
        "patch_resource_limit",
        "patch_elapsed_limit",
        "patch_output_truncated",
        "workspace_tool",
        "ok",
        "empty",
        "eof",
        "binary",
        "invalid_text",
        "missing",
        "omitted",
        "no_matches",
        "link_not_followed",
        "drift",
        "filesystem",
        "denied",
        "true",
        "false",
        "null",
    }
)
_CODING_JSON_SYNTAX_TOKENS: Final[frozenset[str]] = frozenset(
    {"{", "}", '"', ":", ","}
)

_DEFAULT_COMMAND_ARGV_ITEMS: Final[int] = 128
_DEFAULT_COMMAND_ARGV_CHARACTERS: Final[int] = 32_000
_DEFAULT_COMMAND_TIMEOUT_SECONDS: Final[float] = 30.0
_DEFAULT_COMMAND_STDOUT_BYTES: Final[int] = 32_000
_DEFAULT_COMMAND_STDERR_BYTES: Final[int] = 32_000
_DEFAULT_COMMAND_OUTPUT_CHARACTERS: Final[int] = 32_000
_DEFAULT_COMMAND_ARGUMENT_BYTES: Final[int] = 64_000
_COMMAND_REQUIRED_CAPABILITIES: Final[frozenset[IsolationCapability]] = frozenset(
    {
        IsolationCapability.DIRECT_ARGV,
        IsolationCapability.WORKING_DIRECTORY_CONTROL,
        IsolationCapability.ALLOWLISTED_ENVIRONMENT,
        IsolationCapability.NO_STDIN,
        IsolationCapability.WALL_TIME_LIMIT,
        IsolationCapability.BOUNDED_OUTPUT,
        IsolationCapability.DIRECT_CHILD_TERMINATION,
        IsolationCapability.DIRECT_CHILD_REAP,
    }
)


WORKSPACE_READ_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "minLength": 1,
            "maxLength": _SCHEMA_MAX_PATH_CHARACTERS,
        },
        "start_line": {
            "type": "integer",
            "minimum": 1,
            "maximum": _SCHEMA_MAX_START_LINE,
        },
        "line_count": {
            "type": "integer",
            "minimum": 1,
            "maximum": _SCHEMA_MAX_LINE_COUNT,
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}

WORKSPACE_SEARCH_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": _SCHEMA_MAX_QUERY_CHARACTERS,
        },
        "path": {
            "type": "string",
            "minLength": 1,
            "maxLength": _SCHEMA_MAX_PATH_CHARACTERS,
        },
        "glob": {
            "type": "string",
            "minLength": 1,
            "maxLength": _SCHEMA_MAX_GLOB_CHARACTERS,
        },
        "case_sensitive": {"type": "boolean"},
        "max_matches": {
            "type": "integer",
            "minimum": 1,
            "maximum": _SCHEMA_MAX_MATCHES,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

WORKSPACE_PATCH_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": ["create", "update", "delete"],
        },
        "path": {
            "type": "string",
            "minLength": 1,
            "maxLength": _SCHEMA_MAX_PATH_CHARACTERS,
        },
        "content": {
            "type": "string",
            "maxLength": _SCHEMA_MAX_PATCH_CONTENT_CHARACTERS,
        },
        "expected_sha256": {
            "type": "string",
            "pattern": r"^[0-9a-f]{64}$",
        },
        "replacements": {
            "type": "array",
            "minItems": 1,
            "maxItems": _SCHEMA_MAX_PATCH_REPLACEMENTS,
            "items": {
                "type": "object",
                "properties": {
                    "old": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": _SCHEMA_MAX_PATCH_REPLACEMENT_CHARACTERS,
                    },
                    "new": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": _SCHEMA_MAX_PATCH_REPLACEMENT_CHARACTERS,
                    },
                    "expected_occurrences": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": _SCHEMA_MAX_PATCH_OCCURRENCES,
                    },
                },
                "required": ["old", "new", "expected_occurrences"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["operation", "path"],
    "additionalProperties": False,
    "allOf": [
        {
            "if": {"properties": {"operation": {"const": "create"}}},
            "then": {
                "required": ["content"],
                "not": {
                    "anyOf": [
                        {"required": ["expected_sha256"]},
                        {"required": ["replacements"]},
                    ]
                },
            },
        },
        {
            "if": {"properties": {"operation": {"const": "update"}}},
            "then": {
                "required": ["expected_sha256", "replacements"],
                "not": {"required": ["content"]},
            },
        },
        {
            "if": {"properties": {"operation": {"const": "delete"}}},
            "then": {
                "required": ["expected_sha256"],
                "not": {"anyOf": [{"required": ["content"]}, {"required": ["replacements"]}]},
            },
        },
    ],
}

WORKSPACE_COMMAND_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {
        "argv": {
            "type": "array",
            "minItems": 1,
            "maxItems": _SCHEMA_MAX_COMMAND_ARGV_ITEMS,
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": _SCHEMA_MAX_COMMAND_ARGUMENT_LENGTH,
                "pattern": r"^[^\x00]*$",
            },
        },
        "cwd": {
            "type": "string",
            "minLength": 1,
            "maxLength": _SCHEMA_MAX_PATH_CHARACTERS,
        },
        "timeout_seconds": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": _SCHEMA_MAX_COMMAND_TIMEOUT_SECONDS,
        },
    },
    "required": ["argv"],
    "additionalProperties": False,
}

# Keep both names available for callers that distinguish tool and input schemas.
WORKSPACE_READ_INPUT_SCHEMA = WORKSPACE_READ_SCHEMA
WORKSPACE_SEARCH_INPUT_SCHEMA = WORKSPACE_SEARCH_SCHEMA
WORKSPACE_PATCH_INPUT_SCHEMA = WORKSPACE_PATCH_SCHEMA
WORKSPACE_COMMAND_INPUT_SCHEMA = WORKSPACE_COMMAND_SCHEMA


@dataclass(frozen=True, slots=True)
class CommandExecutable:
    """A trusted executable mapping used by ``workspace_command``.

    The model selects the opaque ``identity``.  The trusted composition owns
    the resolved path and may mark a mapping as a shell interpreter.  Shell
    mappings still require the command factory's explicit shell opt-in.
    """

    identity: str
    path: Path
    shell: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identity, str)
            or not self.identity.strip()
            or len(self.identity) > 128
            or "\x00" in self.identity
            or any(ord(character) < 32 for character in self.identity)
            or "/" in self.identity
            or "\\" in self.identity
            or ":" in self.identity
        ):
            raise ValueError("executable identity must be an opaque non-empty name")
        if not isinstance(self.path, Path):
            raise TypeError("executable path must be a pathlib.Path")
        try:
            resolved = self.path.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise ValueError("executable path could not be resolved") from error
        if resolved.exists() and not resolved.is_file():
            raise ValueError("executable path must identify a file")
        if not isinstance(self.shell, bool):
            raise TypeError("executable shell flag must be a boolean")
        object.__setattr__(self, "path", resolved)


@dataclass(frozen=True, slots=True)
class _PreparedCommandAction(PreparedAction):
    """Prepared command action carrying the executable resolved before approval."""

    executable: CommandExecutable = field(kw_only=True)


@dataclass(frozen=True, slots=True)
class CommandToolLimits:
    """Trusted argv, process, stream, and model-output limits."""

    max_argv_items: int = _DEFAULT_COMMAND_ARGV_ITEMS
    max_argv_characters: int = _DEFAULT_COMMAND_ARGV_CHARACTERS
    max_timeout_seconds: float = _DEFAULT_COMMAND_TIMEOUT_SECONDS
    max_stdout_bytes: int = _DEFAULT_COMMAND_STDOUT_BYTES
    max_stderr_bytes: int = _DEFAULT_COMMAND_STDERR_BYTES
    max_output_characters: int = _DEFAULT_COMMAND_OUTPUT_CHARACTERS
    max_argument_bytes: int = _DEFAULT_COMMAND_ARGUMENT_BYTES

    def __post_init__(self) -> None:
        for name in (
            "max_argv_items",
            "max_argv_characters",
            "max_stdout_bytes",
            "max_stderr_bytes",
            "max_output_characters",
            "max_argument_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_argv_items > _SCHEMA_MAX_COMMAND_ARGV_ITEMS:
            raise ValueError("max_argv_items exceeds the command schema bound")
        if self.max_argv_characters > _SCHEMA_MAX_COMMAND_ARGUMENT_CHARACTERS:
            raise ValueError("max_argv_characters exceeds the command schema bound")
        if (
            isinstance(self.max_timeout_seconds, bool)
            or not isinstance(self.max_timeout_seconds, (int, float))
            or self.max_timeout_seconds <= 0
            or not math.isfinite(float(self.max_timeout_seconds))
        ):
            raise ValueError("max_timeout_seconds must be a finite number greater than zero")
        if self.max_timeout_seconds > _SCHEMA_MAX_COMMAND_TIMEOUT_SECONDS:
            raise ValueError("max_timeout_seconds exceeds the command schema bound")

    @property
    def timeout_seconds(self) -> float:
        return float(self.max_timeout_seconds)

    @property
    def stdout_limit_bytes(self) -> int:
        return self.max_stdout_bytes

    @property
    def stderr_limit_bytes(self) -> int:
        return self.max_stderr_bytes

    @property
    def stdout_max_bytes(self) -> int:
        return self.max_stdout_bytes

    @property
    def stderr_max_bytes(self) -> int:
        return self.max_stderr_bytes


@dataclass(frozen=True, slots=True)
class CodingToolLimits:
    """Trusted collection and projection limits owned by the coding adapters."""

    max_argument_bytes: int = _DEFAULT_MAX_ARGUMENT_BYTES
    max_read_source_bytes: int = _DEFAULT_MAX_READ_SOURCE_BYTES
    max_read_line_count: int = _DEFAULT_MAX_READ_LINE_COUNT
    max_read_line_characters: int = _DEFAULT_MAX_READ_LINE_CHARACTERS
    max_read_elapsed_seconds: float = _DEFAULT_MAX_READ_ELAPSED_SECONDS
    max_read_output_characters: int = _DEFAULT_MAX_READ_OUTPUT_CHARACTERS
    max_search_visited_files: int = _DEFAULT_MAX_SEARCH_VISITED_FILES
    max_search_source_bytes: int = _DEFAULT_MAX_SEARCH_SOURCE_BYTES
    max_search_file_bytes: int = _DEFAULT_MAX_SEARCH_FILE_BYTES
    max_search_matches: int = _DEFAULT_MAX_SEARCH_MATCHES
    max_search_line_characters: int = _DEFAULT_MAX_SEARCH_LINE_CHARACTERS
    max_search_elapsed_seconds: float = _DEFAULT_MAX_SEARCH_ELAPSED_SECONDS
    max_search_output_characters: int = _DEFAULT_MAX_SEARCH_OUTPUT_CHARACTERS
    max_query_characters: int = _DEFAULT_MAX_QUERY_CHARACTERS
    max_glob_characters: int = _DEFAULT_MAX_GLOB_CHARACTERS
    max_patch_content_bytes: int = _DEFAULT_MAX_PATCH_CONTENT_BYTES
    max_patch_replacements: int = _DEFAULT_MAX_PATCH_REPLACEMENTS
    max_patch_replacement_characters: int = _DEFAULT_MAX_PATCH_REPLACEMENT_CHARACTERS
    max_patch_elapsed_seconds: float = _DEFAULT_MAX_PATCH_ELAPSED_SECONDS
    max_patch_output_characters: int = _DEFAULT_MAX_PATCH_OUTPUT_CHARACTERS

    def __post_init__(self) -> None:
        for name in (
            "max_argument_bytes",
            "max_read_source_bytes",
            "max_read_line_count",
            "max_read_line_characters",
            "max_read_output_characters",
            "max_search_visited_files",
            "max_search_source_bytes",
            "max_search_file_bytes",
            "max_search_matches",
            "max_search_line_characters",
            "max_search_output_characters",
            "max_query_characters",
            "max_glob_characters",
            "max_patch_content_bytes",
            "max_patch_replacements",
            "max_patch_replacement_characters",
            "max_patch_output_characters",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, minimum in (
            ("max_read_output_characters", _MIN_READ_OUTPUT_CHARACTERS),
            ("max_search_output_characters", _MIN_SEARCH_OUTPUT_CHARACTERS),
            ("max_patch_output_characters", _MIN_PATCH_OUTPUT_CHARACTERS),
        ):
            if getattr(self, name) < minimum:
                raise ValueError(f"{name} must fit the compact structured output header")
        for name in ("max_read_elapsed_seconds", "max_search_elapsed_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be a finite number greater than zero")
            object.__setattr__(self, name, float(value))
        value = self.max_patch_elapsed_seconds
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError("max_patch_elapsed_seconds must be a finite number greater than zero")
        object.__setattr__(self, "max_patch_elapsed_seconds", float(value))


DEFAULT_CODING_TOOL_LIMITS = CodingToolLimits()
CODING_TOOL_LIMITS = DEFAULT_CODING_TOOL_LIMITS


class _SearchElapsedLimit(Exception):
    """Internal cooperative search ceiling, converted to bounded evidence."""


class _ReadElapsedLimit(Exception):
    """Internal cooperative read ceiling, converted to bounded evidence."""


class _PatchBudget:
    """Cooperative elapsed budget for one patch call."""

    def __init__(self, context: ToolExecutionContext, duration_seconds: float) -> None:
        self._context = context
        self._deadline = time.monotonic() + duration_seconds

    def check(self, *, effect_state: EffectState = EffectState.NONE) -> None:
        self._context.run_context.check_active()
        if time.monotonic() >= self._deadline:
            raise ActionExecutionError(
                ToolErrorCode.RESOURCE_LIMIT,
                "workspace patch elapsed limit exceeded",
                effect_state=effect_state,
                diagnostics=("patch_elapsed_limit",),
            )

    def check_preparation(self) -> None:
        self._context.run_context.check_active()
        if time.monotonic() >= self._deadline:
            raise ActionPreparationError(
                ToolErrorCode.RESOURCE_LIMIT,
                "workspace patch elapsed limit exceeded",
            )


@dataclass(frozen=True, slots=True)
class _PendingPatchBudget:
    action: PreparedAction
    context: ToolExecutionContext
    context_run_id: str
    budget: _PatchBudget


_PENDING_PATCH_BUDGET: ContextVar[tuple[_PendingPatchBudget, ...]] = ContextVar(
    "dqagent_pending_patch_budget",
    default=(),
)


def _active_patch_budget(
    action: PreparedAction,
    context: ToolExecutionContext,
) -> _PatchBudget | None:
    for pending in reversed(_PENDING_PATCH_BUDGET.get()):
        if (
            pending.action is action
            and pending.context is context
            and pending.context_run_id == context.run_context.run_id
        ):
            return pending.budget
    return None


def _cleanup_patch_budget(
    action: PreparedAction | None,
    context: ToolExecutionContext,
) -> None:
    if action is None:
        return
    pending = _PENDING_PATCH_BUDGET.get()
    run_id = context.run_context.run_id
    for index in range(len(pending) - 1, -1, -1):
        entry = pending[index]
        if (
            entry.action is action
            and entry.context is context
            and entry.context_run_id == run_id
        ):
            _PENDING_PATCH_BUDGET.set(pending[:index] + pending[index + 1 :])
            return


def _check_patch_budget(
    budget: _PatchBudget | None,
    *,
    preparation: bool,
    effect_state: EffectState = EffectState.NONE,
) -> None:
    if budget is None:
        return
    if preparation:
        budget.check_preparation()
    else:
        budget.check(effect_state=effect_state)


@dataclass(slots=True)
class _ReadProjection:
    lines: list[tuple[int, str]]
    source_bytes: int = 0
    lines_seen: int = 0
    source_limit: bool = False
    line_limit: bool = False
    elapsed_limit: bool = False
    filesystem_error: bool = False
    binary: bool = False
    invalid_text: bool = False


@dataclass(slots=True)
class _SearchMatch:
    logical_path: PurePosixPath
    line: int
    column: int
    projection: str
    line_limit: bool


@dataclass(slots=True)
class _SearchFileResult:
    matches: list[_SearchMatch]
    source_bytes: int
    source_limit: bool = False
    match_limit: bool = False
    line_limit: bool = False
    filesystem_error: bool = False
    binary: bool = False
    invalid_text: bool = False


@dataclass(frozen=True, slots=True)
class _PatchReplacement:
    old: str
    new: str
    expected_occurrences: int


def _read_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="workspace_read",
        description="Read a bounded numbered window from one contained UTF-8 workspace file.",
        input_schema=WORKSPACE_READ_SCHEMA,
    )


def _search_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="workspace_search",
        description="Search contained workspace files for one bounded literal query.",
        input_schema=WORKSPACE_SEARCH_SCHEMA,
    )


def _patch_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="workspace_patch",
        description=(
            "Apply one governed create, exact replacement update, or digest-checked delete "
            "to one contained UTF-8 workspace file."
        ),
        input_schema=WORKSPACE_PATCH_SCHEMA,
    )


def _command_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="workspace_command",
        description=(
            "Run one trusted allowlisted executable with bounded direct arguments in a "
            "contained workspace directory."
        ),
        input_schema=WORKSPACE_COMMAND_SCHEMA,
    )


def _opaque_executable_identity(value: str) -> str:
    candidate = Path(value).name if "/" in value or "\\" in value or ":" in value else value
    if not candidate or any(
        character in candidate for character in ("/", "\\", ":", "\x00")
    ):
        raise ValueError("executable allowlist identity must be an opaque name")
    return candidate


def _normalize_executable_allowlist(
    allowlist: Mapping[str, str | Path | CommandExecutable] | None,
    *,
    shell_executables: Iterable[str],
) -> tuple[tuple[str, CommandExecutable], ...]:
    if allowlist is None:
        return ()
    if not isinstance(allowlist, Mapping):
        raise TypeError("executable allowlist must be a mapping")
    if isinstance(shell_executables, (str, bytes)):
        raise TypeError("shell executables must be an iterable of names")
    shell_names = tuple(shell_executables)
    if any(not isinstance(name, str) or not name.strip() for name in shell_names):
        raise ValueError("shell executable identities must be non-empty text")
    entries: list[tuple[str, CommandExecutable]] = []
    for raw_name, raw_value in allowlist.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("executable allowlist names must be non-empty text")
        identity = (
            raw_value.identity
            if isinstance(raw_value, CommandExecutable)
            else _opaque_executable_identity(raw_name)
        )
        if isinstance(raw_value, CommandExecutable):
            executable = raw_value
        else:
            if not isinstance(raw_value, (str, Path)):
                raise TypeError("executable allowlist values must be paths or CommandExecutable")
            executable = CommandExecutable(
                identity,
                Path(raw_value),
                shell=identity in shell_names or raw_name in shell_names,
            )
        if raw_name in shell_names and not executable.shell:
            executable = CommandExecutable(executable.identity, executable.path, shell=True)
        if any(alias == raw_name for alias, _ in entries):
            raise ValueError("executable allowlist identities must be unique")
        entries.append((raw_name, executable))
    return tuple(entries)


def _resolve_command_executable(
    requested: str,
    entries: tuple[tuple[str, CommandExecutable], ...],
    resolver: Callable[[str], CommandExecutable | Path | str | None] | None,
) -> CommandExecutable | None:
    if resolver is not None:
        raw = resolver(requested)
        if raw is None:
            return None
        if isinstance(raw, CommandExecutable):
            return raw
        if isinstance(raw, (str, Path)):
            return CommandExecutable(_opaque_executable_identity(requested), Path(raw))
        raise TypeError("trusted executable resolver returned an invalid value")

    for alias, executable in entries:
        if requested == alias or requested == executable.identity:
            return executable
        try:
            if Path(requested).resolve(strict=False) == executable.path:
                return executable
        except (OSError, RuntimeError, ValueError):
            continue
    return None


def _resolve_command_cwd(
    workspace: Workspace,
    raw_cwd: str,
) -> tuple[PurePosixPath, Path]:
    try:
        if raw_cwd == ".":
            resolved = workspace.resolve_root(purpose=WorkspacePurpose.COMMAND_CWD)
        else:
            logical = workspace.normalize(raw_cwd, purpose=WorkspacePurpose.COMMAND_CWD)
            resolved = workspace.resolve(logical, purpose=WorkspacePurpose.COMMAND_CWD)
    except WorkspaceError as error:
        raise ActionPreparationError(
            ToolErrorCode.CONTAINMENT_DENIED,
            "command working directory is not contained",
        ) from error
    if not resolved.is_directory or resolved.followed_link:
        raise ActionPreparationError(
            ToolErrorCode.CONTAINMENT_DENIED,
            "command working directory must be an existing contained directory",
        )
    return resolved.logical_path, resolved.path


def _prepare_command(
    arguments: Mapping[str, object],
    context: ToolExecutionContext,
    *,
    workspace: Workspace,
    limits: CommandToolLimits,
    executable_entries: tuple[tuple[str, CommandExecutable], ...],
    executable_resolver: Callable[[str], CommandExecutable | Path | str | None] | None,
    allow_shell: bool,
    required_capabilities: frozenset[IsolationCapability],
    environment: Mapping[str, str],
    secret_values: tuple[str, ...],
) -> PreparedAction:
    context.run_context.check_active()
    raw_argv = arguments.get("argv")
    if isinstance(raw_argv, (str, bytes)) or not isinstance(raw_argv, Sequence):
        raise ValueError("argv must be a non-empty array of strings")
    argv = tuple(raw_argv)
    if not argv or len(argv) > limits.max_argv_items:
        raise ValueError("argv exceeds its configured item bound")
    characters = 0
    if any(
        not isinstance(argument, str) or not argument or "\x00" in argument
        for argument in argv
    ):
        raise ValueError("argv arguments must be non-empty NUL-free strings")
    for argument in argv:
        characters += len(argument)
    if characters > limits.max_argv_characters:
        raise ValueError("argv exceeds its configured character bound")

    executable = _resolve_command_executable(argv[0], executable_entries, executable_resolver)
    if executable is None:
        raise ActionPreparationError(
            ToolErrorCode.POLICY_DENIED,
            "command executable is not allowlisted",
        )
    if executable.shell and not allow_shell:
        raise ActionPreparationError(
            ToolErrorCode.POLICY_DENIED,
            "shell executable is not enabled by trusted composition",
        )

    raw_cwd = arguments.get("cwd", ".")
    if not isinstance(raw_cwd, str) or not raw_cwd:
        raise ValueError("cwd must be non-empty text")
    cwd, _cwd_path = _resolve_command_cwd(workspace, raw_cwd)

    raw_timeout = arguments.get("timeout_seconds", limits.max_timeout_seconds)
    if (
        isinstance(raw_timeout, bool)
        or not isinstance(raw_timeout, (int, float))
        or not math.isfinite(float(raw_timeout))
        or float(raw_timeout) <= 0
    ):
        raise ValueError("timeout_seconds must be a finite number greater than zero")
    timeout = float(raw_timeout)
    if timeout > limits.max_timeout_seconds:
        raise ValueError("timeout_seconds may only tighten the configured ceiling")

    normalized_argv = (executable.identity, *argv[1:])
    normalized_fields: dict[str, CanonicalJsonValue] = {
        "argv": tuple(normalized_argv),
        "cwd": str(cwd),
        "environment": tuple(sorted(environment)),
        "shell": executable.shell,
        "timeout_seconds": timeout,
    }
    return _PreparedCommandAction(
        ActionKind.COMMAND,
        EffectKind.PROCESS_EXECUTION,
        workspace.scope.workspace_id,
        cwd=cwd,
        argv=normalized_argv,
        executable_identity=executable.identity,
        environment_identity=tuple(sorted(environment)),
        normalized_fields=normalized_fields,
        required_capabilities=required_capabilities,
        limits=EffectiveLimits(
            max_input_characters=limits.max_argument_bytes,
            max_output_characters=limits.max_output_characters,
            max_duration_seconds=timeout,
            max_argv_items=limits.max_argv_items,
            max_argv_characters=limits.max_argv_characters,
        ),
        display_text=(
            f"workspace_command argv={list(normalized_argv)!r} cwd={cwd} "
            f"timeout_seconds={timeout:g}"
        ),
        secret_values=secret_values,
        executable=executable,
    )


def _effective_limits(
    limits: CodingToolLimits,
    *,
    output_characters: int,
    elapsed_seconds: float,
) -> EffectiveLimits:
    return EffectiveLimits(
        max_input_characters=limits.max_argument_bytes,
        max_output_characters=output_characters,
        max_duration_seconds=elapsed_seconds,
    )


def _as_int(arguments: Mapping[str, object], name: str, default: int) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _as_text(arguments: Mapping[str, object], name: str, *, required: bool = True) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or (required and not value):
        raise ValueError(f"{name} must be non-empty text")
    return value


def _normalize_path(
    workspace: Workspace,
    raw_path: str,
    *,
    purpose: WorkspacePurpose,
    allow_root: bool = False,
) -> PurePosixPath:
    if allow_root and raw_path == ".":
        return PurePosixPath(".")
    return workspace.normalize(raw_path, purpose=purpose)


def _normalize_glob(raw_glob: str, limits: CodingToolLimits) -> str:
    if not raw_glob or len(raw_glob) > limits.max_glob_characters:
        raise ValueError("glob exceeds its bounded relative pattern limit")
    if "\x00" in raw_glob or "\\" in raw_glob:
        raise ValueError("glob must be a relative POSIX pattern")
    if raw_glob.startswith("/") or raw_glob.startswith("//"):
        raise ValueError("glob must be relative")
    if len(raw_glob) >= 2 and raw_glob[1] == ":" and raw_glob[0].isalpha():
        raise ValueError("glob must not contain a drive prefix")
    parts = raw_glob.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("glob contains an ambiguous or parent component")
    return raw_glob


def _prepare_read(
    arguments: Mapping[str, object],
    context: ToolExecutionContext,
    *,
    workspace: Workspace,
    limits: CodingToolLimits,
    secret_values: tuple[str, ...],
) -> PreparedAction:
    context.run_context.check_active()
    raw_path = _as_text(arguments, "path")
    path = _normalize_path(workspace, raw_path, purpose=WorkspacePurpose.READ)
    start_line = _as_int(arguments, "start_line", 1)
    line_count = _as_int(arguments, "line_count", limits.max_read_line_count)
    if start_line < 1 or line_count < 1:
        raise ValueError("read line window must be one-based and positive")
    if line_count > limits.max_read_line_count:
        raise ValueError("read line_count exceeds the configured limit")
    normalized: dict[str, None | bool | int | str] = {
        "path": str(path),
        "start_line": start_line,
        "line_count": line_count,
        "max_source_bytes": limits.max_read_source_bytes,
        "max_line_characters": limits.max_read_line_characters,
    }
    return PreparedAction(
        ActionKind.READ,
        EffectKind.NONE,
        workspace.scope.workspace_id,
        logical_targets=(path,),
        normalized_fields=normalized,
        limits=_effective_limits(
            limits,
            output_characters=limits.max_read_output_characters,
            elapsed_seconds=limits.max_read_elapsed_seconds,
        ),
        display_text=f"workspace_read path={path} start_line={start_line} line_count={line_count}",
        secret_values=secret_values,
    )


def _prepare_search(
    arguments: Mapping[str, object],
    context: ToolExecutionContext,
    *,
    workspace: Workspace,
    limits: CodingToolLimits,
    secret_values: tuple[str, ...],
) -> PreparedAction:
    context.run_context.check_active()
    query = _as_text(arguments, "query")
    if len(query) > limits.max_query_characters or "\x00" in query:
        raise ValueError("query exceeds its bounded literal limit")
    if "\r" in query or "\n" in query:
        raise ValueError("query must be contained within one source line")
    raw_path = arguments.get("path", ".")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("search path must be non-empty text")
    path = _normalize_path(
        workspace,
        raw_path,
        purpose=WorkspacePurpose.SEARCH,
        allow_root=True,
    )
    raw_glob = arguments.get("glob")
    glob = None if raw_glob is None else _normalize_glob(_as_text(arguments, "glob"), limits)
    case_sensitive = arguments.get("case_sensitive", True)
    if not isinstance(case_sensitive, bool):
        raise ValueError("case_sensitive must be a boolean")
    max_matches = _as_int(arguments, "max_matches", limits.max_search_matches)
    if max_matches < 1 or max_matches > limits.max_search_matches:
        raise ValueError("max_matches exceeds the configured limit")
    normalized: dict[str, None | bool | int | str] = {
        "query": query,
        "path": str(path),
        "glob": glob,
        "case_sensitive": case_sensitive,
        "max_matches": max_matches,
        "max_visited_files": limits.max_search_visited_files,
        "max_source_bytes": limits.max_search_source_bytes,
        "max_file_bytes": limits.max_search_file_bytes,
        "max_line_characters": limits.max_search_line_characters,
    }
    return PreparedAction(
        ActionKind.SEARCH,
        EffectKind.NONE,
        workspace.scope.workspace_id,
        logical_targets=(path,),
        normalized_fields=normalized,
        limits=_effective_limits(
            limits,
            output_characters=limits.max_search_output_characters,
            elapsed_seconds=limits.max_search_elapsed_seconds,
        ),
        display_text=(
            f"workspace_search path={path} query={query} "
            f"max_matches={max_matches}"
        ),
        secret_values=secret_values,
    )


def _patch_workspace_error(error: WorkspaceError, *, operation: str) -> ActionPreparationError:
    reason = str(error.reason_code)
    if reason in {WorkspaceReason.PROTECTED.value, WorkspaceReason.SECRET.value}:
        code = ToolErrorCode.PROTECTED_RESOURCE_DENIED
    elif reason in {WorkspaceReason.LINK_ESCAPE.value, WorkspaceReason.CONTAINMENT.value}:
        code = ToolErrorCode.CONTAINMENT_DENIED
    elif reason in {WorkspaceReason.TARGET_MISSING.value, WorkspaceReason.PARENT_MISSING.value}:
        code = (
            ToolErrorCode.PRECONDITION_CONFLICT
            if operation in {"update", "delete"}
            else ToolErrorCode.CONTAINMENT_DENIED
        )
    elif reason == WorkspaceReason.TARGET_KIND.value:
        code = ToolErrorCode.PRECONDITION_CONFLICT
    else:
        code = ToolErrorCode.GOVERNANCE_FAILURE
    return ActionPreparationError(code, "workspace patch target could not be prepared")


def _normalize_patch_path(workspace: Workspace, raw_path: object) -> PurePosixPath:
    if not isinstance(raw_path, str) or not raw_path:
        raise ActionPreparationError(
            ToolErrorCode.INVALID_ARGUMENTS,
            "patch path must be non-empty",
        )
    if any(character in _PATCH_WILDCARD_CHARACTERS for character in raw_path):
        raise ActionPreparationError(
            ToolErrorCode.INVALID_ARGUMENTS,
            "workspace patch paths do not support wildcard patterns",
        )
    try:
        return workspace.normalize(raw_path, purpose=WorkspacePurpose.PATCH)
    except WorkspaceError as error:
        raise _patch_workspace_error(error, operation="create") from error


def _resolve_patch_target(
    workspace: Workspace,
    path: PurePosixPath,
    *,
    operation: str,
) -> ResolvedWorkspacePath:
    try:
        resolved = workspace.resolve(path, purpose=WorkspacePurpose.PATCH)
    except WorkspaceError as error:
        raise _patch_workspace_error(error, operation=operation) from error
    if resolved.followed_link:
        raise ActionPreparationError(
            ToolErrorCode.CONTAINMENT_DENIED,
            "workspace patch targets may not be links",
        )
    if operation == "create":
        if not resolved.is_missing:
            raise ActionPreparationError(
                ToolErrorCode.PRECONDITION_CONFLICT,
                "workspace patch create target already exists",
            )
    elif not resolved.is_file:
        raise ActionPreparationError(
            ToolErrorCode.PRECONDITION_CONFLICT,
            "workspace patch target is not a regular file",
        )
    return resolved


def _revalidate_patch_target(
    workspace: Workspace,
    resolved: ResolvedWorkspacePath,
    *,
    operation: str,
) -> ResolvedWorkspacePath:
    try:
        current = workspace.revalidate(resolved)
    except WorkspaceError as error:
        raise ActionExecutionError(
            ToolErrorCode.PRECONDITION_CONFLICT,
            "workspace patch precondition changed before effect",
            effect_state=EffectState.NONE,
            diagnostics=("patch_precondition_conflict",),
        ) from error
    if current.followed_link:
        raise ActionExecutionError(
            ToolErrorCode.CONTAINMENT_DENIED,
            "workspace patch target became a link",
            effect_state=EffectState.NONE,
            diagnostics=("patch_link_target_denied",),
        )
    if operation == "create" and not current.is_missing:
        raise ActionExecutionError(
            ToolErrorCode.PRECONDITION_CONFLICT,
            "workspace patch create target appeared before effect",
            effect_state=EffectState.NONE,
            diagnostics=("patch_target_race",),
        )
    if operation != "create" and not current.is_file:
        raise ActionExecutionError(
            ToolErrorCode.PRECONDITION_CONFLICT,
            "workspace patch target changed kind before effect",
            effect_state=EffectState.NONE,
            diagnostics=("patch_precondition_conflict",),
        )
    return current


def _read_patch_bytes(
    resolved: ResolvedWorkspacePath,
    *,
    max_bytes: int,
    budget: _PatchBudget | None = None,
    preparation: bool = False,
) -> bytes:
    _check_patch_budget(budget, preparation=preparation)
    try:
        file_size = int(os.stat(resolved.path, follow_symlinks=False).st_size)
    except (OSError, ValueError) as error:
        raise ActionPreparationError(
            ToolErrorCode.EXECUTION_ERROR,
            "workspace patch target could not be inspected",
        ) from error
    if file_size > max_bytes:
        raise ActionPreparationError(
            ToolErrorCode.RESOURCE_LIMIT,
            "workspace patch source exceeds the configured byte limit",
        )
    _check_patch_budget(budget, preparation=preparation)
    try:
        with open(resolved.path, "rb") as handle:
            raw = handle.read(max_bytes + 1)
    except (OSError, ValueError) as error:
        raise ActionPreparationError(
            ToolErrorCode.EXECUTION_ERROR,
            "workspace patch target could not be read",
        ) from error
    _check_patch_budget(budget, preparation=preparation)
    if len(raw) > max_bytes:
        raise ActionPreparationError(
            ToolErrorCode.RESOURCE_LIMIT,
            "workspace patch source exceeds the configured byte limit",
        )
    return raw


def _encode_patch_text(value: object, *, label: str) -> bytes:
    if not isinstance(value, str):
        raise ActionPreparationError(ToolErrorCode.INVALID_ARGUMENTS, f"patch {label} must be text")
    if "\x00" in value:
        raise ActionPreparationError(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"patch {label} must not contain NUL characters",
        )
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ActionPreparationError(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"patch {label} must be valid UTF-8",
        ) from error


def _decode_patch_text(raw: bytes) -> str:
    if b"\x00" in raw:
        raise ActionPreparationError(
            ToolErrorCode.INVALID_ARGUMENTS,
            "workspace patch update does not support binary content",
        )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ActionPreparationError(
            ToolErrorCode.INVALID_ARGUMENTS,
            "workspace patch update requires valid UTF-8 content",
        ) from error


def _expected_patch_digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ActionPreparationError(
            ToolErrorCode.INVALID_ARGUMENTS,
            "patch expected_sha256 must be a lowercase SHA-256 digest",
        )
    return value


def _normalize_patch_replacements(
    raw: object,
    *,
    limits: CodingToolLimits,
    budget: _PatchBudget | None = None,
) -> tuple[_PatchReplacement, ...]:
    _check_patch_budget(budget, preparation=True)
    if not isinstance(raw, list) or not raw:
        raise ActionPreparationError(
            ToolErrorCode.INVALID_ARGUMENTS,
            "patch update requires ordered replacements",
        )
    if len(raw) > limits.max_patch_replacements:
        raise ActionPreparationError(
            ToolErrorCode.RESOURCE_LIMIT,
            "patch replacement count exceeds the configured limit",
        )
    replacements: list[_PatchReplacement] = []
    total_characters = 0
    for item in raw:
        _check_patch_budget(budget, preparation=True)
        if not isinstance(item, Mapping):
            raise ActionPreparationError(
                ToolErrorCode.INVALID_ARGUMENTS,
                "patch replacements must be objects",
            )
        old = item.get("old")
        new = item.get("new")
        expected = item.get("expected_occurrences")
        if not isinstance(old, str) or not old:
            raise ActionPreparationError(
                ToolErrorCode.INVALID_ARGUMENTS,
                "patch replacement old text must be non-empty",
            )
        if not isinstance(new, str) or not new:
            raise ActionPreparationError(
                ToolErrorCode.INVALID_ARGUMENTS,
                "patch replacement new text must be non-empty",
            )
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            raise ActionPreparationError(
                ToolErrorCode.INVALID_ARGUMENTS,
                "patch replacement expected_occurrences must be non-negative",
            )
        _encode_patch_text(old, label="replacement old")
        _encode_patch_text(new, label="replacement new")
        _check_patch_budget(budget, preparation=True)
        total_characters += len(old) + len(new)
        if total_characters > limits.max_patch_replacement_characters:
            raise ActionPreparationError(
                ToolErrorCode.RESOURCE_LIMIT,
                "patch replacement text exceeds the configured limit",
            )
        replacements.append(_PatchReplacement(old, new, expected))
    _check_patch_budget(budget, preparation=True)
    return tuple(replacements)


def _apply_patch_replacements(
    content: str,
    replacements: Sequence[_PatchReplacement],
    *,
    max_bytes: int,
    budget: _PatchBudget | None = None,
    preparation: bool = False,
) -> str:
    current = content
    for replacement in replacements:
        _check_patch_budget(budget, preparation=preparation)
        occurrences = current.count(replacement.old)
        _check_patch_budget(budget, preparation=preparation)
        if occurrences != replacement.expected_occurrences:
            raise ActionPreparationError(
                ToolErrorCode.PRECONDITION_CONFLICT,
                "workspace patch replacement occurrence count conflicted",
            )
        current_bytes = len(_encode_patch_text(current, label="updated content"))
        old_bytes = len(_encode_patch_text(replacement.old, label="replacement old"))
        new_bytes = len(_encode_patch_text(replacement.new, label="replacement new"))
        projected_bytes = current_bytes + occurrences * (new_bytes - old_bytes)
        if projected_bytes > max_bytes:
            raise ActionPreparationError(
                ToolErrorCode.RESOURCE_LIMIT,
                "workspace patch updated content exceeds the configured byte limit",
            )
        _check_patch_budget(budget, preparation=preparation)
        current = current.replace(replacement.old, replacement.new)
        _check_patch_budget(budget, preparation=preparation)
        if len(_encode_patch_text(current, label="updated content")) > max_bytes:
            raise ActionPreparationError(
                ToolErrorCode.RESOURCE_LIMIT,
                "workspace patch updated content exceeds the configured byte limit",
            )
        _check_patch_budget(budget, preparation=preparation)
    return current


def _patch_replacement_identity(
    replacements: Sequence[_PatchReplacement],
) -> list[dict[str, object]]:
    return [
        {
            "old": replacement.old,
            "new": replacement.new,
            "expected_occurrences": replacement.expected_occurrences,
        }
        for replacement in replacements
    ]


def _patch_preconditions(
    path: PurePosixPath,
    *,
    exists: bool,
    digest: str | None = None,
) -> EffectPreconditions:
    return EffectPreconditions(
        (
            EffectPrecondition(
                path,
                expected_kind=PathKind.REGULAR_FILE.value if exists else PathKind.MISSING.value,
                expected_sha256=digest,
                must_exist=exists,
            ),
        )
    )


def _prepare_patch(
    arguments: Mapping[str, object],
    context: ToolExecutionContext,
    *,
    workspace: Workspace,
    limits: CodingToolLimits,
    secret_values: tuple[str, ...],
    budget: _PatchBudget,
) -> PreparedAction:
    budget.check_preparation()
    operation = arguments.get("operation")
    budget.check_preparation()
    if operation not in {"create", "update", "delete"}:
        raise ActionPreparationError(
            ToolErrorCode.INVALID_ARGUMENTS,
            "workspace patch operation is unsupported",
        )
    path = _normalize_patch_path(workspace, arguments.get("path"))
    budget.check_preparation()
    normalized: dict[str, object] = {
        "operation": operation,
        "path": str(path),
        "max_content_bytes": limits.max_patch_content_bytes,
        "max_replacements": limits.max_patch_replacements,
        "max_replacement_characters": limits.max_patch_replacement_characters,
    }

    if operation == "create":
        content = arguments.get("content")
        budget.check_preparation()
        content_bytes = _encode_patch_text(content, label="content")
        budget.check_preparation()
        if len(content_bytes) > limits.max_patch_content_bytes:
            raise ActionPreparationError(
                ToolErrorCode.RESOURCE_LIMIT,
                "workspace patch content exceeds the configured byte limit",
            )
        resolved = _resolve_patch_target(workspace, path, operation=operation)
        budget.check_preparation()
        try:
            workspace.revalidate(resolved)
        except WorkspaceError as error:
            raise ActionPreparationError(
                ToolErrorCode.PRECONDITION_CONFLICT,
                "workspace patch create target changed during preparation",
            ) from error
        budget.check_preparation()
        normalized["content"] = content
        normalized["content_sha256"] = hashlib.sha256(content_bytes).hexdigest()
        budget.check_preparation()
        preconditions = _patch_preconditions(path, exists=False)
    elif operation == "update":
        expected_digest = _expected_patch_digest(arguments.get("expected_sha256"))
        budget.check_preparation()
        replacements = _normalize_patch_replacements(
            arguments.get("replacements"),
            limits=limits,
            budget=budget,
        )
        budget.check_preparation()
        resolved = _resolve_patch_target(workspace, path, operation=operation)
        budget.check_preparation()
        raw = _read_patch_bytes(
            resolved,
            max_bytes=limits.max_patch_content_bytes,
            budget=budget,
            preparation=True,
        )
        budget.check_preparation()
        try:
            workspace.revalidate(resolved)
        except WorkspaceError as error:
            raise ActionPreparationError(
                ToolErrorCode.PRECONDITION_CONFLICT,
                "workspace patch target changed during preparation",
            ) from error
        budget.check_preparation()
        actual_digest = hashlib.sha256(raw).hexdigest()
        budget.check_preparation()
        if actual_digest != expected_digest:
            raise ActionPreparationError(
                ToolErrorCode.PRECONDITION_CONFLICT,
                "workspace patch expected digest does not match",
            )
        content = _decode_patch_text(raw)
        budget.check_preparation()
        _apply_patch_replacements(
            content,
            replacements,
            max_bytes=limits.max_patch_content_bytes,
            budget=budget,
            preparation=True,
        )
        budget.check_preparation()
        normalized["expected_sha256"] = expected_digest
        normalized["replacements"] = _patch_replacement_identity(replacements)
        budget.check_preparation()
        preconditions = _patch_preconditions(path, exists=True, digest=expected_digest)
    else:
        expected_digest = _expected_patch_digest(arguments.get("expected_sha256"))
        budget.check_preparation()
        resolved = _resolve_patch_target(workspace, path, operation=operation)
        budget.check_preparation()
        raw = _read_patch_bytes(
            resolved,
            max_bytes=limits.max_patch_content_bytes,
            budget=budget,
            preparation=True,
        )
        budget.check_preparation()
        try:
            workspace.revalidate(resolved)
        except WorkspaceError as error:
            raise ActionPreparationError(
                ToolErrorCode.PRECONDITION_CONFLICT,
                "workspace patch target changed during preparation",
            ) from error
        budget.check_preparation()
        actual_digest = hashlib.sha256(raw).hexdigest()
        budget.check_preparation()
        if actual_digest != expected_digest:
            raise ActionPreparationError(
                ToolErrorCode.PRECONDITION_CONFLICT,
                "workspace patch expected digest does not match",
            )
        normalized["expected_sha256"] = expected_digest
        budget.check_preparation()
        preconditions = _patch_preconditions(path, exists=True, digest=expected_digest)

    action = PreparedAction(
        ActionKind.PATCH,
        EffectKind.WORKSPACE_MUTATION,
        workspace.scope.workspace_id,
        logical_targets=(path,),
        normalized_fields=cast(Mapping[str, CanonicalJsonValue], normalized),
        preconditions=preconditions,
        limits=_effective_limits(
            limits,
            output_characters=limits.max_patch_output_characters,
            elapsed_seconds=limits.max_patch_elapsed_seconds,
        ),
        display_text=f"workspace_patch operation={operation} path={path}",
        secret_values=secret_values,
    )
    budget.check_preparation()
    return action


def _patch_current_preconditions(
    action: PreparedAction,
    *,
    workspace: Workspace,
    limits: CodingToolLimits,
    budget: _PatchBudget | None = None,
) -> EffectPreconditions:
    path = action.logical_targets[0]
    expected = action.preconditions.items[0]
    try:
        resolved = workspace.resolve(path, purpose=WorkspacePurpose.PATCH)
    except WorkspaceError:
        return EffectPreconditions(
            (EffectPrecondition(path, expected_kind="unavailable", must_exist=None),)
        )
    if resolved.followed_link:
        return EffectPreconditions(
            (EffectPrecondition(path, expected_kind="link", must_exist=True),)
        )
    if resolved.is_missing:
        return _patch_preconditions(path, exists=False)
    if not resolved.is_file:
        return EffectPreconditions(
            (EffectPrecondition(path, expected_kind="other", must_exist=True),)
        )
    if expected.expected_sha256 is None:
        return _patch_preconditions(path, exists=True)
    try:
        _check_patch_budget(budget, preparation=False)
        raw = _read_patch_bytes(
            resolved,
            max_bytes=limits.max_patch_content_bytes,
            budget=budget,
        )
        _check_patch_budget(budget, preparation=False)
    except (ActionExecutionError, ActionPreparationError):
        return EffectPreconditions(
            (EffectPrecondition(path, expected_kind="unavailable", must_exist=True),)
        )
    return _patch_preconditions(
        path,
        exists=True,
        digest=hashlib.sha256(raw).hexdigest(),
    )


def _patch_guard_context_factory(
    workspace: Workspace,
    *,
    base_context: GuardContext | None,
    supplied_factory: GuardContextFactory | None,
    limits: CodingToolLimits,
) -> GuardContextFactory:
    def factory(action: PreparedAction, context: ToolExecutionContext) -> GuardContext:
        selected = (
            supplied_factory(action, context)
            if supplied_factory is not None
            else base_context
        )
        if not isinstance(selected, GuardContext):
            raise TypeError("patch guard context factory returned an invalid GuardContext")
        if selected.workspace.scope != workspace.scope:
            raise ValueError("patch guard context is bound to a different workspace")
        return replace(
            selected,
            current_preconditions=_patch_current_preconditions(
                action,
                workspace=workspace,
                limits=limits,
                budget=_active_patch_budget(action, context),
            ),
        )

    return factory


def _patch_replacements_from_action(action: PreparedAction) -> tuple[_PatchReplacement, ...]:
    raw = action.normalized_fields.get("replacements")
    if not isinstance(raw, tuple):
        raise ValueError("prepared patch replacements are malformed")
    values: list[_PatchReplacement] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("prepared patch replacement is malformed")
        old = item.get("old")
        new = item.get("new")
        expected = item.get("expected_occurrences")
        if (
            not isinstance(old, str)
            or not old
            or not isinstance(new, str)
            or not new
            or isinstance(expected, bool)
            or not isinstance(expected, int)
            or expected < 0
        ):
            raise ValueError("prepared patch replacement is malformed")
        values.append(_PatchReplacement(old, new, expected))
    return tuple(values)


def _patch_execution_precondition_error(error: Exception) -> ActionExecutionError:
    code = (
        error.code
        if isinstance(error, ActionPreparationError)
        else ToolErrorCode.PRECONDITION_CONFLICT
    )
    diagnostic = (
        "patch_resource_limit"
        if code is ToolErrorCode.RESOURCE_LIMIT
        else "patch_precondition_conflict"
    )
    return ActionExecutionError(
        code,
        "workspace patch precondition failed at the effect boundary",
        effect_state=EffectState.NONE,
        diagnostics=(diagnostic,),
    )


def _patch_success(
    action: PreparedAction,
    *,
    operation: str,
    status: str,
    **attributes: object,
) -> ActionExecutionResult:
    header: dict[str, object] = {
        "operation": operation,
        "path": str(action.logical_targets[0]),
        "status": status,
        "tool": "workspace_patch",
        **attributes,
    }
    output, output_limited = _render_bounded_output(
        header,
        (),
        action.limits.max_output_characters,
    )
    diagnostics = ("patch_output_truncated",) if output_limited else ()
    return ActionExecutionResult(output, diagnostics=diagnostics)


def _cleanup_patch_temp(path: Path) -> bool:
    try:
        os.unlink(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _write_patch_temp_and_replace(
    action: PreparedAction,
    *,
    target: Path,
    content: bytes,
    budget: _PatchBudget,
) -> None:
    file_descriptor: int | None = None
    temporary_path: Path | None = None
    handle = None

    def cleanup_once() -> bool:
        nonlocal temporary_path
        if temporary_path is None:
            return True
        path = temporary_path
        temporary_path = None
        return _cleanup_patch_temp(path)

    def close_open_resources() -> None:
        nonlocal file_descriptor
        nonlocal handle
        if handle is not None:
            with suppress(Exception):
                handle.close()
            handle = None
        if file_descriptor is not None:
            with suppress(OSError):
                os.close(file_descriptor)
            file_descriptor = None

    try:
        budget.check()
        try:
            file_descriptor, raw_temporary_path = tempfile.mkstemp(
                prefix=".dqagent-patch-",
                suffix=".tmp",
                dir=str(target.parent),
            )
            temporary_path = Path(raw_temporary_path)
        except (OSError, ValueError) as error:
            raise ActionExecutionError(
                ToolErrorCode.EXECUTION_ERROR,
                "workspace patch temporary file could not be created",
                effect_state=EffectState.NONE,
                diagnostics=("patch_temp_create_failed",),
            ) from error

        budget.check()
        try:
            handle = os.fdopen(file_descriptor, "wb")
            file_descriptor = None
        except Exception as error:
            if file_descriptor is not None:
                with suppress(OSError):
                    os.close(file_descriptor)
                file_descriptor = None
            cleanup_succeeded = cleanup_once()
            diagnostics = ["patch_temp_write_failed"]
            if not cleanup_succeeded:
                diagnostics.append("patch_temp_cleanup_failed")
            raise ActionExecutionError(
                ToolErrorCode.EXECUTION_ERROR,
                "workspace patch temporary file could not be opened",
                effect_state=EffectState.PARTIAL
                if not cleanup_succeeded
                else EffectState.NONE,
                diagnostics=tuple(diagnostics),
            ) from error

        budget.check()
        try:
            written = handle.write(content)
            if written != len(content):
                raise OSError("temporary file short write")
        except Exception as error:
            close_open_resources()
            cleanup_succeeded = cleanup_once()
            diagnostics = ["patch_temp_write_failed"]
            if not cleanup_succeeded:
                diagnostics.append("patch_temp_cleanup_failed")
            raise ActionExecutionError(
                ToolErrorCode.EXECUTION_ERROR,
                "workspace patch temporary file write failed",
                effect_state=EffectState.PARTIAL
                if not cleanup_succeeded
                else EffectState.NONE,
                diagnostics=tuple(diagnostics),
            ) from error

        budget.check()
        try:
            handle.close()
        except Exception as error:
            handle = None
            cleanup_succeeded = cleanup_once()
            diagnostics = ["patch_temp_write_failed"]
            if not cleanup_succeeded:
                diagnostics.append("patch_temp_cleanup_failed")
            raise ActionExecutionError(
                ToolErrorCode.EXECUTION_ERROR,
                "workspace patch temporary file close failed",
                effect_state=EffectState.PARTIAL
                if not cleanup_succeeded
                else EffectState.NONE,
                diagnostics=tuple(diagnostics),
            ) from error
        handle = None

        budget.check()
        try:
            os.replace(temporary_path, target)
        except Exception as error:
            cleanup_succeeded = cleanup_once()
            diagnostics = ["patch_atomic_replace_failed"]
            if not cleanup_succeeded:
                diagnostics.append("patch_temp_cleanup_failed")
            raise ActionExecutionError(
                ToolErrorCode.EXECUTION_ERROR,
                "workspace patch atomic replacement failed",
                effect_state=EffectState.UNKNOWN,
                diagnostics=tuple(diagnostics),
            ) from error
        temporary_path = None
        budget.check(effect_state=EffectState.UNKNOWN)
    except ActionExecutionError as error:
        close_open_resources()
        if temporary_path is not None:
            cleanup_succeeded = cleanup_once()
            if not cleanup_succeeded:
                cleanup_diagnostics = tuple(
                    (*error.diagnostics, "patch_temp_cleanup_failed")
                )
                raise ActionExecutionError(
                    error.code,
                    str(error),
                    effect_state=(
                        EffectState.PARTIAL
                        if error.effect_state is EffectState.NONE
                        else error.effect_state
                    ),
                    diagnostics=cleanup_diagnostics,
                ) from error
        raise
    except Exception as error:
        close_open_resources()
        cleanup_succeeded = cleanup_once()
        diagnostics = ["patch_atomic_replace_failed"]
        if not cleanup_succeeded:
            diagnostics.append("patch_temp_cleanup_failed")
        raise ActionExecutionError(
            ToolErrorCode.EXECUTION_ERROR,
            "workspace patch update failed",
            effect_state=EffectState.UNKNOWN,
            diagnostics=tuple(diagnostics),
        ) from error
    finally:
        close_open_resources()
        if temporary_path is not None:
            cleanup_once()


def _execute_patch_create(
    action: PreparedAction,
    context: ToolExecutionContext,
    *,
    workspace: Workspace,
    limits: CodingToolLimits,
    budget: _PatchBudget,
) -> ActionExecutionResult:
    path = action.logical_targets[0]
    content = action.normalized_fields.get("content")
    try:
        budget.check()
        content_bytes = _encode_patch_text(content, label="content")
        if len(content_bytes) > limits.max_patch_content_bytes:
            raise ActionPreparationError(
                ToolErrorCode.RESOURCE_LIMIT,
                "workspace patch content exceeds the configured byte limit",
            )
        budget.check()
        resolved = _resolve_patch_target(workspace, path, operation="create")
        budget.check()
        _revalidate_patch_target(workspace, resolved, operation="create")
        budget.check()
    except ActionExecutionError:
        raise
    except (ActionPreparationError, WorkspaceError) as error:
        raise _patch_execution_precondition_error(error) from error

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    file_descriptor: int | None = None
    try:
        budget.check()
        file_descriptor = os.open(resolved.path, flags, 0o666)
    except FileExistsError as error:
        raise ActionExecutionError(
            ToolErrorCode.PRECONDITION_CONFLICT,
            "workspace patch create target appeared during exclusive open",
            effect_state=EffectState.NONE,
            diagnostics=("patch_target_race",),
        ) from error
    except OSError as error:
        raise ActionExecutionError(
            ToolErrorCode.EXECUTION_ERROR,
            "workspace patch create target could not be opened",
            effect_state=EffectState.NONE,
            diagnostics=("patch_create_open_failed",),
        ) from error

    handle = None
    try:
        budget.check(effect_state=EffectState.PARTIAL)
        try:
            handle = os.fdopen(file_descriptor, "wb")
            file_descriptor = None
        except Exception as error:
            raise ActionExecutionError(
                ToolErrorCode.EXECUTION_ERROR,
                "workspace patch create file could not be opened for writing",
                effect_state=EffectState.PARTIAL,
                diagnostics=("patch_create_write_failed",),
            ) from error
        try:
            budget.check(effect_state=EffectState.PARTIAL)
        except ActionExecutionError:
            with suppress(Exception):
                handle.close()
            raise
        try:
            written = handle.write(content_bytes)
            if written != len(content_bytes):
                raise OSError("created file short write")
        except Exception as error:
            with suppress(Exception):
                handle.close()
            raise ActionExecutionError(
                ToolErrorCode.EXECUTION_ERROR,
                "workspace patch create write failed",
                effect_state=EffectState.PARTIAL,
                diagnostics=("patch_create_write_failed",),
            ) from error
        try:
            budget.check(effect_state=EffectState.PARTIAL)
        except ActionExecutionError:
            with suppress(Exception):
                handle.close()
            raise
        try:
            handle.close()
        except Exception as error:
            raise ActionExecutionError(
                ToolErrorCode.EXECUTION_ERROR,
                "workspace patch create close failed",
                effect_state=EffectState.PARTIAL,
                diagnostics=("patch_create_close_failed",),
            ) from error
        handle = None
        budget.check(effect_state=EffectState.PARTIAL)
    finally:
        if file_descriptor is not None:
            with suppress(OSError):
                os.close(file_descriptor)
    result = _patch_success(
        action,
        operation="create",
        status="created",
        bytes_written=len(content_bytes),
    )
    budget.check(effect_state=EffectState.UNKNOWN)
    return result


def _execute_patch_update(
    action: PreparedAction,
    context: ToolExecutionContext,
    *,
    workspace: Workspace,
    limits: CodingToolLimits,
    budget: _PatchBudget,
) -> ActionExecutionResult:
    path = action.logical_targets[0]
    expected_digest = action.normalized_fields.get("expected_sha256")
    if not isinstance(expected_digest, str):
        raise ActionExecutionError(
            ToolErrorCode.GOVERNANCE_FAILURE,
            "prepared workspace patch digest is malformed",
            effect_state=EffectState.NONE,
            diagnostics=("patch_precondition_conflict",),
        )
    try:
        budget.check()
        replacements = _patch_replacements_from_action(action)
        budget.check()
        resolved = _resolve_patch_target(workspace, path, operation="update")
        budget.check()
        raw = _read_patch_bytes(
            resolved,
            max_bytes=limits.max_patch_content_bytes,
            budget=budget,
        )
        budget.check()
        _revalidate_patch_target(workspace, resolved, operation="update")
        budget.check()
        if hashlib.sha256(raw).hexdigest() != expected_digest:
            raise ActionPreparationError(
                ToolErrorCode.PRECONDITION_CONFLICT,
                "workspace patch expected digest changed before effect",
            )
        content = _decode_patch_text(raw)
        budget.check()
        updated = _apply_patch_replacements(
            content,
            replacements,
            max_bytes=limits.max_patch_content_bytes,
            budget=budget,
        )
        budget.check()
        updated_bytes = _encode_patch_text(updated, label="updated content")
        budget.check()
        _revalidate_patch_target(workspace, resolved, operation="update")
        budget.check()
    except ActionExecutionError:
        raise
    except (ActionPreparationError, WorkspaceError) as error:
        raise _patch_execution_precondition_error(error) from error
    _write_patch_temp_and_replace(
        action,
        target=resolved.path,
        content=updated_bytes,
        budget=budget,
    )
    budget.check(effect_state=EffectState.UNKNOWN)
    result = _patch_success(
        action,
        operation="update",
        status="updated",
        bytes_written=len(updated_bytes),
        replacement_count=len(replacements),
    )
    budget.check(effect_state=EffectState.UNKNOWN)
    return result


def _execute_patch_delete(
    action: PreparedAction,
    context: ToolExecutionContext,
    *,
    workspace: Workspace,
    limits: CodingToolLimits,
    budget: _PatchBudget,
) -> ActionExecutionResult:
    path = action.logical_targets[0]
    expected_digest = action.normalized_fields.get("expected_sha256")
    if not isinstance(expected_digest, str):
        raise ActionExecutionError(
            ToolErrorCode.GOVERNANCE_FAILURE,
            "prepared workspace patch digest is malformed",
            effect_state=EffectState.NONE,
            diagnostics=("patch_precondition_conflict",),
        )
    try:
        budget.check()
        resolved = _resolve_patch_target(workspace, path, operation="delete")
        budget.check()
        raw = _read_patch_bytes(
            resolved,
            max_bytes=limits.max_patch_content_bytes,
            budget=budget,
        )
        budget.check()
        _revalidate_patch_target(workspace, resolved, operation="delete")
        budget.check()
        if hashlib.sha256(raw).hexdigest() != expected_digest:
            raise ActionPreparationError(
                ToolErrorCode.PRECONDITION_CONFLICT,
                "workspace patch expected digest changed before delete",
            )
        _revalidate_patch_target(workspace, resolved, operation="delete")
        budget.check()
    except ActionExecutionError:
        raise
    except (ActionPreparationError, WorkspaceError) as error:
        raise _patch_execution_precondition_error(error) from error
    try:
        os.remove(resolved.path)
    except FileNotFoundError as error:
        raise ActionExecutionError(
            ToolErrorCode.PRECONDITION_CONFLICT,
            "workspace patch delete target disappeared before removal",
            effect_state=EffectState.UNKNOWN,
            diagnostics=("patch_delete_failed",),
        ) from error
    except OSError as error:
        raise ActionExecutionError(
            ToolErrorCode.EXECUTION_ERROR,
            "workspace patch delete failed",
            effect_state=EffectState.UNKNOWN,
            diagnostics=("patch_delete_failed",),
        ) from error
    budget.check(effect_state=EffectState.UNKNOWN)
    result = _patch_success(
        action,
        operation="delete",
        status="deleted",
        bytes_deleted=len(raw),
    )
    budget.check(effect_state=EffectState.UNKNOWN)
    return result


def _execute_patch(
    action: PreparedAction,
    context: ToolExecutionContext,
    *,
    workspace: Workspace,
    limits: CodingToolLimits,
    budget: _PatchBudget,
) -> ActionExecutionResult:
    budget.check()
    operation = action.normalized_fields.get("operation")
    if operation == "create":
        return _execute_patch_create(
            action,
            context,
            workspace=workspace,
            limits=limits,
            budget=budget,
        )
    if operation == "update":
        return _execute_patch_update(
            action,
            context,
            workspace=workspace,
            limits=limits,
            budget=budget,
        )
    if operation == "delete":
        return _execute_patch_delete(
            action,
            context,
            workspace=workspace,
            limits=limits,
            budget=budget,
        )
    raise ActionExecutionError(
        ToolErrorCode.GOVERNANCE_FAILURE,
        "prepared workspace patch operation is malformed",
        effect_state=EffectState.NONE,
        diagnostics=("patch_precondition_conflict",),
    )


def _field_text(action: PreparedAction, name: str) -> str:
    value = action.normalized_fields[name]
    if not isinstance(value, str):
        raise ValueError(f"prepared field {name} is not text")
    return value


def _field_int(action: PreparedAction, name: str) -> int:
    value = action.normalized_fields[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"prepared field {name} is not an integer")
    return value


def _field_bool(action: PreparedAction, name: str) -> bool:
    value = action.normalized_fields[name]
    if not isinstance(value, bool):
        raise ValueError(f"prepared field {name} is not a boolean")
    return value


def _bounded_text(value: str, maximum: int, marker: str) -> str:
    if len(value) <= maximum:
        return value
    if maximum <= len(marker):
        return marker[:maximum]
    return value[: maximum - len(marker)] + marker


def _project_line(chars: list[str], truncated: bool, maximum: int) -> str:
    text = "".join(chars)
    return _bounded_text(text, maximum, _READ_MARKER) if truncated else text


def _render_header(header: Mapping[str, object]) -> str:
    return json.dumps(
        dict(header),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def _render_bounded_output(
    header: Mapping[str, object],
    lines: Iterable[str],
    maximum: int,
) -> tuple[str, bool]:
    """Render a header and ordered body without exceeding the character bound."""

    if maximum < 1:
        return "_", True
    base = _render_header(header)
    if len(base) > maximum:
        compact = _render_header(
            {
                "output_limit": True,
                "status": "output_limit",
                "tool": header.get("tool", "workspace_tool"),
            }
        ).rstrip("\n")
        if len(compact) <= maximum:
            return compact, True
        return _bounded_text(compact, maximum, _OUTPUT_MARKER), True
    parts = [base]
    used = len(base)
    for line in lines:
        candidate = line if not parts or parts[-1].endswith("\n") else "\n" + line
        candidate += "\n"
        if used + len(candidate) > maximum:
            return "".join(parts), True
        parts.append(candidate)
        used += len(candidate)
    return "".join(parts).rstrip("\n"), False


_CODING_DATA_VALUE_KEYS: Final[frozenset[str]] = frozenset(
    {"glob", "path", "query", "reason"}
)


def _sanitize_coding_output(
    guard_context: GuardContext,
    output: str,
    maximum: int,
    *,
    secret_values: tuple[str, ...],
) -> tuple[str, bool, bool]:
    raw_lines = output.splitlines()
    if not raw_lines:
        raise ValueError("coding output is empty")
    raw_header = json.loads(raw_lines[0])
    if not isinstance(raw_header, dict):
        raise ValueError("coding output header is not an object")
    header = dict(raw_header)
    sanitizer = guard_context.workspace.sanitizer(secrets=secret_values)
    for key in _CODING_DATA_VALUE_KEYS:
        value = header.get(key)
        if isinstance(value, str):
            header[key] = sanitizer.sanitize(value)

    def sanitized_body() -> Iterable[str]:
        for line in raw_lines[1:]:
            yield sanitizer.sanitize(line)

    was_limited = header.get("output_limit") is True or header.get("status") == "output_limit"
    rendered, output_limited = _render_bounded_output(header, sanitized_body(), maximum)
    if output_limited:
        header["output_limit"] = True
        header["status"] = "output_limit"
        rendered, _ = _render_bounded_output(header, sanitized_body(), maximum)
    return rendered, was_limited or output_limited, False


def _coding_output_sanitizer(secret_values: tuple[str, ...]) -> ActionOutputSanitizer:
    def sanitize(
        guard_context: GuardContext,
        output: str,
        max_characters: int,
    ) -> tuple[str, bool, bool]:
        return _sanitize_coding_output(
            guard_context,
            output,
            max_characters,
            secret_values=secret_values,
        )

    return sanitize


def _sanitize_command_value(value: object, sanitizer: Sanitizer) -> object:
    if isinstance(value, str):
        return sanitizer.sanitize(value)
    if isinstance(value, list):
        return [_sanitize_command_value(item, sanitizer) for item in value]
    if isinstance(value, dict):
        return {
            key: _sanitize_command_value(item, sanitizer)
            for key, item in value.items()
        }
    return value


def _command_output_sanitizer(secret_values: tuple[str, ...]) -> ActionOutputSanitizer:
    def sanitize(
        guard_context: GuardContext,
        output: str,
        max_characters: int,
    ) -> tuple[str, bool, bool]:
        raw_lines = output.splitlines()
        if not raw_lines:
            raise ValueError("command output is empty")
        raw_header = json.loads(raw_lines[0])
        if not isinstance(raw_header, dict):
            raise ValueError("command output header is not an object")
        sanitizer = guard_context.workspace.sanitizer(secrets=secret_values)
        header = {
            key: _sanitize_command_value(value, sanitizer)
            for key, value in raw_header.items()
        }
        rendered, truncated = _render_bounded_output(header, (), max_characters)
        return rendered, truncated, False

    return sanitize


def _command_result_payload(action: PreparedAction, result: SubprocessResult) -> str:
    payload: dict[str, object] = {
        "argv": list(action.argv),
        "backend": result.backend_identity,
        "backend_capabilities": [
            capability.value for capability in result.backend_capabilities
        ],
        "cleanup": result.cleanup.status.value,
        "cleanup_succeeded": result.cleanup_succeeded,
        "cwd": str(action.cwd or PurePosixPath(".")),
        "duration_seconds": result.duration_seconds,
        "exit_code": result.returncode,
        "stderr": result.stderr,
        "stderr_decode_replacements": result.stderr_decode_replacements,
        "stderr_truncated": result.stderr_truncated,
        "stdout": result.stdout,
        "stdout_decode_replacements": result.stdout_decode_replacements,
        "stdout_truncated": result.stdout_truncated,
        "status": result.status.value,
        "tool": "workspace_command",
    }
    if result.diagnostic:
        payload["diagnostic"] = result.diagnostic
    if result.spawn_error is not None:
        payload["spawn_error"] = result.spawn_error
    if result.missing_capabilities:
        payload["missing_capabilities"] = [
            capability.value for capability in result.missing_capabilities
        ]
    return _render_header(payload).rstrip("\n")


def _command_diagnostics(result: SubprocessResult) -> tuple[str, ...]:
    diagnostics: list[str] = [f"subprocess_status={result.status.value}"]
    if result.stdout_truncated:
        diagnostics.append("stdout_truncated")
    if result.stderr_truncated:
        diagnostics.append("stderr_truncated")
    if result.decode_replacements:
        diagnostics.append("output_decode_replacements")
    if not result.cleanup_succeeded and result.spawned:
        diagnostics.append("cleanup_incomplete")
    if result.diagnostic:
        diagnostics.append(result.diagnostic)
    return tuple(diagnostics[:8])


def _execute_command(
    action: PreparedAction,
    context: ToolExecutionContext,
    *,
    workspace: Workspace,
    allow_shell: bool,
    environment: Mapping[str, str],
    limits: CommandToolLimits,
    runner: SubprocessRunner,
) -> ActionExecutionResult:
    if not isinstance(action, _PreparedCommandAction):
        raise ActionExecutionError(
            ToolErrorCode.GOVERNANCE_FAILURE,
            "command executable binding is unavailable",
            effect_state=EffectState.NONE,
        )
    executable = action.executable
    if executable.shell and not allow_shell:
        raise ActionExecutionError(
            ToolErrorCode.POLICY_DENIED,
            "shell executable is not enabled by trusted composition",
            effect_state=EffectState.NONE,
        )
    command_cwd = action.cwd or PurePosixPath(".")
    try:
        if command_cwd == PurePosixPath("."):
            resolved_cwd = workspace.resolve_root(purpose=WorkspacePurpose.COMMAND_CWD)
        else:
            resolved_cwd = workspace.resolve(command_cwd, purpose=WorkspacePurpose.COMMAND_CWD)
    except WorkspaceError as error:
        raise ActionExecutionError(
            ToolErrorCode.CONTAINMENT_DENIED,
            "command working directory is no longer contained",
            effect_state=EffectState.NONE,
        ) from error
    if not resolved_cwd.is_directory or resolved_cwd.followed_link:
        raise ActionExecutionError(
            ToolErrorCode.CONTAINMENT_DENIED,
            "command working directory is no longer an existing directory",
            effect_state=EffectState.NONE,
        )

    request = SubprocessRequest(
        argv=(str(executable.path), *action.argv[1:]),
        cwd=resolved_cwd.path,
        environment=environment,
        timeout_seconds=action.limits.max_duration_seconds,
        stdout_limit_bytes=limits.max_stdout_bytes,
        stderr_limit_bytes=limits.max_stderr_bytes,
        required_capabilities=action.required_capabilities,
    )
    result = runner.run(request, context.run_context)
    payload = _command_result_payload(action, result)
    diagnostics = _command_diagnostics(result)
    if result.status in {
        SubprocessStatus.CANCELLED,
        SubprocessStatus.DEADLINE_EXCEEDED,
    }:
        context.run_context.check_active()
    if result.status is SubprocessStatus.NORMAL and result.returncode == 0:
        if not result.cleanup_succeeded:
            raise ActionExecutionError(
                ToolErrorCode.OBSERVATION_FAILURE,
                "command cleanup did not complete",
                effect_state=EffectState.UNKNOWN,
                diagnostics=diagnostics,
                output=payload,
            )
        return ActionExecutionResult(payload, EffectState.COMPLETE, diagnostics)

    if result.status is SubprocessStatus.CAPABILITY_DENIED:
        code = ToolErrorCode.CAPABILITY_MISSING
        effect_state = EffectState.NONE
    elif result.status is SubprocessStatus.TIMEOUT:
        code = ToolErrorCode.TIMEOUT
        effect_state = EffectState.UNKNOWN
    elif result.status is SubprocessStatus.CANCELLED:
        code = ToolErrorCode.EXECUTION_ERROR
        effect_state = EffectState.UNKNOWN
    elif result.status is SubprocessStatus.DEADLINE_EXCEEDED:
        code = ToolErrorCode.TIMEOUT
        effect_state = EffectState.UNKNOWN
    elif result.status is SubprocessStatus.OUTPUT_SANITIZATION_ERROR:
        code = ToolErrorCode.OBSERVATION_FAILURE
        effect_state = EffectState.UNKNOWN
    else:
        code = ToolErrorCode.PROCESS_FAILURE
        effect_state = EffectState.NONE if not result.spawned else EffectState.UNKNOWN
    raise ActionExecutionError(
        code,
        "workspace command did not complete successfully",
        effect_state=effect_state,
        diagnostics=diagnostics,
        output=payload,
    )


def _validate_coding_secret_values(secrets: tuple[str, ...]) -> None:
    if any(
        isinstance(secret, str)
        and secret
        and (
            any(secret in literal for literal in _CODING_FIXED_SERIALIZED_LITERALS)
            or secret in _CODING_JSON_SYNTAX_TOKENS
            or (secret.isascii() and secret.isdigit())
        )
        for secret in secrets
    ):
        raise ValueError("coding tool secret values collide with structured output metadata")


def _read_result_header(
    *,
    status: str,
    path: str | None,
    start_line: int,
    line_count: int,
    returned_lines: int,
    source_bytes: int,
    source_limit: bool = False,
    line_limit: bool = False,
    elapsed_limit: bool = False,
    output_limit: bool = False,
    reason: str | None = None,
) -> dict[str, object]:
    header: dict[str, object] = {
        "line_count": line_count,
        "elapsed_limit": elapsed_limit,
        "line_limit": line_limit,
        "output_limit": output_limit,
        "returned_lines": returned_lines,
        "source_bytes": source_bytes,
        "source_limit": source_limit,
        "start_line": start_line,
        "status": status,
        "tool": "workspace_read",
    }
    if path is not None:
        header["path"] = path
    if reason is not None:
        header["reason"] = reason
    return header


def _safe_boundary_reason(error: WorkspaceError) -> str:
    reason = str(error.reason_code)
    if reason in {WorkspaceReason.PROTECTED.value, WorkspaceReason.SECRET.value}:
        return reason
    if reason == WorkspaceReason.LINK_ESCAPE.value:
        return "link_not_followed"
    if reason == WorkspaceReason.TARGET_MISSING.value:
        return "missing"
    if reason == WorkspaceReason.PARENT_MISSING.value:
        return "missing"
    if reason == WorkspaceReason.DRIFT.value:
        return "drift"
    if reason == WorkspaceReason.FILESYSTEM.value:
        return "filesystem"
    return "denied"


def _read_source(
    path: os.PathLike[str] | str,
    *,
    start_line: int,
    line_count: int,
    source_limit: int,
    line_characters: int,
    check_budget: Callable[[], None] | None = None,
) -> _ReadProjection:
    """Stream one UTF-8 file while retaining only the requested line window."""

    projection = _ReadProjection(lines=[])
    try:
        file_size = int(os.stat(path, follow_symlinks=False).st_size)
    except (OSError, ValueError):
        projection.filesystem_error = True
        return projection
    read_limit = min(file_size, source_limit)
    projection.source_limit = file_size > source_limit
    if file_size == 0:
        return projection

    decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="strict")
    current_line = 1
    current_chars: list[str] = []
    current_characters = 0
    current_truncated = False
    current_has_data = False
    pending_cr = False
    window_end = start_line + line_count - 1
    window_complete = False

    def finish_line() -> None:
        nonlocal window_complete
        nonlocal current_line
        nonlocal current_chars
        nonlocal current_characters
        nonlocal current_truncated
        nonlocal current_has_data
        projection.lines_seen = max(projection.lines_seen, current_line)
        if start_line <= current_line <= window_end:
            projection.lines.append(
                (current_line, _project_line(current_chars, current_truncated, line_characters))
            )
            projection.line_limit = projection.line_limit or current_truncated
        if current_line >= window_end:
            window_complete = True
        current_line += 1
        current_chars = []
        current_characters = 0
        current_truncated = False
        current_has_data = False

    def consume_text(text: str) -> None:
        nonlocal pending_cr
        nonlocal current_characters
        nonlocal current_truncated
        nonlocal current_has_data
        for character in text:
            if check_budget is not None and current_characters % 1024 == 0:
                check_budget()
            if character == "\r":
                finish_line()
                pending_cr = True
                continue
            if character == "\n":
                if pending_cr:
                    pending_cr = False
                else:
                    finish_line()
                continue
            pending_cr = False
            current_has_data = True
            current_characters += 1
            if len(current_chars) < line_characters:
                current_chars.append(character)
            else:
                current_truncated = True

    try:
        with open(path, "rb") as handle:
            remaining = read_limit
            while remaining and not window_complete:
                if check_budget is not None:
                    check_budget()
                raw = handle.read(min(_CHUNK_BYTES, remaining))
                if not raw:
                    break
                projection.source_bytes += len(raw)
                remaining -= len(raw)
                if b"\x00" in raw:
                    projection.binary = True
                    return projection
                try:
                    consume_text(decoder.decode(raw, final=False))
                except UnicodeDecodeError:
                    projection.invalid_text = True
                    return projection
            if projection.source_limit:
                if current_has_data:
                    finish_line()
            elif not window_complete:
                if check_budget is not None:
                    check_budget()
                try:
                    consume_text(decoder.decode(b"", final=True))
                except UnicodeDecodeError:
                    projection.invalid_text = True
                    return projection
                if current_has_data:
                    finish_line()
    except _ReadElapsedLimit:
        projection.elapsed_limit = True
        if current_has_data:
            finish_line()
    except (OSError, ValueError):
        projection.filesystem_error = True
    return projection


def _execute_read(
    action: PreparedAction,
    context: ToolExecutionContext,
    *,
    workspace: Workspace,
    limits: CodingToolLimits,
) -> ActionExecutionResult:
    context.run_context.check_active()
    path = _field_text(action, "path")
    start_line = _field_int(action, "start_line")
    line_count = _field_int(action, "line_count")
    started = time.monotonic()

    def check_budget() -> None:
        context.run_context.check_active()
        if time.monotonic() - started >= limits.max_read_elapsed_seconds:
            raise _ReadElapsedLimit

    try:
        check_budget()
        resolved = workspace.resolve(path, purpose=WorkspacePurpose.READ)
        if resolved.followed_link:
            header = _read_result_header(
                status="omitted",
                path=None,
                start_line=start_line,
                line_count=line_count,
                returned_lines=0,
                source_bytes=0,
                reason="link_not_followed",
            )
            return ActionExecutionResult(
                _render_header(header).rstrip("\n"),
                diagnostics=("link_not_followed",),
            )
        check_budget()
        resolved = workspace.revalidate(resolved)
        if not resolved.is_file:
            raise WorkspaceError(
                reason_code=WorkspaceReason.TARGET_KIND,
                purpose=WorkspacePurpose.READ,
                workspace_id=workspace.scope.workspace_id,
            )
        projection = _read_source(
            resolved.path,
            start_line=start_line,
            line_count=line_count,
            source_limit=limits.max_read_source_bytes,
            line_characters=limits.max_read_line_characters,
            check_budget=check_budget,
        )
        check_budget()
        resolved = workspace.revalidate(resolved)
        del resolved
    except _ReadElapsedLimit:
        header = _read_result_header(
            status="elapsed_limit",
            path=path,
            start_line=start_line,
            line_count=line_count,
            returned_lines=0,
            source_bytes=0,
            elapsed_limit=True,
        )
        return ActionExecutionResult(
            _render_header(header).rstrip("\n"),
            diagnostics=("elapsed_limit",),
        )
    except WorkspaceError as error:
        reason = _safe_boundary_reason(error)
        if reason == "missing":
            header = _read_result_header(
                status="missing",
                path=path,
                start_line=start_line,
                line_count=line_count,
                returned_lines=0,
                source_bytes=0,
            )
            return ActionExecutionResult(
                _render_header(header).rstrip("\n"),
                diagnostics=("missing",),
            )
        header = _read_result_header(
            status="omitted",
            path=None,
            start_line=start_line,
            line_count=line_count,
            returned_lines=0,
            source_bytes=0,
            reason=reason,
        )
        return ActionExecutionResult(_render_header(header).rstrip("\n"), diagnostics=(reason,))

    if projection.binary:
        status = "binary"
        lines: list[str] = []
    elif projection.invalid_text:
        status = "invalid_text"
        lines = []
    elif projection.filesystem_error:
        header = _read_result_header(
            status="omitted",
            path=None,
            start_line=start_line,
            line_count=line_count,
            returned_lines=0,
            source_bytes=projection.source_bytes,
            reason="filesystem",
        )
        return ActionExecutionResult(
            _render_header(header).rstrip("\n"),
            diagnostics=("filesystem",),
        )
    elif projection.elapsed_limit:
        status = "elapsed_limit"
        lines = [f"{number}: {text}" for number, text in projection.lines]
    elif projection.source_limit:
        status = "source_limit"
        lines = [f"{number}: {text}" for number, text in projection.lines]
    elif not projection.lines and projection.source_bytes == 0:
        status = "empty"
        lines = []
    elif projection.line_limit:
        status = "line_limit"
        lines = [f"{number}: {text}" for number, text in projection.lines]
    elif len(projection.lines) < line_count:
        status = "eof"
        lines = [f"{number}: {text}" for number, text in projection.lines]
    else:
        status = "ok"
        lines = [f"{number}: {text}" for number, text in projection.lines]

    header = _read_result_header(
        status=status,
        path=path,
        start_line=start_line,
        line_count=line_count,
        returned_lines=len(projection.lines),
        source_bytes=projection.source_bytes,
        elapsed_limit=projection.elapsed_limit,
        source_limit=projection.source_limit,
        line_limit=projection.line_limit,
    )
    output, output_limited = _render_bounded_output(
        header,
        lines,
        limits.max_read_output_characters,
    )
    if output_limited:
        header["output_limit"] = True
        header["status"] = "output_limit"
        output, _ = _render_bounded_output(
            header,
            lines,
            limits.max_read_output_characters,
        )
    diagnostics = tuple(
        value
        for value, enabled in (
            ("binary", projection.binary),
            ("invalid_text", projection.invalid_text),
            ("source_limit", projection.source_limit),
            ("line_limit", projection.line_limit),
            ("elapsed_limit", projection.elapsed_limit),
            ("output_limit", output_limited),
        )
        if enabled
    )
    return ActionExecutionResult(output, diagnostics=diagnostics)


def _glob_match(
    logical_path: PurePosixPath,
    search_path: PurePosixPath,
    pattern: str | None,
    *,
    case_sensitive: bool,
) -> bool:
    if pattern is None:
        return True
    relative = logical_path
    if search_path != PurePosixPath("."):
        try:
            relative = logical_path.relative_to(search_path)
        except ValueError:
            return False
    candidate = relative.as_posix()
    values = (candidate, logical_path.as_posix())
    if not case_sensitive:
        folded_pattern = pattern.casefold()
        return any(fnmatchcase(value.casefold(), folded_pattern) for value in values)
    return any(fnmatchcase(value, pattern) for value in values)


def _search_file(
    path: os.PathLike[str] | str,
    logical_path: PurePosixPath,
    *,
    query: str,
    case_sensitive: bool,
    byte_limit: int,
    match_capacity: int,
    line_characters: int,
    check_budget: Callable[[], None],
) -> _SearchFileResult:
    result = _SearchFileResult(matches=[], source_bytes=0)
    folded_query = query if case_sensitive else query.casefold()
    query_length = len(folded_query)
    decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="strict")
    current_line = 1
    current_projection: list[str] = []
    current_characters = 0
    current_truncated = False
    current_has_data = False
    pending_cr = False
    line_columns: list[int] = []
    tail: deque[tuple[str, int]] = deque(maxlen=query_length)
    stop_after_file = False
    try:
        source_capped = byte_limit < int(os.stat(path, follow_symlinks=False).st_size)
    except (OSError, ValueError):
        result.filesystem_error = True
        return result

    def finish_line() -> None:
        nonlocal current_line
        nonlocal current_projection
        nonlocal current_characters
        nonlocal current_truncated
        nonlocal current_has_data
        nonlocal line_columns
        if current_truncated:
            result.line_limit = True
        if line_columns:
            projection = _project_line(
                current_projection,
                current_truncated,
                line_characters,
            )
            for column in line_columns:
                if len(result.matches) < match_capacity:
                    result.matches.append(
                        _SearchMatch(
                            logical_path,
                            current_line,
                            column,
                            projection,
                            current_truncated,
                        )
                    )
                else:
                    result.match_limit = True
        current_line += 1
        current_projection = []
        current_characters = 0
        current_truncated = False
        current_has_data = False
        line_columns = []
        tail.clear()

    def consume_text(text: str) -> None:
        nonlocal stop_after_file
        nonlocal pending_cr
        nonlocal current_characters
        nonlocal current_truncated
        nonlocal current_has_data
        for character in text:
            if current_characters % 1024 == 0:
                check_budget()
            if character == "\r":
                finish_line()
                pending_cr = True
                continue
            if character == "\n":
                if pending_cr:
                    pending_cr = False
                else:
                    finish_line()
                continue
            pending_cr = False
            current_has_data = True
            current_characters += 1
            if len(current_projection) < line_characters:
                current_projection.append(character)
            else:
                current_truncated = True
            folded_character = character if case_sensitive else character.casefold()
            for folded_part in folded_character:
                tail.append((folded_part, current_characters))
                if len(tail) != query_length:
                    continue
                candidate = "".join(part for part, _column in tail)
                matched = candidate == folded_query
                if not matched:
                    continue
                column = tail[0][1]
                if line_columns and line_columns[-1] == column:
                    continue
                if len(result.matches) + len(line_columns) < match_capacity:
                    line_columns.append(column)
                else:
                    stop_after_file = True
                    result.match_limit = True
                    return

    try:
        with open(path, "rb") as handle:
            remaining = byte_limit
            while remaining:
                check_budget()
                raw = handle.read(min(_CHUNK_BYTES, remaining))
                if not raw:
                    break
                result.source_bytes += len(raw)
                remaining -= len(raw)
                if b"\x00" in raw:
                    result.binary = True
                    result.matches.clear()
                    return result
                try:
                    consume_text(decoder.decode(raw, final=False))
                except UnicodeDecodeError:
                    result.invalid_text = True
                    result.matches.clear()
                    return result
                if stop_after_file:
                    if current_has_data or current_truncated or line_columns:
                        finish_line()
                    return result
            if not source_capped:
                try:
                    consume_text(decoder.decode(b"", final=True))
                except UnicodeDecodeError:
                    result.invalid_text = True
                    result.matches.clear()
                    return result
                if current_has_data or current_truncated or line_columns:
                    finish_line()
            else:
                result.source_limit = True
                if current_has_data or current_truncated or line_columns:
                    finish_line()
            if stop_after_file:
                result.match_limit = True
    except (OSError, ValueError):
        result.filesystem_error = True
        result.matches.clear()
    return result


def _search_header(
    *,
    query: str,
    path: str | None,
    glob: str | None,
    case_sensitive: bool,
    status: str,
    matches: int,
    visited_files: int,
    source_bytes: int,
    omissions: Mapping[str, int],
    source_limit: bool,
    match_limit: bool,
    line_limit: bool,
    visited_files_limit: bool,
    elapsed_limit: bool,
    output_limit: bool = False,
) -> dict[str, object]:
    header: dict[str, object] = {
        "case_sensitive": case_sensitive,
        "elapsed_limit": elapsed_limit,
        "glob": glob,
        "line_projection_limit": line_limit,
        "match_limit": match_limit,
        "matches": matches,
        "omissions": dict(omissions),
        "output_limit": output_limit,
        "source_bytes": source_bytes,
        "source_limit": source_limit,
        "status": status,
        "tool": "workspace_search",
        "visited_files_limit": visited_files_limit,
        "visited_files": visited_files,
    }
    if path is not None:
        header["path"] = path
    header["query"] = query
    return header


def _execute_search(
    action: PreparedAction,
    context: ToolExecutionContext,
    *,
    workspace: Workspace,
    limits: CodingToolLimits,
) -> ActionExecutionResult:
    context.run_context.check_active()
    query = _field_text(action, "query")
    path = _field_text(action, "path")
    raw_glob = action.normalized_fields["glob"]
    glob = raw_glob if isinstance(raw_glob, str) else None
    case_sensitive = _field_bool(action, "case_sensitive")
    max_matches = _field_int(action, "max_matches")
    max_visited_files = _field_int(action, "max_visited_files")
    max_source_bytes = _field_int(action, "max_source_bytes")
    max_file_bytes = _field_int(action, "max_file_bytes")
    max_line_characters = _field_int(action, "max_line_characters")
    started = time.monotonic()
    source_bytes = 0
    visited_files = 0
    matches: list[_SearchMatch] = []
    omissions = {key: 0 for key in _OMISSION_KEYS}
    source_limit = False
    match_limit = False
    line_limit = False
    visited_files_limit = False
    elapsed_limit = False
    safe_path: str | None = path

    def check_budget() -> None:
        context.run_context.check_active()
        if time.monotonic() - started >= limits.max_search_elapsed_seconds:
            raise _SearchElapsedLimit

    try:
        entries = workspace.walk(path, max_files=max_visited_files, cancel=check_budget)
        for entry in entries:
            check_budget()
            if entry.omitted:
                key = _search_omission_key(entry)
                omissions[key] = omissions.get(key, 0) + 1
                if key in {"protected", "secret", "link", "denied", "drift"}:
                    safe_path = None
                if key == "visited_files_limit":
                    omissions[key] = 1
                    visited_files_limit = True
                continue
            if entry.kind is not PathKind.REGULAR_FILE or entry.path is None:
                omissions["unsupported"] += 1
                continue
            visited_files += 1
            if not _glob_match(
                entry.logical_path,
                PurePosixPath(path),
                glob,
                case_sensitive=case_sensitive,
            ):
                continue
            check_budget()
            try:
                resolved = workspace.resolve(entry.logical_path, purpose=WorkspacePurpose.SEARCH)
                if resolved.followed_link:
                    omissions["link"] += 1
                    safe_path = None
                    continue
                resolved = workspace.revalidate(resolved)
                try:
                    file_size = int(os.stat(resolved.path, follow_symlinks=False).st_size)
                except (OSError, ValueError):
                    omissions["filesystem"] += 1
                    safe_path = None
                    continue
                if file_size > max_file_bytes:
                    omissions["oversized"] += 1
                    continue
                remaining_source = max_source_bytes - source_bytes
                if remaining_source <= 0:
                    source_limit = True
                    break
                scanned = _search_file(
                    resolved.path,
                    entry.logical_path,
                    query=query,
                    case_sensitive=case_sensitive,
                    byte_limit=min(file_size, remaining_source),
                    match_capacity=max_matches - len(matches),
                    line_characters=max_line_characters,
                    check_budget=check_budget,
                )
                source_bytes += scanned.source_bytes
                workspace.revalidate(resolved)
                if scanned.filesystem_error:
                    omissions["filesystem"] += 1
                    safe_path = None
                elif scanned.binary:
                    omissions["binary"] += 1
                elif scanned.invalid_text:
                    omissions["invalid_text"] += 1
                else:
                    matches.extend(scanned.matches)
                    line_limit = line_limit or scanned.line_limit
                    source_limit = source_limit or scanned.source_limit
                    match_limit = match_limit or scanned.match_limit
                if scanned.source_limit:
                    source_limit = True
                if scanned.match_limit:
                    match_limit = True
                    break
                # An exact budget boundary is not itself a truncation. The
                # next matching file, if any, will observe the exhausted
                # budget at the bounded read boundary below.
            except WorkspaceError as error:
                reason = _safe_boundary_reason(error)
                key = "link" if reason == "link_not_followed" else reason
                if key not in omissions:
                    key = "denied"
                omissions[key] += 1
                safe_path = None
    except _SearchElapsedLimit:
        elapsed_limit = True

    matches.sort(
        key=lambda item: (
            item.logical_path.as_posix().casefold(),
            item.logical_path.as_posix(),
            item.line,
            item.column,
        )
    )
    if elapsed_limit:
        status = "elapsed_limit"
    elif source_limit:
        status = "source_limit"
    elif match_limit:
        status = "match_limit"
    elif visited_files_limit:
        status = "visited_files_limit"
    elif line_limit:
        status = "line_limit"
    elif not matches:
        status = "no_matches"
    else:
        status = "ok"
    header = _search_header(
        query=query,
        path=safe_path,
        glob=glob,
        case_sensitive=case_sensitive,
        status=status,
        matches=len(matches),
        visited_files=visited_files,
        source_bytes=source_bytes,
        omissions=omissions,
        source_limit=source_limit,
        match_limit=match_limit,
        line_limit=line_limit,
        visited_files_limit=visited_files_limit,
        elapsed_limit=elapsed_limit,
    )
    def body_lines() -> Iterable[str]:
        for item in matches:
            yield f"{item.logical_path.as_posix()}:{item.line}:{item.column}: {item.projection}"

    output, output_limited = _render_bounded_output(
        header,
        body_lines(),
        limits.max_search_output_characters,
    )
    if output_limited:
        header["output_limit"] = True
        header["status"] = "output_limit"
        output, _ = _render_bounded_output(
            header,
            body_lines(),
            limits.max_search_output_characters,
        )
    limit_diagnostics = tuple(
        value
        for value, enabled in (
            ("source_limit", source_limit),
            ("match_limit", match_limit),
            ("line_limit", line_limit),
            ("elapsed_limit", elapsed_limit),
            ("output_limit", output_limited),
        )
        if enabled
    )
    omission_parts = tuple(
        f"{key}:{count}"
        for key, count in omissions.items()
        if count
    )
    omission_diagnostics = (
        (f"omissions={','.join(omission_parts)}",) if omission_parts else ()
    )
    no_result_diagnostics = ("no_matches",) if status == "no_matches" else ()
    diagnostics = limit_diagnostics + omission_diagnostics + no_result_diagnostics
    return ActionExecutionResult(output, diagnostics=diagnostics)


def _search_omission_key(entry: WorkspaceWalkEntry) -> str:
    reason = entry.omission_reason or "denied"
    if reason == "link_not_followed" or reason == WorkspaceReason.LINK_ESCAPE.value:
        return "link"
    if reason == WorkspaceReason.PROTECTED.value:
        return "protected"
    if reason == WorkspaceReason.SECRET.value:
        return "secret"
    if reason == WorkspaceReason.TARGET_MISSING.value:
        return "missing"
    if reason == WorkspaceReason.PARENT_MISSING.value:
        return "missing"
    if reason == WorkspaceReason.FILESYSTEM.value:
        return "filesystem"
    if reason == "filesystem_error":
        return "filesystem"
    if reason == "unsupported_kind":
        return "unsupported"
    if reason in {"visited_files_limit", "elapsed_limit"}:
        return reason
    if reason in {WorkspaceReason.DRIFT.value, WorkspaceReason.CONTAINMENT.value}:
        return "drift" if reason == WorkspaceReason.DRIFT.value else "denied"
    return "denied"


def _default_guard_context(
    workspace: Workspace,
    limits: CodingToolLimits,
    *,
    output_characters: int,
    elapsed_seconds: float,
    max_governed_calls: int,
) -> GuardContext:
    return GuardContext(
        workspace,
        configured_limits=_effective_limits(
            limits,
            output_characters=output_characters,
            elapsed_seconds=elapsed_seconds,
        ),
        max_governed_calls=max_governed_calls,
    )


def create_workspace_read_tool(
    workspace: Workspace,
    *,
    limits: CodingToolLimits | None = None,
    guard_context: GuardContext | None = None,
    guard_context_factory: GuardContextFactory | None = None,
    policy: ActionPolicy | None = None,
    approval_provider: ApprovalProvider | None = None,
    pre_hooks: Sequence[PreActionHook | PreActionHookSpec] = (),
    post_hooks: Sequence[PostActionHook] = (),
    secret_values: Iterable[str] = (),
    max_governed_calls: int = 1,
) -> ActionTool:
    selected_limits = limits or DEFAULT_CODING_TOOL_LIMITS
    secrets = tuple(secret_values)
    _validate_coding_secret_values(secrets)
    selected_context = guard_context
    if selected_context is None and guard_context_factory is None:
        selected_context = _default_guard_context(
            workspace,
            selected_limits,
            output_characters=selected_limits.max_read_output_characters,
            elapsed_seconds=selected_limits.max_read_elapsed_seconds,
            max_governed_calls=max_governed_calls,
        )
    return ActionTool(
        _read_tool_definition(),
        lambda arguments, context: _prepare_read(
            arguments,
            context,
            workspace=workspace,
            limits=selected_limits,
            secret_values=secrets,
        ),
        lambda action, context: _execute_read(
            action,
            context,
            workspace=workspace,
            limits=selected_limits,
        ),
        guard_context=selected_context,
        guard_context_factory=guard_context_factory,
        policy=policy or DefaultActionPolicy(),
        approval_provider=approval_provider,
        pre_hooks=tuple(pre_hooks),
        post_hooks=tuple(post_hooks),
        secret_values=secrets,
        output_sanitizer=_coding_output_sanitizer(secrets),
        max_argument_bytes=selected_limits.max_argument_bytes,
    )


def create_workspace_search_tool(
    workspace: Workspace,
    *,
    limits: CodingToolLimits | None = None,
    guard_context: GuardContext | None = None,
    guard_context_factory: GuardContextFactory | None = None,
    policy: ActionPolicy | None = None,
    approval_provider: ApprovalProvider | None = None,
    pre_hooks: Sequence[PreActionHook | PreActionHookSpec] = (),
    post_hooks: Sequence[PostActionHook] = (),
    secret_values: Iterable[str] = (),
    max_governed_calls: int = 1,
) -> ActionTool:
    selected_limits = limits or DEFAULT_CODING_TOOL_LIMITS
    secrets = tuple(secret_values)
    _validate_coding_secret_values(secrets)
    selected_context = guard_context
    if selected_context is None and guard_context_factory is None:
        selected_context = _default_guard_context(
            workspace,
            selected_limits,
            output_characters=selected_limits.max_search_output_characters,
            elapsed_seconds=selected_limits.max_search_elapsed_seconds,
            max_governed_calls=max_governed_calls,
        )
    return ActionTool(
        _search_tool_definition(),
        lambda arguments, context: _prepare_search(
            arguments,
            context,
            workspace=workspace,
            limits=selected_limits,
            secret_values=secrets,
        ),
        lambda action, context: _execute_search(
            action,
            context,
            workspace=workspace,
            limits=selected_limits,
        ),
        guard_context=selected_context,
        guard_context_factory=guard_context_factory,
        policy=policy or DefaultActionPolicy(),
        approval_provider=approval_provider,
        pre_hooks=tuple(pre_hooks),
        post_hooks=tuple(post_hooks),
        secret_values=secrets,
        output_sanitizer=_coding_output_sanitizer(secrets),
        max_argument_bytes=selected_limits.max_argument_bytes,
    )


def create_workspace_patch_tool(
    workspace: Workspace,
    *,
    limits: CodingToolLimits | None = None,
    guard_context: GuardContext | None = None,
    guard_context_factory: GuardContextFactory | None = None,
    policy: ActionPolicy | None = None,
    approval_provider: ApprovalProvider | None = None,
    pre_hooks: Sequence[PreActionHook | PreActionHookSpec] = (),
    post_hooks: Sequence[PostActionHook] = (),
    secret_values: Iterable[str] = (),
    max_governed_calls: int = 1,
) -> ActionTool:
    selected_limits = limits or DEFAULT_CODING_TOOL_LIMITS
    secrets = tuple(secret_values)
    _validate_coding_secret_values(secrets)
    selected_context = guard_context
    if selected_context is None and guard_context_factory is None:
        selected_context = _default_guard_context(
            workspace,
            selected_limits,
            output_characters=selected_limits.max_patch_output_characters,
            elapsed_seconds=selected_limits.max_patch_elapsed_seconds,
            max_governed_calls=max_governed_calls,
        )
    dynamic_factory = _patch_guard_context_factory(
        workspace,
        base_context=selected_context,
        supplied_factory=guard_context_factory,
        limits=selected_limits,
    )

    def prepare_patch(
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> PreparedAction:
        budget = _PatchBudget(context, selected_limits.max_patch_elapsed_seconds)
        action = _prepare_patch(
            arguments,
            context,
            workspace=workspace,
            limits=selected_limits,
            secret_values=secrets,
            budget=budget,
        )
        _PENDING_PATCH_BUDGET.set(
            (
                *_PENDING_PATCH_BUDGET.get(),
                _PendingPatchBudget(action, context, context.run_context.run_id, budget),
            )
        )
        return action

    def execute_patch(
        action: PreparedAction,
        context: ToolExecutionContext,
    ) -> ActionExecutionResult:
        budget = _active_patch_budget(action, context)
        if budget is None:
            budget = _PatchBudget(context, action.limits.max_duration_seconds)
        try:
            return _execute_patch(
                action,
                context,
                workspace=workspace,
                limits=selected_limits,
                budget=budget,
            )
        finally:
            _cleanup_patch_budget(action, context)

    return ActionTool(
        _patch_tool_definition(),
        prepare_patch,
        execute_patch,
        guard_context_factory=dynamic_factory,
        policy=policy or DefaultActionPolicy(),
        approval_provider=approval_provider,
        pre_hooks=tuple(pre_hooks),
        post_hooks=tuple(post_hooks),
        secret_values=secrets,
        output_sanitizer=_coding_output_sanitizer(secrets),
        max_argument_bytes=selected_limits.max_argument_bytes,
        action_cleanup=_cleanup_patch_budget,
    )


def create_workspace_command_tool(
    workspace: Workspace,
    *,
    limits: CommandToolLimits | None = None,
    executable_allowlist: Mapping[str, str | Path | CommandExecutable] | None = None,
    executable_resolver: Callable[[str], CommandExecutable | Path | str | None] | None = None,
    environment_allowlist: Mapping[str, str] | Iterable[str] = (),
    environment_source: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | Iterable[str] | None = None,
    env: Mapping[str, str] | Iterable[str] | None = None,
    secret_names: Iterable[str] = (),
    secret_values: Iterable[str] = (),
    shell_executables: Iterable[str] = (),
    allow_shell: bool = False,
    required_capabilities: Iterable[IsolationCapability] = (),
    subprocess_runner: SubprocessRunner | None = None,
    guard_context: GuardContext | None = None,
    guard_context_factory: GuardContextFactory | None = None,
    policy: ActionPolicy | None = None,
    approval_provider: ApprovalProvider | None = None,
    pre_hooks: Sequence[PreActionHook | PreActionHookSpec] = (),
    post_hooks: Sequence[PostActionHook] = (),
    max_governed_calls: int = 1,
) -> ActionTool:
    """Create the governed direct-argv command adapter.

    Executable resolution and environment construction happen entirely in
    trusted composition.  No model argument can add an executable, shell
    mode, or environment variable.
    """

    if not isinstance(workspace, Workspace):
        raise TypeError("workspace command requires a Workspace")
    selected_limits = limits or CommandToolLimits()
    if not isinstance(selected_limits, CommandToolLimits):
        raise TypeError("command limits must be a CommandToolLimits")
    if executable_allowlist is not None and executable_resolver is not None:
        raise ValueError("provide executable_allowlist or executable_resolver, not both")
    if executable_resolver is not None and not callable(executable_resolver):
        raise TypeError("executable resolver must be callable")
    entries = _normalize_executable_allowlist(
        executable_allowlist,
        shell_executables=shell_executables,
    )
    if not isinstance(allow_shell, bool):
        raise TypeError("allow_shell must be a boolean")

    if environment is not None and env is not None:
        raise TypeError("provide environment or env, not both")
    secrets = normalize_secret_values(secret_values)
    selected_environment: Mapping[str, str] | Iterable[str] = (
        environment
        if environment is not None
        else env
        if env is not None
        else environment_allowlist
    )
    trusted_environment = build_minimal_environment(
        selected_environment,
        source=environment_source,
        secret_names=secret_names,
        secret_values=secrets,
    )
    _validate_coding_secret_values(secrets)
    selected_capabilities = normalize_isolation_capabilities(
        (*_COMMAND_REQUIRED_CAPABILITIES, *tuple(required_capabilities))
    )
    runner = subprocess_runner
    if runner is None:
        runner = LocalSubprocessRunner(
            sanitizer=workspace.sanitizer(secrets=secrets),
        )
    runner_capabilities = getattr(runner, "capabilities", None)
    runner_identity = getattr(runner, "backend_identity", None)
    if runner_capabilities is None or runner_identity is None or not callable(
        getattr(runner, "run", None)
    ):
        raise TypeError("subprocess runner must expose identity, capabilities, and run")
    normalized_runner_capabilities = normalize_isolation_capabilities(runner_capabilities)
    if not isinstance(runner_identity, str) or not runner_identity.strip():
        raise ValueError("subprocess runner backend identity must be non-empty text")

    selected_context = guard_context
    if selected_context is None and guard_context_factory is None:
        selected_context = GuardContext(
            workspace,
            configured_limits=EffectiveLimits(
                max_input_characters=selected_limits.max_argument_bytes,
                max_output_characters=selected_limits.max_output_characters,
                max_duration_seconds=selected_limits.max_timeout_seconds,
                max_argv_items=selected_limits.max_argv_items,
                max_argv_characters=selected_limits.max_argv_characters,
            ),
            available_capabilities=normalized_runner_capabilities,
            max_governed_calls=max_governed_calls,
            backend_identity=runner_identity,
        )

    return ActionTool(
        _command_tool_definition(),
        lambda arguments, context: _prepare_command(
            arguments,
            context,
            workspace=workspace,
            limits=selected_limits,
            executable_entries=entries,
            executable_resolver=executable_resolver,
            allow_shell=allow_shell,
            required_capabilities=selected_capabilities,
            environment=trusted_environment,
            secret_values=secrets,
        ),
        lambda action, context: _execute_command(
            action,
            context,
            workspace=workspace,
            allow_shell=allow_shell,
            environment=trusted_environment,
            limits=selected_limits,
            runner=runner,
        ),
        guard_context=selected_context,
        guard_context_factory=guard_context_factory,
        policy=policy or DefaultActionPolicy(),
        approval_provider=approval_provider,
        pre_hooks=tuple(pre_hooks),
        post_hooks=tuple(post_hooks),
        secret_values=secrets,
        output_sanitizer=_command_output_sanitizer(secrets),
        max_argument_bytes=selected_limits.max_argument_bytes,
    )


create_workspace_read_action_tool = create_workspace_read_tool
create_workspace_search_action_tool = create_workspace_search_tool
create_workspace_patch_action_tool = create_workspace_patch_tool
create_workspace_command_action_tool = create_workspace_command_tool


def create_coding_tools(
    workspace: Workspace,
    *,
    limits: CodingToolLimits | None = None,
    guard_context: GuardContext | None = None,
    guard_context_factory: GuardContextFactory | None = None,
    policy: ActionPolicy | None = None,
    approval_provider: ApprovalProvider | None = None,
    pre_hooks: Sequence[PreActionHook | PreActionHookSpec] = (),
    post_hooks: Sequence[PostActionHook] = (),
    secret_values: Iterable[str] = (),
    max_governed_calls: int = 1,
) -> tuple[ActionTool, ActionTool, ActionTool]:
    selected_limits = limits or DEFAULT_CODING_TOOL_LIMITS
    secrets = tuple(secret_values)
    selected_context = guard_context
    if selected_context is None and guard_context_factory is None:
        selected_context = _default_guard_context(
            workspace,
            selected_limits,
            output_characters=max(
                selected_limits.max_read_output_characters,
                selected_limits.max_search_output_characters,
                selected_limits.max_patch_output_characters,
            ),
            elapsed_seconds=max(
                selected_limits.max_read_elapsed_seconds,
                selected_limits.max_search_elapsed_seconds,
                selected_limits.max_patch_elapsed_seconds,
            ),
            max_governed_calls=max_governed_calls,
        )
    return (
        create_workspace_read_tool(
            workspace,
            limits=selected_limits,
            guard_context=selected_context,
            guard_context_factory=guard_context_factory,
            policy=policy,
            approval_provider=approval_provider,
            pre_hooks=pre_hooks,
            post_hooks=post_hooks,
            secret_values=secrets,
            max_governed_calls=max_governed_calls,
        ),
        create_workspace_search_tool(
            workspace,
            limits=selected_limits,
            guard_context=selected_context,
            guard_context_factory=guard_context_factory,
            policy=policy,
            approval_provider=approval_provider,
            pre_hooks=pre_hooks,
            post_hooks=post_hooks,
            secret_values=secrets,
            max_governed_calls=max_governed_calls,
        ),
        create_workspace_patch_tool(
            workspace,
            limits=selected_limits,
            guard_context=selected_context,
            guard_context_factory=guard_context_factory,
            policy=policy,
            approval_provider=approval_provider,
            pre_hooks=pre_hooks,
            post_hooks=post_hooks,
            secret_values=secrets,
            max_governed_calls=max_governed_calls,
        ),
    )


def create_coding_tool_registry(
    workspace: Workspace,
    *,
    limits: CodingToolLimits | None = None,
    guard_context: GuardContext | None = None,
    guard_context_factory: GuardContextFactory | None = None,
    policy: ActionPolicy | None = None,
    approval_provider: ApprovalProvider | None = None,
    pre_hooks: Sequence[PreActionHook | PreActionHookSpec] = (),
    post_hooks: Sequence[PostActionHook] = (),
    secret_values: Iterable[str] = (),
    max_governed_calls: int = 1,
    command_limits: CommandToolLimits | None = None,
    executable_allowlist: Mapping[str, str | Path | CommandExecutable] | None = None,
    executable_resolver: Callable[[str], CommandExecutable | Path | str | None] | None = None,
    environment_allowlist: Mapping[str, str] | Iterable[str] = (),
    environment_source: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | Iterable[str] | None = None,
    env: Mapping[str, str] | Iterable[str] | None = None,
    secret_names: Iterable[str] = (),
    shell_executables: Iterable[str] = (),
    allow_shell: bool = False,
    required_capabilities: Iterable[IsolationCapability] = (),
    subprocess_runner: SubprocessRunner | None = None,
) -> ToolRegistry:
    secrets = normalize_secret_values(secret_values)
    tools = create_coding_tools(
        workspace,
        limits=limits,
        guard_context=guard_context,
        guard_context_factory=guard_context_factory,
        policy=policy,
        approval_provider=approval_provider,
        pre_hooks=pre_hooks,
        post_hooks=post_hooks,
        secret_values=secrets,
        max_governed_calls=max_governed_calls,
    )
    command_tool = create_workspace_command_tool(
        workspace,
        limits=command_limits,
        executable_allowlist=executable_allowlist,
        executable_resolver=executable_resolver,
        environment_allowlist=environment_allowlist,
        environment_source=environment_source,
        environment=environment,
        env=env,
        secret_names=secret_names,
        secret_values=secrets,
        shell_executables=shell_executables,
        allow_shell=allow_shell,
        required_capabilities=required_capabilities,
        subprocess_runner=subprocess_runner,
        guard_context=None,
        guard_context_factory=None,
        policy=policy,
        approval_provider=approval_provider,
        pre_hooks=pre_hooks,
        post_hooks=post_hooks,
        max_governed_calls=max_governed_calls,
    )
    return ToolRegistry((*tools, command_tool))
