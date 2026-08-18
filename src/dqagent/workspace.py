"""Workspace authority and bounded task observation.

This module owns DQAgent-controlled filesystem authority and the task-scoped
baseline/final observation used to establish workspace evidence. Observation
is deliberately not Git-backed and does not provide subprocess isolation.
"""

from __future__ import annotations

import difflib
import hashlib
import heapq
import math
import os
import re
import stat
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from itertools import chain
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, cast

from dqagent.errors import (
    ConfigurationError,
    DQAgentError,
    ErrorCategory,
    RunCancelledError,
    RunDeadlineExceededError,
)

__all__ = [
    "DEFAULT_PROTECTED_PATHS",
    "DEFAULT_SECRET_BASENAMES",
    "WorkspaceBlindSpot",
    "WorkspaceBlindSpotReason",
    "WorkspaceChange",
    "WorkspaceChangeKind",
    "WorkspaceCompleteness",
    "WorkspaceDiff",
    "WorkspaceEntry",
    "WorkspaceEntryKind",
    "WorkspaceObserver",
    "WorkspaceSnapshot",
    "WorkspaceWalkEntry",
    "PathKind",
    "SanitizedText",
    "Sanitizer",
    "Workspace",
    "WorkspaceAccessError",
    "WorkspaceConfigurationError",
    "WorkspaceDriftError",
    "WorkspaceError",
    "WorkspaceLimits",
    "WorkspacePathError",
    "WorkspacePurpose",
    "WorkspaceRevalidation",
    "WorkspaceReason",
    "WorkspaceScope",
    "ResolvedWorkspacePath",
    "sanitize_text",
]


DEFAULT_PROTECTED_PATHS: Final[tuple[PurePosixPath, ...]] = (PurePosixPath(".git"),)
DEFAULT_SECRET_BASENAMES: Final[frozenset[str]] = frozenset(
    {
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "private.key",
        "private.pem",
        "private_key",
    }
)
_DEFAULT_SECRET_SUFFIXES: Final[tuple[str, ...]] = (".key", ".pem")
_REDACTION: Final[str] = "[REDACTED]"
_TRUNCATION_MARKER: Final[str] = "...[truncated]"
_WINDOWS_REPARSE_POINT: Final[int] = 0x0400
_DEFAULT_MAX_LOGICAL_PATH_CHARACTERS: Final[int] = 4_096
_DEFAULT_MAX_LOGICAL_PATH_SEGMENTS: Final[int] = 128
_DEFAULT_MAX_SNAPSHOT_ENTRIES: Final[int] = 10_000
_DEFAULT_MAX_SNAPSHOT_BYTES: Final[int] = 16_000_000
_DEFAULT_MAX_SNAPSHOT_FILE_BYTES: Final[int] = 1_000_000
_DEFAULT_MAX_SNAPSHOT_ELAPSED_SECONDS: Final[float] = 10.0
_DEFAULT_MAX_RENDERED_DIFF_CHARACTERS: Final[int] = 200_000
_WORKSPACE_ROOT_PATH: Final[PurePosixPath] = PurePosixPath(".")
_MAX_RULE_ITEMS: Final[int] = 128
_MAX_RULE_CHARACTERS: Final[int] = 32_000
_MAX_SANITIZER_ITEMS: Final[int] = 128
_MAX_SANITIZER_CHARACTERS: Final[int] = 32_000
_MAX_SANITIZER_HOST_PATH_ITEMS: Final[int] = 32
_MAX_SANITIZER_HOST_PATH_CHARACTERS: Final[int] = 16_384


class WorkspacePurpose(StrEnum):
    """The operation-specific contract applied by the shared resolver."""

    READ = "read"
    SEARCH = "search"
    PATCH = "patch"
    INSTRUCTION = "instruction"
    SKILL = "skill"
    SNAPSHOT = "snapshot"
    COMMAND_CWD = "command_cwd"


class PathKind(StrEnum):
    """Safe filesystem kinds exposed by workspace resolution."""

    MISSING = "missing"
    REGULAR_FILE = "regular_file"
    DIRECTORY = "directory"
    OTHER = "other"


class WorkspaceEntryKind(StrEnum):
    """Filesystem kinds retained by a workspace snapshot.

    A link is represented as a link and is never resolved by the observer.
    """

    REGULAR = "regular"
    REGULAR_FILE = "regular"
    DIRECTORY = "directory"
    LINK = "link"
    SYMLINK = "link"
    OTHER = "other"


class WorkspaceChangeKind(StrEnum):
    """Deterministic baseline-to-final change categories."""

    CREATE = "create"
    UNTRACKED = "create"
    MODIFY = "modify"
    DELETE = "delete"
    TYPE_CHANGE = "type_change"


class WorkspaceBlindSpotReason(StrEnum):
    """Content-free reasons why an observation cannot prove absence/change."""

    PROTECTED = "protected"
    SECRET = "secret"
    VOLATILE_EXCLUSION = "volatile_exclusion"
    IGNORED = "ignored"
    ENTRIES_LIMIT = "entries_limit"
    PER_FILE_BYTES_LIMIT = "per_file_bytes_limit"
    AGGREGATE_BYTES_LIMIT = "aggregate_bytes_limit"
    ELAPSED_LIMIT = "elapsed_limit"
    CANCELLED = "cancelled"
    FILESYSTEM_ERROR = "filesystem_error"
    LINK_NOT_FOLLOWED = "link_not_followed"
    UNSUPPORTED_KIND = "unsupported_kind"
    PATH_UNREPRESENTABLE = "path_unrepresentable"
    CONTENT_CHANGED_DURING_READ = "content_changed_during_read"
    RENDERED_DIFF_LIMIT = "rendered_diff_limit"


class WorkspaceReason(StrEnum):
    """Stable, content-free reasons for workspace boundary failures."""

    INVALID_SCOPE = "invalid_scope"
    INVALID_LIMIT = "invalid_limit"
    INVALID_RULE = "invalid_rule"
    PATH_EMPTY = "path_empty"
    PATH_NUL = "path_nul"
    PATH_ABSOLUTE = "path_absolute"
    PATH_DRIVE = "path_drive"
    PATH_UNC = "path_unc"
    PATH_BACKSLASH = "path_backslash"
    PATH_PARENT = "path_parent"
    PATH_AMBIGUOUS = "path_ambiguous"
    PATH_TOO_LONG = "path_too_long"
    TOO_MANY_SEGMENTS = "too_many_segments"
    TARGET_MISSING = "target_missing"
    PARENT_MISSING = "parent_missing"
    TARGET_KIND = "target_kind"
    CONTAINMENT = "containment"
    LINK_ESCAPE = "link_escape"
    PROTECTED = "protected"
    SECRET = "secret"
    DRIFT = "drift"
    FILESYSTEM = "filesystem"


class WorkspaceError(DQAgentError, ValueError):
    """Base error raised at the workspace authority boundary.

    The message and event attributes intentionally omit host paths, raw OS
    errors, and denied content. Callers that need to make a control decision
    should use ``reason_code`` rather than parse the message.
    """

    category = ErrorCategory.CONFIGURATION

    def __init__(
        self,
        message: str = "workspace operation failed",
        *,
        reason_code: WorkspaceReason | str,
        purpose: WorkspacePurpose | str | None = None,
        workspace_id: str | None = None,
        logical_path: PurePosixPath | None = None,
        expose_logical_path: bool = False,
    ) -> None:
        self.reason_code = str(reason_code)
        self.purpose = _purpose_value(purpose) if purpose is not None else None
        self.workspace_id = workspace_id
        self.logical_path = (
            str(logical_path) if expose_logical_path and logical_path is not None else None
        )
        details = [f"reason={self.reason_code}"]
        if self.purpose is not None:
            details.append(f"purpose={self.purpose}")
        safe_message = f"{message} ({', '.join(details)})"
        super().__init__(safe_message, category=self.category)

    @property
    def reason(self) -> str:
        """Compatibility alias for code that uses reason instead of reason_code."""

        return self.reason_code

    @property
    def event_attributes(self) -> MappingProxyType[str, object]:
        """Return content-free attributes suitable for events and diagnostics."""

        attributes: dict[str, object] = {"reason_code": self.reason_code}
        if self.purpose is not None:
            attributes["purpose"] = self.purpose
        if self.workspace_id is not None:
            attributes["workspace_id"] = self.workspace_id
        if self.logical_path is not None:
            attributes["logical_path"] = self.logical_path
        return MappingProxyType(attributes)


class WorkspaceConfigurationError(WorkspaceError, ConfigurationError):
    """Raised when trusted workspace composition is invalid."""


class WorkspacePathError(WorkspaceError):
    """Raised when a logical path cannot be resolved safely."""


class WorkspaceAccessError(WorkspacePathError):
    """Raised when a protected or secret resource is denied."""


class WorkspaceDriftError(WorkspacePathError):
    """Raised when a previously resolved target no longer has the same authority."""


@dataclass(frozen=True, slots=True, init=False)
class WorkspaceLimits:
    """Limits owned by path handling and snapshot/diff observation only."""

    max_logical_path_characters: int
    max_logical_path_segments: int
    max_snapshot_entries: int
    max_snapshot_bytes: int
    max_snapshot_elapsed_seconds: float
    max_rendered_diff_characters: int
    max_snapshot_file_bytes: int

    def __init__(
        self,
        max_logical_path_characters: int = _DEFAULT_MAX_LOGICAL_PATH_CHARACTERS,
        max_logical_path_segments: int = _DEFAULT_MAX_LOGICAL_PATH_SEGMENTS,
        max_snapshot_entries: int = _DEFAULT_MAX_SNAPSHOT_ENTRIES,
        max_snapshot_bytes: int = _DEFAULT_MAX_SNAPSHOT_BYTES,
        max_snapshot_elapsed_seconds: float = _DEFAULT_MAX_SNAPSHOT_ELAPSED_SECONDS,
        max_rendered_diff_characters: int = _DEFAULT_MAX_RENDERED_DIFF_CHARACTERS,
        max_snapshot_file_bytes: int = _DEFAULT_MAX_SNAPSHOT_FILE_BYTES,
        *,
        max_path_characters: int | None = None,
        max_path_segments: int | None = None,
        snapshot_max_entries: int | None = None,
        snapshot_max_bytes: int | None = None,
        snapshot_max_file_bytes: int | None = None,
        max_per_file_bytes: int | None = None,
        snapshot_max_elapsed_seconds: float | None = None,
        max_diff_characters: int | None = None,
    ) -> None:
        values: dict[str, int | float] = {
            "max_logical_path_characters": max_logical_path_characters,
            "max_logical_path_segments": max_logical_path_segments,
            "max_snapshot_entries": max_snapshot_entries,
            "max_snapshot_bytes": max_snapshot_bytes,
            "max_snapshot_elapsed_seconds": max_snapshot_elapsed_seconds,
            "max_rendered_diff_characters": max_rendered_diff_characters,
            "max_snapshot_file_bytes": max_snapshot_file_bytes,
        }
        aliases: tuple[tuple[str, int | float | None, str], ...] = (
            ("max_logical_path_characters", max_path_characters, "max_path_characters"),
            ("max_logical_path_segments", max_path_segments, "max_path_segments"),
            ("max_snapshot_entries", snapshot_max_entries, "snapshot_max_entries"),
            ("max_snapshot_bytes", snapshot_max_bytes, "snapshot_max_bytes"),
            (
                "max_snapshot_elapsed_seconds",
                snapshot_max_elapsed_seconds,
                "snapshot_max_elapsed_seconds",
            ),
            ("max_rendered_diff_characters", max_diff_characters, "max_diff_characters"),
            ("max_snapshot_file_bytes", snapshot_max_file_bytes, "snapshot_max_file_bytes"),
            ("max_snapshot_file_bytes", max_per_file_bytes, "max_per_file_bytes"),
        )
        for field_name, alias_value, alias_name in aliases:
            if alias_value is None:
                continue
            default_value = {
                "max_logical_path_characters": _DEFAULT_MAX_LOGICAL_PATH_CHARACTERS,
                "max_logical_path_segments": _DEFAULT_MAX_LOGICAL_PATH_SEGMENTS,
                "max_snapshot_entries": _DEFAULT_MAX_SNAPSHOT_ENTRIES,
                "max_snapshot_bytes": _DEFAULT_MAX_SNAPSHOT_BYTES,
                "max_snapshot_elapsed_seconds": _DEFAULT_MAX_SNAPSHOT_ELAPSED_SECONDS,
                "max_rendered_diff_characters": _DEFAULT_MAX_RENDERED_DIFF_CHARACTERS,
                "max_snapshot_file_bytes": _DEFAULT_MAX_SNAPSHOT_FILE_BYTES,
            }[field_name]
            if values[field_name] != default_value and values[field_name] != alias_value:
                raise ValueError(f"{field_name} and {alias_name} specify different values")
            values[field_name] = alias_value

        for field_name in (
            "max_logical_path_characters",
            "max_logical_path_segments",
            "max_snapshot_entries",
            "max_snapshot_bytes",
            "max_rendered_diff_characters",
            "max_snapshot_file_bytes",
        ):
            _validate_positive_int(field_name, values[field_name])
        _validate_positive_float(
            "max_snapshot_elapsed_seconds", values["max_snapshot_elapsed_seconds"]
        )

        for field_name, value in values.items():
            object.__setattr__(self, field_name, value)

    @property
    def max_path_characters(self) -> int:
        return self.max_logical_path_characters

    @property
    def max_path_segments(self) -> int:
        return self.max_logical_path_segments

    @property
    def snapshot_max_entries(self) -> int:
        return self.max_snapshot_entries

    @property
    def snapshot_max_bytes(self) -> int:
        return self.max_snapshot_bytes

    @property
    def snapshot_max_file_bytes(self) -> int:
        return self.max_snapshot_file_bytes

    @property
    def max_per_file_bytes(self) -> int:
        return self.max_snapshot_file_bytes

    @property
    def snapshot_max_elapsed_seconds(self) -> float:
        return self.max_snapshot_elapsed_seconds

    @property
    def max_diff_characters(self) -> int:
        return self.max_rendered_diff_characters


@dataclass(frozen=True, slots=True)
class WorkspaceBlindSpot:
    """A safe logical region whose state could not be fully observed."""

    logical_path: PurePosixPath
    reason: WorkspaceBlindSpotReason | str
    subtree: bool = False
    aggregate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.logical_path, PurePosixPath):
            raise TypeError("blind-spot path must be a PurePosixPath")
        if not isinstance(self.reason, (WorkspaceBlindSpotReason, str)) or not str(
            self.reason
        ):
            raise ValueError("blind-spot reason must not be empty")
        if not isinstance(self.subtree, bool):
            raise TypeError("blind-spot subtree must be a boolean")
        if not isinstance(self.aggregate, bool):
            raise TypeError("blind-spot aggregate must be a boolean")

    @property
    def path(self) -> PurePosixPath:
        """Compatibility alias for consumers that call it a path."""

        return self.logical_path

    @property
    def reason_code(self) -> str:
        return str(self.reason)

    @property
    def is_aggregate(self) -> bool:
        return self.aggregate


@dataclass(frozen=True, slots=True)
class WorkspaceCompleteness:
    """Observation evidence and its scoped proof boundaries.

    ``global_complete`` is intentionally independent from ``target_complete``:
    a target can be completely observed while another workspace region remains
    a blind spot. Render truncation affects display evidence, not the structured
    file comparison itself.
    """

    inventory_complete: bool
    content_complete: bool
    blind_spots: tuple[WorkspaceBlindSpot, ...] = ()
    target_paths: tuple[PurePosixPath, ...] = ()
    target_complete: bool = False
    forbidden_paths: tuple[PurePosixPath, ...] = ()
    forbidden_complete: bool = True
    global_complete: bool = False
    rendered_diff_complete: bool = True
    rendered_diff_omission_reason: WorkspaceBlindSpotReason | str | None = None
    observed_entries: int = 0
    observed_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "inventory_complete",
            "content_complete",
            "target_complete",
            "forbidden_complete",
            "global_complete",
            "rendered_diff_complete",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if not isinstance(self.blind_spots, tuple) or not all(
            isinstance(item, WorkspaceBlindSpot) for item in self.blind_spots
        ):
            raise TypeError("blind_spots must be a tuple of WorkspaceBlindSpot")
        if not isinstance(self.target_paths, tuple) or not all(
            isinstance(item, PurePosixPath) for item in self.target_paths
        ):
            raise TypeError("target_paths must be a tuple of PurePosixPath")
        if not isinstance(self.forbidden_paths, tuple) or not all(
            isinstance(item, PurePosixPath) for item in self.forbidden_paths
        ):
            raise TypeError("forbidden_paths must be a tuple of PurePosixPath")
        if self.rendered_diff_omission_reason is not None and not isinstance(
            self.rendered_diff_omission_reason, (WorkspaceBlindSpotReason, str)
        ):
            raise TypeError("rendered diff omission reason must be text or None")
        for name in ("observed_entries", "observed_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def complete(self) -> bool:
        """Whether the entire observed scope has proof-level evidence."""

        return (
            self.inventory_complete
            and self.content_complete
            and self.global_complete
        )

    @property
    def observation_complete(self) -> bool:
        return (
            self.inventory_complete
            and self.content_complete
            and self.global_complete
        )

    @property
    def is_complete(self) -> bool:
        return self.complete

    @property
    def omissions(self) -> tuple[WorkspaceBlindSpot, ...]:
        return self.blind_spots

    @property
    def omission_reasons(self) -> tuple[str, ...]:
        reasons = [item.reason_code for item in self.blind_spots]
        if self.rendered_diff_omission_reason is not None:
            reasons.append(str(self.rendered_diff_omission_reason))
        return tuple(dict.fromkeys(reasons))

    @property
    def has_blind_spots(self) -> bool:
        return bool(self.blind_spots)

    @property
    def target_evidence_complete(self) -> bool:
        return self.target_complete

    @property
    def forbidden_evidence_complete(self) -> bool:
        return self.forbidden_complete

    @property
    def global_evidence_complete(self) -> bool:
        return self.global_complete

    @property
    def rendered_complete(self) -> bool:
        return self.rendered_diff_complete


@dataclass(frozen=True, slots=True)
class WorkspaceEntry:
    """Immutable, safe metadata for one observed workspace entry.

    ``digest`` is present only when the complete file bytes were read. Text is
    retained only for bounded UTF-8 projection; binary and oversized entries
    retain metadata without content.
    """

    logical_path: PurePosixPath
    kind: WorkspaceEntryKind
    size: int
    digest: str | None = None
    text: str | None = None
    content_complete: bool = True
    omission_reason: WorkspaceBlindSpotReason | str | None = None
    binary: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.logical_path, PurePosixPath):
            raise TypeError("entry path must be a PurePosixPath")
        if not isinstance(self.kind, WorkspaceEntryKind):
            raise TypeError("entry kind must be a WorkspaceEntryKind")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("entry size must be a non-negative integer")
        if self.digest is not None and (
            not isinstance(self.digest, str) or not self.digest
        ):
            raise ValueError("entry digest must be non-empty text when present")
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("entry text must be text or None")
        if not isinstance(self.content_complete, bool) or not isinstance(self.binary, bool):
            raise TypeError("entry completeness and binary flags must be boolean")
        if self.kind is not WorkspaceEntryKind.REGULAR and self.text is not None:
            raise ValueError("only regular entries may retain text")
        if self.binary and self.text is not None:
            raise ValueError("binary entries must not retain text")
        if self.omission_reason is not None and not isinstance(
            self.omission_reason, (WorkspaceBlindSpotReason, str)
        ):
            raise TypeError("entry omission reason must be text or None")

    @property
    def path(self) -> PurePosixPath:
        return self.logical_path

    @property
    def content_digest(self) -> str | None:
        return self.digest

    @property
    def fingerprint(self) -> str | None:
        return self.digest

    @property
    def has_full_digest(self) -> bool:
        return self.digest is not None and self.content_complete

    @property
    def is_regular(self) -> bool:
        return self.kind is WorkspaceEntryKind.REGULAR


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    """One bounded immutable observation of a workspace."""

    workspace_id: str
    captured_at: datetime
    entries: tuple[WorkspaceEntry, ...]
    completeness: WorkspaceCompleteness

    def __post_init__(self) -> None:
        if not isinstance(self.workspace_id, str) or not self.workspace_id:
            raise ValueError("snapshot workspace ID must not be empty")
        if not isinstance(self.captured_at, datetime):
            raise TypeError("snapshot timestamp must be a datetime")
        if not isinstance(self.entries, tuple) or not all(
            isinstance(item, WorkspaceEntry) for item in self.entries
        ):
            raise TypeError("snapshot entries must be a tuple of WorkspaceEntry")
        if not isinstance(self.completeness, WorkspaceCompleteness):
            raise TypeError("snapshot completeness must be WorkspaceCompleteness")
        paths = [item.logical_path for item in self.entries]
        if paths != sorted(paths, key=_logical_key):
            raise ValueError("snapshot entries must be in stable logical-path order")
        if len(set(paths)) != len(paths):
            raise ValueError("snapshot entries must have unique logical paths")

    @property
    def entry_map(self) -> MappingProxyType[str, WorkspaceEntry]:
        return MappingProxyType({str(item.logical_path): item for item in self.entries})

    @property
    def observation_complete(self) -> bool:
        return self.completeness.observation_complete

    @property
    def complete(self) -> bool:
        return self.completeness.complete

    @property
    def omissions(self) -> tuple[WorkspaceBlindSpot, ...]:
        return self.completeness.blind_spots


@dataclass(frozen=True, slots=True)
class WorkspaceChange:
    """One certain baseline-to-final change."""

    logical_path: PurePosixPath
    kind: WorkspaceChangeKind
    before: WorkspaceEntry | None
    after: WorkspaceEntry | None
    comparison_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.logical_path, PurePosixPath):
            raise TypeError("change path must be a PurePosixPath")
        if not isinstance(self.kind, WorkspaceChangeKind):
            raise TypeError("change kind must be a WorkspaceChangeKind")
        if self.before is None and self.after is None:
            raise ValueError("a change must have a before or after entry")
        if not isinstance(self.comparison_complete, bool):
            raise TypeError("comparison completeness must be a boolean")

    @property
    def path(self) -> PurePosixPath:
        return self.logical_path

    @property
    def old_entry(self) -> WorkspaceEntry | None:
        return self.before

    @property
    def new_entry(self) -> WorkspaceEntry | None:
        return self.after

    @property
    def is_untracked(self) -> bool:
        return self.kind is WorkspaceChangeKind.CREATE

    @property
    def certain(self) -> bool:
        return self.comparison_complete


@dataclass(frozen=True, slots=True)
class WorkspaceDiff:
    """Deterministic structured diff plus bounded unified projection."""

    baseline: WorkspaceSnapshot
    final: WorkspaceSnapshot
    changes: tuple[WorkspaceChange, ...]
    rendered_diff: str
    completeness: WorkspaceCompleteness

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, WorkspaceSnapshot) or not isinstance(
            self.final, WorkspaceSnapshot
        ):
            raise TypeError("diff snapshots must be WorkspaceSnapshot values")
        if not isinstance(self.changes, tuple) or not all(
            isinstance(item, WorkspaceChange) for item in self.changes
        ):
            raise TypeError("diff changes must be a tuple of WorkspaceChange")
        paths = [item.logical_path for item in self.changes]
        if paths != sorted(paths, key=_logical_key):
            raise ValueError("diff changes must be in stable logical-path order")
        if len(set(paths)) != len(paths):
            raise ValueError("diff changes must have unique logical paths")
        if not isinstance(self.rendered_diff, str):
            raise TypeError("rendered diff must be text")
        if not isinstance(self.completeness, WorkspaceCompleteness):
            raise TypeError("diff completeness must be WorkspaceCompleteness")

    @property
    def create(self) -> tuple[WorkspaceChange, ...]:
        return self.creates

    @property
    def creates(self) -> tuple[WorkspaceChange, ...]:
        return tuple(item for item in self.changes if item.kind is WorkspaceChangeKind.CREATE)

    @property
    def untracked(self) -> tuple[WorkspaceChange, ...]:
        return self.creates

    @property
    def modifies(self) -> tuple[WorkspaceChange, ...]:
        return tuple(item for item in self.changes if item.kind is WorkspaceChangeKind.MODIFY)

    @property
    def deletes(self) -> tuple[WorkspaceChange, ...]:
        return tuple(item for item in self.changes if item.kind is WorkspaceChangeKind.DELETE)

    @property
    def type_changes(self) -> tuple[WorkspaceChange, ...]:
        return tuple(
            item for item in self.changes if item.kind is WorkspaceChangeKind.TYPE_CHANGE
        )

    @property
    def changed_paths(self) -> tuple[PurePosixPath, ...]:
        return tuple(item.logical_path for item in self.changes)

    @property
    def omissions(self) -> tuple[WorkspaceBlindSpot, ...]:
        return self.completeness.blind_spots

    @property
    def unified_diff(self) -> str:
        return self.rendered_diff

    @property
    def diff_records(self) -> tuple[WorkspaceChange, ...]:
        return self.changes

    @property
    def records(self) -> tuple[WorkspaceChange, ...]:
        return self.changes

    @property
    def target_complete(self) -> bool:
        return self.completeness.target_complete

    @property
    def forbidden_complete(self) -> bool:
        return self.completeness.forbidden_complete

    @property
    def global_complete(self) -> bool:
        return self.completeness.global_complete

    @property
    def complete(self) -> bool:
        return self.completeness.complete


@dataclass(frozen=True, slots=True)
class WorkspaceScope:
    """Trusted, immutable authority for one existing workspace root."""

    workspace_id: str
    root: Path
    limits: WorkspaceLimits = field(default_factory=WorkspaceLimits)
    protected_paths: tuple[PurePosixPath, ...] = ()
    secret_paths: tuple[PurePosixPath, ...] = ()
    volatile_paths: tuple[PurePosixPath, ...] = ()
    ignored_paths: tuple[PurePosixPath, ...] = ()
    volatile_exclusions: tuple[PurePosixPath, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.workspace_id, str) or not self.workspace_id.strip():
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_SCOPE)
        if "\x00" in self.workspace_id:
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_SCOPE)
        if "/" in self.workspace_id or "\\" in self.workspace_id:
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_SCOPE)
        if not isinstance(self.root, Path):
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_SCOPE)
        if not isinstance(self.limits, WorkspaceLimits):
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_SCOPE)

        try:
            canonical_root = self.root.resolve(strict=True)
            root_is_directory = canonical_root.is_dir()
        except (OSError, RuntimeError):
            canonical_root = self.root
            root_is_directory = False
        if not root_is_directory:
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_SCOPE)

        protected = _normalize_rules(self.protected_paths, "protected_paths")
        protected = _unique_rules((*DEFAULT_PROTECTED_PATHS, *protected))
        secret = _normalize_rules(self.secret_paths, "secret_paths")
        volatile = _unique_rules(
            (
                *_normalize_rules(self.volatile_paths, "volatile_paths"),
                *_normalize_rules(self.volatile_exclusions, "volatile_exclusions"),
            )
        )
        ignored = _unique_rules(_normalize_rules(self.ignored_paths, "ignored_paths"))
        object.__setattr__(self, "root", canonical_root)
        object.__setattr__(self, "protected_paths", protected)
        object.__setattr__(self, "secret_paths", secret)
        object.__setattr__(self, "volatile_paths", volatile)
        object.__setattr__(self, "volatile_exclusions", volatile)
        object.__setattr__(self, "ignored_paths", ignored)


@dataclass(frozen=True, slots=True)
class _PathSignature:
    """Metadata used to detect authority drift without reading content."""

    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int


@dataclass(frozen=True, slots=True)
class WorkspaceRevalidation:
    """Effect-boundary checks retained by a resolved path."""

    workspace_id: str
    logical_path: PurePosixPath
    purpose: WorkspacePurpose
    authority_parent: PurePosixPath
    authority_parent_signature: _PathSignature
    target_signature: _PathSignature | None
    target_must_remain_missing: bool


@dataclass(frozen=True, slots=True)
class ResolvedWorkspacePath:
    """A safe path projection for a trusted caller.

    ``path`` is canonical and may be used by the caller that owns the
    workspace operation. It is not suitable for model-facing diagnostics.
    """

    workspace_id: str
    purpose: WorkspacePurpose
    logical_path: PurePosixPath
    path: Path
    kind: PathKind
    authority_parent: PurePosixPath
    authority_parent_path: Path
    revalidation: WorkspaceRevalidation
    followed_link: bool = False

    @property
    def absolute_path(self) -> Path:
        """Alias used by filesystem adapters."""

        return self.path

    @property
    def exists(self) -> bool:
        return self.kind is not PathKind.MISSING

    @property
    def is_missing(self) -> bool:
        return self.kind is PathKind.MISSING

    @property
    def is_file(self) -> bool:
        return self.kind is PathKind.REGULAR_FILE

    @property
    def is_directory(self) -> bool:
        return self.kind is PathKind.DIRECTORY


@dataclass(frozen=True, slots=True)
class WorkspaceWalkEntry:
    """One safe item yielded by the shared non-following workspace walker.

    ``path`` is an internal trusted projection for regular files only. Omitted
    entries intentionally retain only a logical path and a reason, so callers
    can count omissions without resolving or exposing denied targets.
    """

    logical_path: PurePosixPath
    path: Path | None = None
    kind: PathKind = PathKind.OTHER
    omission_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.logical_path, PurePosixPath):
            raise TypeError("walk logical path must be a PurePosixPath")
        if self.path is not None and not isinstance(self.path, Path):
            raise TypeError("walk path must be a Path or None")
        if not isinstance(self.kind, PathKind):
            raise TypeError("walk kind must be a PathKind")
        if self.omission_reason is not None and (
            not isinstance(self.omission_reason, str) or not self.omission_reason
        ):
            raise ValueError("walk omission reason must be non-empty text")
        if self.kind is PathKind.REGULAR_FILE and self.path is None:
            raise ValueError("regular walk entries require a trusted path")
        if self.omission_reason is None and self.kind is not PathKind.REGULAR_FILE:
            raise ValueError("non-file walk entries require an omission reason")

    @property
    def omitted(self) -> bool:
        return self.omission_reason is not None


@dataclass(frozen=True, slots=True)
class SanitizedText:
    """Bounded text plus evidence about redaction and truncation."""

    text: str
    redacted: bool
    truncated: bool
    original_characters: int


class Sanitizer:
    """Shared literal secret and host-path sanitizer."""

    def __init__(
        self,
        *,
        secrets: Iterable[str] = (),
        host_paths: Iterable[Path | str] = (),
        replacement: str = _REDACTION,
    ) -> None:
        if not isinstance(replacement, str) or not replacement:
            raise ValueError("sanitizer replacement must not be empty")
        normalized_secrets = _normalize_literals(secrets, "secrets")
        if any(
            secret in replacement or replacement in secret for secret in normalized_secrets
        ):
            raise ValueError(
                "sanitizer replacement must not contain or be contained by a configured secret"
            )
        normalized_paths = _normalize_host_paths(host_paths)
        self._secrets: tuple[str, ...] = normalized_secrets
        self._host_paths: tuple[str, ...] = normalized_paths
        self._replacement: str = replacement

    @property
    def secrets(self) -> tuple[str, ...]:
        return self._secrets

    @property
    def host_paths(self) -> tuple[str, ...]:
        return self._host_paths

    def sanitize(self, value: str, *, max_characters: int | None = None) -> str:
        return self.sanitize_with_evidence(value, max_characters=max_characters).text

    def sanitize_with_evidence(
        self, value: str, *, max_characters: int | None = None
    ) -> SanitizedText:
        if not isinstance(value, str):
            raise TypeError("sanitizer input must be text")
        if max_characters is not None:
            _validate_non_negative_int("maximum sanitized characters", max_characters)

        rendered = value
        redacted = False
        for secret in self._secrets:
            if secret in rendered:
                rendered = rendered.replace(secret, self._replacement)
                redacted = True

        for host_path in self._host_paths:
            pattern = re.escape(host_path)
            flags = re.IGNORECASE if os.name == "nt" else 0
            rendered, replacements = re.subn(pattern, self._replacement, rendered, flags=flags)
            redacted = redacted or replacements > 0

        original_characters = len(value)
        if max_characters is None or len(rendered) <= max_characters:
            output = rendered
            truncated = False
        else:
            output = _truncate_text(rendered, max_characters, _TRUNCATION_MARKER)
            truncated = True
        if any(secret in output for secret in self._secrets):
            raise ValueError("sanitizer output would contain a configured secret")
        return SanitizedText(output, redacted, truncated, original_characters)


def sanitize_text(
    value: str,
    *,
    secrets: Iterable[str] = (),
    host_paths: Iterable[Path | str] = (),
    max_characters: int | None = None,
    replacement: str = _REDACTION,
) -> str:
    """Redact literal secrets and host paths before applying a character bound."""

    return Sanitizer(
        secrets=secrets,
        host_paths=host_paths,
        replacement=replacement,
    ).sanitize(value, max_characters=max_characters)


class Workspace:
    """Resolve DQAgent-owned logical paths under one trusted scope."""

    def __init__(self, scope: WorkspaceScope) -> None:
        if not isinstance(scope, WorkspaceScope):
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_SCOPE)
        self._scope = scope

    @property
    def scope(self) -> WorkspaceScope:
        return self._scope

    @property
    def root(self) -> Path:
        """Return the canonical root for trusted filesystem adapters."""

        return self._scope.root

    def sanitizer(
        self,
        *,
        secrets: Iterable[str] = (),
        host_paths: Iterable[Path | str] = (),
    ) -> Sanitizer:
        """Create a sanitizer that also redacts this workspace's host root."""

        return Sanitizer(
            secrets=secrets,
            host_paths=chain((self._scope.root,), host_paths),
        )

    def sanitize(
        self,
        value: str,
        *,
        secrets: Iterable[str] = (),
        host_paths: Iterable[Path | str] = (),
        max_characters: int | None = None,
    ) -> str:
        return self.sanitizer(secrets=secrets, host_paths=host_paths).sanitize(
            value, max_characters=max_characters
        )

    def resolve(
        self,
        logical_path: str | PurePosixPath,
        *,
        purpose: WorkspacePurpose | str,
        allow_missing: bool | None = None,
    ) -> ResolvedWorkspacePath:
        """Resolve one logical path without reading file content.

        Only ``PATCH`` may resolve a missing final target. The missing target
        must have an existing contained parent; no directories are created.
        """

        resolved_purpose = _coerce_purpose(purpose)
        parsed = self._parse_path(logical_path, resolved_purpose)
        if allow_missing is not None and not isinstance(allow_missing, bool):
            raise TypeError("allow_missing must be a boolean or None")
        missing_allowed = (
            resolved_purpose is WorkspacePurpose.PATCH
            if allow_missing is None
            else allow_missing
        )
        if missing_allowed and resolved_purpose is not WorkspacePurpose.PATCH:
            raise WorkspacePathError(
                reason_code=WorkspaceReason.TARGET_MISSING,
                purpose=resolved_purpose,
                workspace_id=self._scope.workspace_id,
                logical_path=parsed,
                expose_logical_path=False,
            )

        self._deny_if_protected_or_secret(parsed, resolved_purpose)
        return self._resolve_components(parsed, resolved_purpose, missing_allowed)

    def resolve_root(
        self, *, purpose: WorkspacePurpose = WorkspacePurpose.SNAPSHOT
    ) -> ResolvedWorkspacePath:
        """Resolve the trusted root for snapshot and observation adapters."""

        if purpose not in {
            WorkspacePurpose.SNAPSHOT,
            WorkspacePurpose.SEARCH,
            WorkspacePurpose.COMMAND_CWD,
        }:
            raise WorkspacePathError(
                reason_code=WorkspaceReason.TARGET_KIND,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            )
        try:
            root_info = os.lstat(self._scope.root)
            if _is_link_or_reparse(root_info) or not stat.S_ISDIR(root_info.st_mode):
                raise OSError
            canonical_root = self._scope.root.resolve(strict=True)
            if canonical_root != self._scope.root or not canonical_root.is_dir():
                raise OSError
            signature = _signature(canonical_root)
        except (OSError, RuntimeError):
            raise WorkspaceDriftError(
                reason_code=WorkspaceReason.DRIFT,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            ) from None
        logical = PurePosixPath(".")
        revalidation = WorkspaceRevalidation(
            workspace_id=self._scope.workspace_id,
            logical_path=logical,
            purpose=purpose,
            authority_parent=logical,
            authority_parent_signature=signature,
            target_signature=signature,
            target_must_remain_missing=False,
        )
        return ResolvedWorkspacePath(
            workspace_id=self._scope.workspace_id,
            purpose=purpose,
            logical_path=logical,
            path=self._scope.root,
            kind=PathKind.DIRECTORY,
            authority_parent=logical,
            authority_parent_path=self._scope.root,
            revalidation=revalidation,
        )

    def revalidate(self, resolved: ResolvedWorkspacePath) -> ResolvedWorkspacePath:
        """Re-check containment and authority immediately before an effect."""

        if not isinstance(resolved, ResolvedWorkspacePath):
            raise TypeError("resolved path must be a ResolvedWorkspacePath")
        if resolved.workspace_id != self._scope.workspace_id:
            raise WorkspaceDriftError(
                reason_code=WorkspaceReason.DRIFT,
                purpose=resolved.purpose,
                workspace_id=self._scope.workspace_id,
            )
        if resolved.logical_path == PurePosixPath("."):
            try:
                current = self.resolve_root(purpose=resolved.purpose)
            except WorkspaceError as error:
                raise WorkspaceDriftError(
                    reason_code=WorkspaceReason.DRIFT,
                    purpose=resolved.purpose,
                    workspace_id=self._scope.workspace_id,
                ) from error
        else:
            try:
                current = self.resolve(
                    resolved.logical_path,
                    purpose=resolved.purpose,
                    allow_missing=resolved.is_missing,
                )
            except WorkspaceError as error:
                raise WorkspaceDriftError(
                    reason_code=WorkspaceReason.DRIFT,
                    purpose=resolved.purpose,
                    workspace_id=self._scope.workspace_id,
                ) from error

        previous = resolved.revalidation
        now = current.revalidation
        if not _same_authority(
            previous.authority_parent_signature,
            now.authority_parent_signature,
        ):
            raise WorkspaceDriftError(
                reason_code=WorkspaceReason.DRIFT,
                purpose=resolved.purpose,
                workspace_id=self._scope.workspace_id,
            )
        if previous.target_must_remain_missing:
            if not current.is_missing:
                raise WorkspaceDriftError(
                    reason_code=WorkspaceReason.DRIFT,
                    purpose=resolved.purpose,
                    workspace_id=self._scope.workspace_id,
                )
        elif previous.target_signature != now.target_signature:
            raise WorkspaceDriftError(
                reason_code=WorkspaceReason.DRIFT,
                purpose=resolved.purpose,
                workspace_id=self._scope.workspace_id,
            )
        return current

    def normalize(
        self,
        logical_path: str | PurePosixPath,
        *,
        purpose: WorkspacePurpose | str,
    ) -> PurePosixPath:
        """Validate one logical path without resolving or reading it."""

        return self._parse_path(logical_path, _coerce_purpose(purpose))

    def walk(
        self,
        logical_path: str | PurePosixPath = _WORKSPACE_ROOT_PATH,
        *,
        max_files: int,
        cancel: Callable[[], bool] | object | None = None,
    ) -> Iterator[WorkspaceWalkEntry]:
        """Yield regular files in deterministic order without following links.

        Directory candidates are retained only up to ``max_files`` per active
        directory. A discarded candidate produces one content-free limit
        omission after the selected prefix has been walked. The method is a
        shared authority boundary for read-only tools; it never returns file
        content and never follows a denied or link entry.
        """

        _validate_positive_int("maximum walked files", max_files)
        if (isinstance(logical_path, PurePosixPath) and logical_path == _WORKSPACE_ROOT_PATH) or (
            isinstance(logical_path, str) and logical_path == "."
        ):
            normalized = _WORKSPACE_ROOT_PATH
        else:
            normalized = self._parse_path(logical_path, WorkspacePurpose.SEARCH)

        try:
            if normalized == PurePosixPath("."):
                resolved = self.resolve_root(purpose=WorkspacePurpose.SEARCH)
            else:
                resolved = self.resolve(normalized, purpose=WorkspacePurpose.SEARCH)
        except WorkspaceError as error:
            yield WorkspaceWalkEntry(
                normalized,
                kind=(
                    PathKind.MISSING
                    if error.reason_code
                    in {WorkspaceReason.TARGET_MISSING, WorkspaceReason.PARENT_MISSING}
                    else PathKind.OTHER
                ),
                omission_reason=str(error.reason_code),
            )
            return

        if resolved.followed_link:
            yield WorkspaceWalkEntry(
                normalized,
                kind=PathKind.OTHER,
                omission_reason=WorkspaceBlindSpotReason.LINK_NOT_FOLLOWED.value,
            )
            return
        if resolved.is_file:
            yield WorkspaceWalkEntry(normalized, resolved.path, PathKind.REGULAR_FILE)
            return
        if not resolved.is_directory:
            yield WorkspaceWalkEntry(
                normalized,
                kind=resolved.kind,
                omission_reason=WorkspaceBlindSpotReason.UNSUPPORTED_KIND.value,
            )
            return

        stack: list[_WorkspaceWalkFrame] = []
        try:
            candidates, omitted = _bounded_walk_candidates(
                resolved.path,
                normalized,
                max_files,
                cancel=cancel,
            )
        except OSError:
            yield WorkspaceWalkEntry(
                normalized,
                kind=PathKind.DIRECTORY,
                omission_reason=WorkspaceBlindSpotReason.FILESYSTEM_ERROR.value,
            )
            return
        stack.append(_WorkspaceWalkFrame(resolved.path, normalized, candidates, omitted))
        visited_files = 0

        while stack:
            _walk_check_cancel(cancel)
            frame = stack[-1]
            if frame.index >= len(frame.candidates):
                stack.pop()
                if frame.omitted:
                    yield WorkspaceWalkEntry(
                        frame.logical_path,
                        kind=PathKind.DIRECTORY,
                        omission_reason="visited_files_limit",
                    )
                continue
            if visited_files >= max_files:
                yield WorkspaceWalkEntry(
                    frame.logical_path,
                    kind=PathKind.DIRECTORY,
                    omission_reason="visited_files_limit",
                )
                return

            candidate = frame.candidates[frame.index]
            frame.index += 1
            child = candidate.child
            logical = _join_logical(frame.logical_path, child.name)
            if self.is_protected(logical):
                yield WorkspaceWalkEntry(
                    logical,
                    kind=PathKind.OTHER,
                    omission_reason=WorkspaceBlindSpotReason.PROTECTED.value,
                )
                continue
            if self.is_secret(logical):
                yield WorkspaceWalkEntry(
                    logical,
                    kind=PathKind.OTHER,
                    omission_reason=WorkspaceBlindSpotReason.SECRET.value,
                )
                continue

            try:
                file_info = child.stat(follow_symlinks=False)
            except (OSError, ValueError):
                yield WorkspaceWalkEntry(
                    logical,
                    kind=PathKind.OTHER,
                    omission_reason=WorkspaceBlindSpotReason.FILESYSTEM_ERROR.value,
                )
                continue

            kind = _entry_kind(file_info)
            if kind is WorkspaceEntryKind.LINK:
                yield WorkspaceWalkEntry(
                    logical,
                    kind=PathKind.OTHER,
                    omission_reason=WorkspaceBlindSpotReason.LINK_NOT_FOLLOWED.value,
                )
                continue
            if kind is WorkspaceEntryKind.DIRECTORY:
                try:
                    child_candidates, child_omitted = _bounded_walk_candidates(
                        Path(child.path),
                        logical,
                        max_files,
                        cancel=cancel,
                    )
                except OSError:
                    yield WorkspaceWalkEntry(
                        logical,
                        kind=PathKind.DIRECTORY,
                        omission_reason=WorkspaceBlindSpotReason.FILESYSTEM_ERROR.value,
                    )
                    continue
                stack.append(
                    _WorkspaceWalkFrame(
                        Path(child.path),
                        logical,
                        child_candidates,
                        child_omitted,
                    )
                )
                continue
            if kind is not WorkspaceEntryKind.REGULAR:
                yield WorkspaceWalkEntry(
                    logical,
                    kind=PathKind.OTHER,
                    omission_reason=WorkspaceBlindSpotReason.UNSUPPORTED_KIND.value,
                )
                continue
            if visited_files >= max_files:
                yield WorkspaceWalkEntry(
                    frame.logical_path,
                    kind=PathKind.DIRECTORY,
                    omission_reason="visited_files_limit",
                )
                return
            visited_files += 1
            try:
                current = self.resolve(logical, purpose=WorkspacePurpose.SEARCH)
                if current.followed_link:
                    raise WorkspacePathError(
                        reason_code=WorkspaceReason.LINK_ESCAPE,
                        purpose=WorkspacePurpose.SEARCH,
                        workspace_id=self.scope.workspace_id,
                    )
                if not current.is_file:
                    raise WorkspacePathError(
                        reason_code=WorkspaceReason.TARGET_KIND,
                        purpose=WorkspacePurpose.SEARCH,
                        workspace_id=self.scope.workspace_id,
                    )
            except WorkspaceError as error:
                reason = (
                    WorkspaceBlindSpotReason.LINK_NOT_FOLLOWED.value
                    if error.reason_code == WorkspaceReason.LINK_ESCAPE
                    else str(error.reason_code)
                )
                yield WorkspaceWalkEntry(logical, kind=PathKind.OTHER, omission_reason=reason)
                continue
            yield WorkspaceWalkEntry(logical, current.path, PathKind.REGULAR_FILE)

    def walk_contained(
        self,
        logical_path: str | PurePosixPath = _WORKSPACE_ROOT_PATH,
        *,
        max_files: int,
        cancel: Callable[[], bool] | object | None = None,
    ) -> Iterator[WorkspaceWalkEntry]:
        """Alias for the shared non-following read/search walker."""

        return self.walk(logical_path, max_files=max_files, cancel=cancel)

    def is_protected(self, logical_path: PurePosixPath) -> bool:
        return _matches_any(logical_path, self._scope.protected_paths)

    def is_secret(self, logical_path: PurePosixPath) -> bool:
        return _matches_any(logical_path, self._scope.secret_paths) or _default_secret_match(
            logical_path
        )

    def _parse_path(
        self, logical_path: str | PurePosixPath, purpose: WorkspacePurpose
    ) -> PurePosixPath:
        if isinstance(logical_path, PurePosixPath):
            raw = str(logical_path)
        elif isinstance(logical_path, str):
            raw = logical_path
        else:
            raise TypeError("workspace logical path must be text or PurePosixPath")

        if not raw:
            raise WorkspacePathError(
                reason_code=WorkspaceReason.PATH_EMPTY,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            )
        if "\x00" in raw:
            raise WorkspacePathError(
                reason_code=WorkspaceReason.PATH_NUL,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            )
        if "\\" in raw:
            reason = (
                WorkspaceReason.PATH_UNC
                if raw.startswith("\\\\")
                else WorkspaceReason.PATH_BACKSLASH
            )
            raise WorkspacePathError(
                reason_code=reason,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            )
        windows_path = re.match(r"^[A-Za-z]:", raw)
        if windows_path:
            raise WorkspacePathError(
                reason_code=WorkspaceReason.PATH_DRIVE,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            )
        if os.name == "nt" and ":" in raw:
            raise WorkspacePathError(
                reason_code=WorkspaceReason.PATH_DRIVE,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            )
        if raw.startswith("//"):
            raise WorkspacePathError(
                reason_code=WorkspaceReason.PATH_UNC,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            )
        if raw.startswith("/"):
            raise WorkspacePathError(
                reason_code=WorkspaceReason.PATH_ABSOLUTE,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            )
        if len(raw) > self._scope.limits.max_logical_path_characters:
            raise WorkspacePathError(
                reason_code=WorkspaceReason.PATH_TOO_LONG,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            )

        raw_parts = raw.split("/")
        if any(part == "" or part == "." for part in raw_parts):
            raise WorkspacePathError(
                reason_code=WorkspaceReason.PATH_AMBIGUOUS,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            )
        if any(part == ".." for part in raw_parts):
            raise WorkspacePathError(
                reason_code=WorkspaceReason.PATH_PARENT,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            )
        if len(raw_parts) > self._scope.limits.max_logical_path_segments:
            raise WorkspacePathError(
                reason_code=WorkspaceReason.TOO_MANY_SEGMENTS,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            )
        return PurePosixPath(*raw_parts)

    def _deny_if_protected_or_secret(
        self, logical_path: PurePosixPath, purpose: WorkspacePurpose
    ) -> None:
        if self.is_protected(logical_path):
            raise WorkspaceAccessError(
                reason_code=WorkspaceReason.PROTECTED,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            )
        if self.is_secret(logical_path):
            raise WorkspaceAccessError(
                reason_code=WorkspaceReason.SECRET,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            )

    def _resolve_components(
        self,
        logical_path: PurePosixPath,
        purpose: WorkspacePurpose,
        missing_allowed: bool,
    ) -> ResolvedWorkspacePath:
        lexical = self._scope.root
        canonical_parent = self._scope.root
        followed_link = False
        parts = logical_path.parts
        for index, part in enumerate(parts):
            candidate = lexical / part
            is_final = index == len(parts) - 1
            try:
                candidate_lstat = os.lstat(candidate)
            except FileNotFoundError:
                if not is_final:
                    raise WorkspacePathError(
                        reason_code=WorkspaceReason.PARENT_MISSING,
                        purpose=purpose,
                        workspace_id=self._scope.workspace_id,
                    ) from None
                if not missing_allowed:
                    raise WorkspacePathError(
                        reason_code=WorkspaceReason.TARGET_MISSING,
                        purpose=purpose,
                        workspace_id=self._scope.workspace_id,
                    ) from None
                try:
                    canonical_parent = canonical_parent.resolve(strict=True)
                    self._ensure_contained(canonical_parent, purpose)
                    parent_logical = self._logical_from_canonical(canonical_parent)
                    self._deny_if_protected_or_secret(parent_logical, purpose)
                    missing_path = canonical_parent / part
                    missing_logical = self._logical_from_canonical(missing_path)
                    self._deny_if_protected_or_secret(missing_logical, purpose)
                    parent_signature = _signature(canonical_parent)
                except WorkspacePathError:
                    raise
                except (OSError, RuntimeError):
                    raise WorkspacePathError(
                        reason_code=WorkspaceReason.CONTAINMENT,
                        purpose=purpose,
                        workspace_id=self._scope.workspace_id,
                    ) from None
                return self._build_result(
                    logical_path,
                    purpose,
                    missing_path,
                    PathKind.MISSING,
                    canonical_parent,
                    parent_signature,
                    None,
                    followed_link,
                )
            except OSError:
                raise WorkspacePathError(
                    reason_code=WorkspaceReason.FILESYSTEM,
                    purpose=purpose,
                    workspace_id=self._scope.workspace_id,
                ) from None

            if _is_link_or_reparse(candidate_lstat):
                followed_link = True
                try:
                    next_canonical = candidate.resolve(strict=True)
                except (OSError, RuntimeError):
                    raise WorkspacePathError(
                        reason_code=WorkspaceReason.LINK_ESCAPE,
                        purpose=purpose,
                        workspace_id=self._scope.workspace_id,
                    ) from None
                try:
                    self._ensure_contained(next_canonical, purpose)
                except WorkspacePathError as error:
                    raise WorkspacePathError(
                        reason_code=WorkspaceReason.LINK_ESCAPE,
                        purpose=purpose,
                        workspace_id=self._scope.workspace_id,
                    ) from error
            else:
                next_canonical = candidate

            if not is_final:
                try:
                    if not next_canonical.is_dir():
                        raise WorkspacePathError(
                            reason_code=WorkspaceReason.TARGET_KIND,
                            purpose=purpose,
                            workspace_id=self._scope.workspace_id,
                        )
                except OSError:
                    raise WorkspacePathError(
                        reason_code=WorkspaceReason.FILESYSTEM,
                        purpose=purpose,
                        workspace_id=self._scope.workspace_id,
                    ) from None
                canonical_parent = next_canonical
                lexical = candidate
                continue

            try:
                canonical_target = next_canonical.resolve(strict=True)
                self._ensure_contained(canonical_target, purpose)
                kind = _path_kind(canonical_target)
                target_signature = _signature(canonical_target)
                parent_path = canonical_target.parent
                parent_signature = _signature(parent_path)
            except WorkspacePathError:
                raise
            except (OSError, RuntimeError):
                raise WorkspacePathError(
                    reason_code=WorkspaceReason.CONTAINMENT,
                    purpose=purpose,
                    workspace_id=self._scope.workspace_id,
                ) from None

            self._deny_if_protected_or_secret(
                self._logical_from_canonical(canonical_target), purpose
            )
            self._validate_kind(kind, purpose, logical_path)
            return self._build_result(
                logical_path,
                purpose,
                canonical_target,
                kind,
                parent_path,
                parent_signature,
                target_signature,
                followed_link,
            )

        raise AssertionError("workspace path resolution must return or raise")

    def _build_result(
        self,
        logical_path: PurePosixPath,
        purpose: WorkspacePurpose,
        canonical_path: Path,
        kind: PathKind,
        parent_path: Path,
        parent_signature: _PathSignature,
        target_signature: _PathSignature | None,
        followed_link: bool,
    ) -> ResolvedWorkspacePath:
        self._ensure_contained(canonical_path, purpose, allow_missing=kind is PathKind.MISSING)
        canonical_logical = self._logical_from_canonical(parent_path)
        revalidation = WorkspaceRevalidation(
            workspace_id=self._scope.workspace_id,
            logical_path=logical_path,
            purpose=purpose,
            authority_parent=canonical_logical,
            authority_parent_signature=parent_signature,
            target_signature=target_signature,
            target_must_remain_missing=kind is PathKind.MISSING,
        )
        return ResolvedWorkspacePath(
            workspace_id=self._scope.workspace_id,
            purpose=purpose,
            logical_path=logical_path,
            path=canonical_path,
            kind=kind,
            authority_parent=canonical_logical,
            authority_parent_path=parent_path,
            revalidation=revalidation,
            followed_link=followed_link,
        )

    def _validate_kind(
        self, kind: PathKind, purpose: WorkspacePurpose, logical_path: PurePosixPath
    ) -> None:
        allowed = {
            WorkspacePurpose.READ: {PathKind.REGULAR_FILE},
            WorkspacePurpose.SEARCH: {PathKind.REGULAR_FILE, PathKind.DIRECTORY},
            WorkspacePurpose.PATCH: {PathKind.REGULAR_FILE},
            WorkspacePurpose.INSTRUCTION: {PathKind.REGULAR_FILE},
            WorkspacePurpose.SKILL: {PathKind.REGULAR_FILE},
            WorkspacePurpose.SNAPSHOT: {PathKind.REGULAR_FILE, PathKind.DIRECTORY},
            WorkspacePurpose.COMMAND_CWD: {PathKind.DIRECTORY},
        }[purpose]
        if kind not in allowed:
            raise WorkspacePathError(
                reason_code=WorkspaceReason.TARGET_KIND,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
                logical_path=logical_path,
                expose_logical_path=False,
            )

    def _ensure_contained(
        self,
        candidate: Path,
        purpose: WorkspacePurpose,
        *,
        allow_missing: bool = False,
    ) -> None:
        try:
            canonical = candidate.resolve(strict=not allow_missing)
            canonical.relative_to(self._scope.root)
        except (OSError, RuntimeError, ValueError):
            raise WorkspacePathError(
                reason_code=WorkspaceReason.CONTAINMENT,
                purpose=purpose,
                workspace_id=self._scope.workspace_id,
            ) from None

    def _logical_from_canonical(self, candidate: Path) -> PurePosixPath:
        try:
            relative = candidate.resolve(strict=False).relative_to(self._scope.root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkspacePathError(
                reason_code=WorkspaceReason.CONTAINMENT,
                purpose=WorkspacePurpose.SNAPSHOT,
                workspace_id=self._scope.workspace_id,
            ) from exc
        return PurePosixPath(relative.as_posix()) if relative.parts else PurePosixPath(".")


@dataclass(slots=True)
class _ObservationState:
    """Mutable state private to one bounded capture."""

    started: float
    entries: list[WorkspaceEntry] = field(default_factory=list)
    blind_spots: list[WorkspaceBlindSpot] = field(default_factory=list)
    observed_bytes: int = 0
    inventory_complete: bool = True
    content_complete: bool = True
    stop_reason: WorkspaceBlindSpotReason | None = None
    target_paths: tuple[PurePosixPath, ...] = ()
    blind_spot_keys: set[tuple[str, str, bool, bool]] = field(default_factory=set)
    explicit_blind_spot_direct_keys: set[tuple[str, str]] = field(default_factory=set)


@dataclass(slots=True)
class _DirectoryCandidate:
    """One retained directory child in the bounded selection heap."""

    sort_key: tuple[str, str]
    child: os.DirEntry[str]

    def __lt__(self, other: _DirectoryCandidate) -> bool:
        # ``heapq`` keeps the largest logical key at the root so the bounded
        # selection can replace it when a smaller child is discovered.
        return self.sort_key > other.sort_key


@dataclass(slots=True)
class _WorkspaceWalkFrame:
    directory: Path
    logical_path: PurePosixPath
    candidates: tuple[_DirectoryCandidate, ...]
    omitted: bool
    index: int = 0


def _bounded_walk_candidates(
    directory: Path,
    logical_directory: PurePosixPath,
    capacity: int,
    *,
    cancel: Callable[[], bool] | object | None = None,
) -> tuple[tuple[_DirectoryCandidate, ...], bool]:
    """Select a deterministic prefix without retaining an unbounded directory."""

    candidates: list[_DirectoryCandidate] = []
    omitted = False
    with os.scandir(directory) as iterator:
        for child in iterator:
            _walk_check_cancel(cancel)
            candidate = _DirectoryCandidate(
                sort_key=(_logical_key(_join_logical(logical_directory, child.name)), child.name),
                child=child,
            )
            if len(candidates) < capacity:
                heapq.heappush(candidates, candidate)
                continue
            omitted = True
            if candidate.sort_key < candidates[0].sort_key:
                heapq.heapreplace(candidates, candidate)
    return tuple(sorted(candidates, key=lambda item: item.sort_key)), omitted


def _walk_check_cancel(cancel: Callable[[], bool] | object | None) -> None:
    if cancel is None:
        return
    check_active = getattr(cancel, "check_active", None)
    if callable(check_active):
        check_active()
        return
    if callable(cancel):
        if bool(cancel()):
            raise RunCancelledError("workspace walk cancelled")
        return
    is_set = getattr(cancel, "is_set", None)
    if callable(is_set) and bool(is_set()):
        raise RunCancelledError("workspace walk cancelled")
    is_cancelled = getattr(cancel, "is_cancelled", None)
    if isinstance(is_cancelled, bool) and is_cancelled:
        raise RunCancelledError("workspace walk cancelled")
    if callable(is_cancelled) and bool(is_cancelled()):
        raise RunCancelledError("workspace walk cancelled")
    raise TypeError("cancel must be a callback or cancellation-like object")


class WorkspaceObserver:
    """Capture bounded task snapshots and compare them without Git.

    The observer uses lexical workspace paths plus ``lstat``/non-following
    directory traversal. A symlink or reparse point is retained as a link
    entry and its target is a deliberate blind spot; it is never traversed.
    """

    def __init__(
        self,
        workspace: Workspace,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(workspace, Workspace):
            raise TypeError("workspace observer requires a Workspace")
        self._workspace = workspace
        self._monotonic = monotonic
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def capture(
        self,
        *,
        target_paths: Iterable[str | PurePosixPath] = (),
        cancel: object | None = None,
        cancellation: object | None = None,
    ) -> WorkspaceSnapshot:
        """Capture one bounded snapshot.

        Cancellation is cooperative and becomes an explicit incomplete
        observation. ``cancel`` may be a callback, an ``Event``-like object,
        or a ``RunContext``-like object exposing ``check_active``.
        """

        if cancel is not None and cancellation is not None:
            raise ValueError("cancel and cancellation cannot both be supplied")
        cancel_check = cancel if cancel is not None else cancellation
        normalized_targets = self._normalize_observation_paths(target_paths)
        state = _ObservationState(
            started=self._monotonic(),
            target_paths=normalized_targets,
        )
        try:
            self._workspace.resolve_root()
        except WorkspaceError:
            self._add_blind_spot(
                state,
                PurePosixPath("."),
                WorkspaceBlindSpotReason.FILESYSTEM_ERROR,
                subtree=True,
            )
            state.inventory_complete = False
            state.content_complete = False
            return self._snapshot(state)

        self._walk_directory(self._workspace.root, PurePosixPath("."), state, cancel_check)
        self._stop_reason(state, cancel_check)
        return self._snapshot(state)

    def observe(
        self,
        *,
        target_paths: Iterable[str | PurePosixPath] = (),
        cancel: object | None = None,
        cancellation: object | None = None,
    ) -> WorkspaceSnapshot:
        """Alias for :meth:`capture` used by observation adapters."""

        return self.capture(
            target_paths=target_paths,
            cancel=cancel,
            cancellation=cancellation,
        )

    def snapshot(
        self,
        *,
        target_paths: Iterable[str | PurePosixPath] = (),
        cancel: object | None = None,
        cancellation: object | None = None,
    ) -> WorkspaceSnapshot:
        """Alias for :meth:`capture`."""

        return self.capture(
            target_paths=target_paths,
            cancel=cancel,
            cancellation=cancellation,
        )

    def capture_baseline(
        self,
        *,
        target_paths: Iterable[str | PurePosixPath] = (),
        cancel: object | None = None,
        cancellation: object | None = None,
    ) -> WorkspaceSnapshot:
        return self.capture(
            target_paths=target_paths,
            cancel=cancel,
            cancellation=cancellation,
        )

    def capture_final(
        self,
        *,
        target_paths: Iterable[str | PurePosixPath] = (),
        cancel: object | None = None,
        cancellation: object | None = None,
    ) -> WorkspaceSnapshot:
        return self.capture(
            target_paths=target_paths,
            cancel=cancel,
            cancellation=cancellation,
        )

    def diff(
        self,
        baseline: WorkspaceSnapshot,
        final: WorkspaceSnapshot,
        *,
        target_paths: Iterable[str | PurePosixPath] = (),
        forbidden_paths: Iterable[str | PurePosixPath] = (),
        require_global: bool = False,
    ) -> WorkspaceDiff:
        """Compare two snapshots using kind, size, and full digest evidence."""

        if not isinstance(baseline, WorkspaceSnapshot) or not isinstance(
            final, WorkspaceSnapshot
        ):
            raise TypeError("diff requires two WorkspaceSnapshot values")
        if baseline.workspace_id != self._workspace.scope.workspace_id or (
            final.workspace_id != self._workspace.scope.workspace_id
        ):
            raise ValueError("snapshot workspace ID does not match observer workspace")
        requested_targets = _as_path_values(target_paths)
        resolved_targets = self._normalize_observation_paths(
            requested_targets
            if requested_targets
            else (*baseline.completeness.target_paths, *final.completeness.target_paths)
        )
        resolved_forbidden = self._normalize_observation_paths(forbidden_paths)
        changes = self._compare_entries(baseline, final)
        blind_spots = _merge_blind_spots(
            (*baseline.completeness.blind_spots, *final.completeness.blind_spots)
        )
        rendered, rendered_complete = self._render_changes(changes)
        rendered_reason = (
            None if rendered_complete else WorkspaceBlindSpotReason.RENDERED_DIFF_LIMIT
        )
        completeness = self._build_diff_completeness(
            baseline,
            final,
            blind_spots,
            resolved_targets,
            resolved_forbidden,
            rendered_complete,
            rendered_reason,
        )
        result = WorkspaceDiff(
            baseline=baseline,
            final=final,
            changes=changes,
            rendered_diff=rendered,
            completeness=completeness,
        )
        if require_global and not result.completeness.global_complete:
            return result
        return result

    def compare(
        self,
        baseline: WorkspaceSnapshot,
        final: WorkspaceSnapshot,
        *,
        target_paths: Iterable[str | PurePosixPath] = (),
        forbidden_paths: Iterable[str | PurePosixPath] = (),
        require_global: bool = False,
    ) -> WorkspaceDiff:
        """Alias for :meth:`diff`."""

        return self.diff(
            baseline,
            final,
            target_paths=target_paths,
            forbidden_paths=forbidden_paths,
            require_global=require_global,
        )

    def _walk_directory(
        self,
        directory: Path,
        logical_directory: PurePosixPath,
        state: _ObservationState,
        cancel: object | None,
    ) -> None:
        if self._stop_reason(state, cancel) is not None:
            self._add_blind_spot(
                state,
                logical_directory,
                cast(WorkspaceBlindSpotReason, state.stop_reason),
                subtree=True,
                inventory_unknown=True,
            )
            return
        available_entries = (
            self._workspace.scope.limits.max_snapshot_entries - len(state.entries)
        )
        if available_entries <= 0:
            self._add_blind_spot(
                state,
                logical_directory,
                WorkspaceBlindSpotReason.ENTRIES_LIMIT,
                subtree=True,
            )
            state.inventory_complete = False
            return

        omitted_due_entry_limit = False
        try:
            candidates: list[_DirectoryCandidate] = []
            with os.scandir(directory) as iterator:
                for child in iterator:
                    reason = self._stop_reason(state, cancel)
                    if reason is not None:
                        self._add_blind_spot(
                            state,
                            logical_directory,
                            reason,
                            subtree=True,
                            inventory_unknown=True,
                        )
                        return
                    try:
                        child.name.encode("utf-8")
                    except UnicodeEncodeError:
                        self._add_blind_spot(
                            state,
                            logical_directory,
                            WorkspaceBlindSpotReason.PATH_UNREPRESENTABLE,
                            subtree=True,
                            inventory_unknown=True,
                        )
                        continue

                    logical_path = _join_logical(logical_directory, child.name)
                    if self._workspace.is_protected(logical_path):
                        if self._add_explicit_blind_spot(
                            state,
                            logical_directory,
                            logical_path,
                            WorkspaceBlindSpotReason.PROTECTED,
                            cancel=cancel,
                            subtree=True,
                        ):
                            return
                        continue
                    if self._workspace.is_secret(logical_path):
                        if self._add_explicit_blind_spot(
                            state,
                            logical_directory,
                            logical_path,
                            WorkspaceBlindSpotReason.SECRET,
                            cancel=cancel,
                            subtree=True,
                        ):
                            return
                        continue
                    exclusion_reason = self._exclusion_reason(logical_path)
                    if exclusion_reason is not None:
                        if self._add_explicit_blind_spot(
                            state,
                            logical_directory,
                            logical_path,
                            exclusion_reason,
                            cancel=cancel,
                            subtree=True,
                        ):
                            return
                        continue

                    candidate = _DirectoryCandidate(
                        sort_key=(_logical_key(PurePosixPath(child.name)), child.name),
                        child=child,
                    )
                    if len(candidates) < available_entries:
                        heapq.heappush(candidates, candidate)
                        continue
                    omitted_due_entry_limit = True
                    if candidate.sort_key < candidates[0].sort_key:
                        heapq.heapreplace(candidates, candidate)

            reason = self._stop_reason(state, cancel)
            if reason is not None:
                self._add_blind_spot(
                    state,
                    logical_directory,
                    reason,
                    subtree=True,
                    inventory_unknown=True,
                )
                return
            ordered_candidates: list[_DirectoryCandidate] = []
            while candidates:
                reason = self._stop_reason(state, cancel)
                if reason is not None:
                    self._add_blind_spot(
                        state,
                        logical_directory,
                        reason,
                        subtree=True,
                        inventory_unknown=True,
                    )
                    return
                ordered_candidates.append(heapq.heappop(candidates))
        except (OSError, ValueError):
            self._add_blind_spot(
                state,
                logical_directory,
                WorkspaceBlindSpotReason.FILESYSTEM_ERROR,
                subtree=True,
            )
            state.inventory_complete = False
            state.content_complete = False
            return

        if omitted_due_entry_limit:
            state.inventory_complete = False
            self._add_blind_spot(
                state,
                logical_directory,
                WorkspaceBlindSpotReason.ENTRIES_LIMIT,
                subtree=True,
            )

        for candidate in reversed(ordered_candidates):
            child = candidate.child
            logical_path = _join_logical(logical_directory, child.name)
            reason = self._stop_reason(state, cancel)
            if reason is not None:
                self._add_blind_spot(
                    state,
                    logical_directory,
                    reason,
                    subtree=True,
                    inventory_unknown=True,
                )
                return
            if len(state.entries) >= self._workspace.scope.limits.max_snapshot_entries:
                self._add_entry_limit_blind_spot(state, logical_directory, cancel)
                return

            if self._workspace.is_protected(logical_path):
                if self._add_explicit_blind_spot(
                    state,
                    logical_directory,
                    logical_path,
                    WorkspaceBlindSpotReason.PROTECTED,
                    cancel=cancel,
                    subtree=True,
                ):
                    return
                continue
            if self._workspace.is_secret(logical_path):
                if self._add_explicit_blind_spot(
                    state,
                    logical_directory,
                    logical_path,
                    WorkspaceBlindSpotReason.SECRET,
                    cancel=cancel,
                    subtree=True,
                ):
                    return
                continue
            exclusion_reason = self._exclusion_reason(logical_path)
            if exclusion_reason is not None:
                if self._add_explicit_blind_spot(
                    state,
                    logical_directory,
                    logical_path,
                    exclusion_reason,
                    cancel=cancel,
                    subtree=True,
                ):
                    return
                continue

            try:
                file_info = child.stat(follow_symlinks=False)
            except (OSError, ValueError):
                self._add_blind_spot(
                    state,
                    logical_path,
                    WorkspaceBlindSpotReason.FILESYSTEM_ERROR,
                    subtree=True,
                    inventory_unknown=True,
                )
                continue
            kind = _entry_kind(file_info)
            if kind is WorkspaceEntryKind.DIRECTORY:
                state.entries.append(
                    WorkspaceEntry(
                        logical_path=logical_path,
                        kind=kind,
                        size=int(file_info.st_size),
                    )
                )
                self._walk_directory(Path(child.path), logical_path, state, cancel)
                continue
            if kind is WorkspaceEntryKind.LINK:
                state.entries.append(
                    WorkspaceEntry(
                        logical_path=logical_path,
                        kind=kind,
                        size=max(0, int(file_info.st_size)),
                        content_complete=False,
                        omission_reason=WorkspaceBlindSpotReason.LINK_NOT_FOLLOWED,
                    )
                )
                self._add_blind_spot(
                    state,
                    logical_path,
                    WorkspaceBlindSpotReason.LINK_NOT_FOLLOWED,
                    subtree=True,
                    inventory_unknown=False,
                )
                continue
            if kind is not WorkspaceEntryKind.REGULAR:
                state.entries.append(
                    WorkspaceEntry(
                        logical_path=logical_path,
                        kind=kind,
                        size=max(0, int(file_info.st_size)),
                        content_complete=False,
                        omission_reason=WorkspaceBlindSpotReason.UNSUPPORTED_KIND,
                    )
                )
                self._add_blind_spot(
                    state,
                    logical_path,
                    WorkspaceBlindSpotReason.UNSUPPORTED_KIND,
                    inventory_unknown=False,
                )
                continue
            state.entries.append(
                self._observe_regular_file(child, logical_path, file_info, state, cancel)
            )

    def _observe_regular_file(
        self,
        child: os.DirEntry[str],
        logical_path: PurePosixPath,
        file_info: os.stat_result,
        state: _ObservationState,
        cancel: object | None,
    ) -> WorkspaceEntry:
        size = max(0, int(file_info.st_size))
        if size > self._workspace.scope.limits.max_snapshot_file_bytes:
            self._add_blind_spot(
                state,
                logical_path,
                WorkspaceBlindSpotReason.PER_FILE_BYTES_LIMIT,
                inventory_unknown=False,
            )
            return WorkspaceEntry(
                logical_path=logical_path,
                kind=WorkspaceEntryKind.REGULAR,
                size=size,
                content_complete=False,
                omission_reason=WorkspaceBlindSpotReason.PER_FILE_BYTES_LIMIT,
            )
        remaining = self._workspace.scope.limits.max_snapshot_bytes - state.observed_bytes
        if size > remaining:
            self._add_blind_spot(
                state,
                logical_path,
                WorkspaceBlindSpotReason.AGGREGATE_BYTES_LIMIT,
                inventory_unknown=False,
            )
            return WorkspaceEntry(
                logical_path=logical_path,
                kind=WorkspaceEntryKind.REGULAR,
                size=size,
                content_complete=False,
                omission_reason=WorkspaceBlindSpotReason.AGGREGATE_BYTES_LIMIT,
            )
        try:
            raw, unchanged, interrupted = _read_regular_file(
                Path(child.path),
                file_info,
                max_bytes=remaining,
                should_stop=lambda: self._stop_reason(state, cancel) is not None,
            )
        except (OSError, ValueError):
            self._add_blind_spot(
                state,
                logical_path,
                WorkspaceBlindSpotReason.FILESYSTEM_ERROR,
                inventory_unknown=True,
            )
            return WorkspaceEntry(
                logical_path=logical_path,
                kind=WorkspaceEntryKind.REGULAR,
                size=size,
                content_complete=False,
                omission_reason=WorkspaceBlindSpotReason.FILESYSTEM_ERROR,
            )
        state.observed_bytes += len(raw)
        if interrupted:
            reason = state.stop_reason or WorkspaceBlindSpotReason.CANCELLED
            self._add_blind_spot(
                state,
                logical_path,
                reason,
                inventory_unknown=False,
            )
            return WorkspaceEntry(
                logical_path=logical_path,
                kind=WorkspaceEntryKind.REGULAR,
                size=size,
                content_complete=False,
                omission_reason=reason,
            )
        post_read_stop_reason = self._stop_reason(state, cancel)
        if post_read_stop_reason is not None:
            self._add_blind_spot(
                state,
                logical_path,
                post_read_stop_reason,
                inventory_unknown=False,
            )
            return WorkspaceEntry(
                logical_path=logical_path,
                kind=WorkspaceEntryKind.REGULAR,
                size=size,
                content_complete=False,
                omission_reason=post_read_stop_reason,
            )
        if not unchanged or len(raw) != size:
            self._add_blind_spot(
                state,
                logical_path,
                WorkspaceBlindSpotReason.CONTENT_CHANGED_DURING_READ,
                inventory_unknown=False,
            )
            return WorkspaceEntry(
                logical_path=logical_path,
                kind=WorkspaceEntryKind.REGULAR,
                size=size,
                digest=None,
                content_complete=False,
                omission_reason=WorkspaceBlindSpotReason.CONTENT_CHANGED_DURING_READ,
            )
        digest = hashlib.sha256(raw).hexdigest()
        text, binary = _project_utf8(raw)
        return WorkspaceEntry(
            logical_path=logical_path,
            kind=WorkspaceEntryKind.REGULAR,
            size=size,
            digest=digest,
            text=text,
            binary=binary,
        )

    def _snapshot(self, state: _ObservationState) -> WorkspaceSnapshot:
        entries = tuple(sorted(state.entries, key=lambda item: _logical_key(item.logical_path)))
        if state.stop_reason is not None and not any(
            item.reason_code == state.stop_reason.value for item in state.blind_spots
        ):
            state.blind_spots.append(
                WorkspaceBlindSpot(PurePosixPath("."), state.stop_reason, subtree=True)
            )
        blind_spots = _merge_blind_spots(state.blind_spots)
        global_complete = state.inventory_complete and state.content_complete and not blind_spots
        completeness = WorkspaceCompleteness(
            inventory_complete=state.inventory_complete and not any(
                item.reason_code
                in {
                    WorkspaceBlindSpotReason.PROTECTED,
                    WorkspaceBlindSpotReason.SECRET,
                    WorkspaceBlindSpotReason.VOLATILE_EXCLUSION,
                    WorkspaceBlindSpotReason.IGNORED,
                    WorkspaceBlindSpotReason.ENTRIES_LIMIT,
                    WorkspaceBlindSpotReason.ELAPSED_LIMIT,
                    WorkspaceBlindSpotReason.CANCELLED,
                    WorkspaceBlindSpotReason.FILESYSTEM_ERROR,
                }
                for item in blind_spots
            ),
            content_complete=state.content_complete and not any(
                item.reason_code
                not in {WorkspaceBlindSpotReason.RENDERED_DIFF_LIMIT}
                for item in blind_spots
            ),
            blind_spots=blind_spots,
            target_paths=state.target_paths,
            target_complete=(
                global_complete
                if not state.target_paths
                else self._scope_is_clear(blind_spots, state.target_paths)
            ),
            global_complete=global_complete,
            observed_entries=len(entries),
            observed_bytes=state.observed_bytes,
        )
        return WorkspaceSnapshot(
            workspace_id=self._workspace.scope.workspace_id,
            captured_at=self._clock(),
            entries=entries,
            completeness=completeness,
        )

    def _compare_entries(
        self, baseline: WorkspaceSnapshot, final: WorkspaceSnapshot
    ) -> tuple[WorkspaceChange, ...]:
        before = {item.logical_path: item for item in baseline.entries}
        after = {item.logical_path: item for item in final.entries}
        paths = sorted(set(before) | set(after), key=_logical_key)
        changes: list[WorkspaceChange] = []
        for path in paths:
            left = before.get(path)
            right = after.get(path)
            if left is None:
                final_clear = _entry_existence_clear(final.completeness.blind_spots, path)
                baseline_clear = _entry_existence_clear(
                    baseline.completeness.blind_spots, path
                )
                if final_clear and baseline_clear:
                    changes.append(WorkspaceChange(path, WorkspaceChangeKind.CREATE, None, right))
                continue
            if right is None:
                baseline_clear = _entry_existence_clear(
                    baseline.completeness.blind_spots, path
                )
                final_clear = _entry_existence_clear(final.completeness.blind_spots, path)
                if baseline_clear and final_clear:
                    changes.append(WorkspaceChange(path, WorkspaceChangeKind.DELETE, left, None))
                continue
            if left.kind is not right.kind:
                changes.append(
                    WorkspaceChange(path, WorkspaceChangeKind.TYPE_CHANGE, left, right)
                )
                continue
            if left.size != right.size:
                changes.append(WorkspaceChange(path, WorkspaceChangeKind.MODIFY, left, right))
                continue
            if left.has_full_digest and right.has_full_digest:
                if left.digest != right.digest:
                    changes.append(WorkspaceChange(path, WorkspaceChangeKind.MODIFY, left, right))
                continue
            if left.kind is WorkspaceEntryKind.DIRECTORY:
                continue
            # Equal metadata without a full digest is unknown, not unchanged
            # and not a certain modification. Completeness carries this gap.
        return tuple(changes)

    def _render_changes(
        self, changes: Sequence[WorkspaceChange]
    ) -> tuple[str, bool]:
        limit = self._workspace.scope.limits.max_rendered_diff_characters
        parts: list[str] = []
        length = 0
        for change in changes:
            prefix = "\n" if parts else ""
            available = limit - length - len(prefix)
            if available <= 0:
                return "".join(parts), False
            fragment, fragment_complete = self._render_bounded_change(change, available)
            if not fragment:
                continue
            candidate = prefix + fragment
            parts.append(candidate)
            length += len(candidate)
            if not fragment_complete:
                return "".join(parts), False
        return "".join(parts), True

    def _render_bounded_change(
        self, change: WorkspaceChange, max_characters: int
    ) -> tuple[str, bool]:
        before = change.before
        after = change.after
        if (
            change.kind is WorkspaceChangeKind.MODIFY
            and before is not None
            and after is not None
            and before.text is not None
            and after.text is not None
        ):
            return _bounded_text_diff(
                before.text,
                after.text,
                change.logical_path.as_posix(),
                max_characters,
            )
        if change.kind is WorkspaceChangeKind.CREATE and after is not None and after.text:
            return _bounded_file_projection(
                f"--- /dev/null\n+++ b/{change.logical_path.as_posix()}",
                after.text,
                "+",
                max_characters,
            )
        if change.kind is WorkspaceChangeKind.DELETE and before is not None and before.text:
            return _bounded_file_projection(
                f"--- a/{change.logical_path.as_posix()}\n+++ /dev/null",
                before.text,
                "-",
                max_characters,
            )
        fragment = self._render_change(change)
        if len(fragment) <= max_characters:
            return fragment, True
        return _truncate_text(fragment, max_characters, _TRUNCATION_MARKER), False

    def _render_change(self, change: WorkspaceChange) -> str:
        before = change.before
        after = change.after
        path = change.logical_path.as_posix()
        if not change.comparison_complete:
            return f"[observation incomplete] {change.kind.value} {path}"
        if (
            change.kind is WorkspaceChangeKind.MODIFY
            and before is not None
            and after is not None
            and before.text is not None
            and after.text is not None
        ):
            lines = difflib.unified_diff(
                before.text.split("\n"),
                after.text.split("\n"),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
            output = "\n".join(lines)
            if output:
                return output
            return f"--- a/{path}\n+++ b/{path}\n[metadata differs; text projection is equal]"
        if change.kind is WorkspaceChangeKind.CREATE:
            return self._render_created(path, after)
        if change.kind is WorkspaceChangeKind.DELETE:
            return self._render_deleted(path, before)
        if change.kind is WorkspaceChangeKind.TYPE_CHANGE:
            return (
                f"[type change] {path}: "
                f"{before.kind.value if before is not None else 'missing'} -> "
                f"{after.kind.value if after is not None else 'missing'}"
            )
        return self._render_metadata_change(path, before, after)

    def _render_created(self, path: str, entry: WorkspaceEntry | None) -> str:
        if entry is not None and entry.text is not None:
            lines = entry.text.split("\n")
            return "\n".join(
                ["--- /dev/null", f"+++ b/{path}", *[f"+{line}" for line in lines]]
            )
        return self._render_metadata_change(path, None, entry)

    def _render_deleted(self, path: str, entry: WorkspaceEntry | None) -> str:
        if entry is not None and entry.text is not None:
            lines = entry.text.split("\n")
            return "\n".join(
                [f"--- a/{path}", "+++ /dev/null", *[f"-{line}" for line in lines]]
            )
        return self._render_metadata_change(path, entry, None)

    @staticmethod
    def _render_metadata_change(
        path: str, before: WorkspaceEntry | None, after: WorkspaceEntry | None
    ) -> str:
        def describe(entry: WorkspaceEntry | None) -> str:
            if entry is None:
                return "missing"
            digest = "digest" if entry.has_full_digest else "digest-omitted"
            projection = "binary" if entry.binary else "metadata"
            return f"{entry.kind.value}, size={entry.size}, {digest}, {projection}"

        return f"[workspace change] {path}: {describe(before)} -> {describe(after)}"

    def _build_diff_completeness(
        self,
        baseline: WorkspaceSnapshot,
        final: WorkspaceSnapshot,
        blind_spots: tuple[WorkspaceBlindSpot, ...],
        target_paths: tuple[PurePosixPath, ...],
        forbidden_paths: tuple[PurePosixPath, ...],
        rendered_complete: bool,
        rendered_reason: WorkspaceBlindSpotReason | None,
    ) -> WorkspaceCompleteness:
        inventory_complete = (
            baseline.completeness.inventory_complete and final.completeness.inventory_complete
        )
        content_complete = (
            baseline.completeness.content_complete and final.completeness.content_complete
        )
        global_complete = (
            baseline.completeness.global_complete
            and final.completeness.global_complete
            and not blind_spots
        )
        target_complete = (
            global_complete
            if not target_paths
            else self._scope_is_clear(blind_spots, target_paths)
        )
        forbidden_complete = (
            True
            if not forbidden_paths
            else self._scope_is_clear(blind_spots, forbidden_paths)
        )
        return WorkspaceCompleteness(
            inventory_complete=inventory_complete,
            content_complete=content_complete,
            blind_spots=blind_spots,
            target_paths=target_paths,
            target_complete=target_complete,
            forbidden_paths=forbidden_paths,
            forbidden_complete=forbidden_complete,
            global_complete=global_complete,
            rendered_diff_complete=rendered_complete,
            rendered_diff_omission_reason=rendered_reason,
            observed_entries=min(
                baseline.completeness.observed_entries, final.completeness.observed_entries
            ),
            observed_bytes=min(
                baseline.completeness.observed_bytes, final.completeness.observed_bytes
            ),
        )

    def _exclusion_reason(
        self, logical_path: PurePosixPath
    ) -> WorkspaceBlindSpotReason | None:
        if _matches_any(logical_path, self._workspace.scope.volatile_paths):
            return WorkspaceBlindSpotReason.VOLATILE_EXCLUSION
        if _matches_any(logical_path, self._workspace.scope.ignored_paths):
            return WorkspaceBlindSpotReason.IGNORED
        return None

    def _scope_is_clear(
        self,
        blind_spots: Iterable[WorkspaceBlindSpot],
        paths: Iterable[PurePosixPath],
    ) -> bool:
        return _scope_is_clear(
            blind_spots,
            paths,
            summary_matcher=self._summary_matches_path,
        )

    def _summary_matches_path(self, reason_code: str, path: PurePosixPath) -> bool:
        if reason_code == WorkspaceBlindSpotReason.PROTECTED.value:
            return self._workspace.is_protected(path)
        if reason_code == WorkspaceBlindSpotReason.SECRET.value:
            return self._workspace.is_secret(path)
        if reason_code == WorkspaceBlindSpotReason.VOLATILE_EXCLUSION.value:
            return _matches_any(path, self._workspace.scope.volatile_paths)
        if reason_code == WorkspaceBlindSpotReason.IGNORED.value:
            return _matches_any(path, self._workspace.scope.ignored_paths)
        return False

    def _normalize_observation_paths(
        self, paths: Iterable[str | PurePosixPath]
    ) -> tuple[PurePosixPath, ...]:
        paths = _as_path_values(paths)
        normalized: dict[str, PurePosixPath] = {}
        for value in paths:
            if isinstance(value, PurePosixPath) and value == PurePosixPath("."):
                path = value
            elif isinstance(value, str) and value == ".":
                path = PurePosixPath(".")
            else:
                path = self._workspace._parse_path(value, WorkspacePurpose.SNAPSHOT)
            normalized[_logical_key(path)] = path
        return tuple(normalized[key] for key in sorted(normalized))

    def _stop_reason(
        self, state: _ObservationState, cancel: object | None
    ) -> WorkspaceBlindSpotReason | None:
        if state.stop_reason is not None:
            return state.stop_reason
        if self._monotonic() - state.started >= (
            self._workspace.scope.limits.max_snapshot_elapsed_seconds
        ):
            state.stop_reason = WorkspaceBlindSpotReason.ELAPSED_LIMIT
            state.inventory_complete = False
            state.content_complete = False
            return state.stop_reason
        if _cancel_requested(cancel):
            state.stop_reason = WorkspaceBlindSpotReason.CANCELLED
            state.inventory_complete = False
            state.content_complete = False
            return state.stop_reason
        return None

    def _add_entry_limit_blind_spot(
        self,
        state: _ObservationState,
        logical_directory: PurePosixPath,
        cancel: object | None,
    ) -> None:
        reason = self._stop_reason(state, cancel)
        if reason is not None:
            self._add_blind_spot(
                state,
                logical_directory,
                reason,
                subtree=True,
                inventory_unknown=True,
            )
            return
        state.inventory_complete = False
        self._add_blind_spot(
            state,
            logical_directory,
            WorkspaceBlindSpotReason.ENTRIES_LIMIT,
            subtree=True,
        )

    def _add_explicit_blind_spot(
        self,
        state: _ObservationState,
        logical_directory: PurePosixPath,
        logical_path: PurePosixPath,
        reason: WorkspaceBlindSpotReason,
        *,
        cancel: object | None,
        subtree: bool,
    ) -> bool:
        reason_code = reason.value
        stop_reason = self._stop_reason(state, cancel)
        if stop_reason is not None:
            self._add_blind_spot(
                state,
                logical_directory,
                stop_reason,
                subtree=True,
                inventory_unknown=True,
            )
            return True

        directory_key = _logical_key(logical_directory)
        aggregate_key = (directory_key, reason_code, False, True)
        if aggregate_key in state.blind_spot_keys:
            state.inventory_complete = False
            state.content_complete = False
        elif (directory_key, reason_code) in state.explicit_blind_spot_direct_keys:
            self._add_blind_spot(
                state,
                logical_directory,
                reason,
                aggregate=True,
                inventory_unknown=True,
            )
        else:
            self._add_blind_spot(
                state,
                logical_path,
                reason,
                subtree=subtree,
                inventory_unknown=True,
            )
            state.explicit_blind_spot_direct_keys.add((directory_key, reason_code))

        stop_reason = self._stop_reason(state, cancel)
        if stop_reason is not None:
            self._add_blind_spot(
                state,
                logical_directory,
                stop_reason,
                subtree=True,
                inventory_unknown=True,
            )
            return True
        return False

    @staticmethod
    def _add_blind_spot(
        state: _ObservationState,
        logical_path: PurePosixPath,
        reason: WorkspaceBlindSpotReason,
        *,
        subtree: bool = False,
        aggregate: bool = False,
        inventory_unknown: bool = False,
    ) -> None:
        key = (_logical_key(logical_path), reason.value, subtree, aggregate)
        if key not in state.blind_spot_keys:
            state.blind_spot_keys.add(key)
            state.blind_spots.append(
                WorkspaceBlindSpot(logical_path, reason, subtree, aggregate)
            )
        if inventory_unknown:
            state.inventory_complete = False
        state.content_complete = False


def _purpose_value(purpose: WorkspacePurpose | str) -> str:
    return purpose.value if isinstance(purpose, WorkspacePurpose) else str(purpose)


def _join_logical(directory: PurePosixPath, name: str) -> PurePosixPath:
    """Join a scandir name without resolving or normalizing filesystem links."""

    return PurePosixPath(name) if directory == PurePosixPath(".") else directory / name


def _as_path_values(
    paths: Iterable[str | PurePosixPath] | str | PurePosixPath,
) -> tuple[str | PurePosixPath, ...]:
    if isinstance(paths, (str, PurePosixPath)):
        return (paths,)
    return tuple(paths)


def _entry_kind(file_info: os.stat_result) -> WorkspaceEntryKind:
    if _is_link_or_reparse(file_info):
        return WorkspaceEntryKind.LINK
    if stat.S_ISREG(file_info.st_mode):
        return WorkspaceEntryKind.REGULAR
    if stat.S_ISDIR(file_info.st_mode):
        return WorkspaceEntryKind.DIRECTORY
    return WorkspaceEntryKind.OTHER


def _read_regular_file(
    path: Path,
    before: os.stat_result,
    *,
    max_bytes: int,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[bytes, bool, bool]:
    """Read in bounded chunks and detect size drift or cooperative stop."""

    expected_size = max(0, int(before.st_size))
    if max_bytes < 0:
        raise ValueError("maximum file read bytes must be non-negative")
    chunks: list[bytes] = []
    remaining = min(expected_size, max_bytes)
    with path.open("rb") as handle:
        while remaining:
            if should_stop is not None and should_stop():
                return b"".join(chunks), False, True
            chunk = handle.read(min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(handle.fileno())
    # Windows may report zero device/inode values for directory enumeration,
    # so identity fields cannot be used as a portable content proof here.
    unchanged = (
        stat.S_IFMT(after.st_mode) == stat.S_IFMT(before.st_mode)
        and int(after.st_size) == expected_size
        and len(raw) == expected_size
    )
    return raw, unchanged, False


def _project_utf8(raw: bytes) -> tuple[str | None, bool]:
    """Return normalized UTF-8 text, or safe binary metadata mode."""

    if b"\x00" in raw:
        return None, True
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, True
    return text.replace("\r\n", "\n").replace("\r", "\n"), False


def _bounded_file_projection(
    header: str,
    text: str,
    marker: str,
    max_characters: int,
) -> tuple[str, bool]:
    if len(header) > max_characters:
        return _truncate_text(header, max_characters, _TRUNCATION_MARKER), False
    lines = [header]
    used = len(header)
    for line in text.split("\n"):
        candidate = f"\n{marker}{line}"
        if used + len(candidate) <= max_characters:
            lines.append(candidate)
            used += len(candidate)
            continue
        remaining = max_characters - used
        lines.append(_truncate_text(candidate, remaining, _TRUNCATION_MARKER))
        return "".join(lines), False
    return "".join(lines), True


def _bounded_text_diff(
    before: str,
    after: str,
    logical_path: str,
    max_characters: int,
) -> tuple[str, bool]:
    if before == after:
        metadata = f"--- a/{logical_path}\n+++ b/{logical_path}\n"
        metadata += "[metadata differs; text projection is equal]"
        if len(metadata) <= max_characters:
            return metadata, True
        return _truncate_text(metadata, max_characters, _TRUNCATION_MARKER), False

    rendered = ""
    for line in difflib.unified_diff(
        before.split("\n"),
        after.split("\n"),
        fromfile=f"a/{logical_path}",
        tofile=f"b/{logical_path}",
        lineterm="",
    ):
        candidate = ("\n" if rendered else "") + line
        if len(rendered) + len(candidate) <= max_characters:
            rendered += candidate
            continue
        remaining = max_characters - len(rendered)
        return rendered + _truncate_text(candidate, remaining, _TRUNCATION_MARKER), False
    return rendered, True


def _cancel_requested(cancel: object | None) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        try:
            return bool(cancel())
        except (RunCancelledError, RunDeadlineExceededError):
            return True
    check_active = getattr(cancel, "check_active", None)
    if callable(check_active):
        try:
            check_active()
        except (RunCancelledError, RunDeadlineExceededError):
            return True
        return False
    is_cancelled = getattr(cancel, "is_cancelled", None)
    if isinstance(is_cancelled, bool):
        return is_cancelled
    if callable(is_cancelled):
        return bool(is_cancelled())
    is_set = getattr(cancel, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    raise TypeError("cancel must be a callback or cancellation-like object")


def _merge_blind_spots(
    blind_spots: Iterable[WorkspaceBlindSpot],
) -> tuple[WorkspaceBlindSpot, ...]:
    merged: dict[tuple[str, str, bool], WorkspaceBlindSpot] = {}
    for item in blind_spots:
        key = (_logical_key(item.logical_path), item.reason_code, item.aggregate)
        previous = merged.get(key)
        if previous is None or (item.subtree and not previous.subtree):
            merged[key] = item
    return tuple(
        sorted(
            merged.values(),
            key=lambda item: (
                _logical_key(item.logical_path),
                item.reason_code,
                item.aggregate,
            ),
        )
    )


def _path_is_within(path: PurePosixPath, parent: PurePosixPath) -> bool:
    path_parts = tuple(
        part.casefold() for part in path.parts
    ) if os.name == "nt" else path.parts
    parent_parts = tuple(
        part.casefold() for part in parent.parts
    ) if os.name == "nt" else parent.parts
    return len(path_parts) >= len(parent_parts) and path_parts[: len(parent_parts)] == parent_parts


def _paths_intersect(
    requested: PurePosixPath, blind_spot: WorkspaceBlindSpot
) -> bool:
    if requested == blind_spot.logical_path:
        return True
    if blind_spot.subtree and _path_is_within(requested, blind_spot.logical_path):
        return True
    return _path_is_within(blind_spot.logical_path, requested)


def _path_clear(
    blind_spots: Iterable[WorkspaceBlindSpot], path: PurePosixPath
) -> bool:
    return not any(_paths_intersect(path, item) for item in blind_spots)


def _entry_existence_clear(
    blind_spots: Iterable[WorkspaceBlindSpot], path: PurePosixPath
) -> bool:
    """Allow an observed entry diff unless its own existence was omitted."""

    inventory_unknown_reasons = {
        WorkspaceBlindSpotReason.PROTECTED.value,
        WorkspaceBlindSpotReason.SECRET.value,
        WorkspaceBlindSpotReason.VOLATILE_EXCLUSION.value,
        WorkspaceBlindSpotReason.IGNORED.value,
        WorkspaceBlindSpotReason.ENTRIES_LIMIT.value,
        WorkspaceBlindSpotReason.ELAPSED_LIMIT.value,
        WorkspaceBlindSpotReason.CANCELLED.value,
        WorkspaceBlindSpotReason.FILESYSTEM_ERROR.value,
        WorkspaceBlindSpotReason.PATH_UNREPRESENTABLE.value,
    }
    return not any(
        (
            item.reason_code in inventory_unknown_reasons
            and (
                item.logical_path == path
                or (item.subtree and _path_is_within(path, item.logical_path))
            )
        )
        or (
            item.reason_code not in inventory_unknown_reasons
            and item.subtree
            and item.logical_path != path
            and _path_is_within(path, item.logical_path)
        )
        for item in blind_spots
    )


def _scope_is_clear(
    blind_spots: Iterable[WorkspaceBlindSpot],
    paths: Iterable[PurePosixPath],
    *,
    summary_matcher: Callable[[str, PurePosixPath], bool] | None = None,
) -> bool:
    requested = tuple(paths)
    if not requested:
        return False
    relevant = tuple(
        item
        for item in blind_spots
        if any(
            (
                _paths_intersect(path, item)
                if not item.aggregate
                else (
                    path == item.logical_path
                    or _path_is_within(item.logical_path, path)
                    or (
                        summary_matcher is not None
                        and summary_matcher(item.reason_code, path)
                    )
                )
            )
            for path in requested
        )
    )
    return not relevant


def _coerce_purpose(purpose: WorkspacePurpose | str) -> WorkspacePurpose:
    try:
        return purpose if isinstance(purpose, WorkspacePurpose) else WorkspacePurpose(purpose)
    except ValueError:
        raise WorkspacePathError(reason_code=WorkspaceReason.INVALID_RULE) from None


def _validate_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_non_negative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive_float(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number greater than zero")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")


def _normalize_rules(
    values: Iterable[PurePosixPath], label: str
) -> tuple[PurePosixPath, ...]:
    try:
        iterator = iter(values)
    except TypeError as error:
        raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_RULE) from error
    normalized: list[PurePosixPath] = []
    total_characters = 0
    for index in range(_MAX_RULE_ITEMS + 1):
        try:
            value = next(iterator)
        except StopIteration:
            break
        except Exception as error:
            raise WorkspaceConfigurationError(
                reason_code=WorkspaceReason.INVALID_RULE
            ) from error
        if index >= _MAX_RULE_ITEMS:
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_RULE)
        if not isinstance(value, PurePosixPath):
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_RULE)
        raw = str(value)
        if not raw or raw == "." or raw.startswith("/") or "\\" in raw or "\x00" in raw:
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_RULE)
        if any(part in {"", ".", ".."} for part in raw.split("/")):
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_RULE)
        if len(raw) > 4_096:
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_RULE)
        total_characters += len(raw)
        if total_characters > _MAX_RULE_CHARACTERS:
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_RULE)
        normalized.append(PurePosixPath(*raw.split("/")))
    return tuple(normalized)


def _unique_rules(values: Iterable[PurePosixPath]) -> tuple[PurePosixPath, ...]:
    unique: dict[str, PurePosixPath] = {}
    for value in values:
        unique[_logical_key(value)] = value
    return tuple(unique[key] for key in sorted(unique))


def _logical_key(path: PurePosixPath) -> str:
    rendered = path.as_posix()
    return rendered.casefold() if os.name == "nt" else rendered


def _matches_any(path: PurePosixPath, rules: Iterable[PurePosixPath]) -> bool:
    path_parts = tuple(part.casefold() for part in path.parts) if os.name == "nt" else path.parts
    for rule in rules:
        rule_parts = (
            tuple(part.casefold() for part in rule.parts) if os.name == "nt" else rule.parts
        )
        if len(path_parts) >= len(rule_parts) and path_parts[: len(rule_parts)] == rule_parts:
            return True
    return False


def _default_secret_match(path: PurePosixPath) -> bool:
    basename = path.name.casefold() if os.name == "nt" else path.name
    if basename == ".env" or basename.startswith(".env."):
        return True
    if basename in DEFAULT_SECRET_BASENAMES:
        return True
    return any(basename.endswith(suffix) for suffix in _DEFAULT_SECRET_SUFFIXES)


def _is_link_or_reparse(file_info: os.stat_result) -> bool:
    if stat.S_ISLNK(file_info.st_mode):
        return True
    attributes = getattr(file_info, "st_file_attributes", 0)
    return isinstance(attributes, int) and bool(attributes & _WINDOWS_REPARSE_POINT)


def _path_kind(path: Path) -> PathKind:
    file_info = os.stat(path)
    if stat.S_ISREG(file_info.st_mode):
        return PathKind.REGULAR_FILE
    if stat.S_ISDIR(file_info.st_mode):
        return PathKind.DIRECTORY
    return PathKind.OTHER


def _signature(path: Path) -> _PathSignature:
    file_info = os.stat(path)
    return _PathSignature(
        device=int(file_info.st_dev),
        inode=int(file_info.st_ino),
        mode=stat.S_IFMT(file_info.st_mode),
        size=int(file_info.st_size),
        modified_ns=int(file_info.st_mtime_ns),
    )


def _same_authority(left: _PathSignature, right: _PathSignature) -> bool:
    """Compare the identity that authorizes a parent path, not its contents."""

    return (
        left.device,
        left.inode,
        left.mode,
    ) == (
        right.device,
        right.inode,
        right.mode,
    )


def _normalize_literals(values: Iterable[str], label: str) -> tuple[str, ...]:
    try:
        iterator = iter(values)
    except TypeError as error:
        raise TypeError(f"{label} must be an iterable of strings") from error
    normalized: list[str] = []
    total_characters = 0
    for index in range(_MAX_SANITIZER_ITEMS + 1):
        try:
            value = next(iterator)
        except StopIteration:
            break
        except Exception as error:
            raise TypeError(f"{label} must be an iterable of strings") from error
        if index >= _MAX_SANITIZER_ITEMS:
            raise ValueError(f"{label} exceeds its item bound")
        if not isinstance(value, str):
            raise TypeError(f"{label} must contain text")
        if value:
            total_characters += len(value)
            if total_characters > _MAX_SANITIZER_CHARACTERS:
                raise ValueError(f"{label} exceeds its character bound")
            normalized.append(value)
    return tuple(sorted(set(normalized), key=lambda item: (-len(item), item)))


def _normalize_host_paths(values: Iterable[Path | str]) -> tuple[str, ...]:
    try:
        iterator = iter(values)
    except TypeError as error:
        raise TypeError("host_paths must be an iterable") from error
    candidates: set[str] = set()
    total_characters = 0
    for index in range(_MAX_SANITIZER_HOST_PATH_ITEMS + 1):
        try:
            value = next(iterator)
        except StopIteration:
            break
        except Exception as error:
            raise TypeError("host_paths must be an iterable") from error
        if index >= _MAX_SANITIZER_HOST_PATH_ITEMS:
            raise ValueError("host_paths exceed their item bound")
        if isinstance(value, Path):
            rendered = str(value)
        elif isinstance(value, str):
            rendered = value
        else:
            raise TypeError("host_paths must contain Path or text values")
        if not rendered:
            continue
        total_characters += len(rendered)
        if total_characters > _MAX_SANITIZER_HOST_PATH_CHARACTERS:
            raise ValueError("host_paths exceed their character bound")
        candidates.add(rendered)
        candidates.add(rendered.replace("\\", "/"))
        candidates.add(rendered.replace("/", "\\"))
    return tuple(sorted(candidates, key=lambda item: (-len(item), item)))


def _truncate_text(value: str, max_characters: int, marker: str) -> str:
    if max_characters == 0:
        return ""
    if max_characters <= len(marker):
        return marker[:max_characters]
    return value[: max_characters - len(marker)] + marker
