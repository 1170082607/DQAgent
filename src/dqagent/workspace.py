"""Workspace authority, protected paths, and bounded outward text handling.

This module owns only DQAgent-controlled filesystem authority. It does not
provide subprocess isolation, tool limits, or observation algorithms.
"""

from __future__ import annotations

import math
import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final

from dqagent.errors import ConfigurationError, DQAgentError, ErrorCategory

__all__ = [
    "DEFAULT_PROTECTED_PATHS",
    "DEFAULT_SECRET_BASENAMES",
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
_DEFAULT_MAX_SNAPSHOT_ELAPSED_SECONDS: Final[float] = 10.0
_DEFAULT_MAX_RENDERED_DIFF_CHARACTERS: Final[int] = 200_000


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

    def __init__(
        self,
        max_logical_path_characters: int = _DEFAULT_MAX_LOGICAL_PATH_CHARACTERS,
        max_logical_path_segments: int = _DEFAULT_MAX_LOGICAL_PATH_SEGMENTS,
        max_snapshot_entries: int = _DEFAULT_MAX_SNAPSHOT_ENTRIES,
        max_snapshot_bytes: int = _DEFAULT_MAX_SNAPSHOT_BYTES,
        max_snapshot_elapsed_seconds: float = _DEFAULT_MAX_SNAPSHOT_ELAPSED_SECONDS,
        max_rendered_diff_characters: int = _DEFAULT_MAX_RENDERED_DIFF_CHARACTERS,
        *,
        max_path_characters: int | None = None,
        max_path_segments: int | None = None,
        snapshot_max_entries: int | None = None,
        snapshot_max_bytes: int | None = None,
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
    def snapshot_max_elapsed_seconds(self) -> float:
        return self.max_snapshot_elapsed_seconds

    @property
    def max_diff_characters(self) -> int:
        return self.max_rendered_diff_characters


@dataclass(frozen=True, slots=True)
class WorkspaceScope:
    """Trusted, immutable authority for one existing workspace root."""

    workspace_id: str
    root: Path
    limits: WorkspaceLimits = field(default_factory=WorkspaceLimits)
    protected_paths: tuple[PurePosixPath, ...] = ()
    secret_paths: tuple[PurePosixPath, ...] = ()

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
        object.__setattr__(self, "root", canonical_root)
        object.__setattr__(self, "protected_paths", protected)
        object.__setattr__(self, "secret_paths", secret)


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
            host_paths=(self._scope.root, *tuple(host_paths)),
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

        if purpose is not WorkspacePurpose.SNAPSHOT:
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


def _purpose_value(purpose: WorkspacePurpose | str) -> str:
    return purpose.value if isinstance(purpose, WorkspacePurpose) else str(purpose)


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
    normalized: list[PurePosixPath] = []
    for value in values:
        if not isinstance(value, PurePosixPath):
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_RULE)
        raw = str(value)
        if not raw or raw == "." or raw.startswith("/") or "\\" in raw or "\x00" in raw:
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_RULE)
        if any(part in {"", ".", ".."} for part in raw.split("/")):
            raise WorkspaceConfigurationError(reason_code=WorkspaceReason.INVALID_RULE)
        if len(raw) > 4_096:
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
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{label} must contain text")
        if value:
            normalized.append(value)
    return tuple(sorted(set(normalized), key=lambda item: (-len(item), item)))


def _normalize_host_paths(values: Iterable[Path | str]) -> tuple[str, ...]:
    candidates: set[str] = set()
    for value in values:
        if isinstance(value, Path):
            rendered = str(value)
        elif isinstance(value, str):
            rendered = value
        else:
            raise TypeError("host_paths must contain Path or text values")
        if not rendered:
            continue
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
