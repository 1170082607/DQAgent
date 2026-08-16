"""Contained, target-aware repository instruction loading.

This module selects mutable ``AGENTS.md`` guidance as request-scoped data.  It
does not project the data into a prompt, grant action authority, scan the
repository, or load skills and transitive resources.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final

from dqagent.errors import ContextError
from dqagent.workspace import (
    ResolvedWorkspacePath,
    Workspace,
    WorkspaceError,
    WorkspacePathError,
    WorkspacePurpose,
    WorkspaceReason,
)

__all__ = [
    "RepositoryAuthority",
    "RepositoryContext",
    "RepositoryContextError",
    "RepositoryContextLimits",
    "RepositoryInstructionLoader",
    "RepositoryOmission",
    "RepositoryOmissionReason",
    "RepositoryProvenance",
    "RepositoryResource",
    "RepositoryResourceKind",
    "RepositorySelection",
    "RepositorySelectionReason",
]


_DEFAULT_INDIVIDUAL_BYTES: Final[int] = 64_000
_DEFAULT_AGGREGATE_CHARACTERS: Final[int] = 128_000
_DEFAULT_TARGETS: Final[int] = 64
_DEFAULT_RESOURCES: Final[int] = 128
_DEFAULT_OMISSIONS: Final[int] = 128
_DEFAULT_INSTRUCTION_FILENAME: Final[str] = "AGENTS.md"


class RepositoryResourceKind(StrEnum):
    """Kinds owned by the repository-context resource boundary."""

    INSTRUCTION = "instruction"


class RepositoryAuthority(StrEnum):
    """Authority classification for mutable repository content."""

    REPOSITORY_GUIDANCE = "repository_guidance"


class RepositorySelectionReason(StrEnum):
    """Why one instruction file belongs to a target-scoped projection."""

    ROOT_ANCESTOR = "root_ancestor"
    TARGET_ANCESTOR = "target_ancestor"


class RepositoryOmissionReason(StrEnum):
    """Content-free reasons for omitting an otherwise applicable resource."""

    INDIVIDUAL_LIMIT = "individual_limit"
    AGGREGATE_LIMIT = "aggregate_limit"


class RepositoryContextError(ContextError):
    """Raised when repository context cannot be prepared safely.

    The error deliberately contains only a stable reason.  Workspace errors
    may contain raw operating-system details in their cause, so callers must
    never render the cause as part of outward diagnostics.
    """

    def __init__(
        self,
        message: str = "repository context preparation failed",
        *,
        reason_code: str,
    ) -> None:
        self.reason_code = reason_code
        super().__init__(f"{message} (reason={reason_code})")

    @property
    def event_attributes(self) -> MappingProxyType[str, object]:
        """Return bounded, content-free evidence for diagnostics."""

        return MappingProxyType({"reason_code": self.reason_code})


@dataclass(frozen=True, slots=True)
class RepositoryContextLimits:
    """Trusted limits owned by the repository-context loader."""

    max_individual_bytes: int = _DEFAULT_INDIVIDUAL_BYTES
    max_aggregate_characters: int = _DEFAULT_AGGREGATE_CHARACTERS
    max_targets: int = _DEFAULT_TARGETS
    max_resources: int = _DEFAULT_RESOURCES
    max_omissions: int = _DEFAULT_OMISSIONS

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_individual_bytes, bool)
            or not isinstance(self.max_individual_bytes, int)
            or self.max_individual_bytes < 1
        ):
            raise ValueError("max_individual_bytes must be a positive integer")
        if (
            isinstance(self.max_aggregate_characters, bool)
            or not isinstance(self.max_aggregate_characters, int)
            or self.max_aggregate_characters < 0
        ):
            raise ValueError("max_aggregate_characters must be a non-negative integer")
        for name in ("max_targets", "max_resources", "max_omissions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def max_instruction_bytes(self) -> int:
        """Compatibility name describing the individual instruction bound."""

        return self.max_individual_bytes


@dataclass(frozen=True, slots=True)
class RepositoryProvenance:
    """Stable source identity and digest for one repository resource."""

    source: PurePosixPath
    digest: str | None

    def __post_init__(self) -> None:
        _validate_relative_path(self.source, field_name="resource source", allow_root=False)
        if self.digest is not None and (
            not isinstance(self.digest, str)
            or len(self.digest) != 64
            or any(character not in "0123456789abcdef" for character in self.digest)
        ):
            raise ValueError("resource digest must be a lowercase SHA-256 digest or None")


@dataclass(frozen=True, slots=True)
class RepositorySelection:
    """Deterministic applicability evidence for one resource."""

    applicable_subtree: PurePosixPath
    reason: RepositorySelectionReason
    target_paths: tuple[PurePosixPath, ...] = ()

    def __post_init__(self) -> None:
        _validate_relative_path(
            self.applicable_subtree,
            field_name="applicable subtree",
            allow_root=True,
        )
        if not isinstance(self.reason, RepositorySelectionReason):
            raise TypeError("selection reason must be a RepositorySelectionReason")
        if not isinstance(self.target_paths, tuple):
            raise TypeError("selection target paths must be a tuple")
        for target in self.target_paths:
            _validate_relative_path(target, field_name="selection target", allow_root=False)


@dataclass(frozen=True, slots=True)
class RepositoryResource:
    """One atomically admitted repository instruction."""

    kind: RepositoryResourceKind
    key: str
    content: str
    provenance: RepositoryProvenance
    selection: RepositorySelection
    authority: RepositoryAuthority = RepositoryAuthority.REPOSITORY_GUIDANCE
    character_count: int = 0
    byte_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RepositoryResourceKind):
            raise TypeError("resource kind must be a RepositoryResourceKind")
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("resource key must not be empty")
        if not isinstance(self.content, str):
            raise TypeError("resource content must be text")
        if not isinstance(self.provenance, RepositoryProvenance):
            raise TypeError("resource provenance must be a RepositoryProvenance")
        if self.key != self.provenance.source.as_posix():
            raise ValueError("resource key must match its normalized source")
        if not isinstance(self.selection, RepositorySelection):
            raise TypeError("resource selection must be a RepositorySelection")
        if not isinstance(self.authority, RepositoryAuthority):
            raise TypeError("resource authority must be a RepositoryAuthority")
        try:
            encoded = self.content.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("resource content must be valid UTF-8 text") from error
        expected_digest = hashlib.sha256(encoded).hexdigest()
        if self.provenance.digest != expected_digest:
            raise ValueError("resource digest does not match resource content")
        if self.character_count != len(self.content):
            raise ValueError("resource character count does not match resource content")
        if self.byte_count != len(encoded):
            raise ValueError("resource byte count does not match resource content")

    @property
    def source(self) -> PurePosixPath:
        return self.provenance.source

    @property
    def digest(self) -> str:
        """Return the admitted content digest."""

        assert self.provenance.digest is not None
        return self.provenance.digest

    @property
    def applicable_subtree(self) -> PurePosixPath:
        return self.selection.applicable_subtree

    @property
    def selection_reason(self) -> RepositorySelectionReason:
        return self.selection.reason


@dataclass(frozen=True, slots=True)
class RepositoryOmission:
    """Bounded evidence for a resource omitted without retaining its body."""

    kind: RepositoryResourceKind
    key: str
    provenance: RepositoryProvenance
    selection: RepositorySelection
    reason: RepositoryOmissionReason
    authority: RepositoryAuthority = RepositoryAuthority.REPOSITORY_GUIDANCE
    character_count: int | None = None
    byte_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RepositoryResourceKind):
            raise TypeError("omission kind must be a RepositoryResourceKind")
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("omission key must not be empty")
        if not isinstance(self.provenance, RepositoryProvenance):
            raise TypeError("omission provenance must be a RepositoryProvenance")
        if self.key != self.provenance.source.as_posix():
            raise ValueError("omission key must match its normalized source")
        if not isinstance(self.selection, RepositorySelection):
            raise TypeError("omission selection must be a RepositorySelection")
        if not isinstance(self.reason, RepositoryOmissionReason):
            raise TypeError("omission reason must be a RepositoryOmissionReason")
        if not isinstance(self.authority, RepositoryAuthority):
            raise TypeError("omission authority must be a RepositoryAuthority")
        for name, value in (
            ("character_count", self.character_count),
            ("byte_count", self.byte_count),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"omission {name} must be a non-negative integer or None")

    @property
    def source(self) -> PurePosixPath:
        return self.provenance.source

    @property
    def digest(self) -> str | None:
        return self.provenance.digest

    @property
    def applicable_subtree(self) -> PurePosixPath:
        return self.selection.applicable_subtree

    @property
    def selection_reason(self) -> RepositorySelectionReason:
        return self.selection.reason


@dataclass(frozen=True, slots=True)
class RepositoryContext:
    """Immutable selected and omitted repository guidance for one request."""

    resources: tuple[RepositoryResource, ...] = ()
    omissions: tuple[RepositoryOmission, ...] = ()
    target_paths: tuple[PurePosixPath, ...] = ()
    aggregate_characters: int = 0
    max_aggregate_characters: int = _DEFAULT_AGGREGATE_CHARACTERS

    def __post_init__(self) -> None:
        if not isinstance(self.resources, tuple) or not all(
            isinstance(item, RepositoryResource) for item in self.resources
        ):
            raise TypeError("repository resources must be a tuple of RepositoryResource")
        if not isinstance(self.omissions, tuple) or not all(
            isinstance(item, RepositoryOmission) for item in self.omissions
        ):
            raise TypeError("repository omissions must be a tuple of RepositoryOmission")
        if not isinstance(self.target_paths, tuple):
            raise TypeError("repository target paths must be a tuple")
        for target in self.target_paths:
            _validate_relative_path(target, field_name="repository target", allow_root=False)
        if (
            isinstance(self.aggregate_characters, bool)
            or not isinstance(self.aggregate_characters, int)
            or self.aggregate_characters < 0
        ):
            raise ValueError("aggregate_characters must be a non-negative integer")
        if (
            isinstance(self.max_aggregate_characters, bool)
            or not isinstance(self.max_aggregate_characters, int)
            or self.max_aggregate_characters < 0
        ):
            raise ValueError("max_aggregate_characters must be a non-negative integer")
        if self.aggregate_characters != sum(item.character_count for item in self.resources):
            raise ValueError("aggregate_characters does not match selected resources")

    @property
    def selected(self) -> tuple[RepositoryResource, ...]:
        """Alias emphasizing that resources are selected, not policy."""

        return self.resources

    @property
    def instructions(self) -> tuple[RepositoryResource, ...]:
        return self.resources

    @property
    def selected_characters(self) -> int:
        return self.aggregate_characters

    def event_attributes(self) -> MappingProxyType[str, object]:
        """Return bounded evidence without any instruction body."""

        selected = [
            {
                "kind": resource.kind.value,
                "key": resource.key,
                "source": resource.source.as_posix(),
                "digest": resource.digest,
                "applicable_subtree": resource.applicable_subtree.as_posix(),
                "selection_reason": resource.selection_reason.value,
                "authority": resource.authority.value,
                "character_count": resource.character_count,
            }
            for resource in self.resources
        ]
        omitted = [
            {
                "kind": omission.kind.value,
                "key": omission.key,
                "source": omission.source.as_posix(),
                "digest": omission.digest,
                "applicable_subtree": omission.applicable_subtree.as_posix(),
                "selection_reason": omission.selection_reason.value,
                "authority": omission.authority.value,
                "reason": omission.reason.value,
                "character_count": omission.character_count,
                "byte_count": omission.byte_count,
            }
            for omission in self.omissions
        ]
        return MappingProxyType(
            {
                "target_paths": [path.as_posix() for path in self.target_paths],
                "selected_count": len(self.resources),
                "omitted_count": len(self.omissions),
                "aggregate_characters": self.aggregate_characters,
                "max_aggregate_characters": self.max_aggregate_characters,
                "selected": selected,
                "omitted": omitted,
            }
        )


@dataclass(frozen=True, slots=True)
class _InstructionCandidate:
    logical_path: PurePosixPath
    resolved_path: ResolvedWorkspacePath
    applicable_subtree: PurePosixPath
    target_paths: tuple[PurePosixPath, ...]
    source_identity: str


class RepositoryInstructionLoader:
    """Load only fixed-name instructions on explicit target ancestor chains."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        filename: str = _DEFAULT_INSTRUCTION_FILENAME,
        limits: RepositoryContextLimits | None = None,
        require_root_instruction: bool = False,
    ) -> None:
        if not isinstance(workspace, Workspace):
            raise TypeError("repository instruction loader requires a Workspace")
        _validate_instruction_filename(filename)
        if not isinstance(require_root_instruction, bool):
            raise TypeError("require_root_instruction must be a boolean")
        self._workspace = workspace
        self._filename = filename
        self._limits = limits or RepositoryContextLimits()
        self._require_root_instruction = require_root_instruction

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def repository_root(self) -> Path:
        """The canonical root fixed by trusted workspace composition."""

        return self._workspace.root

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def limits(self) -> RepositoryContextLimits:
        return self._limits

    def load(
        self,
        target_paths: Iterable[str | PurePosixPath] | str | PurePosixPath,
        *,
        mandatory: bool = False,
    ) -> RepositoryContext:
        """Load applicable instructions for explicit logical target paths.

        ``target_paths`` is the structured request field.  This method accepts
        no user prose and never searches for targets or instructions outside
        the ancestor chains derived from these paths.
        """

        if not isinstance(mandatory, bool):
            raise TypeError("mandatory must be a boolean")
        resolved_targets = self._resolve_targets(target_paths)
        candidates = self._collect_candidates(resolved_targets)
        if len(candidates) > self._limits.max_resources:
            raise self._error("resource_limit")

        resources: list[RepositoryResource] = []
        omissions: list[RepositoryOmission] = []
        aggregate_characters = 0
        for candidate in candidates:
            loaded = self._read_candidate(candidate, mandatory=mandatory)
            if loaded is None:
                continue
            if isinstance(loaded, RepositoryOmission):
                self._append_omission(omissions, loaded)
                continue
            content, raw_size, digest = loaded
            character_count = len(content)
            if aggregate_characters + character_count > self._limits.max_aggregate_characters:
                if mandatory:
                    raise self._error(RepositoryOmissionReason.AGGREGATE_LIMIT.value)
                self._append_omission(
                    omissions,
                    RepositoryOmission(
                        kind=RepositoryResourceKind.INSTRUCTION,
                        key=candidate.source_identity,
                        provenance=RepositoryProvenance(
                            PurePosixPath(candidate.source_identity), digest
                        ),
                        selection=self._selection(candidate),
                        reason=RepositoryOmissionReason.AGGREGATE_LIMIT,
                        character_count=character_count,
                        byte_count=raw_size,
                    ),
                )
                continue

            resource = RepositoryResource(
                kind=RepositoryResourceKind.INSTRUCTION,
                key=candidate.source_identity,
                content=content,
                provenance=RepositoryProvenance(
                    PurePosixPath(candidate.source_identity), digest
                ),
                selection=self._selection(candidate),
                character_count=character_count,
                byte_count=raw_size,
            )
            resources.append(resource)
            aggregate_characters += character_count

        return RepositoryContext(
            resources=tuple(resources),
            omissions=tuple(omissions),
            target_paths=tuple(item.logical_path for item in resolved_targets),
            aggregate_characters=aggregate_characters,
            max_aggregate_characters=self._limits.max_aggregate_characters,
        )

    def _resolve_targets(
        self,
        target_paths: Iterable[str | PurePosixPath] | str | PurePosixPath,
    ) -> tuple[ResolvedWorkspacePath, ...]:
        if isinstance(target_paths, (str, PurePosixPath)):
            raw_targets: tuple[str | PurePosixPath, ...] = (target_paths,)
        else:
            try:
                iterator = iter(target_paths)
            except Exception as error:
                raise self._error("invalid_target") from error
            raw_targets_list: list[str | PurePosixPath] = []
            for _ in range(self._limits.max_targets):
                try:
                    raw_targets_list.append(next(iterator))
                except StopIteration:
                    break
                except Exception as error:
                    raise self._error("invalid_target") from error
            else:
                try:
                    next(iterator)
                except StopIteration:
                    pass
                except Exception as error:
                    raise self._error("invalid_target") from error
                else:
                    raise self._error("target_limit")
            raw_targets = tuple(raw_targets_list)

        resolved_by_logical: dict[str, ResolvedWorkspacePath] = {}
        for raw_target in raw_targets:
            try:
                resolved = self._workspace.resolve(
                    raw_target,
                    purpose=WorkspacePurpose.PATCH,
                )
            except WorkspacePathError as error:
                if error.reason_code != WorkspaceReason.TARGET_KIND:
                    raise self._wrap_workspace_error(error) from error
                try:
                    resolved = self._workspace.resolve(
                        raw_target,
                        purpose=WorkspacePurpose.SEARCH,
                    )
                except WorkspaceError as search_error:
                    raise self._wrap_workspace_error(search_error) from search_error
            except (TypeError, ValueError) as error:
                raise self._error("invalid_target") from error
            except WorkspaceError as error:
                raise self._wrap_workspace_error(error) from error
            resolved_by_logical[resolved.logical_path.as_posix()] = resolved

        return tuple(
            sorted(
                resolved_by_logical.values(),
                key=lambda item: item.logical_path.as_posix(),
            )
        )

    def _collect_candidates(
        self, targets: tuple[ResolvedWorkspacePath, ...]
    ) -> tuple[_InstructionCandidate, ...]:
        by_identity: dict[str, _InstructionCandidate] = {}
        for target in targets:
            logical_target = target.logical_path
            parent = logical_target if target.is_directory else logical_target.parent
            ancestors = _ancestor_chain(parent)
            for applicable_subtree in ancestors:
                logical_instruction = _join_logical(applicable_subtree, self._filename)
                try:
                    resolved = self._workspace.resolve(
                        logical_instruction,
                        purpose=WorkspacePurpose.INSTRUCTION,
                    )
                except WorkspaceError as error:
                    if error.reason_code == WorkspaceReason.TARGET_MISSING:
                        if (
                            self._require_root_instruction
                            and applicable_subtree == PurePosixPath(".")
                        ):
                            raise self._error("mandatory_missing") from error
                        continue
                    raise self._wrap_workspace_error(error) from error

                source_identity = self._source_identity(resolved.path)
                existing = by_identity.get(source_identity)
                if existing is None:
                    by_identity[source_identity] = _InstructionCandidate(
                        logical_path=logical_instruction,
                        resolved_path=resolved,
                        applicable_subtree=applicable_subtree,
                        target_paths=(logical_target,),
                        source_identity=source_identity,
                    )
                elif logical_target not in existing.target_paths:
                    by_identity[source_identity] = _InstructionCandidate(
                        logical_path=existing.logical_path,
                        resolved_path=existing.resolved_path,
                        applicable_subtree=existing.applicable_subtree,
                        target_paths=tuple(
                            sorted(
                                (*existing.target_paths, logical_target),
                                key=lambda item: item.as_posix(),
                            )
                        ),
                        source_identity=existing.source_identity,
                    )

        return tuple(
            sorted(
                by_identity.values(),
                key=lambda item: (
                    _path_depth(item.applicable_subtree),
                    item.applicable_subtree.as_posix(),
                    item.source_identity,
                ),
            )
        )

    def _read_candidate(
        self,
        candidate: _InstructionCandidate,
        *,
        mandatory: bool,
    ) -> tuple[str, int, str] | RepositoryOmission | None:
        resolved = candidate.resolved_path
        try:
            current = self._workspace.revalidate(resolved)
            size = current.path.stat().st_size
            if size > self._limits.max_individual_bytes:
                if mandatory:
                    raise self._error(RepositoryOmissionReason.INDIVIDUAL_LIMIT.value)
                return self._oversize_omission(candidate, size)
            with current.path.open("rb") as stream:
                raw = stream.read(self._limits.max_individual_bytes + 1)
            self._workspace.revalidate(current)
        except RepositoryContextError:
            raise
        except (OSError, RuntimeError, WorkspaceError) as error:
            if isinstance(error, WorkspaceError):
                raise self._wrap_workspace_error(error) from error
            raise self._error("instruction_unreadable") from error

        if len(raw) > self._limits.max_individual_bytes:
            if mandatory:
                raise self._error(RepositoryOmissionReason.INDIVIDUAL_LIMIT.value)
            return self._oversize_omission(candidate, len(raw))
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise self._error("invalid_text") from error
        if "\x00" in content:
            raise self._error("invalid_text")
        return content, len(raw), hashlib.sha256(raw).hexdigest()

    def _oversize_omission(
        self, candidate: _InstructionCandidate, size: int
    ) -> RepositoryOmission:
        return RepositoryOmission(
            kind=RepositoryResourceKind.INSTRUCTION,
            key=candidate.source_identity,
            provenance=RepositoryProvenance(
                PurePosixPath(candidate.source_identity),
                None,
            ),
            selection=self._selection(candidate),
            reason=RepositoryOmissionReason.INDIVIDUAL_LIMIT,
            byte_count=size,
        )

    def _append_omission(
        self,
        omissions: list[RepositoryOmission],
        omission: RepositoryOmission,
    ) -> None:
        if len(omissions) >= self._limits.max_omissions:
            raise self._error("omission_limit")
        omissions.append(omission)

    def _selection(self, candidate: _InstructionCandidate) -> RepositorySelection:
        reason = (
            RepositorySelectionReason.ROOT_ANCESTOR
            if candidate.applicable_subtree == PurePosixPath(".")
            else RepositorySelectionReason.TARGET_ANCESTOR
        )
        return RepositorySelection(
            applicable_subtree=candidate.applicable_subtree,
            reason=reason,
            target_paths=candidate.target_paths,
        )

    def _source_identity(self, path: Path) -> str:
        try:
            relative = path.resolve(strict=True).relative_to(self._workspace.root)
        except (OSError, RuntimeError, ValueError) as error:
            raise self._error("containment") from error
        if not relative.parts:
            raise self._error("containment")
        return PurePosixPath(relative.as_posix()).as_posix()

    def _wrap_workspace_error(self, error: WorkspaceError) -> RepositoryContextError:
        return self._error(str(error.reason_code))

    def _error(self, reason_code: str) -> RepositoryContextError:
        return RepositoryContextError(reason_code=reason_code)

def _validate_instruction_filename(filename: str) -> None:
    if (
        not isinstance(filename, str)
        or not filename.strip()
        or "\x00" in filename
        or "/" in filename
        or "\\" in filename
        or filename in {".", ".."}
    ):
        raise ValueError("instruction filename must be one contained basename")


def _validate_relative_path(
    path: PurePosixPath,
    *,
    field_name: str,
    allow_root: bool,
) -> None:
    if not isinstance(path, PurePosixPath):
        raise TypeError(f"{field_name} must be a PurePosixPath")
    if path.is_absolute() or any(part in {"", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must be a contained relative path")
    if not allow_root and path == PurePosixPath("."):
        raise ValueError(f"{field_name} must identify a file")


def _ancestor_chain(parent: PurePosixPath) -> tuple[PurePosixPath, ...]:
    if parent == PurePosixPath("."):
        return (parent,)
    return (
        PurePosixPath("."),
        *(PurePosixPath(*parent.parts[:index]) for index in range(1, len(parent.parts) + 1)),
    )


def _join_logical(directory: PurePosixPath, filename: str) -> PurePosixPath:
    if directory == PurePosixPath("."):
        return PurePosixPath(filename)
    return PurePosixPath(*directory.parts, filename)


def _path_depth(path: PurePosixPath) -> int:
    return 0 if path == PurePosixPath(".") else len(path.parts)
