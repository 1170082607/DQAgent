"""Contained repository instructions and explicitly selected skill loading.

This module selects mutable repository guidance as request-scoped data.  It
does not project the data into a prompt, grant action authority, or load
transitive skill resources.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
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
    "RepositoryContextLoader",
    "RepositoryInstructionLoader",
    "RepositoryOmission",
    "RepositoryOmissionReason",
    "RepositoryProvenance",
    "RepositoryResource",
    "RepositoryResourceKind",
    "RepositorySelection",
    "RepositorySelectionReason",
    "RepositorySkillLoader",
    "SkillBody",
    "SkillCatalogEntry",
    "RepositorySkillBody",
    "RepositorySkillCatalogEntry",
]


_DEFAULT_INDIVIDUAL_BYTES: Final[int] = 64_000
_DEFAULT_AGGREGATE_CHARACTERS: Final[int] = 128_000
_DEFAULT_TARGETS: Final[int] = 64
_DEFAULT_RESOURCES: Final[int] = 128
_DEFAULT_OMISSIONS: Final[int] = 128
_DEFAULT_INSTRUCTION_FILENAME: Final[str] = "AGENTS.md"
_DEFAULT_SKILL_FILENAME: Final[str] = "SKILL.md"
_DEFAULT_CATALOG_ENTRIES: Final[int] = 64
_DEFAULT_CATALOG_ENTRY_BYTES: Final[int] = 16_000
_DEFAULT_CATALOG_CHARACTERS: Final[int] = 32_000
_DEFAULT_SKILL_BODY_BYTES: Final[int] = 64_000
_DEFAULT_SKILL_ROOTS: Final[int] = 16


class RepositoryResourceKind(StrEnum):
    """Kinds owned by the repository-context resource boundary."""

    INSTRUCTION = "instruction"
    SKILL_CATALOG = "skill_catalog"
    SKILL_BODY = "skill_body"


class RepositoryAuthority(StrEnum):
    """Authority classification for mutable repository content."""

    REPOSITORY_GUIDANCE = "repository_guidance"


class RepositorySelectionReason(StrEnum):
    """Why one instruction file belongs to a target-scoped projection."""

    ROOT_ANCESTOR = "root_ancestor"
    TARGET_ANCESTOR = "target_ancestor"
    SKILL_ROOT = "skill_root"
    EXPLICIT_KEY = "explicit_key"


class RepositoryOmissionReason(StrEnum):
    """Content-free reasons for omitting an otherwise applicable resource."""

    INDIVIDUAL_LIMIT = "individual_limit"
    AGGREGATE_LIMIT = "aggregate_limit"
    CATALOG_LIMIT = "catalog_limit"
    SKILL_BODY_LIMIT = "skill_body_limit"
    SKILL_MISSING = "skill_missing"
    SKILL_INVALID = "skill_invalid"
    CONTEXT_LIMIT = "context_limit"


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
    max_catalog_entries: int = _DEFAULT_CATALOG_ENTRIES
    max_catalog_entry_bytes: int = _DEFAULT_CATALOG_ENTRY_BYTES
    max_catalog_characters: int = _DEFAULT_CATALOG_CHARACTERS
    max_skill_body_bytes: int = _DEFAULT_SKILL_BODY_BYTES
    max_skill_roots: int = _DEFAULT_SKILL_ROOTS

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
        for name in (
            "max_targets",
            "max_resources",
            "max_omissions",
            "max_catalog_entries",
            "max_catalog_entry_bytes",
            "max_skill_body_bytes",
            "max_skill_roots",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_catalog_characters, bool)
            or not isinstance(self.max_catalog_characters, int)
            or self.max_catalog_characters < 0
        ):
            raise ValueError("max_catalog_characters must be a non-negative integer")

    @property
    def max_instruction_bytes(self) -> int:
        """Compatibility name describing the individual instruction bound."""

        return self.max_individual_bytes

    @property
    def max_skill_catalog_entries(self) -> int:
        return self.max_catalog_entries

    @property
    def max_skill_catalog_entry_bytes(self) -> int:
        return self.max_catalog_entry_bytes

    @property
    def max_skill_catalog_characters(self) -> int:
        return self.max_catalog_characters


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
        if self.kind is RepositoryResourceKind.INSTRUCTION and (
            self.key != self.provenance.source.as_posix()
        ):
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
        if self.kind is RepositoryResourceKind.INSTRUCTION and (
            self.key != self.provenance.source.as_posix()
        ):
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
class SkillCatalogEntry:
    """Metadata exposed before a caller explicitly selects a skill body."""

    key: str
    name: str
    description: str
    provenance: RepositoryProvenance
    selection: RepositorySelection
    authority: RepositoryAuthority = RepositoryAuthority.REPOSITORY_GUIDANCE
    character_count: int = 0
    byte_count: int = 0

    def __post_init__(self) -> None:
        _validate_skill_key(self.key)
        _validate_skill_text(self.name, "skill name")
        _validate_skill_text(self.description, "skill description")
        if not isinstance(self.provenance, RepositoryProvenance):
            raise TypeError("skill catalog provenance must be a RepositoryProvenance")
        if not isinstance(self.selection, RepositorySelection):
            raise TypeError("skill catalog selection must be a RepositorySelection")
        if not isinstance(self.authority, RepositoryAuthority):
            raise TypeError("skill catalog authority must be a RepositoryAuthority")
        canonical = f"{self.key}\n{self.name}\n{self.description}".encode()
        expected_digest = hashlib.sha256(canonical).hexdigest()
        if self.provenance.digest != expected_digest:
            raise ValueError("skill catalog digest does not match metadata")
        if self.character_count != len(canonical.decode("utf-8")):
            raise ValueError("skill catalog character count does not match metadata")
        if self.byte_count != len(canonical):
            raise ValueError("skill catalog byte count does not match metadata")

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
class SkillBody:
    """One complete, atomically admitted ``SKILL.md`` body."""

    key: str
    name: str
    description: str
    body: str
    provenance: RepositoryProvenance
    selection: RepositorySelection
    authority: RepositoryAuthority = RepositoryAuthority.REPOSITORY_GUIDANCE
    character_count: int = 0
    byte_count: int = 0

    def __post_init__(self) -> None:
        _validate_skill_key(self.key)
        _validate_skill_text(self.name, "skill name")
        _validate_skill_text(self.description, "skill description")
        if not isinstance(self.body, str):
            raise TypeError("skill body must be text")
        if not isinstance(self.provenance, RepositoryProvenance):
            raise TypeError("skill body provenance must be a RepositoryProvenance")
        if not isinstance(self.selection, RepositorySelection):
            raise TypeError("skill body selection must be a RepositorySelection")
        if not isinstance(self.authority, RepositoryAuthority):
            raise TypeError("skill body authority must be a RepositoryAuthority")
        try:
            encoded = self.body.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("skill body must be valid UTF-8 text") from error
        expected_digest = hashlib.sha256(encoded).hexdigest()
        if self.provenance.digest != expected_digest:
            raise ValueError("skill body digest does not match body")
        if self.character_count != len(self.body):
            raise ValueError("skill body character count does not match body")
        if self.byte_count != len(encoded):
            raise ValueError("skill body byte count does not match body")

    @property
    def content(self) -> str:
        """Compatibility alias for generic repository resource renderers."""

        return self.body

    @property
    def source(self) -> PurePosixPath:
        return self.provenance.source

    @property
    def digest(self) -> str:
        assert self.provenance.digest is not None
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
    skill_catalog: tuple[SkillCatalogEntry, ...] = ()
    selected_skill: SkillBody | None = None
    skill_omissions: tuple[RepositoryOmission, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.resources, tuple) or not all(
            isinstance(item, RepositoryResource) for item in self.resources
        ):
            raise TypeError("repository resources must be a tuple of RepositoryResource")
        if not isinstance(self.omissions, tuple) or not all(
            isinstance(item, RepositoryOmission) for item in self.omissions
        ):
            raise TypeError("repository omissions must be a tuple of RepositoryOmission")
        if not isinstance(self.skill_catalog, tuple) or not all(
            isinstance(item, SkillCatalogEntry) for item in self.skill_catalog
        ):
            raise TypeError("skill catalog must be a tuple of SkillCatalogEntry")
        if self.selected_skill is not None and not isinstance(self.selected_skill, SkillBody):
            raise TypeError("selected skill must be a SkillBody or None")
        if not isinstance(self.skill_omissions, tuple) or not all(
            isinstance(item, RepositoryOmission) for item in self.skill_omissions
        ):
            raise TypeError("skill omissions must be a tuple of RepositoryOmission")
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
        keys = [item.key for item in self.skill_catalog]
        if len(keys) != len(set(keys)):
            raise ValueError("skill catalog keys must be unique")
        if self.selected_skill is not None and self.selected_skill.key not in set(keys):
            raise ValueError("selected skill must belong to the skill catalog")

    @property
    def selected(self) -> tuple[RepositoryResource, ...]:
        """Alias emphasizing that resources are selected, not policy."""

        return self.resources

    @property
    def instructions(self) -> tuple[RepositoryResource, ...]:
        return tuple(
            resource
            for resource in self.resources
            if resource.kind is RepositoryResourceKind.INSTRUCTION
        )

    @property
    def selected_characters(self) -> int:
        return self.aggregate_characters

    @property
    def catalog(self) -> tuple[SkillCatalogEntry, ...]:
        return self.skill_catalog

    @property
    def skills(self) -> tuple[SkillCatalogEntry, ...]:
        return self.skill_catalog

    @property
    def skill_body(self) -> SkillBody | None:
        return self.selected_skill

    @property
    def selected_bodies(self) -> tuple[SkillBody, ...]:
        return () if self.selected_skill is None else (self.selected_skill,)

    @property
    def all_omissions(self) -> tuple[RepositoryOmission, ...]:
        return (*self.omissions, *self.skill_omissions)

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
            for resource in self.instructions
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
            for omission in self.all_omissions
        ]
        catalog = [
            {
                "kind": RepositoryResourceKind.SKILL_CATALOG.value,
                "key": entry.key,
                "source": entry.source.as_posix(),
                "digest": entry.digest,
                "selection_reason": entry.selection_reason.value,
                "authority": entry.authority.value,
                "character_count": entry.character_count,
            }
            for entry in self.skill_catalog
        ]
        selected_skill = None
        if self.selected_skill is not None:
            selected_skill = {
                "kind": RepositoryResourceKind.SKILL_BODY.value,
                "key": self.selected_skill.key,
                "source": self.selected_skill.source.as_posix(),
                "digest": self.selected_skill.digest,
                "selection_reason": self.selected_skill.selection_reason.value,
                "authority": self.selected_skill.authority.value,
                "character_count": self.selected_skill.character_count,
            }
        return MappingProxyType(
            {
                "target_paths": [path.as_posix() for path in self.target_paths],
                "selected_count": len(self.instructions)
                + len(self.skill_catalog)
                + (1 if self.selected_skill is not None else 0),
                "omitted_count": len(self.all_omissions),
                "aggregate_characters": self.aggregate_characters,
                "max_aggregate_characters": self.max_aggregate_characters,
                "selected": selected,
                "skill_catalog": catalog,
                "selected_skill": selected_skill,
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

@dataclass(frozen=True, slots=True)
class _SkillRoot:
    identifier: str
    path: Path
    is_workspace: bool


@dataclass(frozen=True, slots=True)
class _SkillCandidate:
    root: _SkillRoot
    directory: Path
    path: Path
    default_key: str
    source_identity: str
    selection: RepositorySelection
    resolved: ResolvedWorkspacePath | None = None
    missing: bool = False


@dataclass(frozen=True, slots=True)
class _SkillMetadata:
    key: str
    name: str
    description: str
    character_count: int
    byte_count: int


class RepositorySkillLoader:
    """Discover a bounded catalog and load at most one explicit skill body."""

    def __init__(
        self,
        workspace: Workspace | None = None,
        *,
        skill_roots: Mapping[str, Path] | Iterable[Path] | Path = (),
        filename: str = _DEFAULT_SKILL_FILENAME,
        limits: RepositoryContextLimits | None = None,
    ) -> None:
        if workspace is not None and not isinstance(workspace, Workspace):
            raise TypeError("repository skill loader requires a Workspace or None")
        _validate_instruction_filename(filename)
        self._workspace = workspace
        self._filename = filename
        self._limits = limits or RepositoryContextLimits()
        self._roots = self._normalize_roots(skill_roots)

    @property
    def workspace(self) -> Workspace | None:
        return self._workspace

    @property
    def skill_roots(self) -> tuple[Path, ...]:
        return tuple(root.path for root in self._roots)

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def limits(self) -> RepositoryContextLimits:
        return self._limits

    def load(
        self,
        skill_keys: Iterable[str] | str | None = (),
    ) -> RepositoryContext:
        """Return a catalog and optionally one explicitly selected body.

        ``skill_keys`` is the structured caller field used by ``CodingRequest``.
        It is intentionally not inferred from a user message.  The loader never
        follows references from a body and never searches beyond configured roots
        or their immediate skill directories.
        """

        candidates = self._discover_candidates()
        if len(candidates) > self._limits.max_catalog_entries:
            raise self._error("skill_catalog_limit")

        catalog: list[SkillCatalogEntry] = []
        omissions: list[RepositoryOmission] = []
        omitted_by_key: dict[str, RepositoryOmission] = {}
        by_key: dict[str, tuple[SkillCatalogEntry, _SkillCandidate]] = {}
        catalog_characters = 0
        for candidate in candidates:
            loaded = self._load_catalog_entry(candidate)
            if isinstance(loaded, RepositoryOmission):
                self._append_omission(omissions, loaded)
                omitted_by_key[loaded.key] = loaded
                continue
            entry = loaded
            if catalog_characters + entry.character_count > self._limits.max_catalog_characters:
                omission = self._skill_omission(
                    candidate,
                    RepositoryOmissionReason.CATALOG_LIMIT,
                    kind=RepositoryResourceKind.SKILL_CATALOG,
                    byte_count=entry.byte_count,
                    key=entry.key,
                )
                self._append_omission(omissions, omission)
                omitted_by_key[entry.key] = omission
                continue
            if entry.key in by_key:
                raise self._error("duplicate_skill_key")
            by_key[entry.key] = (entry, candidate)
            catalog.append(entry)
            catalog_characters += entry.character_count

        catalog.sort(key=lambda item: (item.key, item.source.as_posix()))
        selected_keys = self._normalize_skill_keys(skill_keys)
        selected_skill: SkillBody | None = None
        if selected_keys:
            selected_key = selected_keys[0]
            selected = by_key.get(selected_key)
            if selected is None:
                omitted = omitted_by_key.get(selected_key)
                if omitted is not None:
                    raise self._error(omitted.reason.value)
                raise self._error("unknown_skill_key")
            entry, candidate = selected
            selected_skill = self._load_body(
                entry,
                candidate,
                selection=RepositorySelection(
                    entry.applicable_subtree,
                    RepositorySelectionReason.EXPLICIT_KEY,
                ),
                omissions=omissions,
            )

        return RepositoryContext(
            skill_catalog=tuple(catalog),
            selected_skill=selected_skill,
            skill_omissions=tuple(omissions),
        )

    def _normalize_roots(
        self,
        skill_roots: Mapping[str, Path] | Iterable[Path] | Path,
    ) -> tuple[_SkillRoot, ...]:
        configured: list[tuple[str, Path]] = []
        if isinstance(skill_roots, Path):
            configured.append(("skill-root-1", skill_roots))
        elif isinstance(skill_roots, Mapping):
            if len(skill_roots) > self._limits.max_skill_roots:
                raise self._error("skill_root_limit")
            for identifier, root in sorted(skill_roots.items(), key=lambda item: str(item[0])):
                if not isinstance(identifier, str) or not identifier.strip():
                    raise self._error("invalid_skill_root")
                configured.append((identifier, root))
        else:
            try:
                iterator = iter(skill_roots)
            except Exception as error:
                raise self._error("invalid_skill_root") from error
            for index in range(self._limits.max_skill_roots):
                try:
                    root = next(iterator)
                except StopIteration:
                    break
                except Exception as error:
                    raise self._error("invalid_skill_root") from error
                configured.append((f"skill-root-{index + 1}", root))
            else:
                try:
                    next(iterator)
                except StopIteration:
                    pass
                except Exception as error:
                    raise self._error("invalid_skill_root") from error
                else:
                    raise self._error("skill_root_limit")

        if len(configured) > self._limits.max_skill_roots:
            raise self._error("skill_root_limit")

        roots: list[_SkillRoot] = []
        if self._workspace is not None:
            roots.append(_SkillRoot("workspace", self._workspace.root, True))
        for identifier, root in configured:
            try:
                _validate_skill_root_identifier(identifier)
            except (TypeError, ValueError) as error:
                raise self._error("invalid_skill_root") from error
            if not isinstance(root, Path):
                raise self._error("invalid_skill_root")
            try:
                canonical = root.resolve(strict=True)
                if not canonical.is_dir() or canonical != root.absolute():
                    raise OSError
            except (OSError, RuntimeError):
                raise self._error("invalid_skill_root") from None
            roots.append(_SkillRoot(identifier, canonical, False))
        return tuple(roots)

    def _discover_candidates(self) -> tuple[_SkillCandidate, ...]:
        candidates: list[_SkillCandidate] = []
        for root in self._roots:
            root_skill = root.path / self._filename
            root_skill_exists = self._path_exists(root_skill)
            if root_skill_exists:
                candidates.append(self._make_candidate(root, root.path, root_skill))
                if len(candidates) > self._limits.max_catalog_entries:
                    raise self._error("skill_catalog_limit")

            child_directories: list[Path] = []
            try:
                for entry in root.path.iterdir():
                    if entry.name == self._filename:
                        continue
                    is_directory = self._is_directory(entry)
                    if is_directory is None:
                        raise self._error("skill_catalog_unreadable")
                    if not is_directory:
                        continue
                    if root.is_workspace and not self._path_exists(entry / self._filename):
                        continue
                    child_directories.append(entry)
                    if len(candidates) + len(child_directories) > self._limits.max_catalog_entries:
                        raise self._error("skill_catalog_limit")
            except RepositoryContextError:
                raise
            except (OSError, RuntimeError) as error:
                raise self._error("skill_catalog_unreadable") from error

            child_directories.sort(key=lambda item: item.name)
            for directory in child_directories:
                skill_path = directory / self._filename
                candidates.append(
                    self._make_candidate(
                        root,
                        directory,
                        skill_path,
                        missing=not self._path_exists(skill_path),
                    )
                )

        return tuple(
            sorted(
                candidates,
                key=lambda item: (item.default_key, item.source_identity),
            )
        )

    def _make_candidate(
        self,
        root: _SkillRoot,
        directory: Path,
        path: Path,
        *,
        missing: bool = False,
    ) -> _SkillCandidate:
        try:
            relative_file = path.relative_to(root.path)
        except ValueError as error:
            raise self._error("skill_containment") from error
        source = (
            PurePosixPath(relative_file.as_posix()).as_posix()
            if root.is_workspace
            else PurePosixPath(root.identifier, *relative_file.parts).as_posix()
        )
        subtree = PurePosixPath(".") if root.is_workspace else PurePosixPath(root.identifier)
        selection = RepositorySelection(subtree, RepositorySelectionReason.SKILL_ROOT)
        resolved: ResolvedWorkspacePath | None = None
        if not missing:
            if root.is_workspace:
                try:
                    resolved = self._workspace.resolve(  # type: ignore[union-attr]
                        PurePosixPath(relative_file.as_posix()),
                        purpose=WorkspacePurpose.SKILL,
                    )
                except WorkspaceError as error:
                    raise self._wrap_workspace_error(error, "skill") from error
            else:
                try:
                    canonical = path.resolve(strict=True)
                    canonical.relative_to(root.path)
                    if not canonical.is_file():
                        raise OSError
                    path = canonical
                except (OSError, RuntimeError, ValueError) as error:
                    raise self._error("skill_containment") from error
        relative_directory = directory.relative_to(root.path)
        return _SkillCandidate(
            root=root,
            directory=directory,
            path=path if resolved is None else resolved.path,
            default_key=relative_directory.name or root.identifier,
            source_identity=source,
            selection=selection,
            resolved=resolved,
            missing=missing,
        )

    def _load_catalog_entry(
        self,
        candidate: _SkillCandidate,
    ) -> SkillCatalogEntry | RepositoryOmission:
        if candidate.missing:
            return self._skill_omission(
                candidate,
                RepositoryOmissionReason.SKILL_MISSING,
                kind=RepositoryResourceKind.SKILL_CATALOG,
            )
        raw = self._read_prefix(candidate)
        metadata_raw = _skill_metadata_bytes(raw)
        try:
            text = metadata_raw.decode("utf-8")
        except UnicodeDecodeError:
            return self._skill_omission(
                candidate,
                RepositoryOmissionReason.SKILL_INVALID,
                kind=RepositoryResourceKind.SKILL_CATALOG,
                byte_count=len(metadata_raw),
            )
        if "\x00" in text:
            return self._skill_omission(
                candidate,
                RepositoryOmissionReason.SKILL_INVALID,
                kind=RepositoryResourceKind.SKILL_CATALOG,
                byte_count=len(metadata_raw),
            )
        metadata = _parse_skill_metadata(text, candidate.default_key)
        if metadata is None:
            reason = (
                RepositoryOmissionReason.CATALOG_LIMIT
                if len(metadata_raw) > self._limits.max_catalog_entry_bytes
                else RepositoryOmissionReason.SKILL_INVALID
            )
            return self._skill_omission(
                candidate,
                reason,
                kind=RepositoryResourceKind.SKILL_CATALOG,
                byte_count=len(metadata_raw),
            )
        if len(metadata_raw) > self._limits.max_catalog_entry_bytes:
            return self._skill_omission(
                candidate,
                RepositoryOmissionReason.CATALOG_LIMIT,
                kind=RepositoryResourceKind.SKILL_CATALOG,
                byte_count=len(metadata_raw),
            )
        if metadata.character_count > self._limits.max_catalog_characters:
            return self._skill_omission(
                candidate,
                RepositoryOmissionReason.CATALOG_LIMIT,
                kind=RepositoryResourceKind.SKILL_CATALOG,
                byte_count=metadata.byte_count,
            )
        try:
            _validate_skill_key(metadata.key)
        except (TypeError, ValueError):
            return self._skill_omission(
                candidate,
                RepositoryOmissionReason.SKILL_INVALID,
                kind=RepositoryResourceKind.SKILL_CATALOG,
                byte_count=metadata.byte_count,
                key=candidate.default_key,
            )
        digest = hashlib.sha256(
            f"{metadata.key}\n{metadata.name}\n{metadata.description}".encode()
        ).hexdigest()
        return SkillCatalogEntry(
            key=metadata.key,
            name=metadata.name,
            description=metadata.description,
            provenance=RepositoryProvenance(PurePosixPath(candidate.source_identity), digest),
            selection=candidate.selection,
            character_count=metadata.character_count,
            byte_count=metadata.byte_count,
        )

    def _load_body(
        self,
        entry: SkillCatalogEntry,
        candidate: _SkillCandidate,
        *,
        selection: RepositorySelection,
        omissions: list[RepositoryOmission],
    ) -> SkillBody | None:
        raw = self._read_body_bytes(candidate, omissions, selection=selection)
        if raw is None:
            return None
        try:
            body = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise self._error("skill_invalid") from error
        if not body.strip() or "\x00" in body:
            raise self._error("skill_invalid")
        metadata = _parse_skill_metadata(body, candidate.default_key)
        if metadata is None or (
            metadata.key != entry.key
            or metadata.name != entry.name
            or metadata.description != entry.description
        ):
            raise self._error("skill_invalid")
        digest = hashlib.sha256(raw).hexdigest()
        return SkillBody(
            key=entry.key,
            name=entry.name,
            description=entry.description,
            body=body,
            provenance=RepositoryProvenance(PurePosixPath(candidate.source_identity), digest),
            selection=selection,
            character_count=len(body),
            byte_count=len(raw),
        )

    def _read_prefix(self, candidate: _SkillCandidate) -> bytes:
        return self._read_bytes(candidate, self._limits.max_catalog_entry_bytes, "skill_catalog")

    def _read_body_bytes(
        self,
        candidate: _SkillCandidate,
        omissions: list[RepositoryOmission],
        *,
        selection: RepositorySelection,
    ) -> bytes | None:
        limit = self._limits.max_skill_body_bytes
        try:
            if candidate.root.is_workspace:
                workspace = self._workspace
                resolved = candidate.resolved
                if workspace is None or resolved is None:
                    raise self._error("invalid_skill_root")
                current = workspace.revalidate(resolved)
                size = current.path.stat().st_size
                if size > limit:
                    self._append_omission(
                        omissions,
                        self._skill_omission(
                            candidate,
                            RepositoryOmissionReason.SKILL_BODY_LIMIT,
                            kind=RepositoryResourceKind.SKILL_BODY,
                            byte_count=size,
                            selection=selection,
                        ),
                    )
                    return None
                with current.path.open("rb") as stream:
                    raw = stream.read(limit + 1)
                workspace.revalidate(current)
            else:
                size = candidate.path.stat().st_size
                before = _file_signature(candidate.path)
                if size > limit:
                    self._append_omission(
                        omissions,
                        self._skill_omission(
                            candidate,
                            RepositoryOmissionReason.SKILL_BODY_LIMIT,
                            kind=RepositoryResourceKind.SKILL_BODY,
                            byte_count=size,
                            selection=selection,
                        ),
                    )
                    return None
                with candidate.path.open("rb") as stream:
                    raw = stream.read(limit + 1)
                if before != _file_signature(candidate.path):
                    raise OSError
        except RepositoryContextError:
            raise
        except (OSError, RuntimeError, WorkspaceError) as error:
            if isinstance(error, WorkspaceError):
                raise self._wrap_workspace_error(error, "skill") from error
            raise self._error("skill_unreadable") from error
        if len(raw) > limit:
            self._append_omission(
                omissions,
                self._skill_omission(
                    candidate,
                    RepositoryOmissionReason.SKILL_BODY_LIMIT,
                    kind=RepositoryResourceKind.SKILL_BODY,
                    byte_count=len(raw),
                    selection=selection,
                ),
            )
            return None
        return raw

    def _read_bytes(self, candidate: _SkillCandidate, limit: int, kind: str) -> bytes:
        try:
            if candidate.root.is_workspace:
                workspace = self._workspace
                resolved = candidate.resolved
                if workspace is None or resolved is None:
                    raise self._error("invalid_skill_root")
                current = workspace.revalidate(resolved)
                with current.path.open("rb") as stream:
                    raw = stream.read(limit + 1)
                workspace.revalidate(current)
            else:
                before = _file_signature(candidate.path)
                with candidate.path.open("rb") as stream:
                    raw = stream.read(limit + 1)
                if before != _file_signature(candidate.path):
                    raise OSError
        except RepositoryContextError:
            raise
        except (OSError, RuntimeError, WorkspaceError) as error:
            if isinstance(error, WorkspaceError):
                raise self._wrap_workspace_error(error, "skill") from error
            raise self._error(f"{kind}_unreadable") from error
        return raw

    def _skill_omission(
        self,
        candidate: _SkillCandidate,
        reason: RepositoryOmissionReason,
        *,
        kind: RepositoryResourceKind,
        byte_count: int | None = None,
        key: str | None = None,
        selection: RepositorySelection | None = None,
    ) -> RepositoryOmission:
        return RepositoryOmission(
            kind=kind,
            key=key or candidate.default_key,
            provenance=RepositoryProvenance(PurePosixPath(candidate.source_identity), None),
            selection=candidate.selection if selection is None else selection,
            reason=reason,
            byte_count=byte_count,
        )

    def _normalize_skill_keys(
        self,
        skill_keys: Iterable[str] | str | None,
    ) -> tuple[str, ...]:
        if skill_keys is None:
            return ()
        if isinstance(skill_keys, str):
            values: tuple[str, ...] = (skill_keys,)
        else:
            try:
                iterator = iter(skill_keys)
            except Exception as error:
                raise self._error("invalid_skill_key") from error
            values_list: list[str] = []
            for _ in range(2):
                try:
                    values_list.append(next(iterator))
                except StopIteration:
                    break
                except Exception as error:
                    raise self._error("invalid_skill_key") from error
            values = tuple(values_list)
        if len(values) > 1:
            raise self._error("multiple_skill_keys")
        if not values:
            return ()
        key = values[0]
        try:
            _validate_skill_key(key)
        except (TypeError, ValueError) as error:
            raise self._error("invalid_skill_key") from error
        return (key,)

    def _append_omission(
        self,
        omissions: list[RepositoryOmission],
        omission: RepositoryOmission,
    ) -> None:
        if len(omissions) >= self._limits.max_omissions:
            raise self._error("omission_limit")
        omissions.append(omission)

    def _path_exists(self, path: Path) -> bool:
        try:
            return path.exists()
        except (OSError, RuntimeError) as error:
            raise self._error("skill_catalog_unreadable") from error

    @staticmethod
    def _is_directory(path: Path) -> bool | None:
        try:
            return path.is_dir()
        except (OSError, RuntimeError):
            return None

    def _wrap_workspace_error(
        self,
        error: WorkspaceError,
        prefix: str,
    ) -> RepositoryContextError:
        reason = str(error.reason_code)
        if reason in {WorkspaceReason.TARGET_MISSING, WorkspaceReason.PARENT_MISSING}:
            return self._error("skill_missing")
        if reason in {
            WorkspaceReason.TARGET_KIND,
            WorkspaceReason.INVALID_SCOPE,
            WorkspaceReason.FILESYSTEM,
        }:
            return self._error(f"{prefix}_unreadable")
        return self._error(reason)

    def _error(self, reason_code: str) -> RepositoryContextError:
        return RepositoryContextError(reason_code=reason_code)


class RepositoryContextLoader:
    """Compose one immutable instruction/catalog/body projection for a run."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        skill_roots: Mapping[str, Path] | Iterable[Path] | Path = (),
        instruction_filename: str = _DEFAULT_INSTRUCTION_FILENAME,
        skill_filename: str = _DEFAULT_SKILL_FILENAME,
        limits: RepositoryContextLimits | None = None,
        require_root_instruction: bool = False,
    ) -> None:
        if not isinstance(workspace, Workspace):
            raise TypeError("repository context loader requires a Workspace")
        shared_limits = limits or RepositoryContextLimits()
        self._limits = shared_limits
        self._instructions = RepositoryInstructionLoader(
            workspace,
            filename=instruction_filename,
            limits=shared_limits,
            require_root_instruction=require_root_instruction,
        )
        self._skills = RepositorySkillLoader(
            workspace,
            skill_roots=skill_roots,
            filename=skill_filename,
            limits=shared_limits,
        )

    @property
    def workspace(self) -> Workspace:
        return self._instructions.workspace

    @property
    def instructions(self) -> RepositoryInstructionLoader:
        return self._instructions

    @property
    def skills(self) -> RepositorySkillLoader:
        return self._skills

    def load(
        self,
        target_paths: Iterable[str | PurePosixPath] | str | PurePosixPath,
        *,
        skill_keys: Iterable[str] | str | None = (),
        mandatory: bool = False,
    ) -> RepositoryContext:
        instructions = self._instructions.load(target_paths, mandatory=mandatory)
        skills = self._skills.load(skill_keys)
        if len(instructions.omissions) + len(skills.skill_omissions) > self._limits.max_omissions:
            raise RepositoryContextError(reason_code="omission_limit")
        return RepositoryContext(
            resources=instructions.resources,
            omissions=instructions.omissions,
            target_paths=instructions.target_paths,
            aggregate_characters=instructions.aggregate_characters,
            max_aggregate_characters=instructions.max_aggregate_characters,
            skill_catalog=skills.skill_catalog,
            selected_skill=skills.selected_skill,
            skill_omissions=skills.skill_omissions,
        )


RepositorySkillCatalogEntry = SkillCatalogEntry
RepositorySkillBody = SkillBody


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


def _validate_skill_key(key: str) -> None:
    if (
        not isinstance(key, str)
        or not key.strip()
        or key != key.strip()
        or "\x00" in key
        or "/" in key
        or "\\" in key
        or any(character.isspace() for character in key)
        or key in {".", ".."}
    ):
        raise ValueError("skill key must be one stable non-path identifier")


def _validate_skill_root_identifier(identifier: str) -> None:
    if (
        not isinstance(identifier, str)
        or not identifier.strip()
        or identifier != identifier.strip()
        or "\x00" in identifier
        or "/" in identifier
        or "\\" in identifier
        or any(character.isspace() for character in identifier)
        or identifier in {".", ".."}
    ):
        raise ValueError("skill root identifier must be one stable non-path identifier")


def _validate_skill_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{field_name} must be non-empty UTF-8 text")


def _skill_metadata_bytes(raw: bytes) -> bytes:
    """Return only the bounded byte lines needed to parse skill metadata."""

    lines = raw.splitlines(keepends=True)
    first_index = next(
        (index for index, line in enumerate(lines) if line.strip()),
        None,
    )
    if first_index is None:
        return b""

    first = lines[first_index]
    metadata_lines = [first]
    if first.strip() == b"---":
        for line in lines[first_index + 1 :]:
            metadata_lines.append(line)
            if line.strip() == b"---":
                break
    elif first.strip().startswith(b"#"):
        for line in lines[first_index + 1 :]:
            if line.strip():
                metadata_lines.append(line)
                break
    return b"".join(metadata_lines)


def _parse_skill_metadata(text: str, default_key: str) -> _SkillMetadata | None:
    """Parse only bounded front matter or the first markdown heading/paragraph."""

    lines = text.splitlines(keepends=True)
    if not lines:
        return None

    first_index = next(
        (index for index, line in enumerate(lines) if line.strip()),
        None,
    )
    if first_index is None:
        return None

    fields: dict[str, str] = {}
    key = default_key
    if lines[first_index].strip() == "---":
        closing_index: int | None = None
        for index in range(first_index + 1, len(lines)):
            line = lines[index].strip()
            if line == "---":
                closing_index = index
                break
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                return None
            field, value = line.split(":", 1)
            field = field.strip().lower()
            value = value.strip()
            if field not in {"key", "name", "description"}:
                continue
            if not value:
                return None
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if field in fields:
                return None
            fields[field] = value
        if closing_index is None:
            return None
        key = fields.get("key", default_key)
        name = fields.get("name", key)
        description = fields.get("description")
        if description is None:
            return None
    else:
        content_lines = [line.strip() for line in lines if line.strip()]
        first = content_lines[0]
        if first.startswith("#"):
            name = first.lstrip("#").strip()
            description = content_lines[1] if len(content_lines) > 1 else ""
        else:
            key = default_key
            name = default_key
            description = first
    try:
        _validate_skill_text(key, "skill key")
        _validate_skill_text(name, "skill name")
        _validate_skill_text(description, "skill description")
    except (TypeError, ValueError):
        return None
    canonical = f"{key}\n{name}\n{description}"
    return _SkillMetadata(
        key=key,
        name=name,
        description=description,
        character_count=len(canonical),
        byte_count=len(canonical.encode("utf-8")),
    )


def _file_signature(path: Path) -> tuple[int, int, int]:
    info = path.stat()
    return (info.st_ino, info.st_size, info.st_mtime_ns)


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
