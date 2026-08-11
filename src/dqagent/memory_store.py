"""Transactional storage contract for policy-governed long-term memory."""

from dataclasses import dataclass, fields
from threading import Lock
from typing import Protocol, TypeAlias

from dqagent.errors import (
    MemoryConflictError,
    MemoryCorruptChangeError,
    MemoryNotFoundError,
    MemoryValidationError,
)
from dqagent.memory import (
    MemoryLifecycleStatus,
    MemoryRecord,
    MemoryScope,
    MemoryTombstone,
)

_LIFECYCLE_STABLE_FIELDS = tuple(
    field.name
    for field in fields(MemoryRecord)
    if field.name not in {"revision", "status", "updated_at"}
)


@dataclass(frozen=True, slots=True)
class MemoryScopeSnapshot:
    """One immutable, exact-scope view and its compare-and-swap revision."""

    scope: MemoryScope
    revision: int = 0
    records: tuple[MemoryRecord, ...] = ()
    tombstones: tuple[MemoryTombstone, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, MemoryScope):
            raise MemoryValidationError("memory snapshot scope must be a MemoryScope")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise MemoryValidationError("memory scope revision must be a non-negative integer")
        if not isinstance(self.records, tuple) or not all(
            isinstance(record, MemoryRecord) for record in self.records
        ):
            raise MemoryValidationError("memory snapshot records must be a tuple of MemoryRecord")
        if not isinstance(self.tombstones, tuple) or not all(
            isinstance(tombstone, MemoryTombstone) for tombstone in self.tombstones
        ):
            raise MemoryValidationError(
                "memory snapshot tombstones must be a tuple of MemoryTombstone"
            )
        _validate_snapshot_contents(self)


@dataclass(frozen=True, slots=True)
class AddMemory:
    record: MemoryRecord


@dataclass(frozen=True, slots=True)
class RefreshMemory:
    record: MemoryRecord


@dataclass(frozen=True, slots=True)
class SupersedeMemory:
    superseded: MemoryRecord
    replacement: MemoryRecord


@dataclass(frozen=True, slots=True)
class ExpireMemory:
    record: MemoryRecord


@dataclass(frozen=True, slots=True)
class ForgetMemory:
    tombstone: MemoryTombstone


MemoryChange: TypeAlias = (
    AddMemory | RefreshMemory | SupersedeMemory | ExpireMemory | ForgetMemory
)


@dataclass(frozen=True, slots=True)
class MemoryChangeSet:
    """Caller-decided record transitions committed as one scope transaction."""

    scope: MemoryScope
    changes: tuple[MemoryChange, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, MemoryScope):
            raise MemoryValidationError("memory change-set scope must be a MemoryScope")
        if not isinstance(self.changes, tuple):
            raise MemoryValidationError("memory changes must be a tuple")


class MemoryStore(Protocol):
    """Provider-neutral exact-scope load and optimistic transaction boundary."""

    def load(self, scope: MemoryScope) -> MemoryScopeSnapshot: ...

    def apply(
        self,
        change_set: MemoryChangeSet,
        *,
        expected_revision: int,
    ) -> MemoryScopeSnapshot: ...


class InMemoryMemoryStore:
    """Thread-safe process-local adapter proving the transactional store contract."""

    def __init__(self) -> None:
        self._snapshots: dict[MemoryScope, MemoryScopeSnapshot] = {}
        self._lock = Lock()

    def load(self, scope: MemoryScope) -> MemoryScopeSnapshot:
        if not isinstance(scope, MemoryScope):
            raise MemoryValidationError("memory store scope must be a MemoryScope")
        with self._lock:
            return self._snapshots.get(scope, MemoryScopeSnapshot(scope))

    def apply(
        self,
        change_set: MemoryChangeSet,
        *,
        expected_revision: int,
    ) -> MemoryScopeSnapshot:
        if not isinstance(change_set, MemoryChangeSet):
            raise MemoryCorruptChangeError("memory store requires a MemoryChangeSet")
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise MemoryCorruptChangeError(
                "expected memory scope revision must be a non-negative integer"
            )

        with self._lock:
            current = self._snapshots.get(
                change_set.scope,
                MemoryScopeSnapshot(change_set.scope),
            )
            if current.revision != expected_revision:
                raise MemoryConflictError(
                    "memory scope revision conflict for "
                    f"'{change_set.scope.kind.value}:{change_set.scope.scope_id}': "
                    f"expected {expected_revision}, found {current.revision}"
                )
            candidate = _apply_changes(current, change_set)
            self._snapshots[change_set.scope] = candidate
            return candidate


def _apply_changes(
    current: MemoryScopeSnapshot,
    change_set: MemoryChangeSet,
) -> MemoryScopeSnapshot:
    if not change_set.changes:
        raise MemoryCorruptChangeError("memory change set must not be empty")

    records = {record.memory_id: record for record in current.records}
    tombstones = {tombstone.memory_id: tombstone for tombstone in current.tombstones}
    touched: set[str] = set()

    for change in change_set.changes:
        if isinstance(change, AddMemory):
            _apply_add(change_set.scope, change.record, records, tombstones, touched)
        elif isinstance(change, RefreshMemory):
            _apply_refresh(change_set.scope, change.record, records, touched)
        elif isinstance(change, SupersedeMemory):
            _apply_supersede(
                change_set.scope,
                change,
                records,
                tombstones,
                touched,
            )
        elif isinstance(change, ExpireMemory):
            _apply_expire(change_set.scope, change.record, records, touched)
        elif isinstance(change, ForgetMemory):
            _apply_forget(change_set.scope, change.tombstone, records, tombstones, touched)
        else:
            raise MemoryCorruptChangeError(
                f"unsupported memory change: {type(change).__name__}"
            )

    try:
        return MemoryScopeSnapshot(
            scope=change_set.scope,
            revision=current.revision + 1,
            records=tuple(sorted(records.values(), key=lambda record: record.memory_id)),
            tombstones=tuple(
                sorted(tombstones.values(), key=lambda tombstone: tombstone.memory_id)
            ),
        )
    except MemoryValidationError as error:
        raise MemoryCorruptChangeError(
            f"memory change violates store invariant: {error}"
        ) from error


def _apply_add(
    scope: MemoryScope,
    record: MemoryRecord,
    records: dict[str, MemoryRecord],
    tombstones: dict[str, MemoryTombstone],
    touched: set[str],
) -> None:
    _require_record(record)
    _require_scope(scope, record.scope)
    _claim(record.memory_id, touched)
    if record.memory_id in records or record.memory_id in tombstones:
        raise MemoryCorruptChangeError(f"memory ID '{record.memory_id}' already exists")
    if (
        record.revision != 1
        or record.status is not MemoryLifecycleStatus.ACTIVE
        or record.supersedes_id is not None
    ):
        raise MemoryCorruptChangeError(
            "added memory must be a first-revision active record without a supersedes ID"
        )
    records[record.memory_id] = record


def _apply_refresh(
    scope: MemoryScope,
    refreshed: MemoryRecord,
    records: dict[str, MemoryRecord],
    touched: set[str],
) -> None:
    _require_record(refreshed)
    _require_scope(scope, refreshed.scope)
    _claim(refreshed.memory_id, touched)
    current = _find_record(refreshed.memory_id, records)
    if current.status is not MemoryLifecycleStatus.ACTIVE:
        raise MemoryCorruptChangeError("only an active memory can be refreshed")
    if refreshed.status is not MemoryLifecycleStatus.ACTIVE:
        raise MemoryCorruptChangeError("a refreshed memory must remain active")
    _require_next_revision(current, refreshed)
    _require_equal_fields(
        current,
        refreshed,
        (
            "memory_id",
            "scope",
            "kind",
            "topic",
            "content",
            "supersedes_id",
            "created_at",
            "schema_version",
        ),
        transition="refresh",
    )
    _require_later_update(current, refreshed)
    records[refreshed.memory_id] = refreshed


def _apply_supersede(
    scope: MemoryScope,
    change: SupersedeMemory,
    records: dict[str, MemoryRecord],
    tombstones: dict[str, MemoryTombstone],
    touched: set[str],
) -> None:
    _require_record(change.superseded)
    _require_record(change.replacement)
    _require_scope(scope, change.superseded.scope)
    _require_scope(scope, change.replacement.scope)
    _claim(change.superseded.memory_id, touched)
    _claim(change.replacement.memory_id, touched)
    current = _find_record(change.superseded.memory_id, records)
    if current.status is not MemoryLifecycleStatus.ACTIVE:
        raise MemoryCorruptChangeError("only an active memory can be superseded")
    if change.superseded.status is not MemoryLifecycleStatus.SUPERSEDED:
        raise MemoryCorruptChangeError("superseded record must have superseded status")
    _require_next_revision(current, change.superseded)
    _require_equal_fields(
        current,
        change.superseded,
        _LIFECYCLE_STABLE_FIELDS,
        transition="supersede",
    )
    _require_later_update(current, change.superseded)

    replacement = change.replacement
    if replacement.memory_id in records or replacement.memory_id in tombstones:
        raise MemoryCorruptChangeError(f"memory ID '{replacement.memory_id}' already exists")
    if (
        replacement.revision != 1
        or replacement.status is not MemoryLifecycleStatus.ACTIVE
        or replacement.supersedes_id != current.memory_id
    ):
        raise MemoryCorruptChangeError(
            "replacement must be a first-revision active record linked to the superseded memory"
        )
    _require_equal_fields(
        current,
        replacement,
        ("scope", "kind", "topic"),
        transition="supersede replacement",
    )
    if replacement.created_at != change.superseded.updated_at:
        raise MemoryCorruptChangeError(
            "replacement creation and superseded update timestamps must match"
        )
    records[current.memory_id] = change.superseded
    records[replacement.memory_id] = replacement


def _apply_expire(
    scope: MemoryScope,
    expired: MemoryRecord,
    records: dict[str, MemoryRecord],
    touched: set[str],
) -> None:
    _require_record(expired)
    _require_scope(scope, expired.scope)
    _claim(expired.memory_id, touched)
    current = _find_record(expired.memory_id, records)
    if current.status is not MemoryLifecycleStatus.ACTIVE:
        raise MemoryCorruptChangeError("only an active memory can expire")
    if expired.status is not MemoryLifecycleStatus.EXPIRED:
        raise MemoryCorruptChangeError("expired record must have expired status")
    _require_next_revision(current, expired)
    _require_equal_fields(
        current,
        expired,
        _LIFECYCLE_STABLE_FIELDS,
        transition="expire",
    )
    _require_later_update(current, expired)
    records[expired.memory_id] = expired


def _apply_forget(
    scope: MemoryScope,
    tombstone: MemoryTombstone,
    records: dict[str, MemoryRecord],
    tombstones: dict[str, MemoryTombstone],
    touched: set[str],
) -> None:
    if not isinstance(tombstone, MemoryTombstone):
        raise MemoryCorruptChangeError("forget requires a MemoryTombstone")
    _require_scope(scope, tombstone.scope)
    _claim(tombstone.memory_id, touched)
    current = _find_record(tombstone.memory_id, records)
    if tombstone.revision != current.revision:
        raise MemoryCorruptChangeError(
            "forgotten memory tombstone must retain the latest record revision"
        )
    if tombstone.forgotten_at < current.updated_at:
        raise MemoryCorruptChangeError("memory cannot be forgotten before its latest update")
    del records[tombstone.memory_id]
    tombstones[tombstone.memory_id] = tombstone


def _validate_snapshot_contents(snapshot: MemoryScopeSnapshot) -> None:
    record_ids: set[str] = set()
    active_keys: set[tuple[object, str]] = set()
    for record in snapshot.records:
        if record.scope != snapshot.scope:
            raise MemoryValidationError("memory snapshot contains a record from another scope")
        if record.memory_id in record_ids:
            raise MemoryValidationError("memory snapshot contains duplicate record IDs")
        record_ids.add(record.memory_id)
        if record.status is MemoryLifecycleStatus.ACTIVE:
            key = (record.kind, record.topic)
            if key in active_keys:
                raise MemoryValidationError(
                    "memory scope may contain at most one active record per kind and topic"
                )
            active_keys.add(key)

    tombstone_ids: set[str] = set()
    for tombstone in snapshot.tombstones:
        if tombstone.scope != snapshot.scope:
            raise MemoryValidationError("memory snapshot contains a tombstone from another scope")
        if tombstone.memory_id in tombstone_ids:
            raise MemoryValidationError("memory snapshot contains duplicate tombstone IDs")
        tombstone_ids.add(tombstone.memory_id)
    if record_ids & tombstone_ids:
        raise MemoryValidationError("a memory ID cannot be both a record and a tombstone")


def _require_record(record: object) -> None:
    if not isinstance(record, MemoryRecord):
        raise MemoryCorruptChangeError("memory change requires a MemoryRecord")


def _require_scope(expected: MemoryScope, actual: MemoryScope) -> None:
    if actual != expected:
        raise MemoryCorruptChangeError("memory change contains an item from another scope")


def _claim(memory_id: str, touched: set[str]) -> None:
    if memory_id in touched:
        raise MemoryCorruptChangeError(
            f"memory ID '{memory_id}' is changed more than once in one change set"
        )
    touched.add(memory_id)


def _find_record(memory_id: str, records: dict[str, MemoryRecord]) -> MemoryRecord:
    try:
        return records[memory_id]
    except KeyError as error:
        raise MemoryNotFoundError(f"memory record '{memory_id}' does not exist") from error


def _require_next_revision(current: MemoryRecord, changed: MemoryRecord) -> None:
    if changed.revision != current.revision + 1:
        raise MemoryCorruptChangeError(
            f"memory record '{current.memory_id}' revision must advance by one"
        )


def _require_later_update(current: MemoryRecord, changed: MemoryRecord) -> None:
    if changed.updated_at <= current.updated_at:
        raise MemoryCorruptChangeError(
            f"memory record '{current.memory_id}' update timestamp must advance"
        )


def _require_equal_fields(
    current: MemoryRecord,
    changed: MemoryRecord,
    names: tuple[str, ...],
    *,
    transition: str,
) -> None:
    if any(getattr(current, name) != getattr(changed, name) for name in names):
        raise MemoryCorruptChangeError(
            f"memory {transition} changes fields that must remain stable"
        )
