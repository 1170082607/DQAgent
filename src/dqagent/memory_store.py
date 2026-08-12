"""Transactional storage contract for policy-governed long-term memory."""

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Protocol, TypeAlias, cast

from dqagent.errors import (
    MemoryConflictError,
    MemoryCorruptChangeError,
    MemoryNotFoundError,
    MemoryValidationError,
)
from dqagent.errors import MemoryError as DQMemoryError
from dqagent.memory import (
    MemoryConfidence,
    MemoryConfirmation,
    MemoryForgetReason,
    MemoryKind,
    MemoryLifecycleStatus,
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
    MemoryScopeKind,
    MemorySensitivity,
    MemorySourceType,
    MemoryTombstone,
)

MEMORY_DATABASE_SCHEMA_VERSION = 1
_PROVENANCE_IDENTITY_PREFIX = "dqagent-provenance-v1:"

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
        _validate_apply_arguments(change_set, expected_revision)

        with self._lock:
            current = self._snapshots.get(
                change_set.scope,
                MemoryScopeSnapshot(change_set.scope),
            )
            if current.revision != expected_revision:
                raise _revision_conflict(
                    change_set.scope,
                    expected_revision,
                    current.revision,
                )
            candidate = _apply_changes(current, change_set)
            self._snapshots[change_set.scope] = candidate
            return candidate


class SqliteMemoryStore:
    """SQLite-backed exact-scope store with cross-connection transactional CAS."""

    def __init__(self, path: Path, *, timeout_seconds: float = 5.0) -> None:
        if not isinstance(path, Path):
            raise MemoryValidationError("memory database path must be a Path")
        if path == Path(":memory:"):
            raise MemoryValidationError(
                "memory database path must identify a durable filesystem database"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise MemoryValidationError("memory database timeout must be greater than zero")
        self._path = path
        self._timeout_seconds = timeout_seconds
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    _initialize_sqlite_schema(connection)
                    _verify_sqlite_schema(connection)
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise
        except DQMemoryError:
            raise
        except (OSError, sqlite3.DatabaseError):
            raise DQMemoryError("cannot initialize memory database") from None

    def load(self, scope: MemoryScope) -> MemoryScopeSnapshot:
        if not isinstance(scope, MemoryScope):
            raise MemoryValidationError("memory store scope must be a MemoryScope")
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN")
                try:
                    _verify_sqlite_schema(connection)
                    snapshot = _load_sqlite_snapshot(connection, scope)
                    connection.commit()
                    return snapshot
                except BaseException:
                    connection.rollback()
                    raise
        except DQMemoryError:
            raise
        except (sqlite3.DatabaseError, TypeError, ValueError) as error:
            raise DQMemoryError("cannot load memory scope") from error

    def apply(
        self,
        change_set: MemoryChangeSet,
        *,
        expected_revision: int,
    ) -> MemoryScopeSnapshot:
        _validate_apply_arguments(change_set, expected_revision)
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    _verify_sqlite_schema(connection)
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO memory_scopes(scope_kind, scope_id, revision)
                        VALUES (?, ?, 0)
                        """,
                        (change_set.scope.kind.value, change_set.scope.scope_id),
                    )
                    current = _load_sqlite_snapshot(connection, change_set.scope)
                    if current.revision != expected_revision:
                        raise _revision_conflict(
                            change_set.scope,
                            expected_revision,
                            current.revision,
                        )
                    candidate = _apply_changes(current, change_set)
                    _write_sqlite_snapshot(
                        connection,
                        candidate,
                        expected_revision=expected_revision,
                    )
                    connection.commit()
                    return candidate
                except BaseException:
                    connection.rollback()
                    raise
        except (MemoryConflictError, MemoryCorruptChangeError, MemoryNotFoundError):
            raise
        except sqlite3.IntegrityError as error:
            raise MemoryCorruptChangeError(
                "memory change violates database constraints"
            ) from error
        except DQMemoryError:
            raise
        except (sqlite3.DatabaseError, TypeError, ValueError) as error:
            raise DQMemoryError("cannot apply memory changes") from error

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=self._timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA secure_delete = ON")
        return connection


def _initialize_sqlite_schema(connection: sqlite3.Connection) -> None:
    statements = (
        """
        CREATE TABLE IF NOT EXISTS memory_metadata (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_scopes (
            scope_kind TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            revision INTEGER NOT NULL CHECK (revision >= 0),
            PRIMARY KEY (scope_kind, scope_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_records (
            scope_kind TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            kind TEXT NOT NULL,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL NOT NULL,
            sensitivity TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_item_digest TEXT NOT NULL,
            extractor_identity TEXT NOT NULL,
            extracted_at TEXT NOT NULL,
            source_id TEXT,
            source_revision INTEGER,
            run_id TEXT,
            confirmation_digest TEXT NOT NULL,
            confirmed_at TEXT NOT NULL,
            status TEXT NOT NULL,
            valid_from TEXT NOT NULL,
            expires_at TEXT,
            supersedes_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            PRIMARY KEY (scope_kind, scope_id, memory_id),
            FOREIGN KEY (scope_kind, scope_id)
                REFERENCES memory_scopes(scope_kind, scope_id) ON DELETE CASCADE
        )
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS memory_one_active_topic
        ON memory_records(scope_kind, scope_id, kind, topic)
        WHERE status = 'active'
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_tombstones (
            scope_kind TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            forgotten_at TEXT NOT NULL,
            reason TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            PRIMARY KEY (scope_kind, scope_id, memory_id),
            FOREIGN KEY (scope_kind, scope_id)
                REFERENCES memory_scopes(scope_kind, scope_id) ON DELETE CASCADE
        )
        """,
    )
    for statement in statements:
        connection.execute(statement)
    connection.execute(
        "INSERT OR IGNORE INTO memory_metadata(key, value) VALUES (?, ?)",
        ("schema_version", MEMORY_DATABASE_SCHEMA_VERSION),
    )


def _verify_sqlite_schema(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT value FROM memory_metadata WHERE key = ?",
        ("schema_version",),
    ).fetchone()
    if row is None or row["value"] != MEMORY_DATABASE_SCHEMA_VERSION:
        raise DQMemoryError("unsupported memory database schema version")


def _load_sqlite_snapshot(
    connection: sqlite3.Connection,
    scope: MemoryScope,
) -> MemoryScopeSnapshot:
    parameters = (scope.kind.value, scope.scope_id)
    revision_row = connection.execute(
        """
        SELECT revision FROM memory_scopes
        WHERE scope_kind = ? AND scope_id = ?
        """,
        parameters,
    ).fetchone()
    if revision_row is None:
        return MemoryScopeSnapshot(scope)
    try:
        revision = _required_int(revision_row, "revision")
        record_rows = connection.execute(
            """
            SELECT * FROM memory_records
            WHERE scope_kind = ? AND scope_id = ?
            ORDER BY memory_id
            """,
            parameters,
        ).fetchall()
        tombstone_rows = connection.execute(
            """
            SELECT * FROM memory_tombstones
            WHERE scope_kind = ? AND scope_id = ?
            ORDER BY memory_id
            """,
            parameters,
        ).fetchall()
        return MemoryScopeSnapshot(
            scope=scope,
            revision=revision,
            records=tuple(_record_from_sqlite_row(row) for row in record_rows),
            tombstones=tuple(_tombstone_from_sqlite_row(row) for row in tombstone_rows),
        )
    except (KeyError, TypeError, ValueError, MemoryValidationError) as error:
        raise DQMemoryError("memory database contains invalid scope data") from error


def _write_sqlite_snapshot(
    connection: sqlite3.Connection,
    snapshot: MemoryScopeSnapshot,
    *,
    expected_revision: int,
) -> None:
    scope_parameters = (snapshot.scope.kind.value, snapshot.scope.scope_id)
    cursor = connection.execute(
        """
        UPDATE memory_scopes SET revision = ?
        WHERE scope_kind = ? AND scope_id = ? AND revision = ?
        """,
        (snapshot.revision, *scope_parameters, expected_revision),
    )
    if cursor.rowcount != 1:
        row = connection.execute(
            """
            SELECT revision FROM memory_scopes
            WHERE scope_kind = ? AND scope_id = ?
            """,
            scope_parameters,
        ).fetchone()
        actual_revision = _required_int(row, "revision") if row is not None else 0
        raise _revision_conflict(snapshot.scope, expected_revision, actual_revision)

    connection.execute(
        "DELETE FROM memory_records WHERE scope_kind = ? AND scope_id = ?",
        scope_parameters,
    )
    connection.execute(
        "DELETE FROM memory_tombstones WHERE scope_kind = ? AND scope_id = ?",
        scope_parameters,
    )
    connection.executemany(
        """
        INSERT INTO memory_records(
            scope_kind, scope_id, memory_id, revision, kind, topic, content,
            confidence, sensitivity, source_type, source_item_digest,
            extractor_identity, extracted_at, source_id, source_revision, run_id,
            confirmation_digest, confirmed_at, status, valid_from, expires_at,
            supersedes_id, created_at, updated_at, schema_version
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (_record_to_sqlite_values(record) for record in snapshot.records),
    )
    connection.executemany(
        """
        INSERT INTO memory_tombstones(
            scope_kind, scope_id, memory_id, revision, forgotten_at, reason, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (_tombstone_to_sqlite_values(tombstone) for tombstone in snapshot.tombstones),
    )


def _record_to_sqlite_values(record: MemoryRecord) -> tuple[object, ...]:
    provenance = record.provenance
    return (
        record.scope.kind.value,
        record.scope.scope_id,
        record.memory_id,
        record.revision,
        record.kind.value,
        record.topic,
        record.content,
        record.confidence.value,
        record.sensitivity.value,
        provenance.source_type.value,
        provenance.source_item_digest,
        _stored_extractor_identity(provenance),
        provenance.extracted_at.isoformat(),
        provenance.source_id,
        provenance.source_revision,
        provenance.run_id,
        record.confirmation.candidate_digest,
        record.confirmation.confirmed_at.isoformat(),
        record.status.value,
        record.valid_from.isoformat(),
        record.expires_at.isoformat() if record.expires_at is not None else None,
        record.supersedes_id,
        record.created_at.isoformat(),
        record.updated_at.isoformat(),
        record.schema_version,
    )


def _tombstone_to_sqlite_values(tombstone: MemoryTombstone) -> tuple[object, ...]:
    return (
        tombstone.scope.kind.value,
        tombstone.scope.scope_id,
        tombstone.memory_id,
        tombstone.revision,
        tombstone.forgotten_at.isoformat(),
        tombstone.reason.value,
        tombstone.schema_version,
    )


def _record_from_sqlite_row(row: sqlite3.Row) -> MemoryRecord:
    scope = MemoryScope(
        MemoryScopeKind(_required_str(row, "scope_kind")),
        _required_str(row, "scope_id"),
    )
    extractor_identity, model_identity, response_identity = _load_extractor_identity(
        _required_str(row, "extractor_identity")
    )
    provenance = MemoryProvenance(
        source_type=MemorySourceType(_required_str(row, "source_type")),
        source_item_digest=_required_str(row, "source_item_digest"),
        extractor_identity=extractor_identity,
        extracted_at=_required_datetime(row, "extracted_at"),
        source_id=_optional_str(row, "source_id"),
        source_revision=_optional_int(row, "source_revision"),
        run_id=_optional_str(row, "run_id"),
        model_identity=model_identity,
        response_identity=response_identity,
    )
    return MemoryRecord(
        memory_id=_required_str(row, "memory_id"),
        revision=_required_int(row, "revision"),
        scope=scope,
        kind=MemoryKind(_required_str(row, "kind")),
        topic=_required_str(row, "topic"),
        content=_required_str(row, "content"),
        confidence=MemoryConfidence(_required_float(row, "confidence")),
        sensitivity=MemorySensitivity(_required_str(row, "sensitivity")),
        provenance=provenance,
        confirmation=MemoryConfirmation(
            _required_str(row, "confirmation_digest"),
            _required_datetime(row, "confirmed_at"),
        ),
        status=MemoryLifecycleStatus(_required_str(row, "status")),
        valid_from=_required_datetime(row, "valid_from"),
        expires_at=_optional_datetime(row, "expires_at"),
        supersedes_id=_optional_str(row, "supersedes_id"),
        created_at=_required_datetime(row, "created_at"),
        updated_at=_required_datetime(row, "updated_at"),
        schema_version=_required_int(row, "schema_version"),
    )


def _tombstone_from_sqlite_row(row: sqlite3.Row) -> MemoryTombstone:
    return MemoryTombstone(
        memory_id=_required_str(row, "memory_id"),
        revision=_required_int(row, "revision"),
        scope=MemoryScope(
            MemoryScopeKind(_required_str(row, "scope_kind")),
            _required_str(row, "scope_id"),
        ),
        forgotten_at=_required_datetime(row, "forgotten_at"),
        reason=MemoryForgetReason(_required_str(row, "reason")),
        schema_version=_required_int(row, "schema_version"),
    )


def _required_str(row: sqlite3.Row, name: str) -> str:
    value = row[name]
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    return value


def _optional_str(row: sqlite3.Row, name: str) -> str | None:
    value = row[name]
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text or null")
    return value


def _required_int(row: sqlite3.Row, name: str) -> int:
    value = row[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return cast(int, value)


def _optional_int(row: sqlite3.Row, name: str) -> int | None:
    value = row[name]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or null")
    return cast(int, value)


def _required_float(row: sqlite3.Row, name: str) -> float:
    value = row[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _required_datetime(row: sqlite3.Row, name: str) -> datetime:
    return datetime.fromisoformat(_required_str(row, name))


def _optional_datetime(row: sqlite3.Row, name: str) -> datetime | None:
    value = _optional_str(row, name)
    return datetime.fromisoformat(value) if value is not None else None


def _stored_extractor_identity(provenance: MemoryProvenance) -> str:
    """Keep the v1 SQLite shape while round-tripping optional provider identities."""

    if provenance.model_identity is None and provenance.response_identity is None:
        return provenance.extractor_identity
    payload = {
        "extractor_identity": provenance.extractor_identity,
        "model_identity": provenance.model_identity,
        "response_identity": provenance.response_identity,
    }
    return _PROVENANCE_IDENTITY_PREFIX + json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _load_extractor_identity(value: str) -> tuple[str, str | None, str | None]:
    if not value.startswith(_PROVENANCE_IDENTITY_PREFIX):
        return value, None, None
    try:
        payload = json.loads(value[len(_PROVENANCE_IDENTITY_PREFIX) :])
    except json.JSONDecodeError as error:
        raise ValueError("stored extractor identity envelope is invalid") from error
    if not isinstance(payload, dict):
        raise TypeError("stored extractor identity envelope must be an object")
    identity = payload.get("extractor_identity")
    model_identity = payload.get("model_identity")
    response_identity = payload.get("response_identity")
    if not isinstance(identity, str):
        raise TypeError("stored extractor identity must be text")
    if model_identity is not None and not isinstance(model_identity, str):
        raise TypeError("stored model identity must be text or null")
    if response_identity is not None and not isinstance(response_identity, str):
        raise TypeError("stored response identity must be text or null")
    return identity, model_identity, response_identity


def _validate_apply_arguments(change_set: object, expected_revision: object) -> None:
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


def _revision_conflict(
    scope: MemoryScope,
    expected_revision: int,
    actual_revision: int,
) -> MemoryConflictError:
    return MemoryConflictError(
        "memory scope revision conflict for "
        f"'{scope.kind.value}:{scope.scope_id}': "
        f"expected {expected_revision}, found {actual_revision}"
    )


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
