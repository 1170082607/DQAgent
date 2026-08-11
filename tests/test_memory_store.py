"""Reusable contract tests for transactional MemoryStore adapters."""

import sqlite3
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier, Event

import pytest

from dqagent.errors import (
    MemoryConflictError,
    MemoryCorruptChangeError,
    MemoryNotFoundError,
    MemoryValidationError,
)
from dqagent.errors import MemoryError as DQMemoryError
from dqagent.memory import (
    MemoryCandidate,
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
from dqagent.memory_store import (
    AddMemory,
    ExpireMemory,
    ForgetMemory,
    InMemoryMemoryStore,
    MemoryChangeSet,
    MemoryScopeSnapshot,
    MemoryStore,
    RefreshMemory,
    SqliteMemoryStore,
    SupersedeMemory,
)

NOW = datetime(2026, 8, 11, 8, 0, tzinfo=UTC)
USER_SCOPE = MemoryScope(MemoryScopeKind.USER, "user-7")
OTHER_SCOPE = MemoryScope(MemoryScopeKind.USER, "user-8")


def make_candidate(
    *,
    scope: MemoryScope = USER_SCOPE,
    topic: str = "response.language",
    content: str = "The user prefers concise Chinese answers.",
    extracted_at: datetime = NOW - timedelta(minutes=3),
    expires_at: datetime | None = NOW + timedelta(days=90),
) -> MemoryCandidate:
    return MemoryCandidate(
        scope=scope,
        kind=MemoryKind.PREFERENCE,
        topic=topic,
        content=content,
        confidence=MemoryConfidence(0.9),
        sensitivity=MemorySensitivity.NON_SENSITIVE,
        provenance=MemoryProvenance(
            source_type=MemorySourceType.USER_DRAFT,
            source_item_digest="a" * 64,
            extractor_identity="contract-fixture",
            extracted_at=extracted_at,
        ),
        valid_from=NOW - timedelta(days=1),
        expires_at=expires_at,
    )


def make_record(
    memory_id: str = "memory-1",
    *,
    candidate: MemoryCandidate | None = None,
    created_at: datetime = NOW - timedelta(minutes=1),
    revision: int = 1,
    supersedes_id: str | None = None,
) -> MemoryRecord:
    candidate = candidate or make_candidate()
    return MemoryRecord.from_candidate(
        candidate,
        memory_id=memory_id,
        revision=revision,
        confirmation=MemoryConfirmation(
            candidate.digest,
            candidate.provenance.extracted_at + timedelta(seconds=1),
        ),
        created_at=created_at,
        updated_at=created_at,
        supersedes_id=supersedes_id,
    )


def refresh_record(record: MemoryRecord, *, updated_at: datetime = NOW) -> MemoryRecord:
    candidate = make_candidate(
        scope=record.scope,
        topic=record.topic,
        content=record.content,
        extracted_at=record.provenance.extracted_at + timedelta(minutes=1),
        expires_at=record.expires_at,
    )
    return MemoryRecord.from_candidate(
        candidate,
        memory_id=record.memory_id,
        revision=record.revision + 1,
        confirmation=MemoryConfirmation(
            candidate.digest,
            candidate.provenance.extracted_at + timedelta(seconds=1),
        ),
        created_at=record.created_at,
        updated_at=updated_at,
        supersedes_id=record.supersedes_id,
    )


def correction(
    record: MemoryRecord,
    *,
    updated_at: datetime = NOW,
) -> SupersedeMemory:
    superseded = replace(
        record,
        revision=record.revision + 1,
        status=MemoryLifecycleStatus.SUPERSEDED,
        updated_at=updated_at,
    )
    replacement = make_record(
        "memory-2",
        candidate=make_candidate(content="The user prefers detailed English answers."),
        created_at=updated_at,
        supersedes_id=record.memory_id,
    )
    return SupersedeMemory(superseded, replacement)


class MemoryStoreContract:
    """Adapter-independent behavioral suite; subclasses provide the ``store`` fixture."""

    __test__ = False

    @pytest.fixture
    def store(self) -> MemoryStore:
        raise NotImplementedError

    def test_load_returns_an_immutable_empty_exact_scope_snapshot(
        self,
        store: MemoryStore,
    ) -> None:

        snapshot = store.load(USER_SCOPE)

        assert snapshot == MemoryScopeSnapshot(USER_SCOPE)
        assert store.load(OTHER_SCOPE) == MemoryScopeSnapshot(OTHER_SCOPE)
        with pytest.raises(FrozenInstanceError):
            snapshot.revision = 4  # type: ignore[misc]

    def test_add_commits_one_scope_revision(self, store: MemoryStore) -> None:
        record = make_record()

        saved = store.apply(
            MemoryChangeSet(USER_SCOPE, (AddMemory(record),)),
            expected_revision=0,
        )

        assert saved.revision == 1
        assert saved.records == (record,)
        assert saved.tombstones == ()
        assert store.load(USER_SCOPE) == saved
        assert store.load(OTHER_SCOPE).records == ()

    def test_duplicate_refresh_updates_one_record_without_adding_another(
        self,
        store: MemoryStore,
    ) -> None:
        original = make_record()
        first = store.apply(
            MemoryChangeSet(USER_SCOPE, (AddMemory(original),)),
            expected_revision=0,
        )
        refreshed = refresh_record(original)

        saved = store.apply(
            MemoryChangeSet(USER_SCOPE, (RefreshMemory(refreshed),)),
            expected_revision=first.revision,
        )

        assert saved.revision == 2
        assert saved.records == (refreshed,)
        assert refreshed.revision == 2
        assert refreshed.provenance != original.provenance

    def test_correction_switches_old_and_new_active_state_once(
        self,
        store: MemoryStore,
    ) -> None:
        original = make_record()
        first = store.apply(
            MemoryChangeSet(USER_SCOPE, (AddMemory(original),)),
            expected_revision=0,
        )
        change = correction(original)

        saved = store.apply(
            MemoryChangeSet(USER_SCOPE, (change,)),
            expected_revision=first.revision,
        )

        assert saved.revision == 2
        assert saved.records == (change.superseded, change.replacement)
        active = [
            record
            for record in saved.records
            if record.status is MemoryLifecycleStatus.ACTIVE
        ]
        assert active == [change.replacement]
        assert change.replacement.supersedes_id == original.memory_id

    def test_expire_atomically_removes_a_record_from_active_state(
        self,
        store: MemoryStore,
    ) -> None:
        original = make_record(
            candidate=make_candidate(expires_at=NOW),
            created_at=NOW - timedelta(minutes=1),
        )
        first = store.apply(
            MemoryChangeSet(USER_SCOPE, (AddMemory(original),)),
            expected_revision=0,
        )
        expired = replace(
            original,
            revision=2,
            status=MemoryLifecycleStatus.EXPIRED,
            updated_at=NOW,
        )

        saved = store.apply(
            MemoryChangeSet(USER_SCOPE, (ExpireMemory(expired),)),
            expected_revision=first.revision,
        )

        assert saved.records == (expired,)
        assert all(
            record.status is not MemoryLifecycleStatus.ACTIVE for record in saved.records
        )

    def test_forget_removes_content_and_provenance_from_every_public_read(
        self,
        store: MemoryStore,
    ) -> None:
        original = make_record()
        first = store.apply(
            MemoryChangeSet(USER_SCOPE, (AddMemory(original),)),
            expected_revision=0,
        )
        tombstone = MemoryTombstone.from_record(
            original,
            forgotten_at=NOW,
            reason=MemoryForgetReason.USER_REQUEST,
        )

        saved = store.apply(
            MemoryChangeSet(USER_SCOPE, (ForgetMemory(tombstone),)),
            expected_revision=first.revision,
        )
        loaded = store.load(USER_SCOPE)

        assert saved == loaded
        assert loaded.records == ()
        assert loaded.tombstones == (tombstone,)
        assert not hasattr(loaded.tombstones[0], "content")
        assert not hasattr(loaded.tombstones[0], "provenance")
        assert original.content not in repr(loaded)

    def test_stale_scope_revision_cannot_overwrite_newer_state(
        self,
        store: MemoryStore,
    ) -> None:
        original = make_record()
        saved = store.apply(
            MemoryChangeSet(USER_SCOPE, (AddMemory(original),)),
            expected_revision=0,
        )

        with pytest.raises(MemoryConflictError, match="revision conflict"):
            store.apply(
                MemoryChangeSet(USER_SCOPE, (RefreshMemory(refresh_record(original)),)),
                expected_revision=0,
            )

        assert store.load(USER_SCOPE) == saved

    def test_concurrent_writers_from_one_revision_have_exactly_one_winner(
        self,
        store: MemoryStore,
    ) -> None:
        barrier = Barrier(2)

        def write(record: MemoryRecord) -> MemoryScopeSnapshot | MemoryConflictError:
            barrier.wait()
            try:
                return store.apply(
                    MemoryChangeSet(USER_SCOPE, (AddMemory(record),)),
                    expected_revision=0,
                )
            except MemoryConflictError as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(
                executor.map(write, (make_record(), make_record("memory-2")))
            )

        assert sum(isinstance(result, MemoryScopeSnapshot) for result in results) == 1
        assert sum(isinstance(result, MemoryConflictError) for result in results) == 1
        assert store.load(USER_SCOPE).revision == 1
        assert len(store.load(USER_SCOPE).records) == 1

    def test_failed_transaction_leaves_no_partial_state(self, store: MemoryStore) -> None:
        original = make_record()
        missing = refresh_record(make_record("missing"))

        with pytest.raises(MemoryNotFoundError, match="does not exist"):
            store.apply(
                MemoryChangeSet(
                    USER_SCOPE,
                    (AddMemory(original), RefreshMemory(missing)),
                ),
                expected_revision=0,
            )

        assert store.load(USER_SCOPE) == MemoryScopeSnapshot(USER_SCOPE)

    def test_change_items_cannot_cross_the_change_set_scope(
        self,
        store: MemoryStore,
    ) -> None:
        other_record = make_record("other-memory", candidate=make_candidate(scope=OTHER_SCOPE))

        with pytest.raises(MemoryCorruptChangeError, match="another scope"):
            store.apply(
                MemoryChangeSet(USER_SCOPE, (AddMemory(other_record),)),
                expected_revision=0,
            )

        assert store.load(USER_SCOPE) == MemoryScopeSnapshot(USER_SCOPE)
        assert store.load(OTHER_SCOPE) == MemoryScopeSnapshot(OTHER_SCOPE)

        original = make_record()
        saved = store.apply(
            MemoryChangeSet(USER_SCOPE, (AddMemory(original),)),
            expected_revision=0,
        )
        wrong_scope_tombstone = replace(
            MemoryTombstone.from_record(
                original,
                forgotten_at=NOW,
                reason=MemoryForgetReason.USER_REQUEST,
            ),
            scope=OTHER_SCOPE,
        )

        with pytest.raises(MemoryCorruptChangeError, match="another scope"):
            store.apply(
                MemoryChangeSet(USER_SCOPE, (ForgetMemory(wrong_scope_tombstone),)),
                expected_revision=saved.revision,
            )

        assert store.load(USER_SCOPE) == saved
        assert store.load(OTHER_SCOPE) == MemoryScopeSnapshot(OTHER_SCOPE)

    def test_failed_transaction_rolls_back_forget_payload_deletion(
        self,
        store: MemoryStore,
    ) -> None:
        original = make_record()
        saved = store.apply(
            MemoryChangeSet(USER_SCOPE, (AddMemory(original),)),
            expected_revision=0,
        )
        tombstone = MemoryTombstone.from_record(
            original,
            forgotten_at=NOW,
            reason=MemoryForgetReason.USER_REQUEST,
        )
        corrupt = MemoryChangeSet(
            USER_SCOPE,
            (ForgetMemory(tombstone), object()),  # type: ignore[arg-type]
        )

        with pytest.raises(MemoryCorruptChangeError, match="unsupported memory change"):
            store.apply(corrupt, expected_revision=saved.revision)

        assert store.load(USER_SCOPE) == saved

    def test_store_enforces_one_active_record_per_scope_kind_and_topic(
        self,
        store: MemoryStore,
    ) -> None:
        first = make_record()
        second = make_record("memory-2")

        with pytest.raises(MemoryCorruptChangeError, match="at most one active"):
            store.apply(
                MemoryChangeSet(USER_SCOPE, (AddMemory(first), AddMemory(second))),
                expected_revision=0,
            )

        assert store.load(USER_SCOPE).records == ()

    def test_active_record_uniqueness_includes_committed_scope_state(
        self,
        store: MemoryStore,
    ) -> None:
        first = make_record()
        saved = store.apply(
            MemoryChangeSet(USER_SCOPE, (AddMemory(first),)),
            expected_revision=0,
        )

        with pytest.raises(MemoryCorruptChangeError, match="at most one active"):
            store.apply(
                MemoryChangeSet(USER_SCOPE, (AddMemory(make_record("memory-2")),)),
                expected_revision=saved.revision,
            )

        assert store.load(USER_SCOPE) == saved

    def test_missing_target_and_corrupt_change_have_stable_error_types(
        self,
        store: MemoryStore,
    ) -> None:

        with pytest.raises(MemoryNotFoundError):
            store.apply(
                MemoryChangeSet(USER_SCOPE, (RefreshMemory(refresh_record(make_record())),)),
                expected_revision=0,
            )
        with pytest.raises(MemoryCorruptChangeError, match="must not be empty"):
            store.apply(MemoryChangeSet(USER_SCOPE, ()), expected_revision=0)

        corrupt = MemoryChangeSet(USER_SCOPE, (object(),))  # type: ignore[arg-type]
        with pytest.raises(MemoryCorruptChangeError, match="unsupported memory change"):
            store.apply(corrupt, expected_revision=0)

    def test_correction_failure_cannot_leave_old_record_inactive(
        self,
        store: MemoryStore,
    ) -> None:
        original = make_record()
        first = store.apply(
            MemoryChangeSet(USER_SCOPE, (AddMemory(original),)),
            expected_revision=0,
        )
        valid = correction(original)
        invalid = SupersedeMemory(
            valid.superseded,
            replace(valid.replacement, supersedes_id="another-memory"),
        )

        with pytest.raises(MemoryCorruptChangeError, match="replacement"):
            store.apply(
                MemoryChangeSet(USER_SCOPE, (invalid,)),
                expected_revision=first.revision,
            )

        assert store.load(USER_SCOPE) == first


class TestInMemoryMemoryStore(MemoryStoreContract):
    __test__ = True

    @pytest.fixture
    def store(self) -> MemoryStore:
        return InMemoryMemoryStore()


class TestSqliteMemoryStore(MemoryStoreContract):
    __test__ = True

    @pytest.fixture
    def store(self, tmp_path: Path) -> MemoryStore:
        return SqliteMemoryStore(tmp_path / "memory.sqlite3")


def test_in_memory_adapter_satisfies_memory_store_protocol() -> None:
    store: MemoryStore = InMemoryMemoryStore()

    assert store.load(USER_SCOPE) == MemoryScopeSnapshot(USER_SCOPE)


def test_sqlite_adapter_initializes_v1_schema_metadata(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"

    store = SqliteMemoryStore(path)

    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT value FROM memory_metadata WHERE key = ?",
            ("schema_version",),
        ).fetchone()
    adapter_connection = store._connect()
    try:
        secure_delete = adapter_connection.execute("PRAGMA secure_delete").fetchone()
    finally:
        adapter_connection.close()
    assert version == (1,)
    assert secure_delete is not None
    assert secure_delete[0] == 1


def test_sqlite_adapter_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    SqliteMemoryStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE memory_metadata SET value = ? WHERE key = ?",
            (99, "schema_version"),
        )

    with pytest.raises(DQMemoryError, match="unsupported memory database schema version"):
        SqliteMemoryStore(path)


def test_sqlite_round_trips_complete_provenance_and_timezone(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    store = SqliteMemoryStore(path)
    local_timezone = timezone(timedelta(hours=8))
    extracted_at = datetime(2026, 8, 11, 15, 55, tzinfo=local_timezone)
    candidate = MemoryCandidate(
        scope=MemoryScope(MemoryScopeKind.PROJECT, "project-4"),
        kind=MemoryKind.EXPERIENCE,
        topic="deployment.rollback",
        content="Verify the health check before completing a rollback.",
        confidence=MemoryConfidence(0.75),
        sensitivity=MemorySensitivity.NON_SENSITIVE,
        provenance=MemoryProvenance(
            source_type=MemorySourceType.COMMITTED_SESSION_TURN,
            source_item_digest="b" * 64,
            extractor_identity="sqlite-roundtrip",
            extracted_at=extracted_at,
            source_id="session-9",
            source_revision=3,
            run_id="run-12",
        ),
        valid_from=extracted_at - timedelta(days=1),
        expires_at=extracted_at + timedelta(days=30),
    )
    record = MemoryRecord.from_candidate(
        candidate,
        memory_id="memory-roundtrip",
        revision=1,
        confirmation=MemoryConfirmation(candidate.digest, extracted_at + timedelta(minutes=1)),
        created_at=extracted_at + timedelta(minutes=2),
    )

    saved = store.apply(
        MemoryChangeSet(candidate.scope, (AddMemory(record),)),
        expected_revision=0,
    )

    assert store.load(candidate.scope) == saved
    assert saved.records == (record,)
    assert saved.records[0].provenance == candidate.provenance
    assert saved.records[0].created_at.utcoffset() == timedelta(hours=8)


def test_two_sqlite_instances_share_committed_state_and_reject_stale_writer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    first_store = SqliteMemoryStore(path)
    second_store = SqliteMemoryStore(path)
    original = make_record()
    first = first_store.apply(
        MemoryChangeSet(USER_SCOPE, (AddMemory(original),)),
        expected_revision=0,
    )
    stale = second_store.load(USER_SCOPE)
    refreshed = refresh_record(original)
    second = first_store.apply(
        MemoryChangeSet(USER_SCOPE, (RefreshMemory(refreshed),)),
        expected_revision=first.revision,
    )

    with pytest.raises(MemoryConflictError, match="expected 1, found 2"):
        second_store.apply(
            MemoryChangeSet(USER_SCOPE, (RefreshMemory(refreshed),)),
            expected_revision=stale.revision,
        )

    assert second_store.load(USER_SCOPE) == second


def test_two_sqlite_instances_have_exactly_one_concurrent_cas_winner(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    stores = (SqliteMemoryStore(path), SqliteMemoryStore(path))
    barrier = Barrier(2)

    def write(item: tuple[SqliteMemoryStore, MemoryRecord]) -> object:
        store, record = item
        barrier.wait()
        try:
            return store.apply(
                MemoryChangeSet(USER_SCOPE, (AddMemory(record),)),
                expected_revision=0,
            )
        except MemoryConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(write, zip(stores, (make_record(), make_record("memory-2")), strict=True))
        )

    assert sum(isinstance(result, MemoryScopeSnapshot) for result in results) == 1
    assert sum(isinstance(result, MemoryConflictError) for result in results) == 1
    assert stores[0].load(USER_SCOPE).revision == 1
    assert len(stores[1].load(USER_SCOPE).records) == 1


def test_sqlite_load_reads_revision_and_records_from_one_transaction_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    writer = SqliteMemoryStore(path)
    original = make_record()
    first = writer.apply(
        MemoryChangeSet(USER_SCOPE, (AddMemory(original),)),
        expected_revision=0,
    )
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)

    records_query_started = Event()
    continue_read = Event()

    class PausingSqliteMemoryStore(SqliteMemoryStore):
        def _connect(self) -> sqlite3.Connection:
            connection = super()._connect()

            def pause_after_revision(statement: str) -> None:
                if "SELECT * FROM memory_records" not in statement:
                    return
                records_query_started.set()
                if not continue_read.wait(timeout=5):
                    raise RuntimeError("timed out waiting to continue snapshot read")

            connection.set_trace_callback(pause_after_revision)
            return connection

    reader = PausingSqliteMemoryStore(path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        loaded_future = executor.submit(reader.load, USER_SCOPE)
        assert records_query_started.wait(timeout=5)
        refreshed = refresh_record(original)
        second = writer.apply(
            MemoryChangeSet(USER_SCOPE, (RefreshMemory(refreshed),)),
            expected_revision=first.revision,
        )
        continue_read.set()
        loaded = loaded_future.result(timeout=5)

    assert loaded == first
    assert reader.load(USER_SCOPE) == second


def test_sqlite_forget_rolls_back_payload_delete_when_tombstone_insert_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    store = SqliteMemoryStore(path)
    original = make_record()
    saved = store.apply(
        MemoryChangeSet(USER_SCOPE, (AddMemory(original),)),
        expected_revision=0,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_tombstone BEFORE INSERT ON memory_tombstones
            BEGIN
                SELECT RAISE(ABORT, 'injected tombstone failure');
            END
            """
        )
    tombstone = MemoryTombstone.from_record(
        original,
        forgotten_at=NOW,
        reason=MemoryForgetReason.USER_REQUEST,
    )

    with pytest.raises(DQMemoryError, match="memory change violates database constraints"):
        store.apply(
            MemoryChangeSet(USER_SCOPE, (ForgetMemory(tombstone),)),
            expected_revision=saved.revision,
        )

    assert store.load(USER_SCOPE) == saved


def test_sqlite_forget_leaves_no_payload_or_provenance_in_public_tables(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    store = SqliteMemoryStore(path)
    original = make_record()
    saved = store.apply(
        MemoryChangeSet(USER_SCOPE, (AddMemory(original),)),
        expected_revision=0,
    )
    tombstone = MemoryTombstone.from_record(
        original,
        forgotten_at=NOW,
        reason=MemoryForgetReason.USER_REQUEST,
    )

    store.apply(
        MemoryChangeSet(USER_SCOPE, (ForgetMemory(tombstone),)),
        expected_revision=saved.revision,
    )

    with sqlite3.connect(path) as connection:
        records = connection.execute("SELECT * FROM memory_records").fetchall()
        tombstone_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(memory_tombstones)")
        }
    assert records == []
    assert "content" not in tombstone_columns
    assert "source_item_digest" not in tombstone_columns
    assert "extractor_identity" not in tombstone_columns


def test_sqlite_partial_unique_index_guards_active_topic_invariant(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    store = SqliteMemoryStore(path)
    store.apply(
        MemoryChangeSet(USER_SCOPE, (AddMemory(make_record()),)),
        expected_revision=0,
    )

    with sqlite3.connect(path) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO memory_records
            SELECT scope_kind, scope_id, ?, revision, kind, topic, content, confidence,
                   sensitivity, source_type, source_item_digest, extractor_identity,
                   extracted_at, source_id, source_revision, run_id, confirmation_digest,
                   confirmed_at, status, valid_from, expires_at, supersedes_id, created_at,
                   updated_at, schema_version
            FROM memory_records WHERE memory_id = ?
            """,
            ("memory-2", "memory-1"),
        )


def test_sqlite_corrupt_row_raises_stable_memory_error_without_content(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    store = SqliteMemoryStore(path)
    original = make_record()
    store.apply(
        MemoryChangeSet(USER_SCOPE, (AddMemory(original),)),
        expected_revision=0,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE memory_records SET status = ? WHERE memory_id = ?",
            ("unknown-status", original.memory_id),
        )

    with pytest.raises(DQMemoryError, match="invalid scope data") as raised:
        store.load(USER_SCOPE)

    assert original.content not in str(raised.value)
    assert str(path) not in str(raised.value)


def test_sqlite_corrupt_schema_raises_stable_memory_error(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    store = SqliteMemoryStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE memory_records RENAME COLUMN content TO payload")

    with pytest.raises(DQMemoryError, match="cannot apply memory changes") as raised:
        store.apply(
            MemoryChangeSet(USER_SCOPE, (AddMemory(make_record()),)),
            expected_revision=0,
        )

    assert str(path) not in str(raised.value)
    assert make_record().content not in str(raised.value)


def test_sqlite_initialization_error_does_not_expose_database_path(tmp_path: Path) -> None:
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied", encoding="utf-8")
    path = parent_file / "memory.sqlite3"

    with pytest.raises(DQMemoryError, match="cannot initialize memory database") as raised:
        SqliteMemoryStore(path)

    assert str(path) not in str(raised.value)
    assert raised.value.__cause__ is None
    assert str(path) not in "".join(traceback.format_exception(raised.value))


def test_sqlite_adapter_rejects_connection_local_memory_database() -> None:
    with pytest.raises(
        MemoryValidationError,
        match="must identify a durable filesystem database",
    ):
        SqliteMemoryStore(Path(":memory:"))
