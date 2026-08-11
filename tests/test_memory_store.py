"""Reusable contract tests for transactional MemoryStore adapters."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from dqagent.errors import (
    MemoryConflictError,
    MemoryCorruptChangeError,
    MemoryNotFoundError,
)
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


def test_in_memory_adapter_satisfies_memory_store_protocol() -> None:
    store: MemoryStore = InMemoryMemoryStore()

    assert store.load(USER_SCOPE) == MemoryScopeSnapshot(USER_SCOPE)
