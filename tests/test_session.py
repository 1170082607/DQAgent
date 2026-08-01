import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from dqagent.errors import SessionConflictError, SessionError
from dqagent.models import Message, Role, ToolCall, ToolErrorCode, ToolOutcome, ToolResult
from dqagent.session import InMemorySessionStore, JsonFileSessionStore, SessionSnapshot


def make_snapshot(session_id: str = "session-1") -> SessionSnapshot:
    return SessionSnapshot(
        session_id,
        (
            Message(Role.USER, "Use the tool"),
            ToolCall("call-1", "example", '{"value":1}'),
            ToolResult(
                "call-1",
                "example",
                "failed safely",
                ToolOutcome.ERROR,
                ToolErrorCode.EXECUTION_ERROR,
            ),
            Message(Role.ASSISTANT, "The tool failed."),
        ),
    )


def test_session_snapshot_round_trips_every_conversation_item() -> None:
    snapshot = make_snapshot()

    restored = SessionSnapshot.from_dict(snapshot.to_dict())

    assert restored == snapshot


def test_in_memory_session_store_uses_compare_and_swap_revisions() -> None:
    store = InMemorySessionStore()
    first = store.save(make_snapshot(), expected_revision=None)
    second = store.save(
        replace(
            first,
            transcript=(
                *first.transcript,
                Message(Role.USER, "Next"),
                Message(Role.ASSISTANT, "Done"),
            ),
        ),
        expected_revision=first.revision,
    )

    assert first.revision == 1
    assert second.revision == 2
    assert first.created_at == second.created_at
    assert second.updated_at is not None

    with pytest.raises(SessionConflictError, match="expected 1, found 2"):
        store.save(first, expected_revision=first.revision)


def test_json_store_persists_atomically_with_hashed_session_name(tmp_path) -> None:
    store = JsonFileSessionStore(tmp_path)
    saved = store.save(make_snapshot("customer/unsafe"), expected_revision=None)

    loaded = JsonFileSessionStore(tmp_path).load("customer/unsafe")

    assert loaded == saved
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].name != "customer/unsafe.json"
    assert json.loads(files[0].read_text(encoding="utf-8"))["revision"] == 1


def test_json_store_rejects_corrupt_or_mismatched_session(tmp_path) -> None:
    store = JsonFileSessionStore(tmp_path)
    store.save(make_snapshot(), expected_revision=None)
    path = next(tmp_path.glob("*.json"))
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(SessionError, match="cannot load session"):
        store.load("session-1")

    path.write_text(
        json.dumps(SessionSnapshot("other").to_dict()),
        encoding="utf-8",
    )
    with pytest.raises(SessionError, match="identity mismatch"):
        store.load("session-1")


def test_session_snapshot_rejects_unknown_schema_and_naive_timestamp() -> None:
    data = make_snapshot().to_dict()
    data["schema_version"] = 99
    with pytest.raises(SessionError, match="unsupported session schema"):
        SessionSnapshot.from_dict(data)

    with pytest.raises(ValueError, match="timezone-aware"):
        replace(make_snapshot(), created_at=datetime.now())


def test_session_snapshot_rejects_incomplete_durable_turn() -> None:
    with pytest.raises(ValueError, match="must end with an assistant message"):
        SessionSnapshot("incomplete", (Message(Role.USER, "not committed"),))


def test_json_store_cleanup_does_not_mask_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_write(*args: object, **kwargs: object) -> int:
        del args, kwargs
        raise OSError("write failed")

    def fail_cleanup(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("cleanup failed")

    monkeypatch.setattr(Path, "write_text", fail_write)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)

    with pytest.raises(SessionError, match="write failed"):
        JsonFileSessionStore(tmp_path).save(make_snapshot(), expected_revision=None)
