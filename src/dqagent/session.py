"""Durable session transcripts with explicit optimistic concurrency."""

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, cast
from uuid import uuid4

from dqagent.errors import SessionConflictError, SessionError
from dqagent.models import (
    ConversationItem,
    Message,
    Role,
    ToolCall,
    ToolErrorCode,
    ToolOutcome,
    ToolResult,
)

SESSION_SCHEMA_VERSION = 1
_JSON_STORE_LOCKS: dict[Path, Lock] = {}
_JSON_STORE_LOCKS_GUARD = Lock()


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """One immutable revision of a complete durable conversation transcript."""

    session_id: str
    transcript: tuple[ConversationItem, ...] = ()
    revision: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session ID must not be empty")
        if self.revision < 0:
            raise ValueError("session revision must be non-negative")
        for label, timestamp in (("created", self.created_at), ("updated", self.updated_at)):
            if timestamp is not None and timestamp.tzinfo is None:
                raise ValueError(f"session {label} timestamp must be timezone-aware")
        # Round-tripping validates every item before it reaches a persistence adapter.
        tuple(_item_from_dict(_item_to_dict(item)) for item in self.transcript)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "transcript": [_item_to_dict(item) for item in self.transcript],
            "revision": self.revision,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SessionSnapshot":
        if data.get("schema_version") != SESSION_SCHEMA_VERSION:
            raise SessionError(
                f"unsupported session schema version: {data.get('schema_version')!r}"
            )
        try:
            raw_transcript = data["transcript"]
            raw_created = data["created_at"]
            raw_updated = data["updated_at"]
            if not isinstance(raw_transcript, list):
                raise TypeError("transcript must be an array")
            if raw_created is not None and not isinstance(raw_created, str):
                raise TypeError("created_at must be a string or null")
            if raw_updated is not None and not isinstance(raw_updated, str):
                raise TypeError("updated_at must be a string or null")
            return cls(
                session_id=cast(str, data["session_id"]),
                transcript=tuple(
                    _item_from_dict(cast(Mapping[str, object], item))
                    for item in raw_transcript
                ),
                revision=cast(int, data["revision"]),
                created_at=datetime.fromisoformat(raw_created) if raw_created else None,
                updated_at=datetime.fromisoformat(raw_updated) if raw_updated else None,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise SessionError(f"invalid session snapshot: {exc}") from exc


class SessionStore(Protocol):
    """Compare-and-swap persistence boundary for durable transcripts."""

    def load(self, session_id: str) -> SessionSnapshot | None: ...

    def save(
        self,
        snapshot: SessionSnapshot,
        *,
        expected_revision: int | None,
    ) -> SessionSnapshot: ...


def _next_snapshot(
    snapshot: SessionSnapshot,
    current: SessionSnapshot | None,
    expected_revision: int | None,
) -> SessionSnapshot:
    actual_revision = current.revision if current is not None else None
    if actual_revision != expected_revision:
        raise SessionConflictError(
            f"session revision conflict for '{snapshot.session_id}': "
            f"expected {expected_revision!r}, found {actual_revision!r}"
        )
    now = datetime.now(UTC)
    return replace(
        snapshot,
        revision=1 if current is None else current.revision + 1,
        created_at=current.created_at if current is not None else now,
        updated_at=now,
    )


class InMemorySessionStore:
    """Thread-safe process-local session store."""

    def __init__(self) -> None:
        self._items: dict[str, SessionSnapshot] = {}
        self._lock = Lock()

    def load(self, session_id: str) -> SessionSnapshot | None:
        _validate_session_id(session_id)
        with self._lock:
            snapshot = self._items.get(session_id)
            return self._clone(snapshot) if snapshot is not None else None

    def save(
        self,
        snapshot: SessionSnapshot,
        *,
        expected_revision: int | None,
    ) -> SessionSnapshot:
        with self._lock:
            saved = _next_snapshot(
                snapshot,
                self._items.get(snapshot.session_id),
                expected_revision,
            )
            self._items[snapshot.session_id] = saved
            return self._clone(saved)

    @staticmethod
    def _clone(snapshot: SessionSnapshot) -> SessionSnapshot:
        return SessionSnapshot.from_dict(snapshot.to_dict())


class JsonFileSessionStore:
    """Single-process JSON session store with hashed names and atomic replacement."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory.resolve()
        with _JSON_STORE_LOCKS_GUARD:
            self._lock = _JSON_STORE_LOCKS.setdefault(self._directory, Lock())

    def load(self, session_id: str) -> SessionSnapshot | None:
        with self._lock:
            return self._load_unlocked(session_id)

    def save(
        self,
        snapshot: SessionSnapshot,
        *,
        expected_revision: int | None,
    ) -> SessionSnapshot:
        with self._lock:
            current = self._load_unlocked(snapshot.session_id)
            saved = _next_snapshot(snapshot, current, expected_revision)
            path = self._path_for(snapshot.session_id)
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                self._directory.mkdir(parents=True, exist_ok=True)
                rendered = json.dumps(
                    saved.to_dict(), indent=2, ensure_ascii=True, allow_nan=False
                ) + "\n"
                temporary.write_text(rendered, encoding="utf-8")
                os.replace(temporary, path)
            except OSError as exc:
                raise SessionError(
                    f"cannot save session '{snapshot.session_id}': {exc}"
                ) from exc
            finally:
                temporary.unlink(missing_ok=True)
            return SessionSnapshot.from_dict(saved.to_dict())

    def _load_unlocked(self, session_id: str) -> SessionSnapshot | None:
        path = self._path_for(session_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SessionError(f"cannot load session '{session_id}': {exc}") from exc
        if not isinstance(raw, dict):
            raise SessionError(f"session '{session_id}' must be a JSON object")
        snapshot = SessionSnapshot.from_dict(cast(dict[str, Any], raw))
        if snapshot.session_id != session_id:
            raise SessionError(f"session identity mismatch for '{session_id}'")
        return snapshot

    def _path_for(self, session_id: str) -> Path:
        _validate_session_id(session_id)
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self._directory / f"{digest}.json"


def _validate_session_id(session_id: str) -> None:
    if not session_id.strip():
        raise ValueError("session ID must not be empty")


def _item_to_dict(item: ConversationItem) -> dict[str, object]:
    if isinstance(item, Message):
        return {"kind": "message", "role": item.role.value, "content": item.content}
    if isinstance(item, ToolCall):
        return {
            "kind": "tool_call",
            "call_id": item.call_id,
            "name": item.name,
            "arguments": item.arguments,
        }
    if isinstance(item, ToolResult):
        return {
            "kind": "tool_result",
            "call_id": item.call_id,
            "name": item.name,
            "output": item.output,
            "outcome": item.outcome.value,
            "error_code": item.error_code.value if item.error_code else None,
        }
    raise TypeError(f"unsupported conversation item: {type(item).__name__}")


def _item_from_dict(data: Mapping[str, object]) -> ConversationItem:
    kind = data.get("kind")
    if kind == "message":
        return Message(Role(cast(str, data["role"])), cast(str, data["content"]))
    if kind == "tool_call":
        return ToolCall(
            cast(str, data["call_id"]),
            cast(str, data["name"]),
            cast(str, data["arguments"]),
        )
    if kind == "tool_result":
        raw_error = data.get("error_code")
        return ToolResult(
            call_id=cast(str, data["call_id"]),
            name=cast(str, data["name"]),
            output=cast(str, data["output"]),
            outcome=ToolOutcome(cast(str, data["outcome"])),
            error_code=(ToolErrorCode(cast(str, raw_error)) if raw_error is not None else None),
        )
    raise ValueError(f"unsupported transcript item kind: {kind!r}")
