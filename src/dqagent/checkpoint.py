"""Checkpoint models and stores for durable workflow execution."""

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, Protocol, cast
from uuid import uuid4

from dqagent.errors import CheckpointConflictError, CheckpointError
from dqagent.events import RunState

CHECKPOINT_SCHEMA_VERSION = 1


def _copy_json_mapping(value: Mapping[str, object], *, label: str) -> dict[str, object]:
    try:
        encoded = json.dumps(dict(value), ensure_ascii=True, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only JSON-compatible values") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], decoded)


@dataclass(frozen=True, slots=True)
class WorkflowCheckpoint:
    workflow_id: str
    definition_id: str
    definition_version: str
    initial_state: Mapping[str, object]
    state: Mapping[str, object]
    current_node: str | None
    completed_nodes: tuple[str, ...]
    status: RunState
    revision: int = 0
    updated_at: datetime | None = None
    last_error: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("workflow ID", self.workflow_id),
            ("definition ID", self.definition_id),
            ("definition version", self.definition_version),
        ):
            if not value.strip():
                raise ValueError(f"{label} must not be empty")
        if self.current_node is not None and not self.current_node.strip():
            raise ValueError("current node must not be empty")
        if self.revision < 0:
            raise ValueError("checkpoint revision must be non-negative")
        if self.status is RunState.COMPLETED and self.current_node is not None:
            raise ValueError("completed checkpoint cannot have a current node")
        if self.status is not RunState.COMPLETED and self.current_node is None:
            raise ValueError("non-completed checkpoint must have a current node")
        if self.updated_at is not None and self.updated_at.tzinfo is None:
            raise ValueError("checkpoint timestamp must be timezone-aware")
        initial_state = _copy_json_mapping(self.initial_state, label="initial state")
        state = _copy_json_mapping(self.state, label="workflow state")
        error = (
            _copy_json_mapping(self.last_error, label="checkpoint error")
            if self.last_error is not None
            else None
        )
        object.__setattr__(self, "initial_state", MappingProxyType(initial_state))
        object.__setattr__(self, "state", MappingProxyType(state))
        object.__setattr__(
            self,
            "last_error",
            MappingProxyType(error) if error is not None else None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "workflow_id": self.workflow_id,
            "definition_id": self.definition_id,
            "definition_version": self.definition_version,
            "initial_state": dict(self.initial_state),
            "state": dict(self.state),
            "current_node": self.current_node,
            "completed_nodes": list(self.completed_nodes),
            "status": self.status.value,
            "revision": self.revision,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_error": dict(self.last_error) if self.last_error is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "WorkflowCheckpoint":
        if data.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError(
                f"unsupported checkpoint schema version: {data.get('schema_version')!r}"
            )
        try:
            raw_completed = data["completed_nodes"]
            raw_updated = data["updated_at"]
            if not isinstance(raw_completed, list) or not all(
                isinstance(item, str) for item in raw_completed
            ):
                raise TypeError("completed_nodes must be an array of strings")
            if raw_updated is not None and not isinstance(raw_updated, str):
                raise TypeError("updated_at must be a string or null")
            initial_state = cast(Mapping[str, object], data["initial_state"])
            state = cast(Mapping[str, object], data["state"])
            last_error = cast(Mapping[str, object] | None, data.get("last_error"))
            return cls(
                workflow_id=cast(str, data["workflow_id"]),
                definition_id=cast(str, data["definition_id"]),
                definition_version=cast(str, data["definition_version"]),
                initial_state=initial_state,
                state=state,
                current_node=cast(str | None, data["current_node"]),
                completed_nodes=tuple(raw_completed),
                status=RunState(cast(str, data["status"])),
                revision=cast(int, data["revision"]),
                updated_at=datetime.fromisoformat(raw_updated) if raw_updated else None,
                last_error=last_error,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"invalid workflow checkpoint: {exc}") from exc


class CheckpointStore(Protocol):
    """Compare-and-swap persistence boundary for one workflow checkpoint."""

    def load(self, workflow_id: str) -> WorkflowCheckpoint | None: ...

    def save(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        expected_revision: int | None,
    ) -> WorkflowCheckpoint: ...


def _next_checkpoint(
    checkpoint: WorkflowCheckpoint,
    current: WorkflowCheckpoint | None,
    expected_revision: int | None,
) -> WorkflowCheckpoint:
    actual_revision = current.revision if current is not None else None
    if actual_revision != expected_revision:
        raise CheckpointConflictError(
            f"checkpoint revision conflict for workflow '{checkpoint.workflow_id}': "
            f"expected {expected_revision!r}, found {actual_revision!r}"
        )
    return replace(
        checkpoint,
        revision=1 if current is None else current.revision + 1,
        updated_at=datetime.now(UTC),
    )


class InMemoryCheckpointStore:
    """Thread-safe process-local store for tests and ephemeral workflows."""

    def __init__(self) -> None:
        self._items: dict[str, WorkflowCheckpoint] = {}
        self._lock = Lock()

    def load(self, workflow_id: str) -> WorkflowCheckpoint | None:
        with self._lock:
            checkpoint = self._items.get(workflow_id)
            return self._clone(checkpoint) if checkpoint is not None else None

    def save(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        expected_revision: int | None,
    ) -> WorkflowCheckpoint:
        with self._lock:
            saved = _next_checkpoint(
                checkpoint,
                self._items.get(checkpoint.workflow_id),
                expected_revision,
            )
            self._items[checkpoint.workflow_id] = saved
            return self._clone(saved)

    @staticmethod
    def _clone(checkpoint: WorkflowCheckpoint) -> WorkflowCheckpoint:
        return WorkflowCheckpoint.from_dict(checkpoint.to_dict())


class JsonFileCheckpointStore:
    """Single-process JSON store with atomic file replacement."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._lock = Lock()

    def load(self, workflow_id: str) -> WorkflowCheckpoint | None:
        with self._lock:
            return self._load_unlocked(workflow_id)

    def save(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        expected_revision: int | None,
    ) -> WorkflowCheckpoint:
        with self._lock:
            current = self._load_unlocked(checkpoint.workflow_id)
            saved = _next_checkpoint(checkpoint, current, expected_revision)
            path = self._path_for(checkpoint.workflow_id)
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                self._directory.mkdir(parents=True, exist_ok=True)
                rendered = json.dumps(
                    saved.to_dict(),
                    indent=2,
                    ensure_ascii=True,
                    allow_nan=False,
                ) + "\n"
                temporary.write_text(rendered, encoding="utf-8")
                os.replace(temporary, path)
            except OSError as exc:
                raise CheckpointError(
                    f"cannot save checkpoint for workflow '{checkpoint.workflow_id}': {exc}"
                ) from exc
            finally:
                temporary.unlink(missing_ok=True)
            return WorkflowCheckpoint.from_dict(saved.to_dict())

    def _load_unlocked(self, workflow_id: str) -> WorkflowCheckpoint | None:
        path = self._path_for(workflow_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(
                f"cannot load checkpoint for workflow '{workflow_id}': {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise CheckpointError(
                f"checkpoint for workflow '{workflow_id}' must be a JSON object"
            )
        checkpoint = WorkflowCheckpoint.from_dict(cast(dict[str, Any], raw))
        if checkpoint.workflow_id != workflow_id:
            raise CheckpointError(
                f"checkpoint identity mismatch for workflow '{workflow_id}'"
            )
        return checkpoint

    def _path_for(self, workflow_id: str) -> Path:
        if not workflow_id.strip():
            raise ValueError("workflow ID must not be empty")
        digest = hashlib.sha256(workflow_id.encode("utf-8")).hexdigest()
        return self._directory / f"{digest}.json"
