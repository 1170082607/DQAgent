import json
from dataclasses import replace

import pytest

from dqagent.checkpoint import (
    InMemoryCheckpointStore,
    JsonFileCheckpointStore,
    WorkflowCheckpoint,
)
from dqagent.errors import CheckpointConflictError, CheckpointError
from dqagent.events import RunState


def make_checkpoint(workflow_id: str = "workflow-1") -> WorkflowCheckpoint:
    return WorkflowCheckpoint(
        workflow_id=workflow_id,
        definition_id="example",
        definition_version="1",
        initial_state={"input": 2},
        state={"input": 2},
        current_node="start",
        completed_nodes=(),
        status=RunState.RUNNING,
    )


def test_in_memory_store_uses_compare_and_swap_revisions() -> None:
    store = InMemoryCheckpointStore()
    first = store.save(make_checkpoint(), expected_revision=None)
    second = store.save(
        replace(first, state={"input": 2, "prepared": True}),
        expected_revision=first.revision,
    )

    assert first.revision == 1
    assert second.revision == 2
    assert second.updated_at is not None
    assert store.load("workflow-1") == second

    with pytest.raises(CheckpointConflictError, match="expected 1, found 2"):
        store.save(first, expected_revision=first.revision)


def test_json_store_persists_atomically_without_using_workflow_id_as_path(tmp_path) -> None:
    store = JsonFileCheckpointStore(tmp_path)
    saved = store.save(make_checkpoint("customer/unsafe"), expected_revision=None)

    loaded = JsonFileCheckpointStore(tmp_path).load("customer/unsafe")

    assert loaded == saved
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].name != "customer/unsafe.json"
    assert json.loads(files[0].read_text(encoding="utf-8"))["revision"] == 1


def test_json_store_rejects_corrupt_checkpoint(tmp_path) -> None:
    store = JsonFileCheckpointStore(tmp_path)
    store.save(make_checkpoint(), expected_revision=None)
    path = next(tmp_path.glob("*.json"))
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(CheckpointError, match="cannot load checkpoint"):
        store.load("workflow-1")


def test_checkpoint_requires_json_state_and_consistent_terminal_position() -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        replace(make_checkpoint(), state={"bad": object()})

    with pytest.raises(ValueError, match="completed checkpoint"):
        replace(make_checkpoint(), status=RunState.COMPLETED)

    with pytest.raises(ValueError, match="non-completed checkpoint"):
        replace(make_checkpoint(), current_node=None)
