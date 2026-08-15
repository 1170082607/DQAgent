from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import dqagent.coding_tools as coding_tools_module
from dqagent.coding_tools import (
    WORKSPACE_PATCH_SCHEMA,
    CodingToolLimits,
    create_coding_tools,
    create_workspace_patch_tool,
)
from dqagent.execution import RunContext
from dqagent.models import ToolCall, ToolErrorCode, ToolOutcome
from dqagent.tool_governance import (
    ApprovalDecision,
    ApprovalOutcome,
    EffectState,
    HookOutcome,
    HookResult,
    ScriptedApprovalProvider,
)
from dqagent.tools import (
    ActionPreparationError,
    ToolExecution,
    ToolExecutionContext,
    ToolRegistry,
    _RunActionRecordCollector,
)
from dqagent.workspace import Workspace, WorkspaceObserver, WorkspaceScope


def make_workspace(tmp_path: Path) -> Workspace:
    return Workspace(WorkspaceScope("fixture", tmp_path))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeMonotonic:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def invoke(
    tool: Any,
    arguments: dict[str, object],
    *,
    run_id: str = "run-t7",
) -> tuple[ToolExecution, tuple[Any, ...]]:
    context = RunContext(run_id=run_id)
    collector = _RunActionRecordCollector(run_id, 1)
    execution_context = ToolExecutionContext(context, record_collector=collector)
    execution = ToolRegistry((tool,)).execute_detailed(
        ToolCall(
            "call-t7",
            tool.definition.name,
            json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        ),
        context,
        execution_context=execution_context,
    )
    return execution, collector.records


def approved_tool(
    workspace: Workspace,
    *,
    provider: ScriptedApprovalProvider | None = None,
    limits: CodingToolLimits | None = None,
    **kwargs: object,
) -> tuple[Any, ScriptedApprovalProvider]:
    selected_provider = provider or ScriptedApprovalProvider([ApprovalOutcome.APPROVE])
    return (
        create_workspace_patch_tool(
            workspace,
            approval_provider=selected_provider,
            limits=limits,
            **kwargs,
        ),
        selected_provider,
    )


def test_t7_schema_is_closed_and_operation_specific() -> None:
    assert WORKSPACE_PATCH_SCHEMA["additionalProperties"] is False
    assert set(WORKSPACE_PATCH_SCHEMA["required"]) == {"operation", "path"}
    replacements = WORKSPACE_PATCH_SCHEMA["properties"]["replacements"]
    assert replacements["items"]["additionalProperties"] is False


def test_patch_preparation_validates_digest_and_occurrences_without_effects(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("one one", encoding="utf-8")
    workspace = make_workspace(tmp_path)
    tool, provider = approved_tool(workspace)
    context = RunContext(run_id="prepare-t7")
    tool_context = ToolExecutionContext(
        context,
        record_collector=_RunActionRecordCollector("prepare-t7", 1),
    )

    with pytest.raises(ActionPreparationError) as stale:
        tool.prepare(
            {
                "operation": "update",
                "path": "target.txt",
                "expected_sha256": "0" * 64,
                "replacements": [
                    {"old": "one", "new": "two", "expected_occurrences": 2}
                ],
            },
            tool_context,
        )
    assert stale.value.code is ToolErrorCode.PRECONDITION_CONFLICT

    with pytest.raises(ActionPreparationError) as occurrence:
        tool.prepare(
            {
                "operation": "update",
                "path": "target.txt",
                "expected_sha256": digest(target),
                "replacements": [
                    {"old": "one", "new": "two", "expected_occurrences": 1}
                ],
            },
            tool_context,
        )
    assert occurrence.value.code is ToolErrorCode.PRECONDITION_CONFLICT
    assert target.read_text(encoding="utf-8") == "one one"
    assert provider.calls == 0


def test_create_update_delete_are_single_file_governed_effects_and_keep_exact_diff(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    workspace = make_workspace(tmp_path)
    observer = WorkspaceObserver(workspace)

    create_tool, create_provider = approved_tool(workspace)
    created, created_records = invoke(
        create_tool,
        {"operation": "create", "path": "target.txt", "content": "one\n"},
        run_id="create-t7",
    )
    assert created.result.outcome is ToolOutcome.SUCCESS
    assert create_provider.calls == 1
    assert target.read_bytes() == b"one\n"
    assert created.action_record is not None
    assert created.action_record.effect_state is EffectState.COMPLETE
    assert created.action_record.executor_attempts == 1
    assert created_records == (created.action_record,)
    created_baseline = observer.capture(target_paths=("target.txt",))

    update_tool, update_provider = approved_tool(workspace)
    update_baseline = observer.capture(target_paths=("target.txt",))
    updated, _ = invoke(
        update_tool,
        {
            "operation": "update",
            "path": "target.txt",
            "expected_sha256": digest(target),
            "replacements": [
                {"old": "one", "new": "two", "expected_occurrences": 1},
                {"old": "two", "new": "three", "expected_occurrences": 1},
            ],
        },
        run_id="update-t7",
    )
    assert updated.result.outcome is ToolOutcome.SUCCESS
    assert update_provider.calls == 1
    assert target.read_bytes() == b"three\n"
    assert updated.action_record is not None
    assert updated.action_record.effect_state is EffectState.COMPLETE
    assert json.loads(updated.result.output)["status"] == "updated"

    update_final = observer.capture(target_paths=("target.txt",))
    update_diff = observer.diff(update_baseline, update_final, target_paths=("target.txt",))
    assert update_diff.rendered_diff == (
        "--- a/target.txt\n+++ b/target.txt\n@@ -1,2 +1,2 @@\n-one\n+three\n "
    )

    delete_tool, delete_provider = approved_tool(workspace)
    deleted, _ = invoke(
        delete_tool,
        {
            "operation": "delete",
            "path": "target.txt",
            "expected_sha256": digest(target),
        },
        run_id="delete-t7",
    )
    assert deleted.result.outcome is ToolOutcome.SUCCESS
    assert delete_provider.calls == 1
    assert not target.exists()
    assert deleted.action_record is not None
    assert deleted.action_record.effect_state is EffectState.COMPLETE

    final = observer.capture(target_paths=("target.txt",))
    diff = observer.diff(created_baseline, final, target_paths=("target.txt",))
    assert [item.logical_path.as_posix() for item in diff.changes] == ["target.txt"]
    assert diff.changes[0].kind.value == "delete"


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
def test_patch_elapsed_limit_never_reports_clean_success_after_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    target = tmp_path / "target.txt"
    if operation != "create":
        target.write_text("old", encoding="utf-8")
    workspace = make_workspace(tmp_path)
    clock = FakeMonotonic()
    monkeypatch.setattr(coding_tools_module.time, "monotonic", clock)

    if operation == "create":
        original_fdopen = coding_tools_module.os.fdopen

        def delayed_fdopen(fd: int, mode: str):
            handle = original_fdopen(fd, mode)
            clock.advance(0.01)
            return handle

        monkeypatch.setattr(coding_tools_module.os, "fdopen", delayed_fdopen)
        arguments = {"operation": "create", "path": "target.txt", "content": "created"}
    elif operation == "update":
        original_replace = coding_tools_module.os.replace

        def delayed_replace(source: object, destination: object) -> None:
            original_replace(source, destination)
            clock.advance(0.01)

        monkeypatch.setattr(coding_tools_module.os, "replace", delayed_replace)
        arguments = {
            "operation": "update",
            "path": "target.txt",
            "expected_sha256": digest(target),
            "replacements": [
                {"old": "old", "new": "new", "expected_occurrences": 1}
            ],
        }
    else:
        original_remove = coding_tools_module.os.remove

        def delayed_remove(path: object) -> None:
            original_remove(path)
            clock.advance(0.01)

        monkeypatch.setattr(coding_tools_module.os, "remove", delayed_remove)
        arguments = {
            "operation": "delete",
            "path": "target.txt",
            "expected_sha256": digest(target),
        }

    tool, _ = approved_tool(
        workspace,
        limits=CodingToolLimits(max_patch_elapsed_seconds=0.001),
    )
    execution, _ = invoke(tool, arguments, run_id=f"elapsed-{operation}-t7")

    assert execution.result.outcome is ToolOutcome.ERROR
    assert execution.result.error_code is ToolErrorCode.RESOURCE_LIMIT
    assert execution.action_record is not None
    assert "patch_elapsed_limit" in execution.action_record.diagnostics
    if operation == "create":
        assert execution.action_record.effect_state is EffectState.PARTIAL
        assert target.exists()
    elif operation == "update":
        assert execution.action_record.effect_state is EffectState.UNKNOWN
        assert target.read_text(encoding="utf-8") == "new"
    else:
        assert execution.action_record.effect_state is EffectState.UNKNOWN
        assert not target.exists()


def test_patch_elapsed_limit_blocks_preparation_before_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    workspace = make_workspace(tmp_path)
    clock = FakeMonotonic()
    monkeypatch.setattr(coding_tools_module.time, "monotonic", clock)
    original_normalize = coding_tools_module._normalize_patch_replacements

    def delayed_normalize(*args: object, **kwargs: object):
        clock.advance(1.0)
        return original_normalize(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        coding_tools_module,
        "_normalize_patch_replacements",
        delayed_normalize,
    )
    tool, provider = approved_tool(
        workspace,
        limits=CodingToolLimits(max_patch_elapsed_seconds=0.001),
    )

    execution, _ = invoke(
        tool,
        {
            "operation": "update",
            "path": "target.txt",
            "expected_sha256": digest(target),
            "replacements": [
                {"old": "old", "new": "new", "expected_occurrences": 1}
            ],
        },
        run_id="elapsed-preparation-t7",
    )

    assert execution.result.outcome is ToolOutcome.ERROR
    assert execution.result.error_code is ToolErrorCode.RESOURCE_LIMIT
    assert target.read_text(encoding="utf-8") == "old"
    assert provider.calls == 0
    assert execution.action_record is None


def test_nested_rejected_patch_restores_outer_elapsed_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer_target = tmp_path / "outer.txt"
    inner_target = tmp_path / "inner.txt"
    outer_target.write_text("old", encoding="utf-8")
    inner_target.write_text("old", encoding="utf-8")
    workspace = make_workspace(tmp_path)
    clock = FakeMonotonic()
    monkeypatch.setattr(coding_tools_module.time, "monotonic", clock)
    limits = CodingToolLimits(max_patch_elapsed_seconds=0.001)
    run_context = RunContext(run_id="nested-budget-t7")
    execution_context = ToolExecutionContext(
        run_context,
        record_collector=_RunActionRecordCollector("nested-budget-t7", 2),
    )
    registry: ToolRegistry
    tool: Any
    inner_execution: ToolExecution | None = None

    def approve_outer(request, _context):
        nonlocal inner_execution
        inner_execution = registry.execute_detailed(
            ToolCall(
                "inner-t7",
                tool.definition.name,
                json.dumps(
                    {
                        "operation": "update",
                        "path": "inner.txt",
                        "expected_sha256": digest(inner_target),
                        "replacements": [
                            {"old": "old", "new": "inner", "expected_occurrences": 1}
                        ],
                    },
                    separators=(",", ":"),
                ),
            ),
            run_context,
            execution_context=execution_context,
        )
        clock.advance(1.0)
        return ApprovalDecision.approve_for(request)

    provider = ScriptedApprovalProvider([approve_outer, ApprovalOutcome.REJECT])
    tool = create_workspace_patch_tool(
        workspace,
        approval_provider=provider,
        limits=limits,
        max_governed_calls=2,
    )
    registry = ToolRegistry((tool,))
    execution = registry.execute_detailed(
        ToolCall(
            "outer-t7",
            tool.definition.name,
            json.dumps(
                {
                    "operation": "update",
                    "path": "outer.txt",
                    "expected_sha256": digest(outer_target),
                    "replacements": [
                        {"old": "old", "new": "outer", "expected_occurrences": 1}
                    ],
                },
                separators=(",", ":"),
            ),
        ),
        run_context,
        execution_context=execution_context,
    )

    assert inner_execution is not None
    assert inner_execution.result.outcome is ToolOutcome.ERROR
    assert inner_execution.result.error_code is ToolErrorCode.APPROVAL_REJECTED
    assert inner_execution.action_record is not None
    assert inner_execution.action_record.executor_attempts == 0
    assert inner_execution.action_record.effect_state is EffectState.NONE
    assert execution.result.outcome is ToolOutcome.ERROR
    assert execution.result.error_code is ToolErrorCode.PRECONDITION_CONFLICT
    assert execution.action_record is not None
    assert execution.action_record.executor_attempts == 0
    assert execution.action_record.effect_state is EffectState.NONE
    assert outer_target.read_text(encoding="utf-8") == "old"
    assert inner_target.read_text(encoding="utf-8") == "old"
    assert coding_tools_module._PENDING_PATCH_BUDGET.get() == ()


def test_create_race_is_exclusive_and_does_not_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "race.txt"
    workspace = make_workspace(tmp_path)
    tool, provider = approved_tool(workspace)
    original_open = coding_tools_module.os.open

    def racing_open(*args: object, **kwargs: object) -> int:
        candidate = Path(args[0])
        if candidate == target:
            target.write_text("racer", encoding="utf-8")
        return original_open(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(coding_tools_module.os, "open", racing_open)
    execution, _ = invoke(
        tool,
        {"operation": "create", "path": "race.txt", "content": "agent"},
        run_id="race-t7",
    )

    assert execution.result.outcome is ToolOutcome.ERROR
    assert execution.result.error_code is ToolErrorCode.PRECONDITION_CONFLICT
    assert target.read_text(encoding="utf-8") == "racer"
    assert provider.calls == 1
    assert execution.action_record is not None
    assert execution.action_record.executor_attempts == 1
    assert execution.action_record.effect_state is EffectState.NONE


def test_stale_target_after_approval_is_not_executed_or_reused(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    workspace = make_workspace(tmp_path)
    requests: list[str] = []

    def approve_then_drift(request, _context):
        requests.append(request.action_digest)
        target.write_text("external", encoding="utf-8")
        return ApprovalDecision.approve_for(request)

    provider = ScriptedApprovalProvider([approve_then_drift, ApprovalOutcome.APPROVE])
    tool, _ = approved_tool(workspace, provider=provider)
    execution, _ = invoke(
        tool,
        {
            "operation": "update",
            "path": "target.txt",
            "expected_sha256": digest(target),
            "replacements": [
                {"old": "old", "new": "new", "expected_occurrences": 1}
            ],
        },
        run_id="drift-t7",
    )

    assert execution.result.outcome is ToolOutcome.ERROR
    assert execution.result.error_code in {
        ToolErrorCode.APPROVAL_MISMATCH,
        ToolErrorCode.PRECONDITION_CONFLICT,
    }
    assert requests
    assert target.read_text(encoding="utf-8") == "external"
    assert execution.action_record is not None
    assert execution.action_record.executor_attempts == 0
    assert execution.action_record.effect_state is EffectState.NONE

    target.write_text("old", encoding="utf-8")
    rerun, _ = invoke(
        tool,
        {
            "operation": "update",
            "path": "target.txt",
            "expected_sha256": digest(target),
            "replacements": [
                {"old": "old", "new": "new", "expected_occurrences": 1}
            ],
        },
        run_id="drift-rerun-t7",
    )
    assert rerun.result.outcome is ToolOutcome.SUCCESS
    assert target.read_text(encoding="utf-8") == "new"
    assert provider.calls == 2
    assert rerun.action_record is not None
    assert rerun.action_record.effect_state is EffectState.COMPLETE


def test_post_hook_failure_keeps_patch_effect_visible(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    workspace = make_workspace(tmp_path)

    class FailingPostHook:
        identity = "t7-failing-post-hook"

        def __call__(self, _input) -> HookResult:
            return HookResult(HookOutcome.FAILED, "post hook failed", self.identity)

    tool, _ = approved_tool(workspace, post_hooks=(FailingPostHook(),))
    execution, _ = invoke(
        tool,
        {
            "operation": "update",
            "path": "target.txt",
            "expected_sha256": digest(target),
            "replacements": [
                {"old": "old", "new": "new", "expected_occurrences": 1}
            ],
        },
        run_id="post-hook-t7",
    )

    assert execution.result.outcome is ToolOutcome.SUCCESS
    assert target.read_text(encoding="utf-8") == "new"
    assert execution.action_record is not None
    assert execution.action_record.effect_state is EffectState.COMPLETE
    assert execution.action_record.post_hook_results[0].outcome is HookOutcome.FAILED


@pytest.mark.parametrize("failure", ["write", "close"])
def test_create_write_and_close_failures_report_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    target = tmp_path / "created.txt"
    workspace = make_workspace(tmp_path)
    tool, provider = approved_tool(workspace)
    original_fdopen = coding_tools_module.os.fdopen

    class FailingHandle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def write(self, content: bytes) -> int:
            if failure == "write":
                self._handle.write(content[:1])
                raise OSError("create write unavailable")
            return self._handle.write(content)

        def close(self) -> None:
            self._handle.close()
            if failure == "close":
                raise OSError("create close unavailable")

    monkeypatch.setattr(
        coding_tools_module.os,
        "fdopen",
        lambda fd, mode: FailingHandle(original_fdopen(fd, mode)),
    )

    execution, _ = invoke(
        tool,
        {"operation": "create", "path": "created.txt", "content": "agent"},
        run_id=f"create-{failure}-t7",
    )

    assert execution.result.outcome is ToolOutcome.ERROR
    assert execution.result.error_code is ToolErrorCode.EXECUTION_ERROR
    assert provider.calls == 1
    assert target.exists()
    if failure == "write":
        assert target.read_bytes() == b"a"
        expected_diagnostic = "patch_create_write_failed"
    else:
        assert target.read_bytes() == b"agent"
        expected_diagnostic = "patch_create_close_failed"
    assert execution.action_record is not None
    assert execution.action_record.effect_state is EffectState.PARTIAL
    assert expected_diagnostic in execution.action_record.diagnostics


@pytest.mark.parametrize(
    ("failure", "expected_state", "expected_code"),
    [
        ("temp", EffectState.NONE, ToolErrorCode.EXECUTION_ERROR),
        ("replace", EffectState.UNKNOWN, ToolErrorCode.EXECUTION_ERROR),
        ("delete", EffectState.UNKNOWN, ToolErrorCode.EXECUTION_ERROR),
    ],
)
def test_patch_temp_replace_and_delete_failures_are_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_state: EffectState,
    expected_code: ToolErrorCode,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    workspace = make_workspace(tmp_path)
    tool, _ = approved_tool(workspace)

    if failure == "temp":
        def fail_temp(**_kwargs: object):
            raise OSError("temp unavailable")

        monkeypatch.setattr(coding_tools_module.tempfile, "mkstemp", fail_temp)
    elif failure == "replace":
        def fail_replace(*_args: object) -> None:
            raise OSError("replace unavailable")

        monkeypatch.setattr(coding_tools_module.os, "replace", fail_replace)
    else:
        def fail_remove(*_args: object) -> None:
            raise OSError("delete unavailable")

        monkeypatch.setattr(coding_tools_module.os, "remove", fail_remove)

    arguments = (
        {
            "operation": "delete",
            "path": "target.txt",
            "expected_sha256": digest(target),
        }
        if failure == "delete"
        else {
            "operation": "update",
            "path": "target.txt",
            "expected_sha256": digest(target),
            "replacements": [
                {"old": "old", "new": "new", "expected_occurrences": 1}
            ],
        }
    )
    execution, _ = invoke(tool, arguments, run_id=f"failure-{failure}-t7")

    assert execution.result.outcome is ToolOutcome.ERROR
    assert execution.result.error_code is expected_code
    assert execution.action_record is not None
    assert execution.action_record.effect_state is expected_state
    if failure != "delete":
        assert target.read_text(encoding="utf-8") == "old"
    else:
        assert target.exists()


def test_patch_temp_cleanup_failure_is_retained_as_partial_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    workspace = make_workspace(tmp_path)
    tool, _ = approved_tool(workspace)
    temporary_paths: list[Path] = []

    original_mkstemp = coding_tools_module.tempfile.mkstemp

    def remember_temp(**kwargs: object):
        fd, name = original_mkstemp(**kwargs)
        temporary_paths.append(Path(name))
        return fd, name

    monkeypatch.setattr(coding_tools_module.tempfile, "mkstemp", remember_temp)
    monkeypatch.setattr(coding_tools_module, "_cleanup_patch_temp", lambda _path: False)

    def fail_replace(*_args: object) -> None:
        raise OSError("replace unavailable")

    monkeypatch.setattr(coding_tools_module.os, "replace", fail_replace)

    execution, _ = invoke(
        tool,
        {
            "operation": "update",
            "path": "target.txt",
            "expected_sha256": digest(target),
            "replacements": [
                {"old": "old", "new": "new", "expected_occurrences": 1}
            ],
        },
        run_id="cleanup-t7",
    )

    assert execution.result.outcome is ToolOutcome.ERROR
    assert execution.action_record is not None
    assert execution.action_record.effect_state is EffectState.UNKNOWN
    assert "patch_temp_cleanup_failed" in execution.action_record.diagnostics
    assert temporary_paths and temporary_paths[0].exists()
    assert target.read_text(encoding="utf-8") == "old"


def test_patch_write_failure_does_not_replace_original_and_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    workspace = make_workspace(tmp_path)
    tool, _ = approved_tool(workspace)
    original_fdopen = coding_tools_module.os.fdopen

    class FailingHandle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def write(self, _content: bytes) -> None:
            raise OSError("write unavailable")

        def close(self) -> None:
            self._handle.close()

    monkeypatch.setattr(
        coding_tools_module.os,
        "fdopen",
        lambda fd, mode: FailingHandle(original_fdopen(fd, mode)),
    )

    execution, _ = invoke(
        tool,
        {
            "operation": "update",
            "path": "target.txt",
            "expected_sha256": digest(target),
            "replacements": [
                {"old": "old", "new": "new", "expected_occurrences": 1}
            ],
        },
        run_id="write-failure-t7",
    )

    assert execution.result.outcome is ToolOutcome.ERROR
    assert execution.action_record is not None
    assert execution.action_record.effect_state is EffectState.NONE
    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".dqagent-patch-*.tmp"))


def test_patch_byte_limit_is_checked_before_approval(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    tool, provider = approved_tool(
        workspace,
        limits=CodingToolLimits(max_patch_content_bytes=1),
    )
    context = RunContext(run_id="limit-t7")
    tool_context = ToolExecutionContext(
        context,
        record_collector=_RunActionRecordCollector("limit-t7", 1),
    )

    with pytest.raises(ActionPreparationError) as error:
        tool.prepare(
            {"operation": "create", "path": "limited.txt", "content": "é"},
            tool_context,
        )
    assert error.value.code is ToolErrorCode.RESOURCE_LIMIT
    assert provider.calls == 0
    assert not (tmp_path / "limited.txt").exists()


def test_patch_replacement_rejects_expansion_before_materializing_result() -> None:
    class TrackingText(str):
        replace_calls = 0

        def replace(self, *args: object, **kwargs: object) -> str:
            type(self).replace_calls += 1
            raise AssertionError("replacement result was materialized before the byte check")

    content = TrackingText("a" * 1_000)
    replacement = coding_tools_module._PatchReplacement("a", "x" * 1_000, 1_000)

    with pytest.raises(ActionPreparationError) as error:
        coding_tools_module._apply_patch_replacements(
            content,
            (replacement,),
            max_bytes=1_000,
        )

    assert error.value.code is ToolErrorCode.RESOURCE_LIMIT
    assert TrackingText.replace_calls == 0


@pytest.mark.parametrize(
    "secret",
    [
        "patch_create_open_failed",
        "patch_create_write_failed",
        "patch_create_close_failed",
    ],
)
def test_patch_secret_values_cannot_collide_with_fixed_diagnostics(
    tmp_path: Path,
    secret: str,
) -> None:
    with pytest.raises(ValueError, match="structured output metadata"):
        create_workspace_patch_tool(make_workspace(tmp_path), secret_values=(secret,))


def test_target_type_drift_after_approval_is_blocked(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    workspace = make_workspace(tmp_path)

    def approve_then_replace_with_directory(request, _context):
        target.unlink()
        target.mkdir()
        return ApprovalDecision.approve_for(request)

    provider = ScriptedApprovalProvider([approve_then_replace_with_directory])
    tool, _ = approved_tool(workspace, provider=provider)
    execution, _ = invoke(
        tool,
        {
            "operation": "update",
            "path": "target.txt",
            "expected_sha256": digest(target),
            "replacements": [
                {"old": "old", "new": "new", "expected_occurrences": 1}
            ],
        },
        run_id="type-drift-t7",
    )

    assert execution.result.outcome is ToolOutcome.ERROR
    assert execution.result.error_code in {
        ToolErrorCode.APPROVAL_MISMATCH,
        ToolErrorCode.PRECONDITION_CONFLICT,
    }
    assert target.is_dir()
    assert execution.action_record is not None
    assert execution.action_record.executor_attempts == 0
    assert execution.action_record.effect_state is EffectState.NONE


def test_target_link_drift_after_approval_is_blocked(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    link_target = tmp_path / "link-target.txt"
    link_target.write_text("external", encoding="utf-8")
    symlink_probe = tmp_path / "symlink-probe"
    try:
        symlink_probe.symlink_to(link_target)
        symlink_probe.unlink()
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    workspace = make_workspace(tmp_path)

    def approve_then_replace_with_link(request, _context):
        target.unlink()
        target.symlink_to(link_target)
        return ApprovalDecision.approve_for(request)

    provider = ScriptedApprovalProvider([approve_then_replace_with_link])
    tool, _ = approved_tool(workspace, provider=provider)
    execution, _ = invoke(
        tool,
        {
            "operation": "update",
            "path": "target.txt",
            "expected_sha256": digest(target),
            "replacements": [
                {"old": "old", "new": "new", "expected_occurrences": 1}
            ],
        },
        run_id="link-drift-t7",
    )

    assert execution.result.outcome is ToolOutcome.ERROR
    assert execution.result.error_code in {
        ToolErrorCode.APPROVAL_MISMATCH,
        ToolErrorCode.CONTAINMENT_DENIED,
        ToolErrorCode.PRECONDITION_CONFLICT,
    }
    assert target.is_symlink()
    assert link_target.read_text(encoding="utf-8") == "external"
    assert execution.action_record is not None
    assert execution.action_record.executor_attempts == 0
    assert execution.action_record.effect_state is EffectState.NONE


def test_combined_coding_tools_honor_patch_limits(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    limits = CodingToolLimits(
        max_patch_output_characters=20_000,
        max_patch_elapsed_seconds=6.0,
    )
    patch_provider = ScriptedApprovalProvider([ApprovalOutcome.APPROVE])
    _, _, patch_tool = create_coding_tools(
        workspace,
        limits=limits,
        approval_provider=patch_provider,
    )

    execution, _ = invoke(
        patch_tool,
        {"operation": "create", "path": "combined.txt", "content": "ok"},
        run_id="combined-limits-t7",
    )

    assert execution.result.outcome is ToolOutcome.SUCCESS
    assert (tmp_path / "combined.txt").read_text(encoding="utf-8") == "ok"


def test_patch_rejects_links_types_binary_and_wildcards_before_approval(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    directory = tmp_path / "directory"
    directory.mkdir()
    workspace = make_workspace(tmp_path)
    tool, provider = approved_tool(workspace)

    invalid_calls = [
        {
            "operation": "update",
            "path": "target*.txt",
            "expected_sha256": digest(target),
            "replacements": [
                {"old": "old", "new": "new", "expected_occurrences": 1}
            ],
        },
        {
            "operation": "update",
            "path": "directory",
            "expected_sha256": "0" * 64,
            "replacements": [
                {"old": "old", "new": "new", "expected_occurrences": 1}
            ],
        },
        {
            "operation": "update",
            "path": "target.txt",
            "expected_sha256": digest(target),
            "replacements": [
                {"old": "", "new": "new", "expected_occurrences": 1}
            ],
        },
    ]
    for index, arguments in enumerate(invalid_calls):
        execution, records = invoke(tool, arguments, run_id=f"invalid-{index}-t7")
        assert execution.result.outcome is ToolOutcome.ERROR
        assert records == ()
    assert provider.calls == 0
