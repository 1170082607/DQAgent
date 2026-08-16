from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import Timer

import pytest

from dqagent.coding_tools import (
    WORKSPACE_COMMAND_SCHEMA,
    CommandExecutable,
    CommandToolLimits,
    create_workspace_command_tool,
)
from dqagent.errors import RunCancelledError
from dqagent.execution import RunContext
from dqagent.models import ToolCall, ToolErrorCode, ToolOutcome
from dqagent.subprocesses import (
    LOCAL_SUBPROCESS_CAPABILITIES,
    CleanupResult,
    CleanupStatus,
    IsolationCapability,
    SubprocessRequest,
    SubprocessResult,
    SubprocessStatus,
)
from dqagent.tool_governance import ApprovalOutcome, ScriptedApprovalProvider
from dqagent.tools import ToolExecutionContext, ToolRegistry, _RunActionRecordCollector
from dqagent.workspace import Workspace, WorkspaceScope


def make_workspace(tmp_path: Path, **kwargs: object) -> Workspace:
    return Workspace(WorkspaceScope("fixture", tmp_path, **kwargs))


def invoke(
    tool: object,
    arguments: dict[str, object],
    *,
    context: RunContext | None = None,
) -> tuple[object, _RunActionRecordCollector]:
    run_context = context or RunContext(run_id="command-t9")
    collector = _RunActionRecordCollector(run_context.run_id, 1)
    execution_context = ToolExecutionContext(run_context, record_collector=collector)
    result = ToolRegistry((tool,)).execute_detailed(
        ToolCall("command-call", "workspace_command", json.dumps(arguments)),
        run_context,
        execution_context=execution_context,
    )
    return result, collector


def command_tool(
    workspace: Workspace,
    *,
    limits: CommandToolLimits | None = None,
    allowlist: dict[str, str | Path | CommandExecutable] | None = None,
    **kwargs: object,
) -> object:
    return create_workspace_command_tool(
        workspace,
        limits=limits,
        executable_allowlist=allowlist or {"python": Path(sys.executable)},
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVE]),
        **kwargs,
    )


def output_header(execution: object) -> dict[str, object]:
    return json.loads(execution.result.output)  # type: ignore[attr-defined]


def test_command_schema_is_closed_and_bounded() -> None:
    assert WORKSPACE_COMMAND_SCHEMA["additionalProperties"] is False
    assert WORKSPACE_COMMAND_SCHEMA["required"] == ["argv"]
    properties = WORKSPACE_COMMAND_SCHEMA["properties"]
    assert set(properties) == {"argv", "cwd", "timeout_seconds"}
    assert properties["argv"]["minItems"] == 1


@pytest.mark.parametrize(
    "arguments",
    [
        {"argv": []},
        {"argv": [""]},
        {"argv": ["python", "\x00"]},
        {"argv": ["python"], "unknown": True},
    ],
)
def test_command_rejects_strict_schema_before_spawn(
    tmp_path: Path,
    arguments: dict[str, object],
) -> None:
    execution, collector = invoke(command_tool(make_workspace(tmp_path)), arguments)

    assert execution.result.outcome is ToolOutcome.ERROR  # type: ignore[attr-defined]
    assert execution.result.error_code is ToolErrorCode.INVALID_ARGUMENTS  # type: ignore[attr-defined]
    assert collector.records == ()


def test_command_uses_allowlisted_direct_argv_and_minimal_environment(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    tool = command_tool(
        workspace,
        allowlist={"python": Path(sys.executable)},
        environment={"SAFE": "allowed", "AUTH_VALUE": "must-not-enter"},
    )
    code = (
        "import os,sys; print(os.environ.get('SAFE','missing')); "
        "print(os.environ.get('AUTH_VALUE','missing')); print(sys.argv[1])"
    )
    execution, collector = invoke(
        tool,
        {"argv": ["python", "-c", code, "literal;not a shell"]},
    )

    assert execution.result.outcome is ToolOutcome.SUCCESS  # type: ignore[attr-defined]
    header = output_header(execution)
    assert header["status"] == "normal"
    assert "allowed" in header["stdout"]
    assert "missing" in header["stdout"]
    assert "literal;not a shell" in header["stdout"]
    assert collector.records[0].executor_attempts == 1


def test_command_disallows_unallowlisted_and_untrusted_shell_before_spawn(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    denied = command_tool(
        workspace,
        allowlist={"python": Path(sys.executable)},
    )
    execution, collector = invoke(denied, {"argv": ["not-allowlisted", "--version"]})
    assert execution.result.error_code is ToolErrorCode.POLICY_DENIED  # type: ignore[attr-defined]
    assert collector.records == ()

    shell = command_tool(
        workspace,
        allowlist={
            "trusted-shell": CommandExecutable(
                "trusted-shell", Path(sys.executable), shell=True
            )
        },
    )
    shell_execution, shell_collector = invoke(
        shell,
        {"argv": ["trusted-shell", "-c", "print('must not run')"]},
    )
    assert shell_execution.result.error_code is ToolErrorCode.POLICY_DENIED  # type: ignore[attr-defined]
    assert shell_collector.records == ()


def test_explicitly_enabled_shell_and_trusted_resolver_are_still_governed(tmp_path: Path) -> None:
    shell = create_workspace_command_tool(
        make_workspace(tmp_path),
        executable_allowlist={
            "trusted-shell": CommandExecutable(
                "trusted-shell", Path(sys.executable), shell=True
            )
        },
        allow_shell=True,
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVE]),
    )
    shell_execution, _ = invoke(
        shell,
        {"argv": ["trusted-shell", "-c", "print('shell interpreter is explicit')"]},
    )
    assert shell_execution.result.outcome is ToolOutcome.SUCCESS  # type: ignore[attr-defined]

    resolved = create_workspace_command_tool(
        make_workspace(tmp_path),
        executable_resolver=lambda requested: Path(sys.executable)
        if requested == "resolved-python"
        else None,
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVE]),
    )
    resolved_execution, _ = invoke(
        resolved,
        {"argv": ["resolved-python", "-c", "print('resolved')"]},
    )
    assert resolved_execution.result.outcome is ToolOutcome.SUCCESS  # type: ignore[attr-defined]


def test_command_freezes_resolved_executable_across_approval(tmp_path: Path) -> None:
    resolver_calls: list[str] = []
    executable_path = Path(sys.executable).resolve()
    replacement_path = tmp_path / "replacement.exe"

    def resolver(requested: str) -> CommandExecutable:
        resolver_calls.append(requested)
        if len(resolver_calls) == 1:
            return CommandExecutable("approved-python", executable_path)
        return CommandExecutable("approved-python", replacement_path)

    class RecordingRunner:
        backend_identity = "recording-runner"
        capabilities = LOCAL_SUBPROCESS_CAPABILITIES

        def __init__(self) -> None:
            self.requests: list[tuple[str, ...]] = []

        def run(
            self,
            request: SubprocessRequest,
            context: RunContext | None = None,
        ) -> SubprocessResult:
            del context
            self.requests.append(request.argv)
            return SubprocessResult(
                status=SubprocessStatus.NORMAL,
                returncode=0,
                spawned=True,
                backend_identity=self.backend_identity,
                backend_capabilities=tuple(self.capabilities),
                cleanup=CleanupResult(
                    status=CleanupStatus.REAPED,
                    reaped=True,
                    streams_drained=True,
                ),
            )

    runner = RecordingRunner()
    tool = create_workspace_command_tool(
        make_workspace(tmp_path),
        executable_resolver=resolver,
        subprocess_runner=runner,
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVE]),
    )

    execution, _ = invoke(tool, {"argv": ["requested", "-c", "pass"]})

    assert execution.result.outcome is ToolOutcome.SUCCESS  # type: ignore[attr-defined]
    assert resolver_calls == ["requested"]
    assert runner.requests == [(str(executable_path), "-c", "pass")]


def test_command_result_projects_backend_capabilities(tmp_path: Path) -> None:
    advertised = (
        IsolationCapability.NO_STDIN,
        IsolationCapability.DIRECT_ARGV,
    )

    class CapabilityRunner:
        backend_identity = "capability-runner"
        capabilities = LOCAL_SUBPROCESS_CAPABILITIES

        def run(
            self,
            request: SubprocessRequest,
            context: RunContext | None = None,
        ) -> SubprocessResult:
            del request, context
            return SubprocessResult(
                status=SubprocessStatus.NORMAL,
                returncode=0,
                spawned=True,
                backend_identity=self.backend_identity,
                backend_capabilities=advertised,
                cleanup=CleanupResult(
                    status=CleanupStatus.REAPED,
                    reaped=True,
                    streams_drained=True,
                ),
            )

    execution, _ = invoke(
        create_workspace_command_tool(
            make_workspace(tmp_path),
            executable_allowlist={"python": Path(sys.executable)},
            subprocess_runner=CapabilityRunner(),
            approval_provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVE]),
        ),
        {"argv": ["python", "-c", "pass"]},
    )

    assert execution.result.outcome is ToolOutcome.SUCCESS  # type: ignore[attr-defined]
    header = output_header(execution)
    assert header["backend_capabilities"] == [
        IsolationCapability.DIRECT_ARGV.value,
        IsolationCapability.NO_STDIN.value,
    ]


def test_command_cwd_and_timeout_can_only_tighten_trusted_limits(tmp_path: Path) -> None:
    limits = CommandToolLimits(max_timeout_seconds=0.5)
    tool = create_workspace_command_tool(
        make_workspace(tmp_path),
        limits=limits,
        executable_allowlist={"python": Path(sys.executable)},
        approval_provider=ScriptedApprovalProvider(
            [ApprovalOutcome.APPROVE, ApprovalOutcome.APPROVE]
        ),
    )
    too_long, no_record = invoke(
        tool,
        {"argv": ["python", "-c", "pass"], "timeout_seconds": 0.6},
    )
    assert too_long.result.error_code is ToolErrorCode.GOVERNANCE_FAILURE  # type: ignore[attr-defined]
    assert no_record.records == ()

    missing_cwd, missing_record = invoke(
        tool,
        {"argv": ["python", "-c", "pass"], "cwd": "missing"},
    )
    assert missing_cwd.result.error_code is ToolErrorCode.CONTAINMENT_DENIED  # type: ignore[attr-defined]
    assert missing_record.records == ()


def test_command_nonzero_projects_bounded_error_and_retains_action_record(tmp_path: Path) -> None:
    tool = command_tool(make_workspace(tmp_path))
    execution, collector = invoke(
        tool,
        {"argv": ["python", "-c", "import sys; print('out'); sys.exit(7)"]},
    )

    assert execution.result.outcome is ToolOutcome.ERROR  # type: ignore[attr-defined]
    assert execution.result.error_code is ToolErrorCode.PROCESS_FAILURE  # type: ignore[attr-defined]
    header = output_header(execution)
    assert header["status"] == "nonzero"
    assert header["exit_code"] == 7
    assert "out" in header["stdout"]
    assert collector.records[0].effect_state.value == "unknown"


def test_command_timeout_reports_cleanup_and_output_truncation(tmp_path: Path) -> None:
    limits = CommandToolLimits(
        max_timeout_seconds=1.0,
        max_stdout_bytes=64,
        max_stderr_bytes=64,
        max_output_characters=2_000,
    )
    tool = create_workspace_command_tool(
        make_workspace(tmp_path),
        limits=limits,
        executable_allowlist={"python": Path(sys.executable)},
        approval_provider=ScriptedApprovalProvider(
            [ApprovalOutcome.APPROVE, ApprovalOutcome.APPROVE]
        ),
    )
    output_execution, output_records = invoke(
        tool,
        {"argv": ["python", "-c", "import os; os.write(1,b'A'*10000)"]},
    )
    output_header_value = output_header(output_execution)
    assert output_execution.result.outcome is ToolOutcome.SUCCESS  # type: ignore[attr-defined]
    assert output_header_value["stdout_truncated"] is True
    assert output_records.records[0].executor_attempts == 1

    timeout_execution, timeout_records = invoke(
        tool,
        {"argv": ["python", "-c", "import time; time.sleep(10)"], "timeout_seconds": 0.05},
    )
    assert timeout_execution.result.error_code is ToolErrorCode.TIMEOUT  # type: ignore[attr-defined]
    assert timeout_records.records[0].executor_attempts == 1
    assert timeout_records.records[0].effect_state.value == "unknown"
    timeout_header = output_header(timeout_execution)
    assert timeout_header["status"] == "timeout"
    assert timeout_header["cleanup_succeeded"] is True


def test_command_missing_required_capability_is_denied_before_spawn(tmp_path: Path) -> None:
    tool = command_tool(
        make_workspace(tmp_path),
        required_capabilities=(IsolationCapability.PROCESS_GROUP_TERMINATION,),
    )
    execution, collector = invoke(tool, {"argv": ["python", "-c", "raise SystemExit(9)"]})

    assert execution.result.error_code is ToolErrorCode.CAPABILITY_MISSING  # type: ignore[attr-defined]
    assert collector.records[0].executor_attempts == 0


def test_command_cleanup_failure_is_an_observation_error(tmp_path: Path) -> None:
    class CleanupFailureRunner:
        backend_identity = "cleanup-failure-runner"
        capabilities = LOCAL_SUBPROCESS_CAPABILITIES

        def run(self, request, context=None):
            del request, context
            return SubprocessResult(
                status=SubprocessStatus.NORMAL,
                returncode=0,
                spawned=True,
                backend_identity=self.backend_identity,
                backend_capabilities=tuple(self.capabilities),
                cleanup=CleanupResult(status=CleanupStatus.FAILED, reaped=False),
            )

    tool = create_workspace_command_tool(
        make_workspace(tmp_path),
        executable_allowlist={"python": Path(sys.executable)},
        subprocess_runner=CleanupFailureRunner(),  # type: ignore[arg-type]
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVE]),
    )
    execution, collector = invoke(tool, {"argv": ["python", "-c", "pass"]})

    assert execution.result.error_code is ToolErrorCode.OBSERVATION_FAILURE  # type: ignore[attr-defined]
    assert collector.records[0].effect_state.value == "unknown"


def test_command_cancellation_stops_and_reaps_the_direct_child(tmp_path: Path) -> None:
    context = RunContext(run_id="command-cancel")
    timer = Timer(0.05, context.cancel, args=("test cancellation",))
    timer.start()
    try:
        with pytest.raises(RunCancelledError):
            invoke(
                command_tool(make_workspace(tmp_path)),
                {"argv": ["python", "-c", "import time; time.sleep(10)"]},
                context=context,
            )
    finally:
        timer.join()
