from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import PurePosixPath

import pytest

import dqagent.tools as tools_module
from dqagent.errors import RunCancelledError
from dqagent.events import RunEventType
from dqagent.execution import RunContext
from dqagent.models import (
    Completion,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
    ToolErrorCode,
    ToolOutcome,
    ToolResult,
)
from dqagent.runtime import AgentRuntime, RetryPolicy
from dqagent.subprocesses import IsolationCapability
from dqagent.tool_governance import (
    ActionExecutionResult,
    ActionKind,
    ActionPolicy,
    ActionRecord,
    DefaultActionPolicy,
    EffectKind,
    EffectState,
    GuardContext,
    HookMode,
    HookOutcome,
    HookResult,
    PolicyDecision,
    PolicyOutcome,
    PreActionHookSpec,
    PreparedAction,
)
from dqagent.tools import (
    ActionTool,
    Tool,
    ToolExecutionContext,
    ToolRegistry,
    _RunActionRecordCollector,
)
from dqagent.workspace import Workspace, WorkspaceScope


class AllowPolicy:
    identity = "test-allow-policy"

    def decide(self, action: PreparedAction) -> PolicyDecision:
        return PolicyDecision(PolicyOutcome.ALLOW, "test_allow", self.identity)


def make_workspace(tmp_path, *, secret_values: tuple[str, ...] = ()) -> Workspace:
    del secret_values
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "target.txt").write_text("old", encoding="utf-8")
    return Workspace(WorkspaceScope("fixture", tmp_path))


def make_action(
    *,
    workspace: Workspace,
    kind: ActionKind = ActionKind.READ,
    required_capabilities: frozenset[IsolationCapability] = frozenset(),
    display_text: str = "read target.txt",
    secret_values: tuple[str, ...] = (),
) -> PreparedAction:
    if kind is ActionKind.READ:
        effect = EffectKind.NONE
        targets = (PurePosixPath("target.txt"),)
        argv: tuple[str, ...] = ()
    elif kind is ActionKind.COMMAND:
        effect = EffectKind.PROCESS_EXECUTION
        targets = ()
        argv = ("python", "-c", "print('ok')")
    else:
        effect = EffectKind.WORKSPACE_MUTATION
        targets = (PurePosixPath("target.txt"),)
        argv = ()
    return PreparedAction(
        kind,
        effect,
        workspace.scope.workspace_id,
        logical_targets=targets,
        argv=argv,
        required_capabilities=required_capabilities,
        normalized_fields={"kind": kind.value},
        display_text=display_text,
        secret_values=secret_values,
    )


def make_tool(
    workspace: Workspace,
    action: PreparedAction,
    executor: object,
    *,
    tool_name: str | None = None,
    policy: ActionPolicy | None = None,
    approval_provider: object | None = None,
    secret_values: tuple[str, ...] = (),
    guard_context: GuardContext | None = None,
    guard_context_factory: object | None = None,
    max_argument_bytes: int = 64_000,
    pre_hooks: tuple = (),
    post_hooks: tuple = (),
) -> ActionTool:
    definition = ToolDefinition(
        tool_name
        or ("workspace_read" if action.action_kind is ActionKind.READ else "workspace_command"),
        "Execute one governed action.",
        {"type": "object", "additionalProperties": False},
    )
    return ActionTool(
        definition,
        lambda arguments, context: action,
        executor,  # type: ignore[arg-type]
        guard_context=guard_context
        or GuardContext(
            workspace,
            available_capabilities=action.required_capabilities,
            max_governed_calls=1,
        ),
        guard_context_factory=guard_context_factory,  # type: ignore[arg-type]
        policy=policy or DefaultActionPolicy(),
        approval_provider=approval_provider,  # type: ignore[arg-type]
        secret_values=secret_values,
        max_argument_bytes=max_argument_bytes,
        pre_hooks=pre_hooks,
        post_hooks=post_hooks,
    )


def run_direct(
    tool: ActionTool,
    *,
    run_id: str = "run-t5",
    collector: _RunActionRecordCollector | None = None,
    events: list[tuple[str, Mapping[str, object]]] | None = None,
    arguments: str = "{}",
):
    context = RunContext(run_id=run_id)
    execution_context = ToolExecutionContext(
        context,
        emit_stage=(
            (lambda stage, attributes: events.append((stage, attributes)))
            if events is not None
            else None
        ),
        record_collector=collector,
    )
    return ToolRegistry((tool,)).execute_detailed(
        ToolCall("call-1", tool.definition.name, arguments),
        context,
        execution_context=execution_context,
    )


def test_governed_pipeline_has_fixed_order_and_at_most_once_record(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    executions: list[str] = []
    action = make_action(workspace=workspace)

    def executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value
        context.run_context.check_active()
        executions.append("executor")
        return "read result"

    tool = make_tool(workspace, action, executor)
    collector = _RunActionRecordCollector("run-t5", 1)
    result = run_direct(tool, collector=collector, events=[(stage, {}) for stage in []])

    assert result.result.outcome is ToolOutcome.SUCCESS
    assert executions == ["executor"]
    assert isinstance(result.action_record, ActionRecord)
    assert result.action_record.executor_attempts == 1
    assert result.action_record.effect_state is EffectState.COMPLETE
    assert collector.records == (result.action_record,)

    collected = collector.close("run-t5")
    assert collected == (result.action_record,)
    assert collector.records == ()


def test_governed_stage_events_are_ordered_and_sanitized(tmp_path) -> None:
    secret = "TOPSECRET"
    workspace = make_workspace(tmp_path, secret_values=(secret,))
    root = str(workspace.root)
    action = make_action(
        workspace=workspace,
        display_text=f"{root} {secret}",
        secret_values=(secret,),
    )

    def executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value, context
        return f"{root} {secret} result"

    events: list[tuple[str, Mapping[str, object]]] = []
    result = run_direct(
        make_tool(workspace, action, executor, secret_values=(secret,)),
        collector=_RunActionRecordCollector("run-t5", 1),
        events=events,
    )

    assert result.result.output == "[REDACTED] [REDACTED] result"
    assert root not in repr(events)
    assert secret not in repr(events)
    assert root not in repr(result.action_record)
    assert secret not in repr(result.action_record)
    assert [stage for stage, _ in events] == [
        RunEventType.ACTION_PREPARED.value,
        RunEventType.ACTION_GUARDS_EVALUATED.value,
        RunEventType.ACTION_POLICY_DECIDED.value,
        RunEventType.ACTION_REVALIDATED.value,
        RunEventType.ACTION_PRE_HOOKS_COMPLETED.value,
        RunEventType.ACTION_EFFECT_REVALIDATED.value,
        RunEventType.ACTION_EXECUTOR_STARTED.value,
        RunEventType.ACTION_EXECUTOR_COMPLETED.value,
        RunEventType.ACTION_POST_HOOKS_COMPLETED.value,
        RunEventType.ACTION_OBSERVED.value,
    ]


def test_approval_rejection_and_required_pre_hook_have_zero_attempts(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    action = make_action(workspace=workspace, kind=ActionKind.COMMAND)
    executions: list[str] = []

    def executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value, context
        executions.append("executor")
        return "must not run"

    class RejectingProvider:
        identity = "rejecting-provider"
        supports_deadline = False

        def request_approval(self, request, context):
            del request, context
            from dqagent.tool_governance import ApprovalDecision, ApprovalOutcome

            return ApprovalDecision(ApprovalOutcome.REJECT, "user rejected")

    collector = _RunActionRecordCollector("run-t5", 1)
    result = run_direct(
        make_tool(
            workspace,
            action,
            executor,
            approval_provider=RejectingProvider(),
        ),
        collector=collector,
    )
    assert result.result.error_code is ToolErrorCode.APPROVAL_REJECTED
    assert executions == []
    assert result.action_record is not None
    assert result.action_record.executor_attempts == 0
    assert result.action_record.approval_outcome.value == "reject"


def test_required_pre_hook_blocks_and_post_hook_failure_does_not_conceal_effect(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    action = make_action(workspace=workspace)
    executions: list[str] = []

    class BlockingHook:
        identity = "blocking-hook"

        def __call__(self, value) -> HookResult:
            del value
            return HookResult(HookOutcome.FAILED, "blocked", self.identity)

    class FailingPostHook:
        identity = "failing-post-hook"

        def __call__(self, value) -> HookResult:
            del value
            return HookResult(HookOutcome.FAILED, "post failed", self.identity)

    def executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value, context
        executions.append("executor")
        return "effect visible"

    blocked = run_direct(
        make_tool(
            workspace,
            action,
            executor,
            pre_hooks=(PreActionHookSpec(BlockingHook(), HookMode.REQUIRED),),
        ),
        collector=_RunActionRecordCollector("run-t5", 1),
    )
    assert blocked.result.error_code is ToolErrorCode.GOVERNANCE_FAILURE
    assert blocked.action_record is not None
    assert blocked.action_record.executor_attempts == 0
    assert executions == []

    completed = run_direct(
        make_tool(
            workspace,
            action,
            executor,
            post_hooks=(FailingPostHook(),),
        ),
        collector=_RunActionRecordCollector("run-t5", 1),
    )
    assert completed.result.outcome is ToolOutcome.SUCCESS
    assert completed.result.output == "effect visible"
    assert completed.action_record is not None
    assert completed.action_record.executor_attempts == 1
    assert completed.action_record.effect_state is EffectState.COMPLETE
    assert completed.action_record.post_hook_results[0].outcome is HookOutcome.FAILED
    assert executions == ["executor"]


def test_partial_and_unknown_effects_are_retained_without_retry(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    partial_action = make_action(workspace=workspace)
    partial_calls: list[int] = []

    def partial_executor(
        value: PreparedAction,
        context: ToolExecutionContext,
    ) -> ActionExecutionResult:
        del value, context
        partial_calls.append(1)
        return ActionExecutionResult("partial", EffectState.PARTIAL)

    partial = run_direct(
        make_tool(workspace, partial_action, partial_executor),
        collector=_RunActionRecordCollector("run-t5", 1),
    )
    assert partial.result.outcome is ToolOutcome.SUCCESS
    assert partial.action_record is not None
    assert partial.action_record.effect_state is EffectState.PARTIAL
    assert partial.action_record.executor_attempts == 1
    assert partial_calls == [1]

    def unknown_executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value, context
        raise RuntimeError("backend failed")

    unknown = run_direct(
        make_tool(workspace, make_action(workspace=workspace), unknown_executor),
        collector=_RunActionRecordCollector("run-t5", 1),
    )
    assert unknown.result.outcome is ToolOutcome.ERROR
    assert unknown.result.error_code is ToolErrorCode.EXECUTION_ERROR
    assert unknown.action_record is not None
    assert unknown.action_record.effect_state is EffectState.UNKNOWN
    assert unknown.action_record.executor_attempts == 1


def test_effect_boundary_revalidation_blocks_executor(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    action = make_action(
        workspace=workspace,
        kind=ActionKind.COMMAND,
        required_capabilities=frozenset({IsolationCapability.DIRECT_ARGV}),
    )
    calls = 0
    executions: list[str] = []
    base = GuardContext(
        workspace,
        available_capabilities=frozenset({IsolationCapability.DIRECT_ARGV}),
        max_governed_calls=1,
    )

    def guard_factory(value: PreparedAction, context: ToolExecutionContext) -> GuardContext:
        del value, context
        nonlocal calls
        calls += 1
        if calls >= 2:
            return replace(base, available_capabilities=frozenset())
        return base

    def executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value, context
        executions.append("executor")
        return "must not run"

    result = run_direct(
        make_tool(
            workspace,
            action,
            executor,
            policy=AllowPolicy(),
            guard_context=base,
            guard_context_factory=guard_factory,
        ),
        collector=_RunActionRecordCollector("run-t5", 1),
    )
    assert result.result.error_code is ToolErrorCode.CAPABILITY_MISSING
    assert executions == []
    assert result.action_record is not None
    assert result.action_record.executor_attempts == 0


def test_oversized_arguments_are_rejected_before_json_parse(tmp_path, monkeypatch) -> None:
    workspace = make_workspace(tmp_path)
    action = make_action(workspace=workspace)
    called = False

    def fail_if_parsed(value):
        nonlocal called
        called = True
        raise AssertionError("json.loads must not receive oversized arguments")

    monkeypatch.setattr(tools_module.json, "loads", fail_if_parsed)
    result = run_direct(
        make_tool(workspace, action, lambda value, context: "unused", max_argument_bytes=4),
        arguments='{"x":true}',
    )
    assert result.result.error_code is ToolErrorCode.ARGUMENT_TOO_LARGE
    assert called is False


def test_unencodable_arguments_are_bounded_parse_failures(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    prepared = False

    def prepare(arguments, context):
        del arguments, context
        nonlocal prepared
        prepared = True
        return make_action(workspace=workspace)

    tool = ActionTool(
        ToolDefinition("workspace_read", "Execute one governed action.", {"type": "object"}),
        prepare,
        lambda value, context: "unused",
        guard_context=GuardContext(workspace),
    )
    result = run_direct(tool, arguments="\ud800")

    assert result.result.error_code is ToolErrorCode.INVALID_ARGUMENTS
    assert prepared is False


def test_reserved_names_cannot_use_legacy_handler_dispatch(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        ToolRegistry(
            (
                Tool(
                    ToolDefinition(
                        "workspace_read",
                        "legacy bypass",
                        {"type": "object"},
                    ),
                    lambda arguments, context: "bypass",
                ),
            )
        )

    action = make_action(workspace=workspace)
    result = run_direct(
        make_tool(workspace, action, lambda value, context: "governed"),
    )
    assert result.result.output == "governed"


def test_excess_governed_call_is_visible_before_executor(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    executions: list[str] = []

    def executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value, context
        executions.append("executor")
        return "once"

    tool = make_tool(workspace, make_action(workspace=workspace), executor)
    context = RunContext(run_id="run-limit")
    collector = _RunActionRecordCollector("run-limit", 1)
    execution_context = ToolExecutionContext(context, record_collector=collector)
    registry = ToolRegistry((tool,))
    first = registry.execute_detailed(
        ToolCall("call-1", tool.definition.name, "{}"),
        context,
        execution_context=execution_context,
    )
    second = registry.execute_detailed(
        ToolCall("call-2", tool.definition.name, "{}"),
        context,
        execution_context=execution_context,
    )

    assert first.result.outcome is ToolOutcome.SUCCESS
    assert second.result.error_code is ToolErrorCode.GOVERNED_CALL_LIMIT
    assert executions == ["executor"]
    assert len(collector.records) == 1
    assert collector.close("run-limit")


def test_collector_append_failure_is_an_observation_failure(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    executions: list[str] = []

    class FailingCollector:
        run_id = "run-collector"
        max_records = 1

        def __init__(self) -> None:
            self.reservations = 0

        def reserved_count(self, run_id: str) -> int:
            assert run_id == self.run_id
            return self.reservations

        def reserve(self, run_id: str) -> object:
            assert run_id == self.run_id
            self.reservations += 1
            return object()

        def append(self, run_id: str, record: ActionRecord, reservation: object) -> None:
            del record, reservation
            assert run_id == self.run_id
            raise RuntimeError("collector unavailable")

    def executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value, context
        executions.append("executor")
        return "effect happened"

    events: list[tuple[str, Mapping[str, object]]] = []
    result = run_direct(
        make_tool(workspace, make_action(workspace=workspace), executor),
        collector=FailingCollector(),  # type: ignore[arg-type]
        events=events,
        run_id="run-collector",
    )
    assert result.result.error_code is ToolErrorCode.OBSERVATION_FAILURE
    assert executions == ["executor"]
    assert result.action_record is not None
    assert result.action_record.observation_failure is True
    assert any(stage == RunEventType.ACTION_RECORD_FAILED.value for stage, _ in events)


def test_collector_run_mismatch_is_an_observation_failure(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    executions: list[str] = []
    collector = _RunActionRecordCollector("other-run", 1)
    events: list[tuple[str, Mapping[str, object]]] = []

    def executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value, context
        executions.append("executor")
        return "must not run"

    result = run_direct(
        make_tool(workspace, make_action(workspace=workspace), executor),
        collector=collector,
        events=events,
        run_id="active-run",
    )

    assert result.result.error_code is ToolErrorCode.OBSERVATION_FAILURE
    assert executions == []
    assert any(stage == RunEventType.ACTION_RECORD_FAILED.value for stage, _ in events)
    assert collector.close("other-run") == ()


def test_control_errors_propagate_and_record_unknown_attempt(tmp_path) -> None:
    workspace = make_workspace(tmp_path)

    def canceling_executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value
        context.run_context.cancel("cancel in executor")
        context.run_context.check_active()
        return "late"

    context = RunContext(run_id="run-control")
    collector = _RunActionRecordCollector("run-control", 1)
    execution_context = ToolExecutionContext(context, record_collector=collector)
    tool = make_tool(workspace, make_action(workspace=workspace), canceling_executor)
    with pytest.raises(RunCancelledError):
        ToolRegistry((tool,)).execute_detailed(
            ToolCall("call-1", tool.definition.name, "{}"),
            context,
            execution_context=execution_context,
        )
    assert collector.records
    assert collector.records[0].executor_attempts == 1
    assert collector.records[0].effect_state is EffectState.UNKNOWN
    assert collector.close("run-control")


def test_runtime_bridge_passes_run_context_and_stage_events(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    action = make_action(workspace=workspace)
    seen_run_ids: list[str] = []

    def executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value
        seen_run_ids.append(context.run_context.run_id)
        return "runtime result"

    tool = make_tool(workspace, action, executor)
    call = ToolCall("call-runtime", tool.definition.name, "{}")

    class ScriptedLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, tools=(), *, context=None):
            del messages, tools
            assert context is not None
            self.calls += 1
            return Completion(tool_calls=(call,)) if self.calls == 1 else Completion("done")

    context = RunContext(run_id="run-runtime")
    collector = _RunActionRecordCollector("run-runtime", 1)
    runtime = AgentRuntime(
        ScriptedLLM(),
        ToolRegistry((tool,)),
        retry_policy=RetryPolicy(max_attempts=1),
    )
    result = runtime.run(
        [Message(Role.USER, "read")],
        context=context,
        tool_context=ToolExecutionContext(context, record_collector=collector),
    )

    assert result.output == Message(Role.ASSISTANT, "done")
    assert seen_run_ids == ["run-runtime"]
    assert any(event.type is RunEventType.ACTION_PREPARED for event in result.events)
    assert collector.close("run-runtime")


def test_runtime_bridge_default_collector_enforces_runwide_limit(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    executions: list[str] = []

    def executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value
        executions.append(context.run_context.run_id)
        return "runtime result"

    first_tool = make_tool(workspace, make_action(workspace=workspace), executor)
    second_tool = make_tool(
        workspace,
        make_action(workspace=workspace),
        executor,
        tool_name="workspace_search",
    )
    calls = (
        ToolCall("call-runtime-1", first_tool.definition.name, "{}"),
        ToolCall("call-runtime-2", second_tool.definition.name, "{}"),
    )

    class ScriptedLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, tools=(), *, context=None):
            del messages, tools, context
            self.calls += 1
            if self.calls <= len(calls):
                return Completion(tool_calls=(calls[self.calls - 1],))
            return Completion("done")

    result = AgentRuntime(
        ScriptedLLM(),
        ToolRegistry((first_tool, second_tool)),
        retry_policy=RetryPolicy(max_attempts=1),
    ).run([Message(Role.USER, "read")], context=RunContext(run_id="run-runtime-default"))

    tool_results = [item for item in result.conversation if isinstance(item, ToolResult)]
    assert [item.error_code for item in tool_results] == [None, ToolErrorCode.GOVERNED_CALL_LIMIT]
    assert executions == ["run-runtime-default"]


def test_direct_governed_dispatch_without_context_fails_closed(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    executions: list[str] = []

    def executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value
        executions.append(context.run_context.run_id)
        return "unexpected side effect"

    tool = make_tool(
        workspace,
        make_action(workspace=workspace),
        executor,
    )
    registry = ToolRegistry((tool,))
    context = RunContext(run_id="run-direct-missing-context")

    first = registry.execute_detailed(
        ToolCall("call-direct-1", tool.definition.name, "{}"),
        context,
    )
    second = registry.execute_detailed(
        ToolCall("call-direct-2", tool.definition.name, "{}"),
        context,
    )

    assert first.result.error_code is ToolErrorCode.OBSERVATION_FAILURE
    assert second.result.error_code is ToolErrorCode.OBSERVATION_FAILURE
    assert first.action_record is None
    assert second.action_record is None
    assert executions == []


def test_workspace_scope_drift_is_denied_before_effect_and_recorded(tmp_path) -> None:
    workspace_a = make_workspace(tmp_path / "root-a")
    workspace_b = make_workspace(tmp_path / "root-b")
    action = make_action(workspace=workspace_a)
    base = GuardContext(workspace_a, max_governed_calls=1)
    factory_calls = 0
    executions: list[str] = []

    def guard_factory(value: PreparedAction, context: ToolExecutionContext) -> GuardContext:
        del value, context
        nonlocal factory_calls
        factory_calls += 1
        return base if factory_calls < 3 else replace(base, workspace=workspace_b)

    def executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value, context
        executions.append("executor")
        return "must not run"

    result = run_direct(
        make_tool(
            workspace_a,
            action,
            executor,
            policy=AllowPolicy(),
            guard_context=base,
            guard_context_factory=guard_factory,
        ),
        collector=_RunActionRecordCollector("run-t5", 1),
    )

    assert result.result.error_code is ToolErrorCode.CONTAINMENT_DENIED
    assert executions == []
    assert result.action_record is not None
    assert result.action_record.executor_attempts == 0
    assert result.action_record.observation_failure is True
    assert result.action_record.guard_results[-1].reason == "workspace_scope_binding_mismatch"


def test_stage_event_reasons_use_factory_workspace_sanitizer(tmp_path) -> None:
    secret = "TOPSECRET"
    workspace_a = make_workspace(tmp_path / "root-a")
    workspace_b = make_workspace(tmp_path / "root-b")
    action = make_action(workspace=workspace_a, secret_values=(secret,))

    class LeakingPolicy:
        identity = "leaking-policy"

        def decide(self, value: PreparedAction) -> PolicyDecision:
            del value
            return PolicyDecision(
                PolicyOutcome.DENY,
                f"foreign-root={workspace_b.root} {secret}",
                self.identity,
            )

    events: list[tuple[str, Mapping[str, object]]] = []

    result = run_direct(
        make_tool(
            workspace_a,
            action,
            lambda value, context: "must not run",
            policy=LeakingPolicy(),
            guard_context=GuardContext(workspace_a, max_governed_calls=1),
            guard_context_factory=lambda value, context: GuardContext(
                workspace_b,
                max_governed_calls=1,
            ),
            secret_values=(secret,),
        ),
        collector=_RunActionRecordCollector("run-t5", 1),
        events=events,
    )

    assert result.result.error_code is ToolErrorCode.POLICY_DENIED
    policy_events = [
        attributes
        for stage, attributes in events
        if stage == RunEventType.ACTION_POLICY_DECIDED.value
    ]
    assert len(policy_events) == 1
    event_text = tuple(
        value
        for _stage, attributes in events
        for value in attributes.values()
        if isinstance(value, str)
    )
    assert all(str(workspace_a.root) not in value for value in event_text)
    assert all(str(workspace_b.root) not in value for value in event_text)
    assert all(secret not in value for value in event_text)


def test_registry_rejects_foreign_run_context_object_before_effect(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    action = make_action(workspace=workspace)
    active_context = RunContext(run_id="same-run")
    foreign_context = RunContext(run_id="same-run")
    executions: list[str] = []

    class CancellingPolicy:
        identity = "cancelling-policy"

        def decide(self, value: PreparedAction) -> PolicyDecision:
            del value
            active_context.cancel("cancelled by active context")
            return PolicyDecision(PolicyOutcome.ALLOW, "allowed", self.identity)

    def executor(value: PreparedAction, context: ToolExecutionContext) -> str:
        del value, context
        executions.append("executor")
        return "must not run"

    result = ToolRegistry(
        (
            make_tool(
                workspace,
                action,
                executor,
                policy=CancellingPolicy(),
            ),
        )
    ).execute_detailed(
        ToolCall("call-foreign-context", "workspace_read", "{}"),
        active_context,
        execution_context=ToolExecutionContext(foreign_context),
    )

    assert result.result.error_code is ToolErrorCode.OBSERVATION_FAILURE
    assert executions == []


def test_post_hook_cancellation_retains_completed_partial_results(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    run_context = RunContext(run_id="run-post-control")
    collector = _RunActionRecordCollector("run-post-control", 1)

    class CompletedHook:
        identity = "completed-hook"

        def __call__(self, value) -> HookResult:
            del value
            return HookResult(HookOutcome.COMPLETED, "completed", self.identity)

    class CancellingHook:
        identity = "cancelling-hook"

        def __call__(self, value) -> HookResult:
            del value
            run_context.cancel("cancelled after hook")
            return HookResult(HookOutcome.COMPLETED, "completed", self.identity)

    execution_context = ToolExecutionContext(run_context, record_collector=collector)
    with pytest.raises(RunCancelledError):
        ToolRegistry(
            (
                make_tool(
                    workspace,
                    make_action(workspace=workspace),
                    lambda value, context: "effect",
                    post_hooks=(CompletedHook(), CancellingHook()),
                ),
            )
        ).execute_detailed(
            ToolCall("call-post-control", "workspace_read", "{}"),
            run_context,
            execution_context=execution_context,
        )

    assert len(collector.records) == 1
    record = collector.records[0]
    assert record.executor_attempts == 1
    assert record.effect_state is EffectState.UNKNOWN
    assert [item.hook_id for item in record.post_hook_results] == ["completed-hook"]
    assert collector.close("run-post-control")


def test_hard_guard_failure_does_not_emit_policy_decided_event(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    action = make_action(
        workspace=workspace,
        kind=ActionKind.COMMAND,
        required_capabilities=frozenset({IsolationCapability.DIRECT_ARGV}),
    )

    class CountingPolicy:
        identity = "counting-policy"

        def __init__(self) -> None:
            self.calls = 0

        def decide(self, value: PreparedAction) -> PolicyDecision:
            del value
            self.calls += 1
            return PolicyDecision(PolicyOutcome.ALLOW, "allowed", self.identity)

    policy = CountingPolicy()
    events: list[tuple[str, Mapping[str, object]]] = []
    result = run_direct(
        make_tool(
            workspace,
            action,
            lambda value, context: "must not run",
            policy=policy,
            guard_context=GuardContext(workspace, max_governed_calls=1),
        ),
        events=events,
    )

    assert result.result.error_code is ToolErrorCode.CAPABILITY_MISSING
    assert policy.calls == 0
    assert [stage for stage, _ in events].count(RunEventType.ACTION_POLICY_DECIDED.value) == 0
