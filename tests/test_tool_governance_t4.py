from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import PurePosixPath

import pytest

from dqagent.errors import RunCancelledError
from dqagent.execution import RunContext
from dqagent.subprocesses import IsolationCapability
from dqagent.tool_governance import (
    ActionKind,
    ApprovalDecision,
    ApprovalOutcome,
    EffectiveLimits,
    EffectKind,
    EffectPrecondition,
    EffectPreconditions,
    EffectState,
    GuardContext,
    HookMode,
    HookOutcome,
    HookResult,
    NonInteractiveApprovalProvider,
    PolicyOutcome,
    PreActionHookSpec,
    PreparedAction,
    ScriptedApprovalProvider,
    authorize_action,
    build_approval_request,
    build_post_action_hook_input,
    build_pre_action_hook_input,
    evaluate_action,
    obtain_approval,
    run_post_hooks,
    run_pre_hooks,
)
from dqagent.workspace import PathKind, Workspace, WorkspaceScope


def make_workspace(tmp_path, *, display_text: str = "patch target.txt") -> Workspace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "target.txt").write_text("old content", encoding="utf-8")
    return Workspace(WorkspaceScope("fixture", tmp_path))


def make_action(
    *,
    preconditions: EffectPreconditions | None = None,
    display: str = "patch target.txt",
):
    return PreparedAction(
        ActionKind.PATCH,
        EffectKind.WORKSPACE_MUTATION,
        "fixture",
        logical_targets=(PurePosixPath("target.txt"),),
        normalized_fields={"replacement": "new content"},
        preconditions=preconditions or EffectPreconditions(),
        limits=EffectiveLimits(),
        display_text=display,
        secret_values=(),
    )


def make_approval_fixture(tmp_path, *, preconditions: EffectPreconditions | None = None):
    workspace = make_workspace(tmp_path)
    action = make_action(preconditions=preconditions, display=f"{workspace.root} TOPSECRET")
    guard_context = GuardContext(
        workspace,
        current_preconditions=preconditions,
    )
    governance = evaluate_action(action, guard_context)
    run_context = RunContext(run_id="run-t4")
    request = build_approval_request(
        governance,
        run_context,
        guard_context,
        secret_values=("TOPSECRET",),
    )
    return workspace, action, guard_context, governance, run_context, request


class EofProvider:
    identity = "eof-provider"
    supports_deadline = False

    def request_approval(self, request, context):
        raise EOFError


class MalformedProvider:
    identity = "malformed-provider"
    supports_deadline = False

    def request_approval(self, request, context):
        return object()


class TimeoutProvider:
    def __init__(self, *, supports_deadline: bool) -> None:
        self.identity = "timeout-provider"
        self.supports_deadline = supports_deadline

    def request_approval(self, request, context):
        raise TimeoutError


def test_approval_request_and_decision_are_bounded_immutable_and_sanitized(tmp_path) -> None:
    workspace, action, guard_context, governance, run_context, request = make_approval_fixture(
        tmp_path
    )

    assert str(workspace.root) not in request.sanitized_action_display
    assert "TOPSECRET" not in request.sanitized_action_display
    assert request.action_digest == action.canonical_digest
    approved = ApprovalDecision.approve_for(request)
    assert approved.approved is True
    with pytest.raises(FrozenInstanceError):
        request.workspace_id = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        approved.outcome = ApprovalOutcome.REJECT  # type: ignore[misc]
    assert request.required_capabilities == ()
    assert request.preconditions == EffectPreconditions()

    response = ApprovalDecision(
        ApprovalOutcome.APPROVE,
        reason=f"{workspace.root} TOPSECRET",
        run_id=request.run_id,
        workspace_id=request.workspace_id,
        action_digest=request.action_digest,
        preconditions=request.preconditions,
        required_capabilities=request.required_capabilities,
        available_capabilities=request.available_capabilities,
        backend_identity=request.backend_identity,
    )
    result = obtain_approval(
        request,
        provider=ScriptedApprovalProvider([response]),
        context=run_context,
        workspace=workspace,
        secret_values=("TOPSECRET",),
        action=action,
        guard_context=guard_context,
    )
    assert result.outcome is ApprovalOutcome.APPROVE
    assert str(workspace.root) not in result.reason
    assert "TOPSECRET" not in result.reason


def test_direct_approval_request_is_sanitized_before_provider_call(tmp_path) -> None:
    workspace, action, guard_context, _governance, run_context, request = make_approval_fixture(
        tmp_path
    )
    unsafe_request = replace(
        request,
        policy_reason=f"review {workspace.root} TOPSECRET",
        sanitized_action_display=f"show {workspace.root} TOPSECRET",
    )

    class CapturingProvider:
        identity = "capturing-provider"
        supports_deadline = False

        def __init__(self) -> None:
            self.calls = 0
            self.seen = None

        def request_approval(self, request, context):
            del context
            self.calls += 1
            self.seen = request
            return ApprovalDecision(
                ApprovalOutcome.REJECT,
                "rejected",
                provider_identity=self.identity,
            )

    provider = CapturingProvider()
    result = obtain_approval(
        unsafe_request,
        provider=provider,
        context=run_context,
        workspace=workspace,
        secret_values=("TOPSECRET",),
        action=action,
        guard_context=guard_context,
    )

    assert result.outcome is ApprovalOutcome.REJECT
    assert provider.calls == 1
    assert provider.seen is not None
    assert str(workspace.root) not in provider.seen.policy_reason
    assert str(workspace.root) not in provider.seen.sanitized_action_display
    assert "TOPSECRET" not in provider.seen.policy_reason
    assert "TOPSECRET" not in provider.seen.sanitized_action_display


def test_approval_and_hooks_fail_closed_on_workspace_binding_mismatch(tmp_path) -> None:
    workspace, action, guard_context, governance, context, request = make_approval_fixture(tmp_path)
    other_workspace = make_workspace(tmp_path / "other")

    provider = ScriptedApprovalProvider([ApprovalOutcome.APPROVE])
    approval = obtain_approval(
        request,
        provider=provider,
        context=context,
        workspace=other_workspace,
        secret_values=("TOPSECRET",),
        action=action,
        guard_context=guard_context,
    )
    assert approval.outcome is ApprovalOutcome.DRIFT
    assert provider.calls == 0

    class LeakingProvider:
        identity = "leaking-provider"
        supports_deadline = False

        def request_approval(self, request, context):
            del request, context
            return ApprovalDecision(
                ApprovalOutcome.REJECT,
                reason=f"provider saw {workspace.root} TOPSECRET",
                provider_identity=self.identity,
            )

    request_only = obtain_approval(
        request,
        provider=LeakingProvider(),
        context=context,
        workspace=other_workspace,
        secret_values=("TOPSECRET",),
    )
    assert request_only.outcome is ApprovalOutcome.MALFORMED
    assert str(workspace.root) not in request_only.reason

    pre_input = build_pre_action_hook_input(
        governance,
        context,
        guard_context,
        approval_decision=ApprovalDecision.approve_for(request),
        secret_values=("TOPSECRET",),
    )
    with pytest.raises(ValueError):
        build_post_action_hook_input(
            pre_input,
            effect_state=EffectState.COMPLETE,
            executor_attempts=1,
            workspace=other_workspace,
            secret_values=("TOPSECRET",),
        )

    pre_events: list[str] = []
    pre_result = run_pre_hooks(
        [RecordingHook("pre", pre_events, HookOutcome.COMPLETED)],
        pre_input,
        context=context,
        workspace=other_workspace,
        secret_values=("TOPSECRET",),
    )
    assert pre_result.proceeded is False
    assert pre_result.results[0].outcome is HookOutcome.MALFORMED
    assert pre_events == []

    post_input = build_post_action_hook_input(
        pre_input,
        effect_state=EffectState.COMPLETE,
        executor_attempts=1,
        workspace=workspace,
        secret_values=("TOPSECRET",),
    )
    post_result = run_post_hooks(
        [RaisingHook(workspace)],
        post_input,
        context=context,
        workspace=other_workspace,
        secret_values=("TOPSECRET",),
    )
    assert post_result.results[0].outcome is HookOutcome.MALFORMED
    assert post_result.results[0].reason == "hook_workspace_binding_drift"


def test_capability_bound_rejects_overlong_raw_iterable_before_full_consumption(tmp_path) -> None:
    _workspace, _action, _guard_context, _governance, _context, request = make_approval_fixture(
        tmp_path
    )

    class CountingCapabilities:
        def __init__(self, count: int) -> None:
            self.remaining = count
            self.consumed = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.remaining == 0:
                raise StopIteration
            self.remaining -= 1
            self.consumed += 1
            return IsolationCapability.DIRECT_ARGV

    capabilities = CountingCapabilities(1_000)
    with pytest.raises(ValueError, match="approval required capabilities"):
        replace(request, required_capabilities=capabilities)
    assert capabilities.consumed == len(IsolationCapability) + 1


@pytest.mark.parametrize(
    "effect_state",
    [EffectState.PARTIAL, EffectState.COMPLETE, EffectState.UNKNOWN],
)
def test_unattempted_post_hook_input_cannot_claim_effect(effect_state, tmp_path) -> None:
    workspace, _action, guard_context, governance, context, request = make_approval_fixture(
        tmp_path
    )
    pre_input = build_pre_action_hook_input(
        governance,
        context,
        guard_context,
        approval_decision=ApprovalDecision.approve_for(request),
        secret_values=("TOPSECRET",),
    )

    with pytest.raises(ValueError, match="unattempted post-hook input"):
        build_post_action_hook_input(
            pre_input,
            effect_state=effect_state,
            executor_attempts=0,
            workspace=workspace,
            secret_values=("TOPSECRET",),
        )


def test_safe_truncation_is_not_treated_as_sanitization_failure_and_secret_iterables_are_reused(
    tmp_path,
) -> None:
    workspace, action, guard_context, governance, run_context, _request = make_approval_fixture(
        tmp_path
    )
    truncated = build_approval_request(
        governance,
        run_context,
        guard_context,
        secret_values=("TOPSECRET",),
        max_text_characters=16,
    )
    assert truncated.sanitization_failed is False
    assert len(truncated.sanitized_action_display) <= 16

    def approve_with_raw_reason(request, context):
        return ApprovalDecision.approve_for(
            request,
            reason=f"{workspace.root} TOPSECRET",
        )

    result = authorize_action(
        governance,
        provider=ScriptedApprovalProvider([approve_with_raw_reason]),
        context=run_context,
        guard_context=guard_context,
        secret_values=(secret for secret in ("TOPSECRET",)),
    )
    assert result.approved is True
    assert str(workspace.root) not in result.reason
    assert "TOPSECRET" not in result.reason


@pytest.mark.parametrize(
    ("provider", "outcome", "reason"),
    [
        (NonInteractiveApprovalProvider(), ApprovalOutcome.UNAVAILABLE, "non_interactive"),
        (EofProvider(), ApprovalOutcome.UNAVAILABLE, "eof"),
        (MalformedProvider(), ApprovalOutcome.MALFORMED, "malformed"),
    ],
)
def test_approval_provider_fail_closed_outcomes(tmp_path, provider, outcome, reason) -> None:
    workspace, action, guard_context, _governance, run_context, request = make_approval_fixture(
        tmp_path
    )
    result = obtain_approval(
        request,
        provider=provider,
        context=run_context,
        workspace=workspace,
        action=action,
        guard_context=guard_context,
    )
    assert result.outcome is outcome
    assert reason in result.reason


def test_scripted_provider_binds_approve_once_and_does_not_cache_grants(tmp_path) -> None:
    workspace, action, guard_context, _governance, run_context, request = make_approval_fixture(
        tmp_path
    )
    provider = ScriptedApprovalProvider([ApprovalOutcome.APPROVE])
    first = obtain_approval(
        request,
        provider=provider,
        context=run_context,
        workspace=workspace,
        action=action,
        guard_context=guard_context,
    )
    second = obtain_approval(
        request,
        provider=provider,
        context=run_context,
        workspace=workspace,
        action=action,
        guard_context=guard_context,
    )
    assert first.outcome is ApprovalOutcome.APPROVE
    assert second.outcome is ApprovalOutcome.UNAVAILABLE
    assert provider.calls == 2


def test_reject_and_identity_mismatch_never_reach_revalidation_effect_boundary(tmp_path) -> None:
    workspace, action, guard_context, _governance, run_context, request = make_approval_fixture(
        tmp_path
    )
    rejected = obtain_approval(
        request,
        provider=ScriptedApprovalProvider(
            [ApprovalDecision(ApprovalOutcome.REJECT, "user_rejected")]
        ),
        context=run_context,
        workspace=workspace,
        action=action,
        guard_context=guard_context,
    )
    mismatched = ApprovalDecision(
        ApprovalOutcome.APPROVE,
        run_id="other-run",
        workspace_id=request.workspace_id,
        action_digest=request.action_digest,
        preconditions=request.preconditions,
        required_capabilities=request.required_capabilities,
        available_capabilities=request.available_capabilities,
        backend_identity=request.backend_identity,
    )
    identity_result = obtain_approval(
        request,
        provider=ScriptedApprovalProvider([mismatched]),
        context=run_context,
        workspace=workspace,
        action=action,
        guard_context=guard_context,
    )
    assert rejected.outcome is ApprovalOutcome.REJECT
    assert identity_result.outcome is ApprovalOutcome.IDENTITY_MISMATCH


def test_approval_drift_is_checked_against_execution_context(tmp_path) -> None:
    workspace, action, guard_context, governance, run_context, request = make_approval_fixture(
        tmp_path
    )
    changed_capabilities = replace(
        guard_context,
        available_capabilities=frozenset({IsolationCapability.DIRECT_ARGV}),
    )
    result = obtain_approval(
        request,
        provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVE]),
        context=run_context,
        workspace=workspace,
        action=action,
        guard_context=changed_capabilities,
    )
    assert result.outcome is ApprovalOutcome.DRIFT

    precondition = EffectPrecondition(
        PurePosixPath("target.txt"),
        expected_kind=PathKind.REGULAR_FILE.value,
        expected_sha256=sha256(b"old content").hexdigest(),
        must_exist=True,
    )
    _workspace, preconditioned_action, initial_context, preconditioned_governance, context, preq = (
        make_approval_fixture(
            tmp_path / "preconditioned",
            preconditions=EffectPreconditions((precondition,)),
        )
    )
    stale_context = replace(initial_context, current_preconditions=EffectPreconditions())
    stale_result = obtain_approval(
        preq,
        provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVE]),
        context=context,
        workspace=_workspace,
        action=preconditioned_action,
        guard_context=stale_context,
    )
    assert stale_result.outcome is ApprovalOutcome.DRIFT
    assert preconditioned_governance.policy_decision.outcome is PolicyOutcome.REQUIRE_APPROVAL


def test_cancellation_is_control_error_and_deadline_timeout_requires_provider_support(
    tmp_path,
) -> None:
    workspace, action, guard_context, _governance, run_context, request = make_approval_fixture(
        tmp_path
    )
    run_context.cancel("user cancelled approval")
    with pytest.raises(RunCancelledError):
        obtain_approval(
            request,
            provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVE]),
            context=run_context,
            workspace=workspace,
            action=action,
            guard_context=guard_context,
        )

    active_context = RunContext(run_id="run-t4")
    supported = obtain_approval(
        request,
        provider=TimeoutProvider(supports_deadline=True),
        context=active_context,
        workspace=workspace,
    )
    unsupported = obtain_approval(
        request,
        provider=TimeoutProvider(supports_deadline=False),
        context=active_context,
        workspace=workspace,
    )
    assert supported.outcome is ApprovalOutcome.TIMED_OUT
    assert unsupported.outcome is ApprovalOutcome.UNAVAILABLE


class RecordingHook:
    def __init__(self, identity: str, events: list[str], outcome: HookOutcome) -> None:
        self.identity = identity
        self.events = events
        self.outcome = outcome

    def __call__(self, input_value):
        self.events.append(self.identity)
        return HookResult(self.outcome, f"{self.identity}-result")


class RaisingHook:
    identity = "raising-hook"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def __call__(self, input_value):
        raise RuntimeError(f"{self.workspace.root} TOPSECRET")


def test_required_and_optional_pre_hooks_are_ordered_and_required_failure_blocks(tmp_path) -> None:
    workspace, _action, guard_context, governance, context, request = make_approval_fixture(
        tmp_path
    )
    approval = ApprovalDecision.approve_for(request)
    pre_input = build_pre_action_hook_input(
        governance,
        context,
        guard_context,
        approval_decision=approval,
        secret_values=("TOPSECRET",),
    )
    events: list[str] = []
    optional_failure = RecordingHook("optional", events, HookOutcome.FAILED)
    success = RecordingHook("success", events, HookOutcome.COMPLETED)
    optional_result = run_pre_hooks(
        [
            PreActionHookSpec(optional_failure, HookMode.OPTIONAL),
            PreActionHookSpec(success, HookMode.REQUIRED),
        ],
        pre_input,
        context=context,
        workspace=workspace,
        secret_values=("TOPSECRET",),
    )
    assert optional_result.proceeded is True
    assert [result.hook_id for result in optional_result.results] == ["optional", "success"]
    assert events == ["optional", "success"]

    events.clear()
    required_failure = RecordingHook("required", events, HookOutcome.FAILED)
    unreachable = RecordingHook("unreachable", events, HookOutcome.COMPLETED)
    blocked = run_pre_hooks(
        [required_failure, unreachable],
        pre_input,
        context=context,
        workspace=workspace,
        secret_values=("TOPSECRET",),
    )
    assert blocked.blocked is True
    assert blocked.can_execute is False
    assert events == ["required"]


def test_post_hooks_run_in_order_after_effect_and_sanitize_failures(tmp_path) -> None:
    workspace, _action, guard_context, governance, context, request = make_approval_fixture(
        tmp_path
    )
    pre_input = build_pre_action_hook_input(
        governance,
        context,
        guard_context,
        approval_decision=ApprovalDecision.approve_for(request),
        secret_values=("TOPSECRET",),
    )
    post_input = build_post_action_hook_input(
        pre_input,
        effect_state=EffectState.COMPLETE,
        executor_attempts=1,
        workspace=workspace,
        secret_values=("TOPSECRET",),
        effect_diagnostic=f"{workspace.root} TOPSECRET",
    )
    events: list[str] = []
    first = RaisingHook(workspace)
    second = RecordingHook("second", events, HookOutcome.COMPLETED)
    result = run_post_hooks(
        [first, second],
        post_input,
        context=context,
        workspace=workspace,
        secret_values=("TOPSECRET",),
    )
    assert result.proceeded is True
    assert [item.hook_id for item in result.results] == ["raising-hook", "second"]
    assert result.results[0].outcome is HookOutcome.FAILED
    assert str(workspace.root) not in result.results[0].sanitized_diagnostics[0]
    assert "TOPSECRET" not in result.results[0].sanitized_diagnostics[0]
    assert events == ["second"]
    assert post_input.effect_state is EffectState.COMPLETE
    assert post_input.executor_attempts == 1
    assert not hasattr(pre_input, "workspace")
    assert not hasattr(pre_input, "subprocess")
    assert not hasattr(pre_input, "executor")


def test_authorize_action_accepts_fresh_context_and_does_not_touch_executor(tmp_path) -> None:
    workspace, _action, guard_context, governance, context, _request = make_approval_fixture(
        tmp_path
    )
    executor_calls = 0
    result = authorize_action(
        governance,
        provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVE]),
        context=context,
        guard_context=guard_context,
        execution_guard_context=replace(guard_context),
        secret_values=("TOPSECRET",),
    )
    assert result.approved is True
    assert executor_calls == 0
