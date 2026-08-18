from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import PurePosixPath

import pytest

from dqagent.subprocesses import IsolationCapability
from dqagent.tool_governance import (
    ActionKind,
    ActionRecord,
    DefaultActionPolicy,
    EffectiveLimits,
    EffectKind,
    EffectPrecondition,
    EffectPreconditions,
    GuardContext,
    GuardName,
    PolicyDecision,
    PolicyOutcome,
    PreparedAction,
    build_action_record,
    evaluate_action,
    evaluate_guards,
)
from dqagent.workspace import PathKind, Workspace, WorkspaceScope

WORKSPACE_ID = "fixture"
OLD_DIGEST = sha256(b"old content").hexdigest()
EMPTY_PRECONDITIONS = EffectPreconditions()
DEFAULT_POLICY_DECISION = PolicyDecision(PolicyOutcome.ALLOW, "test")


def make_workspace(tmp_path, *, protected_paths=(), secret_paths=()):
    (tmp_path / "target.txt").write_text("old content", encoding="utf-8")
    return Workspace(
        WorkspaceScope(
            WORKSPACE_ID,
            tmp_path,
            protected_paths=tuple(protected_paths),
            secret_paths=tuple(secret_paths),
        )
    )


def make_read_action(
    *,
    workspace_id: str = WORKSPACE_ID,
    target: str = "target.txt",
    display_text: str = "read target.txt",
    normalized_fields=None,
    required_capabilities=(),
    limits=None,
):
    return PreparedAction(
        ActionKind.READ,
        EffectKind.NONE,
        workspace_id,
        logical_targets=(PurePosixPath(target),),
        normalized_fields=normalized_fields or {},
        required_capabilities=frozenset(required_capabilities),
        limits=limits or EffectiveLimits(),
        display_text=display_text,
        secret_values=(),
    )


def make_patch_action(*, preconditions=None, limits=None):
    return PreparedAction(
        ActionKind.PATCH,
        EffectKind.WORKSPACE_MUTATION,
        WORKSPACE_ID,
        logical_targets=(PurePosixPath("target.txt"),),
        normalized_fields={"replacement": "new content"},
        preconditions=preconditions or EMPTY_PRECONDITIONS,
        limits=limits or EffectiveLimits(),
        display_text="patch target.txt",
        secret_values=(),
    )


def make_command_action():
    return PreparedAction(
        ActionKind.COMMAND,
        EffectKind.PROCESS_EXECUTION,
        WORKSPACE_ID,
        argv=("python", "-c", "print('ok')"),
        executable_identity="python",
        environment_identity=("PATH",),
        normalized_fields={"argv_mode": "direct"},
        display_text="run command",
        secret_values=(),
    )


def make_context(workspace, **kwargs):
    return GuardContext(workspace, **kwargs)


class RecordingPolicy:
    identity = "recording-policy"

    def __init__(self, response=None) -> None:
        response = response or DEFAULT_POLICY_DECISION
        self.response = response
        self.calls = 0

    def decide(self, action: PreparedAction) -> PolicyDecision:
        assert isinstance(action, PreparedAction)
        self.calls += 1
        return self.response


class RaisingPolicy:
    identity = "raising-policy"

    def decide(self, action: PreparedAction) -> PolicyDecision:
        del action
        raise RuntimeError("dependency failure must not escape")


class MalformedPolicy:
    identity = "malformed-policy"

    def decide(self, action: PreparedAction) -> PolicyDecision:
        del action
        return object()  # type: ignore[return-value]


def test_canonical_json_and_digest_match_golden_identity(tmp_path) -> None:
    action = PreparedAction(
        ActionKind.PATCH,
        EffectKind.WORKSPACE_MUTATION,
        WORKSPACE_ID,
        logical_targets=(PurePosixPath("src/app.py"),),
        cwd=PurePosixPath("src"),
        environment_identity=("LANG", "PATH"),
        normalized_fields={"z": "last", "a": {"end": 3, "start": 1}},
        preconditions=EffectPreconditions(
            (
                EffectPrecondition(
                    PurePosixPath("src/app.py"),
                    expected_kind=PathKind.REGULAR_FILE.value,
                    expected_sha256=OLD_DIGEST,
                    must_exist=True,
                ),
            )
        ),
        required_capabilities=(IsolationCapability.DIRECT_ARGV,),
        limits=EffectiveLimits(max_output_characters=1_000),
        display_text=f"display-only {tmp_path}",
        secret_values=(),
    )

    assert action.canonical_json == (
        '{"action_kind":"patch","argv":[],"canonical_version":1,"cwd":"src",'
        '"effect_kind":"workspace_mutation","effective_limits":{"max_argv_characters":32000,'
        '"max_argv_items":128,"max_duration_seconds":30.0,"max_input_characters":64000,'
        '"max_output_characters":1000},"environment_identity":["LANG","PATH"],'
        '"executable_identity":null,"logical_targets":["src/app.py"],'
        '"normalized_fields":{"a":{"end":3,"start":1},"z":"last"},'
        '"preconditions":[{"expected_kind":"regular_file","expected_sha256":"'
        f"{OLD_DIGEST}"
        '","logical_path":"src/app.py","must_exist":true}],'
        '"required_capabilities":["direct_argv"],"workspace_id":"fixture"}'
    )
    assert action.canonical_digest == (
        "a7819c4a18db8391e22c6623882caa963e188d54aa1e4b636cc6008cfded1416"
    )


def test_digest_is_sensitive_to_effect_fields_but_not_display_text_or_root(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    secret = "TOPSECRET"
    base = make_read_action(
        display_text=f"{workspace.root} {secret}",
        normalized_fields={"query": "needle"},
    )
    display_changed = replace(base, display_text="different presentation", secret_values=())
    field_changed = replace(base, normalized_fields={"query": "different"}, secret_values=())
    target_changed = replace(
        base,
        logical_targets=(PurePosixPath("other.txt"),),
        secret_values=(),
    )

    assert base.canonical_digest == display_changed.canonical_digest
    assert base.canonical_digest != field_changed.canonical_digest
    assert base.canonical_digest != target_changed.canonical_digest
    assert str(workspace.root) not in base.canonical_json
    assert secret not in base.canonical_json
    assert "different presentation" not in base.canonical_json
    assert "PreparedAction" not in base.canonical_json
    with pytest.raises((TypeError, FrozenInstanceError)):
        base.normalized_fields["new"] = "value"  # type: ignore[index]


@pytest.mark.parametrize(
    ("argv", "normalized_fields", "secret_values"),
    [
        (("tool", "--value", "TOPSECRET-123"), {}, ()),
        (("tool", "--value", "literal-secret"), {}, ("literal-secret",)),
        (("tool", "--value"), {"argument": "TOPSECRET-123"}, ()),
        (("tool", "--value"), {"outer": {"value": "nested-secret"}}, ("nested-secret",)),
        (("tool", "--value"), {"target": r"C:\host\root"}, ()),
    ],
)
def test_preparation_rejects_secret_or_absolute_effect_identity(
    argv, normalized_fields, secret_values
) -> None:
    with pytest.raises(ValueError, match="secret-free|absolute"):
        PreparedAction(
            ActionKind.COMMAND,
            EffectKind.PROCESS_EXECUTION,
            WORKSPACE_ID,
            argv=argv,
            executable_identity="tool",
            normalized_fields=normalized_fields,
            secret_values=secret_values,
        )


def test_action_replace_requires_secret_validation_context() -> None:
    action = PreparedAction(
        ActionKind.COMMAND,
        EffectKind.PROCESS_EXECUTION,
        WORKSPACE_ID,
        argv=("tool",),
        normalized_fields={"value": "safe"},
        secret_values=("literal-secret",),
    )

    with pytest.raises(ValueError, match="secret_values"):
        replace(action, normalized_fields={"value": "literal-secret"})
    with pytest.raises(ValueError, match="secret-free"):
        replace(
            action,
            normalized_fields={"value": "literal-secret"},
            secret_values=("literal-secret",),
        )


@pytest.mark.parametrize("field", ("secret_values", "logical_targets", "argv"))
def test_prepared_action_public_iterables_are_bounded(field: str) -> None:
    class InfiniteValues:
        def __init__(self) -> None:
            self.consumed = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.consumed += 1
            if field == "secret_values":
                return f"secret-{self.consumed}"
            if field == "logical_targets":
                return PurePosixPath(f"target-{self.consumed}.txt")
            return f"argv-{self.consumed}"

    values = InfiniteValues()
    kwargs = {
        "logical_targets": (PurePosixPath("target.txt"),),
        "secret_values": (),
        "argv": (),
    }
    kwargs[field] = values

    with pytest.raises(ValueError, match="item bound"):
        PreparedAction(
            ActionKind.READ,
            EffectKind.NONE,
            WORKSPACE_ID,
            **kwargs,
        )

    assert values.consumed == 129


def test_effect_preconditions_public_iterable_is_bounded() -> None:
    consumed = 0

    def preconditions():
        nonlocal consumed
        for index in range(10_000):
            consumed += 1
            yield EffectPrecondition(PurePosixPath(f"target-{index}.txt"))

    with pytest.raises(ValueError, match="item bound"):
        EffectPreconditions(preconditions())

    assert consumed == 33


def test_digest_changes_for_each_effect_identity_field() -> None:
    read = make_read_action()
    command = make_command_action()
    precondition = EffectPreconditions(
        (EffectPrecondition(PurePosixPath("target.txt"), must_exist=True),)
    )
    variants = (
        (read, replace(read, action_kind=ActionKind.SEARCH, secret_values=())),
        (
            read,
            replace(
                read,
                effect_kind=EffectKind.WORKSPACE_MUTATION,
                action_kind=ActionKind.PATCH,
                secret_values=(),
            ),
        ),
        (read, replace(read, workspace_id="other-workspace", secret_values=())),
        (
            read,
            replace(read, logical_targets=(PurePosixPath("other.txt"),), secret_values=()),
        ),
        (command, replace(command, cwd=PurePosixPath("src"), secret_values=())),
        (
            command,
            replace(command, argv=("python", "-c", "print('changed')"), secret_values=()),
        ),
        (command, replace(command, executable_identity="python3", secret_values=())),
        (
            command,
            replace(command, environment_identity=("LANG", "PATH"), secret_values=()),
        ),
        (read, replace(read, normalized_fields={"query": "changed"}, secret_values=())),
        (command, replace(command, preconditions=precondition, secret_values=())),
        (
            command,
            replace(
                command,
                required_capabilities=(IsolationCapability.DIRECT_ARGV,),
                secret_values=(),
            ),
        ),
        (
            command,
            replace(command, limits=EffectiveLimits(max_output_characters=1), secret_values=()),
        ),
    )

    for original, variant in variants:
        assert variant.canonical_digest != original.canonical_digest


def test_max_governed_calls_is_first_and_does_not_call_policy(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    policy = RecordingPolicy()
    decision = evaluate_action(
        make_read_action(),
        make_context(workspace, governed_call_count=1, max_governed_calls=1),
        policy,
    )

    assert [result.name for result in decision.guard_results] == [GuardName.MAX_GOVERNED_CALLS]
    assert decision.policy_decision.outcome is PolicyOutcome.DENY
    assert decision.policy_evaluated is False
    assert policy.calls == 0


def test_successful_guards_have_fixed_order_and_policy_runs_once(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    policy = RecordingPolicy()
    decision = evaluate_action(make_read_action(), make_context(workspace), policy)

    assert [result.name for result in decision.guard_results] == list(
        (
            GuardName.MAX_GOVERNED_CALLS,
            GuardName.WORKSPACE_IDENTITY,
            GuardName.CURRENT_CONTAINMENT,
            GuardName.PROTECTED_SECRET,
            GuardName.LIMITS,
            GuardName.PRECONDITIONS,
            GuardName.CAPABILITIES,
        )
    )
    assert decision.guards_passed is True
    assert decision.admitted is True
    assert policy.calls == 1


@pytest.mark.parametrize(
    ("target", "guard"),
    [("protected.txt", GuardName.PROTECTED_SECRET), ("secret.txt", GuardName.PROTECTED_SECRET)],
)
def test_protected_and_secret_resources_fail_after_containment(tmp_path, target, guard) -> None:
    workspace = make_workspace(
        tmp_path,
        protected_paths=(PurePosixPath("protected.txt"),),
        secret_paths=(PurePosixPath("secret.txt"),),
    )
    decision = evaluate_action(
        make_read_action(target=target),
        make_context(workspace),
        RecordingPolicy(),
    )

    assert [result.name for result in decision.guard_results][-1] is guard
    assert decision.policy_evaluated is False
    assert decision.policy_decision.outcome is PolicyOutcome.DENY


def test_precondition_paths_use_containment_and_protection_guards(tmp_path) -> None:
    workspace = make_workspace(tmp_path, secret_paths=(PurePosixPath("secret.txt"),))
    preconditions = EffectPreconditions(
        (EffectPrecondition(PurePosixPath("secret.txt"), must_exist=False),)
    )
    decision = evaluate_action(
        make_patch_action(preconditions=preconditions),
        make_context(workspace, current_preconditions=preconditions),
    )

    assert [result.name for result in decision.guard_results] == [
        GuardName.MAX_GOVERNED_CALLS,
        GuardName.WORKSPACE_IDENTITY,
        GuardName.CURRENT_CONTAINMENT,
        GuardName.PROTECTED_SECRET,
    ]
    assert decision.guard_results[-1].passed is False
    assert decision.policy_decision.outcome is PolicyOutcome.DENY


def test_limits_preconditions_and_capabilities_are_hard_guards(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    limited = evaluate_action(
        make_read_action(limits=EffectiveLimits(max_output_characters=100)),
        make_context(
            workspace,
            configured_limits=EffectiveLimits(max_output_characters=50),
        ),
    )
    assert limited.guard_results[-1].name is GuardName.LIMITS

    precondition = EffectPrecondition(
        PurePosixPath("target.txt"),
        expected_kind=PathKind.REGULAR_FILE.value,
        expected_sha256=OLD_DIGEST,
        must_exist=True,
    )
    missing_precondition = evaluate_action(
        make_patch_action(preconditions=EffectPreconditions((precondition,))),
        make_context(workspace),
    )
    assert missing_precondition.guard_results[-1].name is GuardName.PRECONDITIONS
    assert missing_precondition.guard_results[-1].passed is False

    capability_missing = evaluate_action(
        make_read_action(required_capabilities=(IsolationCapability.DIRECT_ARGV,)),
        make_context(workspace),
    )
    assert capability_missing.guard_results[-1].name is GuardName.CAPABILITIES
    assert capability_missing.guard_results[-1].passed is False

    all_pass = evaluate_action(
        make_patch_action(preconditions=EffectPreconditions((precondition,))),
        make_context(
            workspace,
            current_preconditions=EffectPreconditions((precondition,)),
            available_capabilities=frozenset({IsolationCapability.DIRECT_ARGV}),
        ),
    )
    assert all_pass.guards_passed is True


@pytest.mark.parametrize(
    ("action_factory", "outcome"),
    [
        (make_read_action, PolicyOutcome.ALLOW),
        (make_command_action, PolicyOutcome.REQUIRE_APPROVAL),
        (make_patch_action, PolicyOutcome.REQUIRE_APPROVAL),
    ],
)
def test_default_policy_has_only_tri_state_outcomes(tmp_path, action_factory, outcome) -> None:
    workspace = make_workspace(tmp_path)
    decision = evaluate_action(action_factory(), make_context(workspace))

    assert isinstance(decision.policy_decision, PolicyDecision)
    assert decision.policy_decision.outcome is outcome
    assert isinstance(DefaultActionPolicy().identity, str)


@pytest.mark.parametrize("policy", [RaisingPolicy(), MalformedPolicy()])
def test_policy_exception_or_malformed_response_fails_closed(tmp_path, policy) -> None:
    workspace = make_workspace(tmp_path)
    decision = evaluate_action(make_read_action(), make_context(workspace), policy)

    assert decision.guards_passed is True
    assert decision.policy_evaluated is True
    assert decision.policy_decision.outcome is PolicyOutcome.DENY
    assert decision.policy_decision.reason in {
        "policy_dependency_failure",
        "malformed_policy_response",
    }


def test_malformed_capability_dependency_fails_closed(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    context = make_context(workspace)
    object.__setattr__(context, "available_capabilities", ("not-a-capability",))
    decision = evaluate_action(
        make_read_action(required_capabilities=(IsolationCapability.DIRECT_ARGV,)),
        context,
    )

    assert decision.guard_results[-1].name is GuardName.CAPABILITIES
    assert decision.guard_results[-1].passed is False
    assert decision.policy_decision.outcome is PolicyOutcome.DENY


def test_call_capacity_is_checked_without_reservation_or_validator_budget(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    context = make_context(workspace, max_governed_calls=1)
    action = make_read_action()

    first = evaluate_guards(action, context)
    second = evaluate_guards(action, context)

    assert all(result.passed for result in first)
    assert all(result.passed for result in second)
    assert context.governed_call_count == 0
    assert not hasattr(context, "validator_calls")


def test_record_is_bounded_and_sanitized_without_executor_attempt(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    secret = "TOPSECRET"
    action = make_read_action(display_text=f"{workspace.root} {secret}")
    decision = evaluate_action(action, make_context(workspace))
    record = build_action_record(
        decision,
        workspace=workspace,
        secret_values=(secret,),
        backend_identity="local-backend",
        diagnostics=tuple(f"{workspace.root} {secret} {'x' * 1000}" for _ in range(10)),
    )

    assert isinstance(record, ActionRecord)
    assert record.executor_attempts == 0
    assert record.effect_state.value == "none"
    assert len(record.sanitized_diagnostics) == 8
    assert record.diagnostics_truncated is True
    rendered = repr(record)
    assert str(workspace.root) not in rendered
    assert secret not in rendered
    assert record.action_digest == action.canonical_digest


def test_record_diagnostics_consumption_is_bounded(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    decision = evaluate_action(make_read_action(), make_context(workspace))

    class BudgetedDiagnostics:
        def __init__(self) -> None:
            self.consumed = 0

        def __iter__(self):
            return self

        def __next__(self) -> str:
            self.consumed += 1
            if self.consumed > 9:
                raise AssertionError("diagnostics were consumed beyond the bound")
            return f"diagnostic-{self.consumed}"

    diagnostics = BudgetedDiagnostics()
    record = build_action_record(
        decision,
        workspace=workspace,
        diagnostics=diagnostics,
    )

    assert diagnostics.consumed == 9
    assert record.sanitized_diagnostics == tuple(
        f"diagnostic-{index}" for index in range(1, 9)
    )
    assert record.diagnostics_truncated is True


def test_public_record_constructor_rejects_unbounded_diagnostics_iterable(tmp_path) -> None:
    workspace = make_workspace(tmp_path)
    decision = evaluate_action(make_read_action(), make_context(workspace))
    record = build_action_record(decision, workspace=workspace)

    class LateFailure:
        def __init__(self) -> None:
            self.consumed = 0

        def __iter__(self):
            return self

        def __next__(self) -> str:
            self.consumed += 1
            if self.consumed > 9:
                raise AssertionError("diagnostics were consumed beyond the public bound")
            return f"diagnostic-{self.consumed}"

    diagnostics = LateFailure()
    with pytest.raises(TypeError, match="tuple of strings"):
        replace(record, sanitized_diagnostics=diagnostics)

    assert diagnostics.consumed == 0


def test_governance_does_not_touch_an_executor(tmp_path) -> None:
    workspace = make_workspace(tmp_path)

    class Executor:
        calls = 0

        def execute(self) -> None:
            self.calls += 1

    executor = Executor()
    decision = evaluate_action(make_read_action(), make_context(workspace))

    assert decision.policy_evaluated is True
    assert executor.calls == 0
