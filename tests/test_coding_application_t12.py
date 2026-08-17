from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path, PurePosixPath
from threading import Lock

import pytest

from dqagent import coding as coding_module
from dqagent import coding_cli
from dqagent.coding import (
    CodingFailureEvidence,
    CodingRequest,
    ForegroundApprovalProvider,
    create_coding_agent_application,
)
from dqagent.coding_cli import main as coding_cli_main
from dqagent.errors import (
    AgentLoopError,
    ConfigurationError,
    RunCancelledError,
    RunDeadlineExceededError,
)
from dqagent.events import RunEventType
from dqagent.execution import RunContext
from dqagent.models import Completion, ConversationItem, ToolCall, ToolDefinition
from dqagent.repository_context import RepositoryContextLoader
from dqagent.tool_governance import (
    ApprovalOutcome,
    NonInteractiveApprovalProvider,
    ScriptedApprovalProvider,
)
from dqagent.validators import (
    TaskVerdict,
    ValidatorDefinition,
    ValidatorResult,
    ValidatorRunner,
    ValidatorStatus,
)
from dqagent.workspace import Workspace, WorkspaceObserver, WorkspaceScope


class StubLLM:
    def __init__(self, completions: Sequence[Completion], *, delay: float = 0.0) -> None:
        self._completions = iter(completions)
        self._delay = delay
        self.requests: list[tuple[tuple[ConversationItem, ...], tuple[ToolDefinition, ...]]] = []

    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolDefinition] = (),
        *,
        context: RunContext | None = None,
    ) -> Completion:
        if self._delay:
            time.sleep(self._delay)
        self.requests.append((tuple(messages), tuple(tools)))
        return next(self._completions)


def make_workspace(tmp_path: Path, **kwargs: object) -> Workspace:
    return Workspace(WorkspaceScope("fixture", tmp_path, **kwargs))


def patch_call(path: Path, old: str, new: str, call_id: str = "patch-1") -> ToolCall:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    arguments = json.dumps(
        {
            "operation": "update",
            "path": "target.txt",
            "expected_sha256": digest,
            "replacements": [
                {"old": old, "new": new, "expected_occurrences": 1},
            ],
        }
    )
    return ToolCall(call_id, "workspace_patch", arguments)


def pass_validator() -> ValidatorDefinition:
    return ValidatorDefinition("pass", (sys.executable, "-c", "pass"))


@pytest.mark.parametrize(
    ("request_factory", "error_type"),
    [
        (lambda: CodingRequest("", ("target.txt",)), ValueError),
        (lambda: CodingRequest("edit", "target.txt"), TypeError),
        (lambda: CodingRequest("edit", ()), ValueError),
        (lambda: CodingRequest("edit", (object(),)), TypeError),
        (lambda: CodingRequest("edit", ("target.txt",), "style"), TypeError),
        (lambda: CodingRequest("edit", ("target.txt",), (object(),)), ValueError),
    ],
)
def test_coding_request_rejects_unbounded_or_malformed_input(
    request_factory, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        request_factory()


def test_coding_composition_rejects_unbounded_retained_evidence(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    with pytest.raises(ValueError, match="evidence bound"):
        create_coding_agent_application(
            workspace,
            StubLLM([Completion("done")]),
            max_governed_calls=129,
        )
    with pytest.raises(ValueError, match="evidence limit"):
        create_coding_agent_application(
            workspace,
            StubLLM([Completion("done")]),
            validators=tuple(
                ValidatorDefinition(f"check-{index}", (sys.executable, "-c", "pass"))
                for index in range(129)
            ),
        )


def test_failure_evidence_rejects_unbounded_or_unsafe_limitations() -> None:
    with pytest.raises(ValueError, match="unbounded"):
        CodingFailureEvidence(observation_limitations=tuple("x" for _ in range(33)))
    with pytest.raises(ValueError, match="malformed"):
        CodingFailureEvidence(observation_limitations=("bad\x01",))


def test_full_success_retains_context_actions_diff_validators_and_order(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("one\n", encoding="utf-8")
    llm = StubLLM(
        [
            Completion(tool_calls=(patch_call(target, "one", "two"),)),
            Completion("finished"),
        ]
    )
    app = create_coding_agent_application(
        make_workspace(tmp_path),
        llm,
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVE]),
        validators=(pass_validator(),),
    )

    result = app.run(CodingRequest("update the target", ("target.txt",)))

    assert result.verdict is TaskVerdict.PASSED
    assert result.agent_result.output.content == "finished"
    assert result.context_evidence.target_paths == (PurePosixPath("target.txt"),)
    assert len(result.action_records) == 1
    assert result.diff.changed_paths == (PurePosixPath("target.txt"),)
    assert result.validator_results[0].status is ValidatorStatus.PASSED
    assert result.baseline_identity != result.final_identity
    assert any(item.startswith("backend:") for item in result.observation_limitations)

    event_types = [event.type for event in result.events]
    positions = {event_type: event_types.index(event_type) for event_type in event_types}
    assert event_types[0] is RunEventType.RUN_STARTED
    assert event_types[-1] is RunEventType.RUN_COMPLETED
    assert positions[RunEventType.CODING_REQUEST_VALIDATED] < positions[
        RunEventType.WORKSPACE_BASELINE_CAPTURED
    ]
    assert positions[RunEventType.WORKSPACE_BASELINE_CAPTURED] < positions[
        RunEventType.REPOSITORY_CONTEXT_LOADED
    ]
    assert positions[RunEventType.REPOSITORY_CONTEXT_LOADED] < positions[
        RunEventType.CONTEXT_ASSEMBLED
    ]
    assert positions[RunEventType.CONTEXT_ASSEMBLED] < positions[RunEventType.MODEL_REQUEST_STARTED]
    assert positions[RunEventType.MODEL_REQUEST_STARTED] < positions[
        RunEventType.WORKSPACE_FINAL_CAPTURED
    ]
    assert positions[RunEventType.WORKSPACE_FINAL_CAPTURED] < positions[
        RunEventType.WORKSPACE_DIFF_COMPUTED
    ]
    assert positions[RunEventType.WORKSPACE_DIFF_COMPUTED] < positions[
        RunEventType.VALIDATORS_STARTED
    ]
    assert positions[RunEventType.VALIDATORS_STARTED] < positions[
        RunEventType.VALIDATORS_COMPLETED
    ]


def test_no_validator_is_explicitly_not_validated(tmp_path: Path) -> None:
    result = create_coding_agent_application(
        make_workspace(tmp_path),
        StubLLM([Completion("read-only result")]),
    ).run(CodingRequest("inspect", ("target.txt",)))

    assert result.verdict is TaskVerdict.NOT_VALIDATED
    assert result.validator_results == ()
    assert result.diff.target_complete
    assert result.blind_spots


@pytest.mark.parametrize(
    ("validator", "expected"),
    [
        (
            ValidatorDefinition("fail", (sys.executable, "-c", "raise SystemExit(7)")),
            TaskVerdict.FAILED,
        ),
        (
            ValidatorDefinition("missing", ("definitely-not-a-validator",)),
            TaskVerdict.INDETERMINATE,
        ),
    ],
)
def test_validator_outcome_controls_verdict(
    tmp_path: Path,
    validator: ValidatorDefinition,
    expected: TaskVerdict,
) -> None:
    result = create_coding_agent_application(
        make_workspace(tmp_path),
        StubLLM([Completion("model says done")]),
        validators=(validator,),
    ).run(CodingRequest("inspect", ("target.txt",)))

    assert result.verdict is expected
    assert result.validator_results[0].status in {
        ValidatorStatus.FAILED,
        ValidatorStatus.UNAVAILABLE,
    }


def test_missing_validator_evidence_is_explicitly_indeterminate(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)

    class IncompleteValidatorRunner:
        def __init__(self, bound_workspace: Workspace) -> None:
            self.workspace = bound_workspace

        def run(self, definitions, context=None):
            del definitions, context
            return ()

    result = create_coding_agent_application(
        workspace,
        StubLLM([Completion("model says done")]),
        validators=(pass_validator(),),
        validator_runner=IncompleteValidatorRunner(workspace),  # type: ignore[arg-type]
    ).run(CodingRequest("inspect", ("target.txt",)))

    assert result.verdict is TaskVerdict.INDETERMINATE
    assert result.validator_results[0].status is ValidatorStatus.NOT_RUN
    assert result.validator_results[0].diagnostics == ("validator_not_run",)


def test_unknown_effect_makes_a_completed_run_indeterminate(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    llm = StubLLM(
        [
            Completion(
                tool_calls=(
                    ToolCall(
                        "command-1",
                        "workspace_command",
                        '{"argv":["python","-c","raise SystemExit(3)"]}',
                    ),
                )
            ),
            Completion("recovered"),
        ]
    )
    app = create_coding_agent_application(
        workspace,
        llm,
        executable_allowlist={"python": Path(sys.executable)},
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVE]),
        validators=(pass_validator(),),
    )

    result = app.run(CodingRequest("run the check", ("target.txt",)))

    assert result.verdict is TaskVerdict.INDETERMINATE
    assert result.action_records[0].effect_state.value == "unknown"
    assert result.validator_results[0].status is ValidatorStatus.PASSED


def test_action_record_collection_failure_keeps_effect_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("one\n", encoding="utf-8")

    collector_type = coding_module._RunActionRecordCollector

    class FailingCollector(collector_type):
        def close(self, run_id: str):
            del run_id
            raise RuntimeError("collector retention failed")

    monkeypatch.setattr(coding_module, "_RunActionRecordCollector", FailingCollector)
    app = create_coding_agent_application(
        make_workspace(tmp_path),
        StubLLM(
            [
                Completion(tool_calls=(patch_call(target, "one", "two"),)),
                Completion("finished"),
            ]
        ),
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVE]),
        validators=(pass_validator(),),
    )

    result = app.run(CodingRequest("update the target", ("target.txt",)))

    assert result.diff.changed_paths == (PurePosixPath("target.txt"),)
    assert result.action_records == ()
    assert result.verdict is TaskVerdict.INDETERMINATE
    assert result.validator_results[0].status is ValidatorStatus.PASSED
    assert "action_record_collection_unavailable" in result.observation_limitations
    assert not getattr(result, "rollback_claimed", False)


def test_coding_composition_rejects_mismatched_workspace_authorities(tmp_path: Path) -> None:
    workspace_a_root = tmp_path / "workspace-a"
    workspace_b_root = tmp_path / "workspace-b"
    workspace_a_root.mkdir()
    workspace_b_root.mkdir()
    (workspace_a_root / "target.txt").write_text("one\n", encoding="utf-8")
    (workspace_b_root / "target.txt").write_text("one\n", encoding="utf-8")
    workspace_a = make_workspace(workspace_a_root)
    workspace_b = make_workspace(workspace_b_root)

    with pytest.raises(ConfigurationError, match="canonical workspace") as raised:
        create_coding_agent_application(
            workspace_a,
            StubLLM([Completion("must not run")]),
            observer=WorkspaceObserver(workspace_b),
            repository_loader=RepositoryContextLoader(workspace_b),
            validator_runner=ValidatorRunner(workspace_b),
        )

    assert raised.value.category.value == "configuration"
    assert (workspace_a_root / "target.txt").read_text(encoding="utf-8") == "one\n"
    assert (workspace_b_root / "target.txt").read_text(encoding="utf-8") == "one\n"


def test_factory_reuses_one_shot_secret_names_for_validator_and_cli_output(
    tmp_path: Path,
) -> None:
    secret_name = "PRIVATE_VALUE"
    validator = ValidatorDefinition(
        "secret-check",
        (
            sys.executable,
            "-c",
            "import os; print(os.environ.get('PRIVATE_' + 'VALUE', 'MISSING'))",
        ),
        environment={secret_name: secret_name},
    )
    app = create_coding_agent_application(
        make_workspace(tmp_path),
        StubLLM([Completion("done")]),
        secret_names=(name for name in (secret_name,)),
        validators=(validator,),
    )

    result = app.run(CodingRequest("inspect", ("target.txt",)))

    assert result.verdict is TaskVerdict.PASSED
    assert result.validator_results[0].stdout.strip() == "MISSING"
    assert secret_name not in result.validator_results[0].stdout
    assert secret_name not in result.validator_results[0].stderr
    assert secret_name not in repr(result)

    rendered = StringIO()
    coding_cli._render_result(
        result,
        workspace=make_workspace(tmp_path),
        secret_values=(),
        output=rendered,
    )
    assert secret_name not in rendered.getvalue()

    tuple_control = create_coding_agent_application(
        make_workspace(tmp_path),
        StubLLM([Completion("done")]),
        secret_names=(secret_name,),
        validators=(validator,),
    ).run(CodingRequest("inspect", ("target.txt",)))
    assert tuple_control.validator_results[0].stdout.strip() == "MISSING"


def test_factory_bounds_secret_name_consumption_before_freezing(tmp_path: Path) -> None:
    consumed = 0

    def names():
        nonlocal consumed
        for index in range(10_000):
            consumed += 1
            yield f"SECRET_{index}"

    with pytest.raises(ValueError, match="secret names exceeds its item bound"):
        create_coding_agent_application(
            make_workspace(tmp_path),
            StubLLM([Completion("must not run")]),
            secret_names=names(),
        )

    assert consumed == 129


def test_factory_rejects_non_terminating_secret_names_at_the_bound(
    tmp_path: Path,
) -> None:
    class NonTerminatingNames:
        def __init__(self) -> None:
            self.consumed = 0

        def __iter__(self) -> Iterator[str]:
            return self

        def __next__(self) -> str:
            self.consumed += 1
            if self.consumed > 129:
                raise AssertionError("secret names were consumed past the bound")
            return f"SECRET_{self.consumed}"

    names = NonTerminatingNames()
    with pytest.raises(ValueError, match="secret names exceeds its item bound"):
        create_coding_agent_application(
            make_workspace(tmp_path),
            StubLLM([Completion("must not run")]),
            secret_names=names,
        )

    assert names.consumed == 129


def test_factory_bounds_secret_value_consumption_before_freezing(tmp_path: Path) -> None:
    consumed = 0

    def values():
        nonlocal consumed
        for index in range(10_000):
            consumed += 1
            yield f"VALUE_{index}"

    with pytest.raises(ValueError, match="secret values exceed their bound"):
        create_coding_agent_application(
            make_workspace(tmp_path),
            StubLLM([Completion("must not run")]),
            secret_values=values(),
        )

    assert consumed == 65


def test_factory_rejects_non_terminating_secret_values_at_the_bound(
    tmp_path: Path,
) -> None:
    class NonTerminatingValues:
        def __init__(self) -> None:
            self.consumed = 0

        def __iter__(self) -> Iterator[str]:
            return self

        def __next__(self) -> str:
            self.consumed += 1
            if self.consumed > 65:
                raise AssertionError("secret values were consumed past the bound")
            return f"VALUE_{self.consumed}"

    values = NonTerminatingValues()
    with pytest.raises(ValueError, match="secret values exceed their bound"):
        create_coding_agent_application(
            make_workspace(tmp_path),
            StubLLM([Completion("must not run")]),
            secret_values=values,
        )

    assert values.consumed == 65


def test_runtime_failure_keeps_original_error_and_attaches_final_observation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("one\n", encoding="utf-8")
    llm = StubLLM([Completion(tool_calls=(patch_call(target, "one", "two"),))])
    app = create_coding_agent_application(
        make_workspace(tmp_path),
        llm,
        max_iterations=1,
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVE]),
    )

    with pytest.raises(AgentLoopError) as raised:
        app.run(CodingRequest("make the edit", ("target.txt",)))

    error = raised.value
    evidence = error.coding_failure_evidence  # type: ignore[attr-defined]
    assert isinstance(evidence, CodingFailureEvidence)
    assert evidence.final is not None
    assert evidence.diff is not None
    assert evidence.diff.changed_paths == (PurePosixPath("target.txt"),)
    assert evidence.action_records[0].executor_attempts == 1
    assert not getattr(evidence, "rollback_claimed", False)


def test_pre_cancelled_run_preserves_control_category_and_evidence(tmp_path: Path) -> None:
    context = RunContext(run_id="pre-cancelled")
    context.cancel("caller cancelled")
    app = create_coding_agent_application(make_workspace(tmp_path), StubLLM([]))

    with pytest.raises(RunCancelledError) as raised:
        app.run(CodingRequest("do not run", ("target.txt",)), context=context)

    assert raised.value.category.value == "cancelled"
    assert isinstance(raised.value.coding_failure_evidence, CodingFailureEvidence)  # type: ignore[attr-defined]


def test_deadline_preserves_control_category_and_does_not_run_validators(tmp_path: Path) -> None:
    llm = StubLLM([Completion("late"), Completion("unused")], delay=0.05)
    app = create_coding_agent_application(
        make_workspace(tmp_path),
        llm,
        validators=(pass_validator(),),
    )

    with pytest.raises(RunDeadlineExceededError) as raised:
        app.run(
            CodingRequest("expire", ("target.txt",)),
            context=RunContext(run_id="deadline", timeout_seconds=0.01),
        )

    evidence = raised.value.coding_failure_evidence  # type: ignore[attr-defined]
    assert raised.value.category.value == "deadline_exceeded"
    assert isinstance(evidence, CodingFailureEvidence)
    assert evidence.validator_results == ()


def test_foreground_approval_displays_exact_sanitized_summary(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("one\n", encoding="utf-8")
    output = StringIO()
    provider = ForegroundApprovalProvider(
        input_fn=lambda prompt: "y",
        output=output,
    )
    app = create_coding_agent_application(
        make_workspace(tmp_path),
        StubLLM(
            [
                Completion(tool_calls=(patch_call(target, "one", "two"),)),
                Completion("approved"),
            ]
        ),
        approval_provider=provider,
    )

    result = app.run(CodingRequest("edit", ("target.txt",)))

    rendered = output.getvalue()
    assert result.diff.changes
    assert "action_digest:" in rendered
    assert "target.txt" in rendered
    assert str(tmp_path) not in rendered


def test_foreground_approval_rejection_is_model_visible_and_has_no_effect(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("one\n", encoding="utf-8")
    output = StringIO()
    app = create_coding_agent_application(
        make_workspace(tmp_path),
        StubLLM(
            [
                Completion(tool_calls=(patch_call(target, "one", "two"),)),
                Completion("approval was declined"),
            ]
        ),
        approval_provider=ForegroundApprovalProvider(
            input_fn=lambda prompt: "n",
            output=output,
        ),
    )

    result = app.run(CodingRequest("edit", ("target.txt",)))

    assert target.read_text(encoding="utf-8") == "one\n"
    assert result.action_records[0].approval_outcome is ApprovalOutcome.REJECT
    assert result.agent_result.output.content == "approval was declined"


def test_invalid_final_observation_keeps_a_bounded_configuration_failure(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)

    class InvalidObserver:
        def __init__(self, bound_workspace: Workspace) -> None:
            self.workspace = bound_workspace

        def capture_baseline(self, **kwargs):
            del kwargs
            return object()

    app = create_coding_agent_application(
        workspace,
        StubLLM([Completion("must not run")]),
        observer=InvalidObserver(workspace),  # type: ignore[arg-type]
    )

    with pytest.raises(ConfigurationError) as raised:
        app.run(CodingRequest("inspect", ("target.txt",)))

    assert raised.value.category.value == "configuration"
    evidence = raised.value.coding_failure_evidence  # type: ignore[attr-defined]
    assert isinstance(evidence, CodingFailureEvidence)
    assert evidence.baseline is None
    assert evidence.final is None


def test_invalid_validator_evidence_preserves_final_observation(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)

    class InvalidValidatorRunner:
        def __init__(self, bound_workspace: Workspace) -> None:
            self.workspace = bound_workspace

        def run(self, definitions, context=None):
            del definitions, context
            return (object(),)

    app = create_coding_agent_application(
        workspace,
        StubLLM([Completion("done")]),
        validators=(pass_validator(),),
        validator_runner=InvalidValidatorRunner(workspace),  # type: ignore[arg-type]
    )

    with pytest.raises(ConfigurationError) as raised:
        app.run(CodingRequest("inspect", ("target.txt",)))

    evidence = raised.value.coding_failure_evidence  # type: ignore[attr-defined]
    assert isinstance(evidence, CodingFailureEvidence)
    assert evidence.final is not None
    assert evidence.diff is not None


def test_noninteractive_approval_fails_closed_without_effect(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("one\n", encoding="utf-8")
    app = create_coding_agent_application(
        make_workspace(tmp_path),
        StubLLM(
            [
                Completion(tool_calls=(patch_call(target, "one", "two"),)),
                Completion("recovered")
            ]
        ),
        approval_provider=NonInteractiveApprovalProvider(),
    )

    result = app.run(CodingRequest("edit", ("target.txt",)))

    assert target.read_text(encoding="utf-8") == "one\n"
    assert result.action_records[0].approval_outcome is ApprovalOutcome.UNAVAILABLE


def test_one_application_serializes_runs_for_one_workspace(tmp_path: Path) -> None:
    active = 0
    maximum = 0
    guard = Lock()

    class SerialLLM:
        def complete(self, messages, tools=(), *, context=None):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            with guard:
                active -= 1
            return Completion("done")

    app = create_coding_agent_application(make_workspace(tmp_path), SerialLLM())  # type: ignore[arg-type]
    requests = (
        CodingRequest("first", ("one.txt",)),
        CodingRequest("second", ("two.txt",)),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(app.run, requests))

    assert [item.verdict for item in results] == [TaskVerdict.NOT_VALIDATED] * 2
    assert maximum == 1


class _CliResult:
    run_id = "cli-run"
    verdict = TaskVerdict.PASSED
    validator_results = (
        ValidatorResult(
            validator_id="check",
            status=ValidatorStatus.PASSED,
            argv=("check",),
            cwd=PurePosixPath("."),
        ),
    )
    diff = type("Diff", (), {"rendered_diff": "absolute-placeholder"})()
    blind_spots = ()


def test_cli_exit_and_output_are_bounded_and_workspace_path_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[CodingRequest] = []

    class FakeApplication:
        def run(self, request: CodingRequest):
            captured.append(request)
            result = _CliResult()
            result.diff.rendered_diff = f"{tmp_path}\\target.txt\n"
            return result

    monkeypatch.setenv("DQAGENT_PROVIDER", "llama_cpp")
    monkeypatch.setenv("DQAGENT_MODEL", "test-model")
    monkeypatch.setattr("dqagent.coding_cli.create_llm_client", lambda settings: object())
    monkeypatch.setattr(
        "dqagent.coding_cli.create_coding_agent_application",
        lambda *args, **kwargs: FakeApplication(),
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    exit_code = coding_cli_main(
        [
            "--workspace",
            str(tmp_path),
            "--message",
            "inspect",
            "--target",
            "target.txt",
            "--skill",
            "style",
        ]
    )

    captured_output = capsys.readouterr()
    assert exit_code == 0
    assert captured == [CodingRequest("inspect", ("target.txt",), ("style",))]
    assert str(tmp_path) not in captured_output.out
    assert "verdict: passed" in captured_output.out


def test_cli_redacts_provider_secret_from_result_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class SecretResult:
        run_id = "cli-secret-run"
        verdict = TaskVerdict.PASSED
        validator_results = (
            ValidatorResult(
                validator_id="check",
                status=ValidatorStatus.PASSED,
                argv=("check",),
                cwd=PurePosixPath("."),
                stdout="DATA_SECRET",
                stderr="DATA_SECRET",
            ),
        )
        diff = type("Diff", (), {"rendered_diff": "DATA_SECRET changed"})()
        blind_spots = ()

    class FakeApplication:
        def run(self, request: CodingRequest):
            del request
            return SecretResult()

    monkeypatch.setenv("DQAGENT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "DATA_SECRET")
    monkeypatch.setenv("DQAGENT_MODEL", "test-model")
    monkeypatch.setattr("dqagent.coding_cli.create_llm_client", lambda settings: object())
    monkeypatch.setattr(
        "dqagent.coding_cli.create_coding_agent_application",
        lambda *args, **kwargs: FakeApplication(),
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    exit_code = coding_cli_main(
        [
            "--workspace",
            str(tmp_path),
            "--message",
            "inspect",
            "--target",
            "target.txt",
        ]
    )

    captured_output = capsys.readouterr()
    assert exit_code == 0
    assert "DATA_SECRET" not in captured_output.out
    assert "[REDACTED]" in captured_output.out


def test_cli_noninteractive_approval_and_failure_output_are_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_provider: list[object] = []
    failure = AgentLoopError("provider detail DATA_SECRET")
    failure.coding_failure_evidence = CodingFailureEvidence()  # type: ignore[attr-defined]

    class FakeApplication:
        def run(self, request: CodingRequest):
            del request
            raise failure

    monkeypatch.setenv("DQAGENT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "DATA_SECRET")
    monkeypatch.setenv("DQAGENT_MODEL", "test-model")
    monkeypatch.setattr("dqagent.coding_cli.create_llm_client", lambda settings: object())

    def compose(*args, **kwargs):
        del args
        captured_provider.append(kwargs["approval_provider"])
        return FakeApplication()

    monkeypatch.setattr("dqagent.coding_cli.create_coding_agent_application", compose)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    exit_code = coding_cli_main(
        [
            "--workspace",
            str(tmp_path),
            "--message",
            "inspect",
            "--target",
            "target.txt",
        ]
    )

    captured_output = capsys.readouterr()
    assert exit_code == 1
    assert captured_provider
    assert isinstance(captured_provider[0], coding_cli.SafeNonInteractiveApprovalProvider)
    assert "error_type: AgentLoopError" in captured_output.err
    assert "failure_evidence: bounded" in captured_output.err
    assert "rollback_claimed: false" in captured_output.err
    assert "DATA_SECRET" not in captured_output.err


def test_cli_configuration_parsers_reject_malformed_trusted_inputs() -> None:
    with pytest.raises(ValueError, match="allow-executable"):
        coding_cli._parse_executable_allowlist(("python",))
    with pytest.raises(ValueError, match="duplicate executable"):
        coding_cli._parse_executable_allowlist(("python=a", "python=b"))
    with pytest.raises(ValueError, match="invalid JSON"):
        coding_cli._parse_validators(("check={",))
    with pytest.raises(ValueError, match="string array"):
        coding_cli._parse_validators(("check=1",))
