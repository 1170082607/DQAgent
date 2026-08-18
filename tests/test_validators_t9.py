from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

import pytest

from dqagent.execution import RunContext
from dqagent.subprocesses import (
    LOCAL_SUBPROCESS_CAPABILITIES,
    CleanupResult,
    CleanupStatus,
    IsolationCapability,
    SubprocessRequest,
    SubprocessResult,
    SubprocessStatus,
)
from dqagent.validators import (
    TaskVerdict,
    ValidatorDefinition,
    ValidatorResult,
    ValidatorRunner,
    ValidatorStatus,
    derive_task_verdict,
)
from dqagent.workspace import Workspace, WorkspaceScope


def make_workspace(tmp_path: Path, **kwargs: object) -> Workspace:
    return Workspace(WorkspaceScope("fixture", tmp_path, **kwargs))


def python_validator(code: str, **kwargs: object) -> ValidatorDefinition:
    return ValidatorDefinition(
        "validator",
        (sys.executable, "-c", code),
        **kwargs,
    )


def test_validator_pass_fail_and_unavailable_statuses_use_real_subprocess_boundary(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    runner = ValidatorRunner(workspace)

    passed = runner.run_one(python_validator("print('passed')"))
    failed = runner.run_one(
        ValidatorDefinition(
            "failed",
            (sys.executable, "-c", "print('failed'); raise SystemExit(3)"),
        )
    )
    unavailable = runner.run_one(
        ValidatorDefinition("missing", (str(tmp_path / "missing-validator"),))
    )

    assert passed.status is ValidatorStatus.PASSED
    assert passed.passed
    assert "passed" in passed.stdout
    assert failed.status is ValidatorStatus.FAILED
    assert failed.returncode == 3
    assert unavailable.status is ValidatorStatus.UNAVAILABLE
    assert unavailable.spawned is False


def test_validator_accepts_trusted_exit_codes_and_freezes_environment_aliases() -> None:
    definition = ValidatorDefinition(
        id="accepted",
        argv=("validator",),
        accepted_exit_codes=(2,),
        env={"SAFE": "yes"},
    )

    assert definition.id == "accepted"
    assert definition.name == "accepted"
    assert definition.env == {"SAFE": "yes"}
    try:
        definition.environment["NEW"] = "value"  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("validator environment must be immutable")


def test_validator_timeout_is_bounded_and_reaped(tmp_path: Path) -> None:
    result = ValidatorRunner(make_workspace(tmp_path)).run_one(
        python_validator("import time; time.sleep(10)", timeout_seconds=0.05)
    )

    assert result.status is ValidatorStatus.TIMED_OUT
    assert result.cleanup.status is CleanupStatus.TERMINATED_AND_REAPED
    assert result.cleanup_succeeded


def test_validator_subprocess_statuses_remain_distinct_and_ordered(tmp_path: Path) -> None:
    class FixedStatusRunner:
        backend_identity = "fixed-status-runner"
        capabilities = LOCAL_SUBPROCESS_CAPABILITIES

        def __init__(self, status: SubprocessStatus, *, cleanup_succeeded: bool = True) -> None:
            self.status = status
            self.cleanup_succeeded = cleanup_succeeded

        def run(
            self,
            request: SubprocessRequest,
            context: RunContext | None = None,
        ) -> SubprocessResult:
            del request, context
            return SubprocessResult(
                status=self.status,
                returncode=1,
                spawned=self.status is not SubprocessStatus.SPAWN_ERROR,
                backend_identity=self.backend_identity,
                backend_capabilities=tuple(self.capabilities),
                cleanup=CleanupResult(
                    status=(
                        CleanupStatus.REAPED
                        if self.cleanup_succeeded
                        else CleanupStatus.FAILED
                    ),
                    reaped=self.cleanup_succeeded,
                    streams_drained=self.cleanup_succeeded,
                ),
            )

    expected = {
        SubprocessStatus.OUTPUT_SANITIZATION_ERROR: ValidatorStatus.ERROR,
        SubprocessStatus.CANCELLED: ValidatorStatus.CANCELLED,
        SubprocessStatus.DEADLINE_EXCEEDED: ValidatorStatus.TIMED_OUT,
        SubprocessStatus.CAPABILITY_DENIED: ValidatorStatus.CAPABILITY_MISSING,
    }
    for process_status, validator_status in expected.items():
        result = ValidatorRunner(
            make_workspace(tmp_path),
            FixedStatusRunner(process_status),
        ).run_one(ValidatorDefinition(str(process_status), ("validator",)))
        assert result.status is validator_status

    accepted = ValidatorRunner(
        make_workspace(tmp_path),
        FixedStatusRunner(SubprocessStatus.NONZERO),
    ).run_one(
        ValidatorDefinition(
            "accepted-exit",
            ("validator",),
            accepted_exit_codes=(1,),
        )
    )
    assert accepted.status is ValidatorStatus.PASSED

    cleanup_uncertain = ValidatorRunner(
        make_workspace(tmp_path),
        FixedStatusRunner(SubprocessStatus.NONZERO, cleanup_succeeded=False),
    )
    cleanup_uncertain_result = cleanup_uncertain.run_one(
        ValidatorDefinition(
            "accepted-exit-cleanup-failure",
            ("validator",),
            accepted_exit_codes=(1,),
        )
    )
    assert cleanup_uncertain_result.status is ValidatorStatus.FAILED


def test_validator_definition_and_result_bounds_are_enforced(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ValidatorDefinition("empty", ())
    with pytest.raises(ValueError):
        ValidatorDefinition("too-many", ("x",) * 129)
    with pytest.raises(TypeError):
        ValidatorDefinition("aliases", ("x",), id="other")

    result = ValidatorRunner(make_workspace(tmp_path)).run_one(
        python_validator("pass")
    )
    assert result.argv_identity == result.argv
    assert result.available
    assert result.cleanup_succeeded


def test_validator_definition_argv_is_bounded_before_materialization() -> None:
    consumed = 0

    def argv():
        nonlocal consumed
        for index in range(10_000):
            consumed += 1
            yield f"argument-{index}"

    with pytest.raises(ValueError, match="validator argv"):
        ValidatorDefinition("bounded", argv())

    assert consumed == 129


@pytest.mark.parametrize("field", ("trusted_ignored_paths", "accepted_exit_codes"))
def test_validator_definition_collections_are_bounded(field: str) -> None:
    consumed = 0

    def values():
        nonlocal consumed
        for index in range(10_000):
            consumed += 1
            yield f"ignored-{index}" if field == "trusted_ignored_paths" else index

    kwargs = {
        "trusted_ignored_paths": (),
        "accepted_exit_codes": (0,),
    }
    kwargs[field] = values()

    with pytest.raises(ValueError, match="bound|unbounded"):
        ValidatorDefinition("bounded", ("validator",), **kwargs)

    assert consumed == 33


def test_validator_secret_names_are_bounded_before_materialization(
    tmp_path: Path,
) -> None:
    consumed = 0

    def secret_names():
        nonlocal consumed
        for index in range(10_000):
            consumed += 1
            yield f"SECRET_NAME_{index}"

    with pytest.raises(ValueError, match="secret names exceeds its item bound"):
        ValidatorRunner(
            make_workspace(tmp_path),
            secret_names=secret_names(),
        )

    assert consumed == 129


def test_validator_definition_iterable_is_bounded_before_materialization(
    tmp_path: Path,
) -> None:
    consumed = 0

    def definitions():
        nonlocal consumed
        for index in range(10_000):
            consumed += 1
            yield ValidatorDefinition(f"validator-{index}", ("validator",))

    with pytest.raises(ValueError, match="validator definitions exceed the configured bound"):
        ValidatorRunner(make_workspace(tmp_path), max_validators=1).run(definitions())

    assert consumed == 2


def test_validator_definition_collection_honors_cancellation(
    tmp_path: Path,
) -> None:
    context = RunContext(run_id="validator-definition-cancel")
    context.cancel("cancel before collecting validator definitions")
    consumed = 0

    def definitions():
        nonlocal consumed
        for index in range(10_000):
            consumed += 1
            yield ValidatorDefinition(f"validator-{index}", ("validator",))

    results = ValidatorRunner(make_workspace(tmp_path)).run(definitions(), context)

    assert results == ()
    assert consumed == 0


def test_validator_definition_rejects_malformed_trusted_inputs() -> None:
    with pytest.raises(ValueError):
        ValidatorDefinition("bad/id", ("validator",))
    with pytest.raises(TypeError):
        ValidatorDefinition("bad", "validator")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ValidatorDefinition("bad", object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ValidatorDefinition("bad", ("validator", "\x00"))
    with pytest.raises(ValueError):
        ValidatorDefinition("bad", ("x" * 8_192,) * 5)
    with pytest.raises(ValueError):
        ValidatorDefinition("bad", ("validator",), cwd="/absolute")
    with pytest.raises(ValueError):
        ValidatorDefinition("bad", ("validator",), cwd="a/../b")
    with pytest.raises(ValueError):
        ValidatorDefinition("bad", ("validator",), cwd=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ValidatorDefinition("bad", ("validator",), environment="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ValidatorDefinition("bad", ("validator",), environment={"BAD=NAME": "value"})
    with pytest.raises(TypeError):
        ValidatorDefinition("bad", ("validator",), trusted_ignored_paths=".cache")
    with pytest.raises(TypeError):
        ValidatorDefinition(
            "bad",
            ("validator",),
            stdout_limit=1,
            max_stdout_bytes=2,
        )


def test_validator_runner_reports_configuration_and_backend_failures(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)

    class RaisingRunner:
        backend_identity = "raising-runner"
        capabilities = LOCAL_SUBPROCESS_CAPABILITIES

        def run(
            self,
            request: SubprocessRequest,
            context: RunContext | None = None,
        ) -> SubprocessResult:
            del request, context
            raise RuntimeError("backend unavailable")

    runner = ValidatorRunner(workspace, RaisingRunner())
    unavailable = runner.run_one(
        ValidatorDefinition("bad-cwd", ("validator",), cwd="missing")
    )
    backend_error = runner.run_one(ValidatorDefinition("backend", ("validator",)))

    assert unavailable.status is ValidatorStatus.UNAVAILABLE
    assert backend_error.status is ValidatorStatus.ERROR
    assert runner.run(()) == ()


def test_validator_runs_all_definitions_in_trusted_order_and_does_not_need_approval(
    tmp_path: Path,
) -> None:
    order: list[str] = []

    class RecordingRunner:
        backend_identity = "recording-runner"
        capabilities = LOCAL_SUBPROCESS_CAPABILITIES

        def run(
            self,
            request: SubprocessRequest,
            context: RunContext | None = None,
        ) -> SubprocessResult:
            del context
            order.append(request.argv[-1])
            return SubprocessResult(
                status=SubprocessStatus.NORMAL,
                returncode=0,
                stdout=request.argv[-1],
                spawned=True,
                backend_identity=self.backend_identity,
                backend_capabilities=tuple(self.capabilities),
                cleanup=CleanupResult(
                    status=CleanupStatus.REAPED,
                    reaped=True,
                    streams_drained=True,
                ),
            )

    workspace = make_workspace(tmp_path)
    runner = ValidatorRunner(workspace, RecordingRunner())
    definitions = (
        ValidatorDefinition("first", ("validator", "first")),
        ValidatorDefinition("second", ("validator", "second")),
        ValidatorDefinition("third", ("validator", "third")),
    )

    results = runner.run(definitions)

    assert order == ["first", "second", "third"]
    assert [result.validator_id for result in results] == ["first", "second", "third"]
    assert all(result.status is ValidatorStatus.PASSED for result in results)


def test_validator_missing_capability_is_denied_without_runner_call(tmp_path: Path) -> None:
    calls = 0

    class NoCapabilityRunner:
        backend_identity = "no-capability-runner"
        capabilities = frozenset()

        def run(
            self,
            request: SubprocessRequest,
            context: RunContext | None = None,
        ) -> SubprocessResult:
            nonlocal calls
            del request, context
            calls += 1
            raise AssertionError("missing capability must deny before spawn")

    result = ValidatorRunner(make_workspace(tmp_path), NoCapabilityRunner()).run_one(
        python_validator(
            "raise SystemExit(1)",
            required_capabilities=(IsolationCapability.DIRECT_ARGV,),
        )
    )

    assert result.status is ValidatorStatus.CAPABILITY_MISSING
    assert calls == 0
    assert result.spawned is False


def test_validator_environment_and_ignored_artifacts_are_trusted(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path, ignored_paths=(PurePosixPath(".cache"),))
    seen: list[SubprocessRequest] = []

    class RecordingRunner:
        backend_identity = "recording-runner"
        capabilities = LOCAL_SUBPROCESS_CAPABILITIES

        def run(
            self,
            request: SubprocessRequest,
            context: RunContext | None = None,
        ) -> SubprocessResult:
            del context
            seen.append(request)
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

    definition = python_validator(
        "pass",
        environment={"SAFE": "yes", "SECRET_TOKEN": "secret"},
        trusted_ignored_paths=(".cache",),
    )
    result = ValidatorRunner(
        workspace,
        RecordingRunner(),
        secret_values=("secret",),
    ).run_one(definition)

    assert result.status is ValidatorStatus.PASSED
    assert dict(seen[0].environment) == {"SAFE": "yes"}
    assert seen[0].cwd == tmp_path.resolve()


def test_untrusted_validator_artifact_path_is_a_composition_failure_before_spawn(
    tmp_path: Path,
) -> None:
    calls = 0

    class RecordingRunner:
        backend_identity = "recording-runner"
        capabilities = LOCAL_SUBPROCESS_CAPABILITIES

        def run(
            self,
            request: SubprocessRequest,
            context: RunContext | None = None,
        ) -> SubprocessResult:
            nonlocal calls
            del request, context
            calls += 1
            raise AssertionError("untrusted artifact path must not spawn")

    result = ValidatorRunner(make_workspace(tmp_path), RecordingRunner()).run_one(
        python_validator("pass", trusted_ignored_paths=(".cache",))
    )

    assert result.status is ValidatorStatus.UNAVAILABLE
    assert calls == 0


def test_validator_cancellation_marks_remaining_validators_without_starting_them(
    tmp_path: Path,
) -> None:
    context = RunContext(run_id="validator-cancel")
    context.cancel("cancel before validators")
    calls = 0

    class RecordingRunner:
        backend_identity = "recording-runner"
        capabilities = LOCAL_SUBPROCESS_CAPABILITIES

        def run(
            self,
            request: SubprocessRequest,
            context: RunContext | None = None,
        ) -> SubprocessResult:
            nonlocal calls
            del request, context
            calls += 1
            raise AssertionError("cancelled validator sequence must not spawn")

    results = ValidatorRunner(make_workspace(tmp_path), RecordingRunner()).run(
        (python_validator("pass"), ValidatorDefinition("second", (sys.executable, "-c", "pass"))),
        context,
    )

    assert [result.status for result in results] == [
        ValidatorStatus.CANCELLED,
        ValidatorStatus.CANCELLED,
    ]
    assert calls == 0


def test_verdict_is_harness_evidence_and_model_success_claim_cannot_override_it(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    passed = ValidatorRunner(workspace).run_one(python_validator("pass"))
    failed = ValidatorRunner(workspace).run_one(
        ValidatorDefinition("failed", (sys.executable, "-c", "raise SystemExit(1)"))
    )
    unavailable = ValidatorResult(
        validator_id="unavailable",
        status=ValidatorStatus.UNAVAILABLE,
        argv=("validator",),
        cwd=PurePosixPath("."),
    )

    assert derive_task_verdict((), model_success_claim=True) is TaskVerdict.NOT_VALIDATED
    assert derive_task_verdict((passed,), model_success_claim=False) is TaskVerdict.PASSED
    assert derive_task_verdict((failed,), model_success_claim=True) is TaskVerdict.FAILED
    assert (
        derive_task_verdict((unavailable,), model_success_claim=True)
        is TaskVerdict.INDETERMINATE
    )
    assert (
        derive_task_verdict((passed,), target_observation_complete=False, model_success_claim=True)
        is TaskVerdict.INDETERMINATE
    )
    assert (
        derive_task_verdict((passed,), workspace_effect_unknown=True, model_success_claim=True)
        is TaskVerdict.INDETERMINATE
    )
