from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dqagent import coding as coding_module
from dqagent import coding_evaluation as coding_evaluation_module
from dqagent.coding import CodingFailureEvidence
from dqagent.coding_evaluation import (
    CodingEvaluationDefinitionError,
    CodingEvaluationRunner,
    CodingEvaluationSuite,
    load_coding_evaluation_suite,
)
from dqagent.coding_evaluation_cli import main as coding_evaluation_cli_main
from dqagent.events import RunEventType
from dqagent.tool_governance import ApprovalOutcome, EffectState
from dqagent.validators import TaskVerdict

SUITE_PATH = Path("evaluations/cases/phase-9-coding-smoke-v1.json")


def load_suite():
    return load_coding_evaluation_suite(SUITE_PATH)


def with_case(case):
    return CodingEvaluationSuite("t13-test", 1, (case,))


def refresh_fixture_digest(case):
    return replace(case, fixture_digest=coding_evaluation_module._case_fixture_digest(case))


def make_validator_failure_case():
    patch_case = load_suite().cases[1]
    expected = replace(
        patch_case.expected,
        run_status="error",
        error_type="RunExecutionError",
        error_category="internal",
        answer_non_empty=False,
        answer_exact=None,
        answer_contains_all=(),
        answer_absent_all=(),
        verdict=None,
        validators=(),
        required_event_types=(
            RunEventType.RUN_STARTED,
            RunEventType.CODING_REQUEST_VALIDATED,
            RunEventType.WORKSPACE_DIFF_COMPUTED,
            RunEventType.VALIDATORS_STARTED,
            RunEventType.RUN_FAILED,
        ),
        limits=replace(
            patch_case.expected.limits,
            max_model_attempts=2,
            max_governed_calls=2,
            max_elapsed_seconds=10,
            max_rendered_diff_characters=200_000,
            max_validator_output_characters=0,
        ),
    )
    return refresh_fixture_digest(
        replace(
            patch_case,
            fixture=replace(patch_case.fixture, failure="validator_runner_error"),
            composition=replace(patch_case.composition, validators=()),
            expected=expected,
        )
    )


def make_unbound_runner_error_case():
    patch_case = load_suite().cases[1]
    expected = replace(
        patch_case.expected,
        run_status="error",
        error_type="RuntimeError",
        error_category=None,
        answer_non_empty=False,
        answer_exact=None,
        answer_contains_all=(),
        answer_absent_all=(),
        verdict=None,
        validators=(),
        governance=(),
        required_event_types=(),
        fixture_consumed=False,
    )
    return refresh_fixture_digest(
        replace(
            patch_case,
            composition=replace(patch_case.composition, validators=()),
            expected=expected,
        )
    )


def test_coding_smoke_suite_uses_real_production_path() -> None:
    report = CodingEvaluationRunner().run(load_suite())

    assert report.passed is True
    assert [result.case_id for result in report.results] == [
        "read-with-explicit-skill",
        "approved-create-with-validator",
        "rejected-create-no-effect",
    ]
    assert all(result.evaluation_passed and result.cleanup_passed for result in report.results)
    report_results = report.to_dict()["results"]
    assert all(result["observation_limitations"] for result in report_results)
    assert all("blind_spot_reasons" in result["diff"] for result in report_results)
    assert all(
        any(
            str(limitation).startswith("backend:")
            for limitation in result["observation_limitations"]
        )
        for result in report_results
    )


def test_evaluator_detects_intentionally_wrong_answer_and_weak_diff() -> None:
    suite = load_suite()
    read_case = suite.cases[0]
    wrong_answer = refresh_fixture_digest(
        replace(
            read_case,
            expected=replace(read_case.expected, answer_exact="intentionally wrong"),
        )
    )
    wrong_answer_report = CodingEvaluationRunner().run(with_case(wrong_answer))
    assert wrong_answer_report.passed is False
    assert any(
        check.name == "answer.exact" and not check.passed
        for check in wrong_answer_report.results[0].checks
    )

    patch_case = suite.cases[1]
    weak_diff = refresh_fixture_digest(
        replace(
            patch_case,
            expected=replace(
                patch_case.expected,
                diff=replace(patch_case.expected.diff, expected_changes=()),
            ),
        ),
    )
    weak_diff_report = CodingEvaluationRunner().run(with_case(weak_diff))
    assert weak_diff_report.passed is False
    assert any(
        check.name == "diff.exact_changes" and not check.passed
        for check in weak_diff_report.results[0].checks
    )


def test_case_digest_binds_request_and_expected_predicates(tmp_path: Path) -> None:
    mutations = (
        "request_skills",
        "request_targets",
        "request_message",
        "expected_diff",
        "expected_governance",
    )
    for mutation in mutations:
        raw = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
        case = raw["cases"][0]
        if mutation == "request_skills":
            case["request"]["skills"] = []
        elif mutation == "request_targets":
            case["request"]["targets"] = ["other.txt"]
        elif mutation == "request_message":
            case["request"]["message"] = "Read another target."
        elif mutation == "expected_diff":
            case["expected"]["diff"]["target_complete"] = False
        else:
            case["expected"]["governance"][0]["guards_passed"] = False
        tampered = tmp_path / f"{mutation}.json"
        tampered.write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(CodingEvaluationDefinitionError, match="fixture digest mismatch"):
            load_coding_evaluation_suite(tampered)


def test_case_order_is_independent_and_fixture_tamper_is_rejected(tmp_path: Path) -> None:
    suite = load_suite()
    forward = CodingEvaluationRunner().run(suite)
    reversed_suite = CodingEvaluationSuite(
        suite.suite_id,
        suite.schema_version,
        tuple(reversed(suite.cases)),
    )
    reverse = CodingEvaluationRunner().run(reversed_suite)

    assert {result.case_id: result.passed for result in forward.results} == {
        result.case_id: result.passed for result in reverse.results
    }
    assert forward.deterministic_fingerprint == reverse.deterministic_fingerprint

    raw = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    raw["cases"][0]["fixture"]["model_completions"][1]["content"] = "tampered fixture"
    tampered = tmp_path / "tampered-coding-suite.json"
    tampered.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(CodingEvaluationDefinitionError, match="fixture digest mismatch"):
        load_coding_evaluation_suite(tampered)

    in_memory_tamper = replace(
        suite.cases[0],
        fixture=replace(
            suite.cases[0].fixture,
            approval_decisions=(ApprovalOutcome.REJECT,),
        ),
    )
    with pytest.raises(CodingEvaluationDefinitionError, match="fixture digest mismatch"):
        CodingEvaluationRunner().run(with_case(in_memory_tamper))


def test_unsafe_report_is_redacted_and_cleanup_failure_does_not_mask_result() -> None:
    suite = load_suite()
    patch_case = suite.cases[1]
    validator = replace(
        patch_case.composition.validators[0],
        validator_id="fixture-secret",
        argv=("{python}", "-c", "print('fixture-secret')"),
    )
    unsafe_case = refresh_fixture_digest(
        replace(
            patch_case,
            composition=replace(
                patch_case.composition,
                validators=(validator,),
                secret_values=("fixture-secret",),
            ),
        )
    )
    unsafe_report = CodingEvaluationRunner().run(with_case(unsafe_case))
    rendered = json.dumps(unsafe_report.to_dict(), ensure_ascii=True)
    assert "fixture-secret" not in rendered
    assert "[REDACTED]" in rendered

    cleanup_case = refresh_fixture_digest(
        replace(
            suite.cases[0],
            fixture=replace(suite.cases[0].fixture, failure="cleanup_failure"),
        )
    )
    cleanup_result = CodingEvaluationRunner().run(with_case(cleanup_case)).results[0]
    assert cleanup_result.evaluation_passed is True
    assert cleanup_result.cleanup_passed is False
    assert cleanup_result.passed is False
    assert cleanup_result.cleanup_error is not None


def test_error_path_evaluates_failure_evidence_predicates() -> None:
    error_case = make_validator_failure_case()
    passing = CodingEvaluationRunner().run(with_case(error_case))

    assert passing.passed is True
    assert passing.results[0].error is not None
    assert all(check.passed for check in passing.results[0].checks)

    tampered_cases = (
        replace(
            error_case,
            expected=replace(error_case.expected, answer_exact="unexpected answer"),
        ),
        replace(
            error_case,
            expected=replace(
                error_case.expected,
                diff=replace(error_case.expected.diff, expected_changes=()),
            ),
        ),
        replace(
            error_case,
            expected=replace(
                error_case.expected,
                validators=load_suite().cases[1].expected.validators,
            ),
        ),
        replace(error_case, expected=replace(error_case.expected, governance=())),
        replace(
            error_case,
            expected=replace(
                error_case.expected,
                required_event_types=(RunEventType.RUN_STARTED, RunEventType.RUN_COMPLETED),
            ),
        ),
        replace(
            error_case,
            expected=replace(
                error_case.expected,
                limits=replace(error_case.expected.limits, max_governed_calls=0),
            ),
        ),
    )
    for tampered in tampered_cases:
        report = CodingEvaluationRunner().run(with_case(refresh_fixture_digest(tampered)))
        assert report.passed is False
        assert report.results[0].evaluation_passed is False


def test_error_path_fails_closed_without_or_with_partial_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = make_unbound_runner_error_case()
    original_run = coding_module.CodingAgentApplication.run
    scenarios = (
        (None, "failure.evidence_available"),
        (CodingFailureEvidence(), "diff.evidence_available"),
    )

    for evidence, failed_check_name in scenarios:
        def failing_run(application, request, *, context=None, evidence=evidence):
            del application, request, context
            error = RuntimeError("synthetic failure evidence gap")
            if evidence is not None:
                error.coding_failure_evidence = evidence  # type: ignore[attr-defined]
            raise error

        monkeypatch.setattr(coding_module.CodingAgentApplication, "run", failing_run)
        result = CodingEvaluationRunner().run(with_case(case)).results[0]

        assert result.evaluation_passed is False
        assert result.passed is False
        assert any(
            check.name == failed_check_name and not check.passed for check in result.checks
        )
        assert result.cleanup_passed is True

    monkeypatch.setattr(coding_module.CodingAgentApplication, "run", original_run)


@pytest.mark.parametrize("observation_mode", ("unknown_effect", "incomplete_target"))
def test_evaluator_preserves_indeterminate_observation_and_report(
    monkeypatch: pytest.MonkeyPatch,
    observation_mode: str,
) -> None:
    patch_case = load_suite().cases[1]
    expected = replace(patch_case.expected, verdict="indeterminate")
    if observation_mode == "unknown_effect":
        expected = replace(
            expected,
            governance=tuple(
                replace(item, effect_state=EffectState.UNKNOWN)
                for item in expected.governance
            ),
        )
    else:
        expected = replace(
            expected,
            diff=replace(expected.diff, target_complete=False),
        )
    case = refresh_fixture_digest(replace(patch_case, expected=expected))
    original_run = coding_module.CodingAgentApplication.run

    def indeterminate_run(application, request, *, context=None):
        result = original_run(application, request, context=context)
        if observation_mode == "unknown_effect":
            action_records = tuple(
                replace(item, effect_state=EffectState.UNKNOWN)
                for item in result.action_records
            )
            return replace(
                result,
                action_records=action_records,
                verdict=TaskVerdict.INDETERMINATE,
            )
        incomplete_diff = replace(
            result.diff,
            completeness=replace(result.diff.completeness, target_complete=False),
        )
        return replace(
            result,
            diff=incomplete_diff,
            verdict=TaskVerdict.INDETERMINATE,
        )

    monkeypatch.setattr(coding_module.CodingAgentApplication, "run", indeterminate_run)
    result = CodingEvaluationRunner().run(with_case(case)).results[0]

    assert result.passed is True
    assert result.evaluation_passed is True
    assert result.verdict == "indeterminate"
    expected_limitation = (
        "workspace_effect_unknown"
        if observation_mode == "unknown_effect"
        else "target_observation_incomplete"
    )
    assert expected_limitation in result.observation_limitations
    assert any(
        check.name == "verdict.exact" and check.passed
        for check in result.checks
    )


def test_materialization_failure_is_sanitized_before_workspace_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = load_suite()
    case = suite.cases[0]

    def fail_materialize(repository) -> None:
        del repository
        raise FileExistsError("fixture materialization collision")

    monkeypatch.setattr(
        coding_evaluation_module._DisposableRepository,
        "_materialize",
        fail_materialize,
    )

    result = CodingEvaluationRunner().run(with_case(case)).results[0]
    rendered = json.dumps(result.to_dict(), ensure_ascii=True)

    assert result.passed is False
    assert result.evaluation_passed is False
    assert result.cleanup_passed is False
    assert result.error is not None
    assert "FileExistsError" in result.error
    assert "dqagent-coding-eval-" not in rendered
    assert "fixture materialization collision" not in rendered
    assert "WinError" not in rendered


def test_runner_cannot_bypass_coding_application(monkeypatch: pytest.MonkeyPatch) -> None:
    case = load_suite().cases[0]
    original_run = coding_module.CodingAgentApplication.run
    calls = 0

    def observed_run(application, request, *, context=None):
        nonlocal calls
        calls += 1
        return original_run(application, request, context=context)

    monkeypatch.setattr(coding_module.CodingAgentApplication, "run", observed_run)
    report = CodingEvaluationRunner().run(with_case(case))

    assert calls == 1
    assert report.passed is True


def test_coding_evaluation_cli_writes_bounded_report(tmp_path: Path) -> None:
    output = tmp_path / "coding-report.json"

    assert coding_evaluation_cli_main(["--suite", str(SUITE_PATH), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"] == {
        "passed": True,
        "passed_cases": 3,
        "failed_cases": 0,
        "executed_cases": 3,
        "skipped_cases": 0,
    }
