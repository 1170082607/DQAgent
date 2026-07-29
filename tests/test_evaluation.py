import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from dqagent.evaluation import (
    EvaluationDefinitionError,
    EvaluationMode,
    EvaluationRunner,
    EvaluationSuite,
    ResourceLimits,
    load_evaluation_suite,
)
from dqagent.execution import RunContext
from dqagent.models import Completion, ConversationItem, ToolDefinition

BASELINE_SUITE = Path("evaluations/cases/phase-3-baseline-v1.json")


class LiveStub:
    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolDefinition] = (),
        *,
        context: RunContext | None = None,
    ) -> Completion:
        del messages, tools, context
        return Completion("EVALUATION_OK")


def test_deterministic_baseline_passes_and_records_metrics() -> None:
    suite = load_evaluation_suite(BASELINE_SUITE)

    report = EvaluationRunner(EvaluationMode.DETERMINISTIC, max_model_attempts=1).run(suite)

    assert report.passed is True
    assert len(report.results) == 4
    assert report.skipped_case_ids == ()
    current_time = next(
        result for result in report.results if result.case_id == "current_time_tool"
    )
    assert current_time.metrics.model_attempts == 2
    assert current_time.metrics.tool_calls == 1
    assert current_time.metrics.total_tokens == 58
    assert report.to_dict()["summary"] == {
        "passed": True,
        "passed_cases": 4,
        "failed_cases": 0,
        "executed_cases": 4,
        "skipped_cases": 0,
    }


def test_live_mode_requires_client_and_skips_deterministic_only_cases() -> None:
    suite = load_evaluation_suite(BASELINE_SUITE)

    with pytest.raises(ValueError, match="requires an LLM client"):
        EvaluationRunner(EvaluationMode.LIVE)

    live_suite = EvaluationSuite(
        suite.suite_id,
        suite.schema_version,
        (suite.cases[0], suite.cases[3]),
    )
    report = EvaluationRunner(EvaluationMode.LIVE, live_client=LiveStub()).run(live_suite)

    assert report.passed is True
    assert report.skipped_case_ids == ("invalid_tool_arguments_recovery",)


def test_failed_predicate_fails_report_without_aborting_suite() -> None:
    suite = load_evaluation_suite(BASELINE_SUITE)
    original = suite.cases[0]
    wrong_answer = replace(
        original,
        expected=replace(
            original.expected,
            answer=replace(original.expected.answer, exact="WRONG"),
        ),
    )

    report = EvaluationRunner(EvaluationMode.DETERMINISTIC).run(
        EvaluationSuite("failing-suite", 1, (wrong_answer,))
    )

    assert report.passed is False
    assert report.results[0].passed is False
    assert any(
        check.name == "answer.exact" and not check.passed
        for check in report.results[0].checks
    )


def test_resource_limits_are_selected_by_evaluation_mode() -> None:
    suite = load_evaluation_suite(BASELINE_SUITE)
    original = suite.cases[0]
    mode_specific = replace(
        original,
        expected=replace(
            original.expected,
            trace=replace(
                original.expected.trace,
                resource_limits={
                    EvaluationMode.DETERMINISTIC: ResourceLimits(
                        max_elapsed_seconds=0.000000001
                    ),
                    EvaluationMode.LIVE: ResourceLimits(max_elapsed_seconds=10),
                },
            ),
        ),
    )
    mode_suite = EvaluationSuite("mode-limits", 1, (mode_specific,))

    deterministic = EvaluationRunner(EvaluationMode.DETERMINISTIC).run(mode_suite)
    live = EvaluationRunner(EvaluationMode.LIVE, live_client=LiveStub()).run(mode_suite)

    assert deterministic.passed is False
    assert live.passed is True


def test_loader_rejects_unsupported_or_invalid_suites(tmp_path: Path) -> None:
    unsupported = tmp_path / "unsupported.json"
    unsupported.write_text('{"schema_version": 2}', encoding="utf-8")

    with pytest.raises(EvaluationDefinitionError, match="unsupported evaluation schema version"):
        load_evaluation_suite(unsupported)

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps({"schema_version": 1, "suite_id": "suite", "cases": []}),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationDefinitionError, match="invalid evaluation suite"):
        load_evaluation_suite(invalid)
