import json
from dataclasses import replace
from pathlib import Path

import pytest

from dqagent import memory_evaluation_cli
from dqagent.context import ContextBuilder
from dqagent.errors import DQAgentError
from dqagent.memory_evaluation import (
    MemoryEvaluationRunner,
    _ScriptedAnswerLLM,
    load_memory_evaluation_suite,
)
from dqagent.models import Message, Role

SUITE_PATH = Path("evaluations/cases/phase-8-memory-v1.json")
BASELINE_PATH = Path("evaluations/baselines/phase-8-memory-deterministic-v1.json")


def test_phase_8_memory_suite_passes_through_production_path() -> None:
    suite = load_memory_evaluation_suite(SUITE_PATH)
    report = MemoryEvaluationRunner().run(suite)
    payload = report.to_dict()

    assert report.passed is True
    assert len(report.results) == 13
    assert payload["summary"]["false_admission_rate"] == 0.0  # type: ignore[index]
    assert payload["summary"]["mean_recall_at_k"] == 1.0  # type: ignore[index]
    assert payload["summary"]["mean_precision_at_k"] == 1.0  # type: ignore[index]
    assert payload["summary"]["scope_leakage_rate"] == 0.0  # type: ignore[index]
    assert payload["summary"]["stale_or_forgotten_recall_rate"] == 0.0  # type: ignore[index]
    assert payload["summary"]["harmful_over_retrieval_rate"] == 0.0  # type: ignore[index]
    assert payload["summary"]["correction_compliance_rate"] == 1.0  # type: ignore[index]
    assert payload["summary"]["direct_answer_predicate_pass_rate"] == 1.0  # type: ignore[index]
    assert payload["summary"]["no_result_correct"] == 1.0  # type: ignore[index]
    assert payload["identities"]["memory_store"] == "sqlite-memory-store-v1"  # type: ignore[index]
    assert payload["identities"]["extractor"] == "phase-8-deterministic-extractor-v1"  # type: ignore[index]

    correction = next(
        result
        for result in report.results
        if result.case_id == "correction-supersedes-old-preference"
    )
    assert correction.stage_event_types[:3] == (
        "run_started",
        "memory_recall_started",
        "memory_recall_completed",
    )
    assert correction.recalls[0].recalled_labels == ("new_concise",)
    assert correction.context.projected_labels == ("new_concise",)
    assert correction.answer.direct_answer_predicate_pass is True


def test_memory_answer_fixture_requires_expected_memory_context() -> None:
    expected = "The answer uses the recalled preference."
    memory = Message(
        Role.USER,
        "[memory-context untrusted_data=true]\nThe user prefers concise answers.\n"
        "[/memory-context]",
    )

    without_memory = _ScriptedAnswerLLM(
        "fixture-test",
        expected,
        required_memory_fragments=("The user prefers concise answers.",),
    )
    missing_context = without_memory.complete((Message(Role.USER, "question"),))
    assert missing_context.content != expected
    assert without_memory.memory_context_used is False

    with_memory = _ScriptedAnswerLLM(
        "fixture-test",
        expected,
        required_memory_fragments=("The user prefers concise answers.",),
    )
    recalled_context = with_memory.complete((Message(Role.USER, "question"), memory))
    assert recalled_context.content == expected
    assert with_memory.memory_context_used is True


def test_answer_utilization_rejects_an_answer_that_ignores_available_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dqagent.memory_evaluation as memory_evaluation

    original_complete = memory_evaluation._ScriptedAnswerLLM.complete

    def ignore_memory(
        llm: _ScriptedAnswerLLM,
        messages: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        completion = original_complete(llm, messages, *args, **kwargs)  # type: ignore[arg-type]
        llm.memory_context_used = False
        return completion

    monkeypatch.setattr(memory_evaluation._ScriptedAnswerLLM, "complete", ignore_memory)
    suite = load_memory_evaluation_suite(SUITE_PATH)
    report = MemoryEvaluationRunner().run(suite)
    result = next(
        item for item in report.results if item.case_id == "confirmed-preference-cross-session"
    )

    assert result.answer.direct_answer_predicate_pass is True
    assert result.answer.answer_utilization_predicate_pass is False


def test_runner_injects_configured_memory_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    import dqagent.memory_evaluation as memory_evaluation

    observed_selectors: list[object] = []
    production_service = memory_evaluation.MemoryService

    class RecordingMemoryService(production_service):
        def __init__(self, *args: object, **kwargs: object) -> None:
            observed_selectors.append(kwargs.get("selector"))
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(memory_evaluation, "MemoryService", RecordingMemoryService)
    suite = load_memory_evaluation_suite(SUITE_PATH)
    configured = replace(suite.config, embedding_dimensions=512)

    report = MemoryEvaluationRunner().run(replace(suite, config=configured))

    expected_identity = "memory-selector-v1:hashing-token-v1:512"
    assert observed_selectors
    assert all(
        getattr(selector, "identity", None) == expected_identity
        for selector in observed_selectors
    )
    assert report.identities["memory_selector"] == expected_identity


def test_disabled_memory_is_excluded_from_no_result_denominator() -> None:
    suite = load_memory_evaluation_suite(SUITE_PATH)

    summary = MemoryEvaluationRunner().run(suite).to_dict()["summary"]

    assert summary["no_result_denominator"] == 12  # type: ignore[index]
    assert summary["expected_no_result_cases"] == 5  # type: ignore[index]
    assert summary["empty_result_cases"] == 5  # type: ignore[index]


def test_memory_budget_metric_rejects_tampered_projection_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dqagent.memory_evaluation as memory_evaluation

    class BudgetTamperingContextBuilder(ContextBuilder):
        def build(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            window = super().build(*args, **kwargs)  # type: ignore[arg-type]
            if window.memory_projection is None:
                return window
            projection = replace(
                window.memory_projection,
                budget=window.memory_projection.budget + 1,
            )
            return replace(window, memory_projection=projection)

    monkeypatch.setattr(memory_evaluation, "ContextBuilder", BudgetTamperingContextBuilder)
    suite = load_memory_evaluation_suite(SUITE_PATH)
    report = MemoryEvaluationRunner().run(suite)
    result = next(
        item for item in report.results if item.case_id == "confirmed-preference-cross-session"
    )

    budget_check = next(
        check for check in result.context.checks if check.name == "context.memory_budget"
    )
    assert budget_check.passed is False


def test_memory_budget_omits_oversized_records_atomically() -> None:
    suite = load_memory_evaluation_suite(SUITE_PATH)
    original = next(
        case for case in suite.cases if case.case_id == "confirmed-preference-cross-session"
    )
    oversized = replace(
        original.candidates[0],
        content="The user prefers concise Chinese answers. " + "x" * 500,
    )
    case = replace(
        original,
        candidates=(oversized,),
        context=replace(original.context, expected_projected_labels=()),
        answer=replace(
            original.answer,
            memory_required_labels=(),
            memory_answer_contains_all=(),
        ),
    )
    config = replace(
        suite.config,
        context_budget=replace(suite.config.context_budget, memory_max_characters=200),
    )

    result = MemoryEvaluationRunner().run(
        replace(suite, config=config, cases=(case,))
    ).results[0]

    assert result.passed is True
    assert result.context.projected_labels == ()
    assert result.context.omitted_count == 1
    assert result.context.memory_context_characters == 0


def test_phase_8_fingerprint_is_reproducible() -> None:
    suite = load_memory_evaluation_suite(SUITE_PATH)
    first = MemoryEvaluationRunner().run(suite)
    second = MemoryEvaluationRunner().run(suite)

    assert first.architecture_fingerprint == second.architecture_fingerprint
    assert first.deterministic_fingerprint == second.deterministic_fingerprint


def test_committed_deterministic_baseline_is_a_regression_gate() -> None:
    suite = load_memory_evaluation_suite(SUITE_PATH)
    report = MemoryEvaluationRunner().run(suite)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert baseline["suite_id"] == suite.suite_id
    assert baseline["mode"] == "deterministic"
    assert baseline["summary"]["passed"] is True
    assert baseline["architecture_fingerprint"] == report.architecture_fingerprint
    assert baseline["deterministic_fingerprint"] == report.deterministic_fingerprint


def test_metric_failure_is_reported_upstream_even_when_answer_passes() -> None:
    suite = load_memory_evaluation_suite(SUITE_PATH)
    original = next(
        case for case in suite.cases if case.case_id == "confirmed-preference-cross-session"
    )
    broken_recall = replace(original.recall, expected_selected_labels=("missing",))
    broken_case = replace(original, recall=broken_recall)
    broken_suite = replace(
        suite,
        cases=tuple(broken_case if case is original else case for case in suite.cases),
    )

    report = MemoryEvaluationRunner().run(broken_suite)
    result = next(item for item in report.results if item.case_id == original.case_id)

    assert report.passed is False
    assert result.answer.direct_answer_predicate_pass is True
    assert result.recalls[0].passed is False
    assert result.context.passed is True


def test_metric_negative_case_exposes_false_admission_and_no_result_failure() -> None:
    suite = load_memory_evaluation_suite(SUITE_PATH)
    original = next(
        case for case in suite.cases if case.case_id == "assistant-false-inference-not-admitted"
    )
    candidate = replace(
        original.candidates[0],
        action="confirm",
    )
    broken_case = replace(
        original,
        candidates=(candidate,),
    )
    broken_suite = replace(suite, cases=(broken_case,))

    report = MemoryEvaluationRunner().run(broken_suite)
    result = report.results[0]

    assert result.admission.false_admission_count == 1
    assert result.admission.false_admission_rate == 1.0
    assert result.admission.passed is False
    assert result.recalls[0].no_result_correct is False
    assert result.recalls[0].precision_at_k is None


def test_loader_rejects_invalid_schema_and_semantics(tmp_path: Path) -> None:
    invalid_schema = tmp_path / "invalid-schema.json"
    invalid_schema.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")

    with pytest.raises(DQAgentError, match="unsupported memory evaluation schema version"):
        load_memory_evaluation_suite(invalid_schema)

    payload = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    payload["cases"][0]["recall"]["expected_no_result"] = True
    invalid_semantics = tmp_path / "invalid-semantics.json"
    invalid_semantics.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DQAgentError, match="expected no-result recall cannot have relevant labels"):
        load_memory_evaluation_suite(invalid_semantics)

    payload = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    payload["cases"][0]["recall"]["relevant_labels"] = []
    invalid_denominator = tmp_path / "invalid-denominator.json"
    invalid_denominator.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DQAgentError, match="non-no-result recall must define relevant labels"):
        load_memory_evaluation_suite(invalid_denominator)


def test_memory_evaluation_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "memory-report.json"

    exit_code = memory_evaluation_cli.main(["--suite", str(SUITE_PATH), "--output", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["summary"]["passed"] is True
    assert report["summary"]["executed_cases"] == 13
