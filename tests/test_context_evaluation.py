import json
from pathlib import Path

import pytest

from dqagent import context_evaluation_cli
from dqagent.context_evaluation import (
    ContextEvaluationRunner,
    load_context_evaluation_suite,
)
from dqagent.errors import ContextError

SUITE = Path("evaluations/cases/phase-6-context-v1.json")


def test_phase_6_context_regression_suite_passes() -> None:
    report = ContextEvaluationRunner().run(load_context_evaluation_suite(SUITE))

    assert report.passed is True
    assert len(report.results) == 3
    assert report.to_dict()["summary"] == {
        "passed": True,
        "passed_cases": 3,
        "failed_cases": 0,
        "executed_cases": 3,
    }
    loss = next(
        result
        for result in report.results
        if result.case_id == "structural_compaction_loss_visible"
    )
    assert loss.omitted_turns == 2


def test_context_evaluation_loader_rejects_invalid_schema(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")

    with pytest.raises(ContextError, match="invalid context evaluation suite"):
        load_context_evaluation_suite(path)


def test_context_evaluation_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    exit_code = context_evaluation_cli.main(["--suite", str(SUITE), "--output", str(output)])

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["passed"] is True
