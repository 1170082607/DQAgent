import json
from pathlib import Path

import pytest

from dqagent.evaluation_cli import main

BASELINE_SUITE = Path("evaluations/cases/phase-3-baseline-v1.json")


def test_deterministic_cli_writes_passing_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "--mode",
            "deterministic",
            "--suite",
            str(BASELINE_SUITE),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["summary"]["passed"] is True
    assert report["summary"]["executed_cases"] == 4


def test_cli_reports_invalid_suite(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    missing = tmp_path / "missing.json"

    assert main(["--suite", str(missing)]) == 2
    assert "cannot load evaluation suite" in capsys.readouterr().err
