from pathlib import Path


def test_ci_runs_and_uploads_the_phase8_memory_evaluation_report() -> None:
    workflow = (Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "dqagent-memory-eval" in workflow
    assert "--output memory-evaluation-report.json" in workflow
    assert "memory-evaluation-report.json" in workflow
