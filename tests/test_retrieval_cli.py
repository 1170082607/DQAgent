import json
from pathlib import Path

import pytest

from dqagent import cli, retrieval_cli, retrieval_evaluation_cli


def test_index_cli_upserts_queries_and_deletes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "policy.md"
    source.write_text("refund policy permits thirty days", encoding="utf-8")
    index = tmp_path / "index.json"

    assert retrieval_cli.main(
        ["--index", str(index), "upsert", "policy", str(source)]
    ) == 0
    upsert = json.loads(capsys.readouterr().out)
    assert upsert["indexed_chunks"] == 1

    assert retrieval_cli.main(
        ["--index", str(index), "query", "refund policy", "--limit", "1"]
    ) == 0
    query = json.loads(capsys.readouterr().out)
    assert query["results"][0]["document_id"] == "policy"
    assert query["results"][0]["citation_id"] == "R1"

    assert retrieval_cli.main(["--index", str(index), "delete", "policy"]) == 0
    deleted = json.loads(capsys.readouterr().out)
    assert deleted["deleted_chunks"] == 1


def test_index_cli_reports_missing_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = retrieval_cli.main(
        [
            "--index",
            str(tmp_path / "index.json"),
            "upsert",
            "missing",
            str(tmp_path / "missing.md"),
        ]
    )

    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


def test_retrieval_evaluation_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    assert retrieval_evaluation_cli.main(["--output", str(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["passed"] is True
    assert report["summary"]["mean_recall_at_k"] == 1.0


def test_main_rejects_retrieval_without_durable_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DQAGENT_MODEL", "test-model")

    exit_code = cli.main(["--retrieval-index", "index.json", "--message", "hello"])

    assert exit_code == 1
    assert "requires --session-id" in capsys.readouterr().err
