import json
from pathlib import Path

import pytest

from dqagent.errors import RetrievalError
from dqagent.retrieval_evaluation import (
    RetrievalEvaluationRunner,
    load_retrieval_evaluation_suite,
)


def test_committed_retrieval_suite_passes_and_reports_recall() -> None:
    suite = load_retrieval_evaluation_suite(
        Path("evaluations/cases/phase-7-retrieval-v1.json")
    )

    report = RetrievalEvaluationRunner().run(suite)

    assert report.passed
    assert len(report.results) == 7
    assert all(
        result.recall_at_k == 1.0
        for result in report.results
        if not result.expected_no_results
    )
    no_answer = next(
        result for result in report.results if result.case_id == "no-answer-audit-retention"
    )
    assert no_answer.no_results is True
    assert no_answer.recall_at_k is None
    assert no_answer.reciprocal_rank is None
    assert report.to_dict()["summary"]["mean_recall_at_k"] == 1.0  # type: ignore[index]
    assert report.to_dict()["summary"]["expected_no_result_cases"] == 1  # type: ignore[index]
    assert report.to_dict()["summary"]["empty_result_cases"] == 1  # type: ignore[index]


def test_loader_rejects_unknown_relevant_document(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "suite_id": "invalid",
        "config": {
            "chunk_characters": 100,
            "chunk_overlap": 0,
            "embedding_dimensions": 64,
            "min_score": 0.05,
        },
        "documents": [{"id": "known", "content": "text", "source": "known.md"}],
        "cases": [
            {
                "id": "missing",
                "query": "text",
                "relevant_document_ids": ["unknown"],
                "top_k": 1,
                "min_recall": 1,
                "min_reciprocal_rank": 1,
            }
        ],
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrievalError, match="unknown document"):
        load_retrieval_evaluation_suite(path)
