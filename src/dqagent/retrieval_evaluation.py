"""Recall-oriented deterministic evaluation for the production retrieval path."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from dqagent.errors import RetrievalError
from dqagent.retrieval import (
    CharacterTextChunker,
    DocumentIngestor,
    HashingEmbeddingProvider,
    InMemoryVectorStore,
    SourceDocument,
    VectorRetriever,
)

RETRIEVAL_EVALUATION_SCHEMA_VERSION = 1
RETRIEVAL_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    case_id: str
    query: str
    relevant_document_ids: tuple[str, ...]
    top_k: int
    min_recall: float
    min_reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationSuite:
    suite_id: str
    schema_version: int
    documents: tuple[SourceDocument, ...]
    cases: tuple[RetrievalEvaluationCase, ...]
    chunk_characters: int
    chunk_overlap: int
    embedding_dimensions: int


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    case_id: str
    passed: bool
    retrieved_document_ids: tuple[str, ...]
    recall_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    suite_id: str
    suite_schema_version: int
    generated_at: datetime
    results: tuple[RetrievalEvaluationResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, object]:
        return {
            "report_schema_version": RETRIEVAL_REPORT_SCHEMA_VERSION,
            "suite_id": self.suite_id,
            "suite_schema_version": self.suite_schema_version,
            "generated_at": self.generated_at.isoformat(),
            "summary": {
                "passed": self.passed,
                "passed_cases": sum(result.passed for result in self.results),
                "failed_cases": sum(not result.passed for result in self.results),
                "executed_cases": len(self.results),
                "mean_recall_at_k": sum(result.recall_at_k for result in self.results)
                / len(self.results),
                "mean_reciprocal_rank": sum(
                    result.reciprocal_rank for result in self.results
                )
                / len(self.results),
            },
            "results": [
                {
                    "case_id": result.case_id,
                    "passed": result.passed,
                    "retrieved_document_ids": list(result.retrieved_document_ids),
                    "metrics": {
                        "recall_at_k": result.recall_at_k,
                        "reciprocal_rank": result.reciprocal_rank,
                    },
                }
                for result in self.results
            ],
        }


class RetrievalEvaluationRunner:
    """Indexes fixtures once, then evaluates retrieval without answer generation."""

    def run(self, suite: RetrievalEvaluationSuite) -> RetrievalEvaluationReport:
        embeddings = HashingEmbeddingProvider(suite.embedding_dimensions)
        store = InMemoryVectorStore()
        ingestor = DocumentIngestor(
            CharacterTextChunker(
                max_characters=suite.chunk_characters,
                overlap_characters=suite.chunk_overlap,
            ),
            embeddings,
            store,
        )
        for document in suite.documents:
            ingestor.upsert(document)
        retriever = VectorRetriever(embeddings, store)
        return RetrievalEvaluationReport(
            suite.suite_id,
            suite.schema_version,
            datetime.now(UTC),
            tuple(self._run_case(case, retriever) for case in suite.cases),
        )

    @staticmethod
    def _run_case(
        case: RetrievalEvaluationCase,
        retriever: VectorRetriever,
    ) -> RetrievalEvaluationResult:
        result = retriever.retrieve(case.query, limit=case.top_k)
        retrieved = tuple(item.chunk.document_id for item in result.chunks)
        relevant = set(case.relevant_document_ids)
        recall = len(relevant.intersection(retrieved)) / len(relevant)
        reciprocal_rank = next(
            (
                1.0 / rank
                for rank, document_id in enumerate(retrieved, start=1)
                if document_id in relevant
            ),
            0.0,
        )
        return RetrievalEvaluationResult(
            case.case_id,
            recall >= case.min_recall and reciprocal_rank >= case.min_reciprocal_rank,
            retrieved,
            recall,
            reciprocal_rank,
        )


def load_retrieval_evaluation_suite(path: Path) -> RetrievalEvaluationSuite:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalError(f"cannot load retrieval evaluation suite '{path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise RetrievalError("retrieval evaluation suite must be a JSON object")
    data = cast(dict[str, Any], raw)
    try:
        Draft202012Validator(_RETRIEVAL_EVALUATION_SCHEMA).validate(data)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "root"
        raise RetrievalError(
            f"invalid retrieval evaluation suite at {location}: {exc.message}"
        ) from exc
    documents = tuple(
        SourceDocument(
            cast(str, item["id"]),
            cast(str, item["content"]),
            cast(str, item["source"]),
            cast(dict[str, str], item.get("metadata", {})),
        )
        for item in cast(list[dict[str, Any]], data["documents"])
    )
    cases = tuple(
        RetrievalEvaluationCase(
            cast(str, item["id"]),
            cast(str, item["query"]),
            tuple(cast(list[str], item["relevant_document_ids"])),
            cast(int, item["top_k"]),
            float(item["min_recall"]),
            float(item["min_reciprocal_rank"]),
        )
        for item in cast(list[dict[str, Any]], data["cases"])
    )
    document_ids = [document.document_id for document in documents]
    case_ids = [case.case_id for case in cases]
    if len(document_ids) != len(set(document_ids)):
        raise RetrievalError("retrieval evaluation document IDs must be unique")
    if len(case_ids) != len(set(case_ids)):
        raise RetrievalError("retrieval evaluation case IDs must be unique")
    known = set(document_ids)
    if any(not set(case.relevant_document_ids).issubset(known) for case in cases):
        raise RetrievalError("retrieval evaluation case references an unknown document")
    config = cast(dict[str, int], data["config"])
    return RetrievalEvaluationSuite(
        cast(str, data["suite_id"]),
        cast(int, data["schema_version"]),
        documents,
        cases,
        config["chunk_characters"],
        config["chunk_overlap"],
        config["embedding_dimensions"],
    )


_RETRIEVAL_EVALUATION_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "suite_id", "config", "documents", "cases"],
    "properties": {
        "schema_version": {"const": RETRIEVAL_EVALUATION_SCHEMA_VERSION},
        "suite_id": {"type": "string", "minLength": 1},
        "config": {
            "type": "object",
            "additionalProperties": False,
            "required": ["chunk_characters", "chunk_overlap", "embedding_dimensions"],
            "properties": {
                "chunk_characters": {"type": "integer", "minimum": 1},
                "chunk_overlap": {"type": "integer", "minimum": 0},
                "embedding_dimensions": {"type": "integer", "minimum": 8},
            },
        },
        "documents": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "content", "source"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "content": {"type": "string", "minLength": 1},
                    "source": {"type": "string", "minLength": 1},
                    "metadata": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                },
            },
        },
        "cases": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "query",
                    "relevant_document_ids",
                    "top_k",
                    "min_recall",
                    "min_reciprocal_rank",
                ],
                "properties": {
                    "id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$"},
                    "query": {"type": "string", "minLength": 1},
                    "relevant_document_ids": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "top_k": {"type": "integer", "minimum": 1},
                    "min_recall": {"type": "number", "minimum": 0, "maximum": 1},
                    "min_reciprocal_rank": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
            },
        },
    },
}
