"""Live answer-level RAG evaluation kept separate from retrieval ranking gates."""

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from dqagent.application import SessionAgentApplication, SessionRunResult
from dqagent.context import ContextBudget, ContextBuilder, PromptAssembler, PromptSection
from dqagent.errors import DQAgentError, RetrievalError
from dqagent.llm import LLMClient
from dqagent.retrieval import (
    CharacterTextChunker,
    DocumentIngestor,
    HashingEmbeddingProvider,
    InMemoryVectorStore,
    SourceDocument,
    VectorRetriever,
)
from dqagent.runtime import AgentRuntime, RetryPolicy
from dqagent.session import InMemorySessionStore
from dqagent.tools import ToolRegistry

ANSWER_EVALUATION_SCHEMA_VERSION = 1
ANSWER_REPORT_SCHEMA_VERSION = 1
_CITATION_PATTERN = re.compile(r"\[(R[1-9]\d*)\]")
_SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


@dataclass(frozen=True, slots=True)
class AnswerClaim:
    text: str
    source_document_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalAnswerEvaluationCase:
    case_id: str
    query: str
    claims: tuple[AnswerClaim, ...]
    answer_fragments: tuple[str, ...]
    forbidden_fragments: tuple[str, ...]
    top_k: int
    expect_no_answer: bool


@dataclass(frozen=True, slots=True)
class RetrievalAnswerEvaluationSuite:
    suite_id: str
    schema_version: int
    documents: tuple[SourceDocument, ...]
    cases: tuple[RetrievalAnswerEvaluationCase, ...]
    chunk_characters: int
    chunk_overlap: int
    embedding_dimensions: int
    retrieval_min_score: float
    context_max_characters: int


@dataclass(frozen=True, slots=True)
class AnswerEvaluationCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class RetrievalAnswerEvaluationResult:
    case_id: str
    passed: bool
    output: str | None
    cited_document_ids: tuple[str, ...]
    citation_coverage: float | None
    checks: tuple[AnswerEvaluationCheck, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalAnswerEvaluationReport:
    suite_id: str
    suite_schema_version: int
    generated_at: datetime
    results: tuple[RetrievalAnswerEvaluationResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, object]:
        citation_coverages = [
            result.citation_coverage
            for result in self.results
            if result.citation_coverage is not None
        ]
        return {
            "report_schema_version": ANSWER_REPORT_SCHEMA_VERSION,
            "suite_id": self.suite_id,
            "suite_schema_version": self.suite_schema_version,
            "generated_at": self.generated_at.isoformat(),
            "summary": {
                "passed": self.passed,
                "passed_cases": sum(result.passed for result in self.results),
                "failed_cases": sum(not result.passed for result in self.results),
                "executed_cases": len(self.results),
                "mean_citation_coverage": (
                    sum(citation_coverages) / len(citation_coverages)
                    if citation_coverages
                    else None
                ),
            },
            "results": [
                {
                    "case_id": result.case_id,
                    "passed": result.passed,
                    "output": result.output,
                    "cited_document_ids": list(result.cited_document_ids),
                    "citation_coverage": result.citation_coverage,
                    "checks": [
                        {
                            "name": check.name,
                            "passed": check.passed,
                            "detail": check.detail,
                        }
                        for check in result.checks
                    ],
                    "error": result.error,
                }
                for result in self.results
            ],
        }


class RetrievalAnswerEvaluationRunner:
    """Runs live-provider answer cases through the production RAG application path."""

    def __init__(self, llm: LLMClient, *, run_timeout_seconds: float = 120.0) -> None:
        self._llm = llm
        self._run_timeout_seconds = run_timeout_seconds

    def run(self, suite: RetrievalAnswerEvaluationSuite) -> RetrievalAnswerEvaluationReport:
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
        results = tuple(self._run_case(case, retriever, suite) for case in suite.cases)
        return RetrievalAnswerEvaluationReport(
            suite.suite_id,
            suite.schema_version,
            datetime.now(UTC),
            results,
        )

    def _run_case(
        self,
        case: RetrievalAnswerEvaluationCase,
        retriever: VectorRetriever,
        suite: RetrievalAnswerEvaluationSuite,
    ) -> RetrievalAnswerEvaluationResult:
        runtime = AgentRuntime(
            self._llm,
            ToolRegistry(),
            retry_policy=RetryPolicy(),
            default_timeout_seconds=self._run_timeout_seconds,
        )
        app = SessionAgentApplication.create(
            runtime,
            InMemorySessionStore(),
            ContextBuilder(
                PromptAssembler(
                    (
                        PromptSection(
                            "rag-answer",
                            "Answer from retrieved evidence only. State when the evidence is "
                            "insufficient, and cite every factual claim in the same sentence "
                            "with its passage ID.",
                        ),
                    )
                ),
                ContextBudget(max_characters=suite.context_max_characters),
            ),
            session_id=f"rag-answer-{case.case_id}",
            retriever=retriever,
            retrieval_limit=case.top_k,
            retrieval_min_score=suite.retrieval_min_score,
        )
        try:
            result = app.run(case.query)
        except DQAgentError as exc:
            check = AnswerEvaluationCheck("run.completed", False, f"run failed: {exc}")
            return RetrievalAnswerEvaluationResult(
                case.case_id,
                False,
                None,
                (),
                None,
                (check,),
                f"{type(exc).__name__}: {exc}",
            )
        return self._evaluate(case, result)

    @staticmethod
    def _evaluate(
        case: RetrievalAnswerEvaluationCase,
        result: SessionRunResult,
    ) -> RetrievalAnswerEvaluationResult:
        output = result.output.content
        resolution = result.citations
        cited_document_ids = (
            tuple(dict.fromkeys(chunk.document_id for chunk in resolution.cited.values()))
            if resolution is not None
            else ()
        )
        checks: list[AnswerEvaluationCheck] = []
        output_folded = output.casefold()
        missing = [
            fragment
            for fragment in case.answer_fragments
            if fragment.casefold() not in output_folded
        ]
        checks.append(
            AnswerEvaluationCheck(
                "answer.fragments",
                not missing,
                "all expected answer fragments found"
                if not missing
                else f"missing fragments: {missing}",
            )
        )
        forbidden = [
            fragment
            for fragment in case.forbidden_fragments
            if fragment.casefold() in output_folded
        ]
        checks.append(
            AnswerEvaluationCheck(
                "answer.forbidden_fragments",
                not forbidden,
                "no forbidden instruction output found"
                if not forbidden
                else f"forbidden fragments found: {forbidden}",
            )
        )
        unknown_ids = resolution.unknown_ids if resolution is not None else ()
        checks.append(
            AnswerEvaluationCheck(
                "citation.unknown_ids",
                not unknown_ids,
                "all answer citations resolve to retrieved passages"
                if not unknown_ids
                else f"unknown citation IDs: {list(unknown_ids)}",
            )
        )
        if case.expect_no_answer:
            no_sources = result.retrieval is not None and not result.retrieval.chunks
            stated_insufficient = not missing
            checks.append(
                AnswerEvaluationCheck(
                    "answer.insufficient_evidence",
                    no_sources and not cited_document_ids and stated_insufficient,
                    "retrieval returned no sources and the answer stated insufficiency"
                    if no_sources and not cited_document_ids and stated_insufficient
                    else "answer had evidence, citations, or no insufficiency statement",
                )
            )
        claim_checks = tuple(
            _evaluate_claim(index, claim, output, result)
            for index, claim in enumerate(case.claims)
        )
        supported_claims = sum(check.passed for check in claim_checks)
        coverage = supported_claims / len(claim_checks) if claim_checks else None
        if coverage is not None:
            checks.append(
                AnswerEvaluationCheck(
                    "citation.claim_coverage",
                    coverage == 1.0,
                    f"supported {supported_claims} of {len(claim_checks)} expected claims",
                )
            )
        checks.extend(claim_checks)
        return RetrievalAnswerEvaluationResult(
            case.case_id,
            all(check.passed for check in checks),
            output,
            cited_document_ids,
            coverage,
            tuple(checks),
        )


def load_retrieval_answer_evaluation_suite(path: Path) -> RetrievalAnswerEvaluationSuite:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalError(
            f"cannot load retrieval answer evaluation suite '{path}': {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise RetrievalError("retrieval answer evaluation suite must be a JSON object")
    data = cast(dict[str, Any], raw)
    try:
        Draft202012Validator(_ANSWER_EVALUATION_SCHEMA).validate(data)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "root"
        raise RetrievalError(
            f"invalid retrieval answer evaluation suite at {location}: {exc.message}"
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
        RetrievalAnswerEvaluationCase(
            cast(str, item["id"]),
            cast(str, item["query"]),
            tuple(
                AnswerClaim(
                    cast(str, claim["text"]),
                    tuple(cast(list[str], claim["source_document_ids"])),
                )
                for claim in cast(list[dict[str, Any]], item["claims"])
            ),
            tuple(cast(list[str], item["answer_fragments"])),
            tuple(cast(list[str], item["forbidden_fragments"])),
            cast(int, item["top_k"]),
            cast(bool, item["expect_no_answer"]),
        )
        for item in cast(list[dict[str, Any]], data["cases"])
    )
    document_ids = [document.document_id for document in documents]
    case_ids = [case.case_id for case in cases]
    if len(document_ids) != len(set(document_ids)):
        raise RetrievalError("retrieval answer evaluation document IDs must be unique")
    if len(case_ids) != len(set(case_ids)):
        raise RetrievalError("retrieval answer evaluation case IDs must be unique")
    known = set(document_ids)
    if any(
        not set(claim.source_document_ids).issubset(known)
        for case in cases
        for claim in case.claims
    ):
        raise RetrievalError("retrieval answer evaluation claim references an unknown document")
    if any(case.expect_no_answer and case.claims for case in cases):
        raise RetrievalError("no-answer cases must not declare factual claims")
    if any(not case.expect_no_answer and not case.claims for case in cases):
        raise RetrievalError("answer cases must declare at least one factual claim")
    config = cast(dict[str, Any], data["config"])
    return RetrievalAnswerEvaluationSuite(
        cast(str, data["suite_id"]),
        cast(int, data["schema_version"]),
        documents,
        cases,
        cast(int, config["chunk_characters"]),
        cast(int, config["chunk_overlap"]),
        cast(int, config["embedding_dimensions"]),
        cast(float, config["retrieval_min_score"]),
        cast(int, config["context_max_characters"]),
    )


def _evaluate_claim(
    index: int,
    claim: AnswerClaim,
    output: str,
    result: SessionRunResult,
) -> AnswerEvaluationCheck:
    claim_folded = claim.text.casefold()
    matching_sentences = [
        sentence.group(0)
        for sentence in _SENTENCE_PATTERN.finditer(output)
        if claim_folded in sentence.group(0).casefold()
    ]
    if not matching_sentences:
        return AnswerEvaluationCheck(
            f"citation.claim[{index}]",
            False,
            f"claim fragment is absent: {claim.text!r}",
        )
    citations = result.retrieval.citations if result.retrieval is not None else {}
    for sentence in matching_sentences:
        for citation_id in _CITATION_PATTERN.findall(sentence):
            chunk = citations.get(citation_id)
            if (
                chunk is not None
                and chunk.document_id in claim.source_document_ids
                and claim_folded in chunk.content.casefold()
            ):
                return AnswerEvaluationCheck(
                    f"citation.claim[{index}]",
                    True,
                    f"claim has local support from {citation_id} ({chunk.document_id})",
                )
    return AnswerEvaluationCheck(
        f"citation.claim[{index}]",
        False,
        "claim has no same-sentence citation to an allowed source containing the claim",
    )


_ANSWER_EVALUATION_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "suite_id", "config", "documents", "cases"],
    "properties": {
        "schema_version": {"const": ANSWER_EVALUATION_SCHEMA_VERSION},
        "suite_id": {"type": "string", "minLength": 1},
        "config": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "chunk_characters",
                "chunk_overlap",
                "embedding_dimensions",
                "retrieval_min_score",
                "context_max_characters",
            ],
            "properties": {
                "chunk_characters": {"type": "integer", "minimum": 1},
                "chunk_overlap": {"type": "integer", "minimum": 0},
                "embedding_dimensions": {"type": "integer", "minimum": 8},
                "retrieval_min_score": {"type": "number"},
                "context_max_characters": {"type": "integer", "minimum": 1},
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
                    "claims",
                    "answer_fragments",
                    "forbidden_fragments",
                    "top_k",
                    "expect_no_answer",
                ],
                "properties": {
                    "id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$"},
                    "query": {"type": "string", "minLength": 1},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["text", "source_document_ids"],
                            "properties": {
                                "text": {"type": "string", "minLength": 1},
                                "source_document_ids": {
                                    "type": "array",
                                    "minItems": 1,
                                    "uniqueItems": True,
                                    "items": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                    },
                    "answer_fragments": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "forbidden_fragments": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "top_k": {"type": "integer", "minimum": 1},
                    "expect_no_answer": {"type": "boolean"},
                },
            },
        },
    },
}
