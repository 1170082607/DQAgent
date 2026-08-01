"""Deterministic regression evaluation for active-context construction."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from dqagent.context import (
    ContextBudget,
    ContextBuilder,
    PromptAssembler,
    PromptSection,
    SummaryMethod,
)
from dqagent.errors import ContextError
from dqagent.models import ConversationItem, Message, Role

CONTEXT_EVALUATION_SCHEMA_VERSION = 1
CONTEXT_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ContextEvaluationCase:
    case_id: str
    prompt_sections: tuple[PromptSection, ...]
    transcript: tuple[ConversationItem, ...]
    user_message: str
    budget: ContextBudget
    contains_all: tuple[str, ...]
    absent_all: tuple[str, ...]
    min_omitted_turns: int
    min_structural_omitted_turns: int
    min_summary_omitted_turns: int
    summary_method: SummaryMethod | None


@dataclass(frozen=True, slots=True)
class ContextEvaluationSuite:
    suite_id: str
    schema_version: int
    cases: tuple[ContextEvaluationCase, ...]


@dataclass(frozen=True, slots=True)
class ContextEvaluationCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ContextEvaluationResult:
    case_id: str
    passed: bool
    estimated_characters: int
    max_characters: int
    retained_turns: int
    omitted_turns: int
    structural_input_turns: int
    structural_omitted_turns: int
    summary_source_turns: int
    summary_omitted_turns: int
    summary_method: SummaryMethod | None
    checks: tuple[ContextEvaluationCheck, ...]


@dataclass(frozen=True, slots=True)
class ContextEvaluationReport:
    suite_id: str
    suite_schema_version: int
    generated_at: datetime
    results: tuple[ContextEvaluationResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, object]:
        return {
            "report_schema_version": CONTEXT_REPORT_SCHEMA_VERSION,
            "suite_id": self.suite_id,
            "suite_schema_version": self.suite_schema_version,
            "generated_at": self.generated_at.isoformat(),
            "summary": {
                "passed": self.passed,
                "passed_cases": sum(result.passed for result in self.results),
                "failed_cases": sum(not result.passed for result in self.results),
                "executed_cases": len(self.results),
            },
            "results": [
                {
                    "case_id": result.case_id,
                    "passed": result.passed,
                    "metrics": {
                        "estimated_characters": result.estimated_characters,
                        "max_characters": result.max_characters,
                        "retained_turns": result.retained_turns,
                        "omitted_turns": result.omitted_turns,
                        "structural_input_turns": result.structural_input_turns,
                        "structural_omitted_turns": result.structural_omitted_turns,
                        "summary_source_turns": result.summary_source_turns,
                        "summary_omitted_turns": result.summary_omitted_turns,
                        "summary_method": (
                            result.summary_method.value if result.summary_method else None
                        ),
                    },
                    "checks": [
                        {
                            "name": check.name,
                            "passed": check.passed,
                            "detail": check.detail,
                        }
                        for check in result.checks
                    ],
                }
                for result in self.results
            ],
        }


class ContextEvaluationRunner:
    """Runs cases through the production ContextBuilder without an LLM provider."""

    def run(self, suite: ContextEvaluationSuite) -> ContextEvaluationReport:
        return ContextEvaluationReport(
            suite_id=suite.suite_id,
            suite_schema_version=suite.schema_version,
            generated_at=datetime.now(UTC),
            results=tuple(self._run_case(case) for case in suite.cases),
        )

    @staticmethod
    def _run_case(case: ContextEvaluationCase) -> ContextEvaluationResult:
        window = ContextBuilder(
            PromptAssembler(case.prompt_sections),
            case.budget,
        ).build(case.transcript, Message(Role.USER, case.user_message))
        rendered = "\n".join(
            item.content for item in window.items if isinstance(item, Message)
        )
        missing = [value for value in case.contains_all if value not in rendered]
        unexpectedly_present = [value for value in case.absent_all if value in rendered]
        actual_method = window.summary.method if window.summary else None
        structural_input_turns = (
            window.summary.structural_input_turns if window.summary else 0
        )
        structural_omitted_turns = (
            window.summary.structural_omitted_turns if window.summary else 0
        )
        summary_source_turns = window.summary.summary_source_turns if window.summary else 0
        summary_omitted_turns = window.summary.summary_omitted_turns if window.summary else 0
        checks = (
            ContextEvaluationCheck(
                "context.contains_all",
                not missing,
                "all required fragments retained" if not missing else f"missing: {missing}",
            ),
            ContextEvaluationCheck(
                "context.absent_all",
                not unexpectedly_present,
                (
                    "all expected losses absent"
                    if not unexpectedly_present
                    else f"unexpectedly present: {unexpectedly_present}"
                ),
            ),
            ContextEvaluationCheck(
                "context.within_budget",
                window.estimated_characters <= window.max_characters,
                f"observed {window.estimated_characters}, maximum {window.max_characters}",
            ),
            ContextEvaluationCheck(
                "context.min_omitted_turns",
                window.omitted_turns >= case.min_omitted_turns,
                f"observed {window.omitted_turns}, minimum {case.min_omitted_turns}",
            ),
            ContextEvaluationCheck(
                "context.min_structural_omitted_turns",
                structural_omitted_turns >= case.min_structural_omitted_turns,
                "observed "
                f"{structural_omitted_turns}, minimum "
                f"{case.min_structural_omitted_turns}",
            ),
            ContextEvaluationCheck(
                "context.min_summary_omitted_turns",
                summary_omitted_turns >= case.min_summary_omitted_turns,
                f"observed {summary_omitted_turns}, minimum {case.min_summary_omitted_turns}",
            ),
            ContextEvaluationCheck(
                "context.summary_method",
                actual_method is case.summary_method,
                (
                    "expected "
                    f"{case.summary_method.value if case.summary_method else None}, got "
                    f"{actual_method.value if actual_method else None}"
                ),
            ),
        )
        return ContextEvaluationResult(
            case_id=case.case_id,
            passed=all(check.passed for check in checks),
            estimated_characters=window.estimated_characters,
            max_characters=window.max_characters,
            retained_turns=window.retained_turns,
            omitted_turns=window.omitted_turns,
            structural_input_turns=structural_input_turns,
            structural_omitted_turns=structural_omitted_turns,
            summary_source_turns=summary_source_turns,
            summary_omitted_turns=summary_omitted_turns,
            summary_method=actual_method,
            checks=checks,
        )


def load_context_evaluation_suite(path: Path) -> ContextEvaluationSuite:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextError(f"cannot load context evaluation suite '{path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise ContextError("context evaluation suite must be a JSON object")
    data = cast(dict[str, Any], raw)
    try:
        Draft202012Validator(_CONTEXT_EVALUATION_SCHEMA).validate(data)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "root"
        raise ContextError(
            f"invalid context evaluation suite at {location}: {exc.message}"
        ) from exc
    cases = tuple(_parse_case(cast(dict[str, Any], item)) for item in data["cases"])
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ContextError("context evaluation case IDs must be unique")
    return ContextEvaluationSuite(
        cast(str, data["suite_id"]),
        cast(int, data["schema_version"]),
        cases,
    )


def _parse_case(data: dict[str, Any]) -> ContextEvaluationCase:
    raw_input = cast(dict[str, Any], data["input"])
    raw_budget = cast(dict[str, int], data["budget"])
    raw_expected = cast(dict[str, Any], data["expected"])
    transcript: list[ConversationItem] = []
    for turn in cast(list[dict[str, str]], raw_input["turns"]):
        transcript.extend(
            (Message(Role.USER, turn["user"]), Message(Role.ASSISTANT, turn["assistant"]))
        )
    raw_method = raw_expected.get("summary_method")
    return ContextEvaluationCase(
        case_id=cast(str, data["id"]),
        prompt_sections=tuple(
            PromptSection(section["name"], section["content"])
            for section in cast(list[dict[str, str]], raw_input.get("prompt_sections", []))
        ),
        transcript=tuple(transcript),
        user_message=cast(str, raw_input["user_message"]),
        budget=ContextBudget(**raw_budget),
        contains_all=tuple(cast(list[str], raw_expected.get("contains_all", []))),
        absent_all=tuple(cast(list[str], raw_expected.get("absent_all", []))),
        min_omitted_turns=cast(int, raw_expected.get("min_omitted_turns", 0)),
        min_structural_omitted_turns=cast(
            int, raw_expected.get("min_structural_omitted_turns", 0)
        ),
        min_summary_omitted_turns=cast(
            int, raw_expected.get("min_summary_omitted_turns", 0)
        ),
        summary_method=SummaryMethod(raw_method) if raw_method is not None else None,
    )


_CONTEXT_EVALUATION_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "suite_id", "cases"],
    "properties": {
        "schema_version": {"const": CONTEXT_EVALUATION_SCHEMA_VERSION},
        "suite_id": {"type": "string", "minLength": 1},
        "cases": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "input", "budget", "expected"],
                "properties": {
                    "id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$"},
                    "input": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["turns", "user_message"],
                        "properties": {
                            "turns": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["user", "assistant"],
                                    "properties": {
                                        "user": {"type": "string", "minLength": 1},
                                        "assistant": {"type": "string", "minLength": 1},
                                    },
                                },
                            },
                            "user_message": {"type": "string", "minLength": 1},
                            "prompt_sections": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["name", "content"],
                                    "properties": {
                                        "name": {"type": "string", "minLength": 1},
                                        "content": {"type": "string", "minLength": 1},
                                    },
                                },
                            },
                        },
                    },
                    "budget": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["max_characters"],
                        "properties": {
                            "max_characters": {"type": "integer", "minimum": 1},
                            "reserved_characters": {"type": "integer", "minimum": 0},
                            "summary_max_characters": {"type": "integer", "minimum": 0},
                            "structural_input_max_characters": {
                                "type": "integer",
                                "minimum": 1,
                            },
                            "min_recent_turns": {"type": "integer", "minimum": 0},
                        },
                    },
                    "expected": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "contains_all": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                            },
                            "absent_all": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                            },
                            "min_omitted_turns": {"type": "integer", "minimum": 0},
                            "min_structural_omitted_turns": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "min_summary_omitted_turns": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "summary_method": {
                                "enum": [method.value for method in SummaryMethod]
                            },
                        },
                    },
                },
            },
        },
    },
}
