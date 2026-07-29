"""Versioned evaluation cases, deterministic fixtures, and behavioral checks."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from dqagent.builtin_tools import create_builtin_tool_registry
from dqagent.errors import DQAgentError, ErrorCategory
from dqagent.execution import RunContext
from dqagent.llm import LLMClient
from dqagent.models import (
    Completion,
    ConversationItem,
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolErrorCode,
    ToolOutcome,
    ToolResult,
)
from dqagent.runtime import (
    AgentRuntime,
    EventSink,
    RetryPolicy,
    RunEvent,
    RunEventType,
    RunState,
)

EVALUATION_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1


class EvaluationDefinitionError(DQAgentError):
    """Raised when an evaluation suite does not match its declared schema."""

    category = ErrorCategory.CONFIGURATION


class EvaluationMode(StrEnum):
    DETERMINISTIC = "deterministic"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class EvaluationInput:
    user_message: str
    system_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class AnswerExpectation:
    non_empty: bool = True
    exact: str | None = None
    contains_all: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolCallExpectation:
    name: str
    arguments: Mapping[str, object]
    outcome: ToolOutcome | None = None
    error_code: ToolErrorCode | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    max_elapsed_seconds: float | None = None
    max_total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class TraceExpectation:
    required_event_types: tuple[RunEventType, ...]
    max_model_attempts: int | None = None
    resource_limits: Mapping[EvaluationMode, ResourceLimits] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_limits",
            MappingProxyType(dict(self.resource_limits)),
        )


@dataclass(frozen=True, slots=True)
class EvaluationExpectation:
    answer: AnswerExpectation
    tool_calls: tuple[ToolCallExpectation, ...]
    trace: TraceExpectation


@dataclass(frozen=True, slots=True)
class EvaluationFixture:
    completions: tuple[Completion, ...]


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    modes: frozenset[EvaluationMode]
    input: EvaluationInput
    expected: EvaluationExpectation
    fixture: EvaluationFixture | None = None


@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    suite_id: str
    schema_version: int
    cases: tuple[EvaluationCase, ...]


@dataclass(frozen=True, slots=True)
class EvaluationCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    elapsed_seconds: float
    model_attempts: int
    tool_calls: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    case_id: str
    passed: bool
    run_id: str
    run_state: RunState
    output: str | None
    metrics: EvaluationMetrics
    checks: tuple[EvaluationCheck, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    suite_id: str
    suite_schema_version: int
    mode: EvaluationMode
    generated_at: datetime
    results: tuple[EvaluationCaseResult, ...]
    skipped_case_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, object]:
        passed_cases = sum(result.passed for result in self.results)
        return {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "suite_id": self.suite_id,
            "suite_schema_version": self.suite_schema_version,
            "mode": self.mode.value,
            "generated_at": self.generated_at.isoformat(),
            "summary": {
                "passed": self.passed,
                "passed_cases": passed_cases,
                "failed_cases": len(self.results) - passed_cases,
                "executed_cases": len(self.results),
                "skipped_cases": len(self.skipped_case_ids),
            },
            "skipped_case_ids": list(self.skipped_case_ids),
            "results": [self._result_to_dict(result) for result in self.results],
        }

    @staticmethod
    def _result_to_dict(result: EvaluationCaseResult) -> dict[str, object]:
        return {
            "case_id": result.case_id,
            "passed": result.passed,
            "run_id": result.run_id,
            "run_state": result.run_state.value,
            "output": result.output,
            "metrics": {
                "elapsed_seconds": result.metrics.elapsed_seconds,
                "model_attempts": result.metrics.model_attempts,
                "tool_calls": result.metrics.tool_calls,
                "input_tokens": result.metrics.input_tokens,
                "output_tokens": result.metrics.output_tokens,
                "total_tokens": result.metrics.total_tokens,
            },
            "checks": [
                {"name": check.name, "passed": check.passed, "detail": check.detail}
                for check in result.checks
            ],
            "error": result.error,
        }


class _CollectingSink(EventSink):
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


class _ScriptedLLM(LLMClient):
    def __init__(self, completions: Sequence[Completion]) -> None:
        self._completions = tuple(completions)
        self._position = 0

    @property
    def consumed_all(self) -> bool:
        return self._position == len(self._completions)

    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolDefinition] = (),
        *,
        context: RunContext | None = None,
    ) -> Completion:
        del messages, tools
        if context is not None:
            context.check_active()
        if self._position >= len(self._completions):
            raise EvaluationDefinitionError("deterministic fixture ran out of completions")
        completion = self._completions[self._position]
        self._position += 1
        return completion


class EvaluationRunner:
    """Runs isolated cases through the production runtime and checks observable behavior."""

    def __init__(
        self,
        mode: EvaluationMode,
        *,
        live_client: LLMClient | None = None,
        run_timeout_seconds: float = 120.0,
        max_model_attempts: int = 3,
    ) -> None:
        if mode is EvaluationMode.LIVE and live_client is None:
            raise ValueError("live evaluation requires an LLM client")
        self._mode = mode
        self._live_client = live_client
        self._run_timeout_seconds = run_timeout_seconds
        self._max_model_attempts = max_model_attempts

    def run(self, suite: EvaluationSuite) -> EvaluationReport:
        results: list[EvaluationCaseResult] = []
        skipped: list[str] = []
        for case in suite.cases:
            if self._mode not in case.modes:
                skipped.append(case.case_id)
                continue
            results.append(self._run_case(case))
        return EvaluationReport(
            suite_id=suite.suite_id,
            suite_schema_version=suite.schema_version,
            mode=self._mode,
            generated_at=datetime.now(UTC),
            results=tuple(results),
            skipped_case_ids=tuple(skipped),
        )

    def _run_case(self, case: EvaluationCase) -> EvaluationCaseResult:
        client = self._client_for(case)
        sink = _CollectingSink()
        runtime = AgentRuntime(
            client,
            create_builtin_tool_registry(),
            retry_policy=RetryPolicy(max_attempts=self._max_model_attempts),
            default_timeout_seconds=self._run_timeout_seconds,
            event_sinks=(sink,),
        )
        context = RunContext(
            run_id=f"eval-{self._mode.value}-{case.case_id}",
            timeout_seconds=self._run_timeout_seconds,
            metadata={"evaluation_case": case.case_id, "evaluation_mode": self._mode.value},
        )
        conversation: list[ConversationItem] = []
        if case.input.system_prompt is not None:
            conversation.append(Message(Role.SYSTEM, case.input.system_prompt))
        conversation.append(Message(Role.USER, case.input.user_message))

        try:
            run_result = runtime.run(conversation, context=context)
        except DQAgentError as exc:
            events = tuple(sink.events)
            metrics = self._metrics(events)
            state = self._terminal_state(events)
            check = EvaluationCheck("run.completed", False, f"run failed: {exc}")
            return EvaluationCaseResult(
                case_id=case.case_id,
                passed=False,
                run_id=context.run_id,
                run_state=state,
                output=None,
                metrics=metrics,
                checks=(check,),
                error=f"{type(exc).__name__}: {exc}",
            )

        checks = self._evaluate(
            case.expected,
            run_result.output.content,
            run_result.conversation,
            run_result.events,
        )
        if isinstance(client, _ScriptedLLM):
            checks = (
                *checks,
                EvaluationCheck(
                    "fixture.completions_consumed",
                    client.consumed_all,
                    (
                        "all scripted completions consumed"
                        if client.consumed_all
                        else "scripted completions remain after the run"
                    ),
                ),
            )
        return EvaluationCaseResult(
            case_id=case.case_id,
            passed=all(check.passed for check in checks),
            run_id=run_result.run_id,
            run_state=run_result.state,
            output=run_result.output.content,
            metrics=self._metrics(run_result.events),
            checks=checks,
        )

    def _client_for(self, case: EvaluationCase) -> LLMClient:
        if self._mode is EvaluationMode.LIVE:
            assert self._live_client is not None
            return self._live_client
        if case.fixture is None:
            raise EvaluationDefinitionError(
                f"case '{case.case_id}' has no deterministic fixture"
            )
        return _ScriptedLLM(case.fixture.completions)

    def _evaluate(
        self,
        expected: EvaluationExpectation,
        output: str,
        conversation: Sequence[ConversationItem],
        events: Sequence[RunEvent],
    ) -> tuple[EvaluationCheck, ...]:
        metrics = self._metrics(events)
        checks = [
            EvaluationCheck(
                "answer.non_empty",
                not expected.answer.non_empty or bool(output.strip()),
                "final answer is non-empty" if output.strip() else "final answer is empty",
            )
        ]
        if expected.answer.exact is not None:
            checks.append(
                EvaluationCheck(
                    "answer.exact",
                    output == expected.answer.exact,
                    f"expected {expected.answer.exact!r}, got {output!r}",
                )
            )
        if expected.answer.contains_all:
            missing = [
                value
                for value in expected.answer.contains_all
                if value.casefold() not in output.casefold()
            ]
            checks.append(
                EvaluationCheck(
                    "answer.contains_all",
                    not missing,
                    (
                        "all required fragments found"
                        if not missing
                        else f"missing fragments: {missing}"
                    ),
                )
            )
        checks.extend(self._check_tool_calls(expected.tool_calls, conversation))
        actual_types = tuple(event.type for event in events)
        checks.append(
            EvaluationCheck(
                "trace.required_event_types",
                self._is_subsequence(expected.trace.required_event_types, actual_types),
                (
                    "required event sequence observed"
                    if self._is_subsequence(expected.trace.required_event_types, actual_types)
                    else f"actual event sequence: {[item.value for item in actual_types]}"
                ),
            )
        )
        if expected.trace.max_model_attempts is not None:
            checks.append(
                EvaluationCheck(
                    "trace.max_model_attempts",
                    metrics.model_attempts <= expected.trace.max_model_attempts,
                    (
                        f"observed {metrics.model_attempts}, "
                        f"maximum {expected.trace.max_model_attempts}"
                    ),
                )
            )
        limits = expected.trace.resource_limits.get(self._mode)
        if limits is not None and limits.max_elapsed_seconds is not None:
            checks.append(
                EvaluationCheck(
                    "trace.max_elapsed_seconds",
                    metrics.elapsed_seconds <= limits.max_elapsed_seconds,
                    (
                        f"observed {metrics.elapsed_seconds:.6f}s, "
                        f"maximum {limits.max_elapsed_seconds:g}s"
                    ),
                )
            )
        if limits is not None and limits.max_total_tokens is not None:
            checks.append(
                EvaluationCheck(
                    "usage.max_total_tokens",
                    metrics.total_tokens is None
                    or metrics.total_tokens <= limits.max_total_tokens,
                    (
                        "provider did not report token usage"
                        if metrics.total_tokens is None
                        else (
                            f"observed {metrics.total_tokens}, "
                            f"maximum {limits.max_total_tokens}"
                        )
                    ),
                )
            )
        return tuple(checks)

    @staticmethod
    def _check_tool_calls(
        expected: Sequence[ToolCallExpectation],
        conversation: Sequence[ConversationItem],
    ) -> tuple[EvaluationCheck, ...]:
        actual = [item for item in conversation if isinstance(item, ToolCall)]
        results = {
            item.call_id: item for item in conversation if isinstance(item, ToolResult)
        }
        checks = [
            EvaluationCheck(
                "tools.count",
                len(actual) == len(expected),
                f"expected {len(expected)} tool calls, observed {len(actual)}",
            )
        ]
        for index, expectation in enumerate(expected):
            if index >= len(actual):
                break
            call = actual[index]
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                arguments = None
            result = results.get(call.call_id)
            passed = (
                call.name == expectation.name
                and arguments == dict(expectation.arguments)
                and (
                    expectation.outcome is None
                    or (result is not None and result.outcome is expectation.outcome)
                )
                and (
                    expectation.error_code is None
                    or (result is not None and result.error_code is expectation.error_code)
                )
            )
            checks.append(
                EvaluationCheck(
                    f"tools.call[{index}]",
                    passed,
                    (
                        f"expected {expectation.name} {dict(expectation.arguments)!r}, "
                        f"got {call.name} {arguments!r}"
                    ),
                )
            )
        return tuple(checks)

    @staticmethod
    def _is_subsequence(
        required: Sequence[RunEventType], actual: Sequence[RunEventType]
    ) -> bool:
        position = 0
        for event_type in actual:
            if position < len(required) and event_type is required[position]:
                position += 1
        return position == len(required)

    @staticmethod
    def _metrics(events: Sequence[RunEvent]) -> EvaluationMetrics:
        completed = [
            event for event in events if event.type is RunEventType.MODEL_REQUEST_COMPLETED
        ]
        input_tokens = EvaluationRunner._sum_usage(completed, "input_tokens")
        output_tokens = EvaluationRunner._sum_usage(completed, "output_tokens")
        total_tokens = EvaluationRunner._sum_usage(completed, "total_tokens")
        return EvaluationMetrics(
            elapsed_seconds=events[-1].elapsed_seconds if events else 0.0,
            model_attempts=sum(
                event.type is RunEventType.MODEL_REQUEST_STARTED for event in events
            ),
            tool_calls=sum(event.type is RunEventType.TOOL_CALL_STARTED for event in events),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _sum_usage(events: Sequence[RunEvent], name: str) -> int | None:
        if not events:
            return None
        values = [event.attributes.get(name) for event in events]
        integers = [
            value
            for value in values
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        return sum(integers) if len(integers) == len(values) else None

    @staticmethod
    def _terminal_state(events: Sequence[RunEvent]) -> RunState:
        return events[-1].state if events else RunState.FAILED


def load_evaluation_suite(path: Path) -> EvaluationSuite:
    """Load and validate a schema-versioned JSON evaluation suite."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDefinitionError(f"cannot load evaluation suite '{path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise EvaluationDefinitionError("evaluation suite must be a JSON object")
    data = cast(dict[str, Any], raw)
    version = data.get("schema_version")
    if version != EVALUATION_SCHEMA_VERSION:
        raise EvaluationDefinitionError(
            f"unsupported evaluation schema version: {version!r}"
        )
    try:
        Draft202012Validator(_EVALUATION_SUITE_SCHEMA_V1).validate(data)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "root"
        raise EvaluationDefinitionError(
            f"invalid evaluation suite at {location}: {exc.message}"
        ) from exc
    return _parse_suite(data)


def _parse_suite(data: dict[str, Any]) -> EvaluationSuite:
    cases = tuple(_parse_case(cast(dict[str, Any], item)) for item in data["cases"])
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise EvaluationDefinitionError("evaluation case IDs must be unique")
    return EvaluationSuite(
        suite_id=cast(str, data["suite_id"]),
        schema_version=cast(int, data["schema_version"]),
        cases=cases,
    )


def _parse_case(data: dict[str, Any]) -> EvaluationCase:
    raw_input = cast(dict[str, Any], data["input"])
    raw_expected = cast(dict[str, Any], data["expected"])
    raw_answer = cast(dict[str, Any], raw_expected["answer"])
    raw_trace = cast(dict[str, Any], raw_expected["trace"])
    fixture = None
    if "fixture" in data:
        raw_fixture = cast(dict[str, Any], data["fixture"])
        fixture = EvaluationFixture(
            tuple(
                _parse_completion(cast(dict[str, Any], item))
                for item in raw_fixture["completions"]
            )
        )
    return EvaluationCase(
        case_id=cast(str, data["id"]),
        modes=frozenset(EvaluationMode(item) for item in data["modes"]),
        input=EvaluationInput(
            user_message=cast(str, raw_input["user_message"]),
            system_prompt=cast(str | None, raw_input.get("system_prompt")),
        ),
        expected=EvaluationExpectation(
            answer=AnswerExpectation(
                non_empty=cast(bool, raw_answer.get("non_empty", True)),
                exact=cast(str | None, raw_answer.get("exact")),
                contains_all=tuple(cast(list[str], raw_answer.get("contains_all", []))),
            ),
            tool_calls=tuple(
                _parse_tool_expectation(cast(dict[str, Any], item))
                for item in raw_expected["tool_calls"]
            ),
            trace=TraceExpectation(
                required_event_types=tuple(
                    RunEventType(item) for item in raw_trace["required_event_types"]
                ),
                max_model_attempts=cast(int | None, raw_trace.get("max_model_attempts")),
                resource_limits={
                    EvaluationMode(mode): ResourceLimits(
                        max_elapsed_seconds=cast(
                            float | None,
                            limits.get("max_elapsed_seconds"),
                        ),
                        max_total_tokens=cast(
                            int | None,
                            limits.get("max_total_tokens"),
                        ),
                    )
                    for mode, limits in cast(
                        dict[str, dict[str, Any]],
                        raw_trace.get("resource_limits", {}),
                    ).items()
                },
            ),
        ),
        fixture=fixture,
    )


def _parse_tool_expectation(data: dict[str, Any]) -> ToolCallExpectation:
    outcome = data.get("outcome")
    error_code = data.get("error_code")
    return ToolCallExpectation(
        name=cast(str, data["name"]),
        arguments=cast(dict[str, object], data["arguments"]),
        outcome=ToolOutcome(outcome) if outcome is not None else None,
        error_code=ToolErrorCode(error_code) if error_code is not None else None,
    )


def _parse_completion(data: dict[str, Any]) -> Completion:
    usage = None
    if "usage" in data:
        raw_usage = cast(dict[str, int], data["usage"])
        usage = TokenUsage(
            input_tokens=raw_usage["input_tokens"],
            output_tokens=raw_usage["output_tokens"],
            total_tokens=raw_usage["total_tokens"],
        )
    return Completion(
        content=cast(str | None, data.get("content")),
        tool_calls=tuple(
            ToolCall(
                call_id=cast(str, item["call_id"]),
                name=cast(str, item["name"]),
                arguments=json.dumps(item["arguments"], separators=(",", ":")),
            )
            for item in cast(list[dict[str, Any]], data.get("tool_calls", []))
        ),
        response_id=cast(str | None, data.get("response_id")),
        model=cast(str | None, data.get("model")),
        usage=usage,
    )


_TOOL_CALL_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["call_id", "name", "arguments"],
    "properties": {
        "call_id": {"type": "string", "minLength": 1},
        "name": {"type": "string", "minLength": 1},
        "arguments": {"type": "object"},
    },
}

_COMPLETION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "content": {"type": "string", "minLength": 1},
        "tool_calls": {"type": "array", "items": _TOOL_CALL_SCHEMA, "minItems": 1},
        "response_id": {"type": "string"},
        "model": {"type": "string"},
        "usage": {
            "type": "object",
            "additionalProperties": False,
            "required": ["input_tokens", "output_tokens", "total_tokens"],
            "properties": {
                "input_tokens": {"type": "integer", "minimum": 0},
                "output_tokens": {"type": "integer", "minimum": 0},
                "total_tokens": {"type": "integer", "minimum": 0},
            },
        },
    },
    "anyOf": [{"required": ["content"]}, {"required": ["tool_calls"]}],
}

_EVALUATION_SUITE_SCHEMA_V1: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "suite_id", "cases"],
    "properties": {
        "schema_version": {"const": EVALUATION_SCHEMA_VERSION},
        "suite_id": {"type": "string", "minLength": 1},
        "cases": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "modes", "input", "expected"],
                "properties": {
                    "id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$"},
                    "modes": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"enum": [mode.value for mode in EvaluationMode]},
                    },
                    "input": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["user_message"],
                        "properties": {
                            "user_message": {"type": "string", "minLength": 1},
                            "system_prompt": {"type": "string", "minLength": 1},
                        },
                    },
                    "expected": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["answer", "tool_calls", "trace"],
                        "properties": {
                            "answer": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "non_empty": {"type": "boolean"},
                                    "exact": {"type": "string", "minLength": 1},
                                    "contains_all": {
                                        "type": "array",
                                        "items": {"type": "string", "minLength": 1},
                                        "uniqueItems": True,
                                    },
                                },
                            },
                            "tool_calls": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["name", "arguments"],
                                    "properties": {
                                        "name": {"type": "string", "minLength": 1},
                                        "arguments": {"type": "object"},
                                        "outcome": {"enum": [item.value for item in ToolOutcome]},
                                        "error_code": {
                                            "enum": [item.value for item in ToolErrorCode]
                                        },
                                    },
                                },
                            },
                            "trace": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["required_event_types"],
                                "properties": {
                                    "required_event_types": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {"enum": [item.value for item in RunEventType]},
                                    },
                                    "max_model_attempts": {"type": "integer", "minimum": 1},
                                    "resource_limits": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            mode.value: {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "properties": {
                                                    "max_elapsed_seconds": {
                                                        "type": "number",
                                                        "exclusiveMinimum": 0,
                                                    },
                                                    "max_total_tokens": {
                                                        "type": "integer",
                                                        "minimum": 0,
                                                    },
                                                },
                                                "minProperties": 1,
                                            }
                                            for mode in EvaluationMode
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "fixture": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["completions"],
                        "properties": {
                            "completions": {
                                "type": "array",
                                "minItems": 1,
                                "items": _COMPLETION_SCHEMA,
                            }
                        },
                    },
                },
                "allOf": [
                    {
                        "if": {
                            "properties": {
                                "modes": {"contains": {"const": EvaluationMode.DETERMINISTIC.value}}
                            }
                        },
                        "then": {"required": ["fixture"]},
                    }
                ],
            },
        },
    },
}
