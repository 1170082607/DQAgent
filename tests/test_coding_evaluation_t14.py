import json
from dataclasses import replace
from pathlib import Path

import pytest

from dqagent.coding_evaluation import (
    CodingEvaluationComposition,
    CodingEvaluationDefinitionError,
    CodingEvaluationRunner,
    CodingEvaluationSuite,
    CodingRepositoryFixture,
    compute_coding_fixture_digest,
    load_coding_evaluation_suite,
)
from dqagent.coding_evaluation_cli import DEFAULT_SUITE, build_parser
from dqagent.models import Completion, ToolErrorCode

SUITE_PATH = Path("evaluations/cases/phase-9-coding-baseline-v1.json")
BASELINE_PATH = Path("evaluations/baselines/phase-9-coding-deterministic-v1.json")


def test_t14_representative_suite_passes_through_production_path() -> None:
    suite = load_coding_evaluation_suite(SUITE_PATH)
    report = CodingEvaluationRunner().run(suite)

    assert len(suite.cases) == 10
    assert report.passed is True
    assert all(result.evaluation_passed and result.cleanup_passed for result in report.results)
    assert all(result.context for result in report.results)
    assert {
        result.verdict for result in report.results
    } == {"passed", "failed", "indeterminate", "not_validated"}


def test_t14_committed_baseline_uses_structural_fingerprint_not_timing() -> None:
    suite = load_coding_evaluation_suite(SUITE_PATH)
    report = CodingEvaluationRunner().run(suite)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert baseline["suite_id"] == suite.suite_id
    assert baseline["suite_schema_version"] == suite.schema_version
    assert baseline["summary"] == {
        "passed": True,
        "passed_cases": 10,
        "failed_cases": 0,
        "executed_cases": 10,
        "skipped_cases": 0,
    }
    assert baseline["deterministic_fingerprint"] == report.deterministic_fingerprint
    assert [
        (item["case_id"], item["fixture_digest"]) for item in baseline["results"]
    ] == [
        (result.case_id, result.fixture_digest) for result in report.results
    ]

    baseline_rendered = json.dumps(baseline, ensure_ascii=True)
    assert '"elapsed_seconds":' not in baseline_rendered


def test_coding_evaluation_cli_defaults_to_t14_suite() -> None:
    assert DEFAULT_SUITE == SUITE_PATH
    assert build_parser().parse_args([]).suite == SUITE_PATH


def refresh_t14_fixture_digest(case):
    from dqagent import coding_evaluation as coding_evaluation_module

    return replace(case, fixture_digest=coding_evaluation_module._case_fixture_digest(case))


def test_t14_traversal_tool_result_trajectory_is_directly_asserted() -> None:
    suite = load_coding_evaluation_suite(SUITE_PATH)
    case = next(item for item in suite.cases if item.case_id == "traversal-protected-secret-denial")

    assert [
        (item.call_id, item.name, item.outcome.value, item.error_code.value)
        for item in case.expected.tool_calls
    ] == [
        ("traversal-1", "workspace_read", "error", "governance_failure"),
        ("protected-1", "workspace_read", "error", "protected_resource_denied"),
        ("secret-1", "workspace_read", "error", "protected_resource_denied"),
    ]

    result = CodingEvaluationRunner().run(
        CodingEvaluationSuite(suite.suite_id, suite.schema_version, (case,))
    ).results[0]

    assert result.passed is True
    assert [
        (item["call_id"], item["name"], item["outcome"], item["error_code"])
        for item in result.to_dict()["tool_calls"]
    ] == [
        ("traversal-1", "workspace_read", "error", "governance_failure"),
        ("protected-1", "workspace_read", "error", "protected_resource_denied"),
        ("secret-1", "workspace_read", "error", "protected_resource_denied"),
    ]


def test_t14_traversal_tool_result_mutations_fail_evaluation() -> None:
    suite = load_coding_evaluation_suite(SUITE_PATH)
    case = next(item for item in suite.cases if item.case_id == "traversal-protected-secret-denial")
    first_completion = case.fixture.model_completions[0]

    unknown_tool_completion = replace(
        first_completion,
        tool_calls=(replace(first_completion.tool_calls[0], name="unknown_tool"),),
    )
    unknown_tool_case = refresh_t14_fixture_digest(
        replace(
            case,
            fixture=replace(
                case.fixture,
                model_completions=(unknown_tool_completion, *case.fixture.model_completions[1:]),
            ),
        )
    )

    deleted_call_case = refresh_t14_fixture_digest(
        replace(
            case,
            fixture=replace(
                case.fixture,
                model_completions=(
                    Completion(content="the first call was deleted", model="t14-scripted"),
                    *case.fixture.model_completions[1:],
                ),
            ),
        )
    )

    wrong_error_case = refresh_t14_fixture_digest(
        replace(
            case,
            expected=replace(
                case.expected,
                tool_calls=(
                    replace(
                        case.expected.tool_calls[0],
                        error_code=ToolErrorCode.PROTECTED_RESOURCE_DENIED,
                    ),
                    *case.expected.tool_calls[1:],
                ),
            ),
        )
    )

    for mutated in (unknown_tool_case, deleted_call_case, wrong_error_case):
        result = CodingEvaluationRunner().run(
            CodingEvaluationSuite(suite.suite_id, suite.schema_version, (mutated,))
        ).results[0]
        assert result.passed is False
        assert result.evaluation_passed is False


def test_t14_case_digest_binds_tool_call_expectations(tmp_path: Path) -> None:
    raw = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    raw["cases"][2]["expected"]["tool_calls"][0]["error_code"] = "protected_resource_denied"
    tampered = tmp_path / "tampered-t14-tool-expectation.json"
    tampered.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(CodingEvaluationDefinitionError, match="fixture digest mismatch"):
        load_coding_evaluation_suite(tampered)


def test_t14_raw_digest_rejects_non_json_iterables_before_consumption() -> None:
    class Sentinel:
        def __init__(self) -> None:
            self.calls = 0

        def __iter__(self):
            return self

        def __next__(self) -> str:
            self.calls += 1
            raise RuntimeError("sentinel: raw digest consumed an iterable")

    raw = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    sentinel = Sentinel()
    raw["cases"][0]["composition"]["secret_values"] = sentinel

    with pytest.raises(TypeError, match="JSON-native containers"):
        compute_coding_fixture_digest(raw["cases"][0])

    assert sentinel.calls == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            ("composition", "secret_values"),
            [f"secret-{index}" for index in range(65)],
            "secret values",
        ),
        (
            ("request", "targets"),
            [f"target-{index}.txt" for index in range(129)],
            "request targets",
        ),
    ),
)
def test_t14_raw_digest_rejects_oversized_arrays_before_hashing(
    field: tuple[str, str],
    value: list[str],
    message: str,
) -> None:
    raw = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    raw["cases"][0][field[0]][field[1]] = value

    with pytest.raises(ValueError, match=message):
        compute_coding_fixture_digest(raw["cases"][0])


@pytest.mark.parametrize(
    ("files", "skill_roots"),
    (
        (
            {"AGENTS.md": "x\n", "Target.txt": "one\n", "target.txt": "two\n"},
            {},
        ),
        (
            {"AGENTS.md": "x\n", "node": "file\n", "node/child.txt": "child\n"},
            {},
        ),
        (
            {"AGENTS.md": "x\n", "skills": "file\n"},
            {"python": "skills/python"},
        ),
        (
            {"AGENTS.md": "x\n", "skills/python": "file\n"},
            {"python": "skills/python"},
        ),
    ),
)
def test_t14_fixture_rejects_portable_identity_collisions(
    files: dict[str, str],
    skill_roots: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="portable fixture path collision"):
        CodingRepositoryFixture(files=files, skill_roots=skill_roots)


def test_t14_fixture_allows_files_inside_a_skill_root() -> None:
    fixture = CodingRepositoryFixture(
        files={"AGENTS.md": "x\n", "skills/python/SKILL.md": "body\n"},
        skill_roots={"python": "skills/python"},
    )

    assert fixture.skill_roots == {"python": "skills/python"}


def test_t14_fixture_iterables_fail_closed_at_bound_plus_one() -> None:
    class InfiniteValues:
        def __init__(self, prefix: str, sentinel_after: int) -> None:
            self.calls = 0
            self._prefix = prefix
            self._sentinel_after = sentinel_after

        def __iter__(self):
            return self

        def __next__(self) -> str:
            self.calls += 1
            if self.calls > self._sentinel_after:
                raise RuntimeError("sentinel: iterator consumed beyond the bound")
            return f"{self._prefix}-{self.calls}"

    secret_values = InfiniteValues("secret", 65)
    with pytest.raises(ValueError, match="secret values.*bound"):
        CodingEvaluationComposition(secret_values=secret_values)
    assert secret_values.calls == 65

    protected_paths = InfiniteValues("protected", 33)
    with pytest.raises(ValueError, match="fixture protected_paths.*bound"):
        CodingRepositoryFixture(files={"AGENTS.md": "x\n"}, protected_paths=protected_paths)
    assert protected_paths.calls == 33


def test_t14_fixture_iterables_enforce_exact_bounds_and_character_budget() -> None:
    composition = CodingEvaluationComposition(
        secret_names=(f"name-{index}" for index in range(64)),
        secret_values=(f"value-{index}" for index in range(64)),
    )
    assert len(composition.secret_names) == 64
    assert len(composition.secret_values) == 64

    with pytest.raises(ValueError, match="secret values character budget"):
        CodingEvaluationComposition(secret_values=("x" * 512 for _ in range(33)))


def test_t14_documentation_states_lifecycle_and_isolation_ceiling() -> None:
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    evaluations_readme = (root / "evaluations" / "README.md").read_text(encoding="utf-8")
    architecture = (root / "docs" / "architecture.md").read_text(encoding="utf-8")
    detailed_design = (root / "docs" / "phase-9-detailed-design.md").read_text(encoding="utf-8")
    roadmap = (root / "docs" / "roadmap.md").read_text(encoding="utf-8")

    assert "T14 remains pending reviewer remediation and fresh closure" in readme
    assert "T14 implemented, pending review" in roadmap
    assert (
        "Status: Implemented in current worktree; T14 pending reviewer remediation "
        "and fresh closure"
        in detailed_design
    )
    assert "Roadmap status: Phase 9 remains `In progress`" in detailed_design

    for document in (readme, evaluations_readme, architecture, detailed_design):
        normalized_document = " ".join(document.split())
        assert "Cases share no workspace, process, approval, cache" not in document
        assert "no workspace, process, approval, cache" not in document
        assert "same evaluator host process" in normalized_document
        assert "Phase 13" in document
