from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

import dqagent.coding_tools as coding_tools_module
from dqagent.coding_tools import (
    WORKSPACE_READ_SCHEMA,
    WORKSPACE_SEARCH_SCHEMA,
    CodingToolLimits,
    create_workspace_read_tool,
    create_workspace_search_tool,
)
from dqagent.errors import RunCancelledError
from dqagent.events import RunEventType
from dqagent.execution import RunContext
from dqagent.models import ToolCall, ToolErrorCode, ToolOutcome
from dqagent.tools import (
    ToolExecution,
    ToolExecutionContext,
    ToolRegistry,
    _RunActionRecordCollector,
)
from dqagent.workspace import (
    Workspace,
    WorkspaceDriftError,
    WorkspaceReason,
    WorkspaceScope,
)


def make_workspace(tmp_path: Path, **kwargs: object) -> Workspace:
    return Workspace(WorkspaceScope("fixture", tmp_path, **kwargs))


def make_symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation requires unavailable privileges")
        raise


def invoke(
    tool: Any,
    arguments: dict[str, object],
    *,
    run_id: str = "run-t6",
    events: list[tuple[str, object]] | None = None,
) -> tuple[ToolExecution, tuple[object, ...]]:
    context = RunContext(run_id=run_id)
    collector = _RunActionRecordCollector(run_id, 1)
    execution_context = ToolExecutionContext(
        context,
        emit_stage=(
            (lambda stage, attributes: events.append((stage, attributes)))
            if events is not None
            else None
        ),
        record_collector=collector,
    )
    execution = ToolRegistry((tool,)).execute_detailed(
        ToolCall(
            "call-t6",
            tool.definition.name,
            json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        ),
        context,
        execution_context=execution_context,
    )
    return execution, collector.records


def read_header(output: str) -> dict[str, object]:
    return json.loads(output.splitlines()[0])


def read_body(output: str) -> list[str]:
    return output.splitlines()[1:]


def test_t6_schemas_are_closed_and_bounded() -> None:
    for schema, required in (
        (WORKSPACE_READ_SCHEMA, {"path"}),
        (WORKSPACE_SEARCH_SCHEMA, {"query"}),
    ):
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == required
        assert schema["properties"]


def test_unknown_arguments_are_rejected_before_governance(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    tool = create_workspace_read_tool(workspace)

    execution, records = invoke(tool, {"path": "file.txt", "unknown": True})

    assert execution.result.outcome is ToolOutcome.ERROR
    assert execution.result.error_code is ToolErrorCode.INVALID_ARGUMENTS
    assert records == ()


def test_read_supports_bom_one_based_numbered_unicode_lines(tmp_path: Path) -> None:
    path = tmp_path / "unicode.txt"
    path.write_bytes("alpha\n汉🙂\nlast".encode("utf-8-sig"))
    workspace = make_workspace(tmp_path)
    tool = create_workspace_read_tool(workspace)

    execution, records = invoke(tool, {"path": "unicode.txt", "start_line": 2, "line_count": 1})

    assert execution.result.outcome is ToolOutcome.SUCCESS
    header = read_header(execution.result.output)
    assert header["status"] == "ok"
    assert header["start_line"] == 2
    assert header["returned_lines"] == 1
    assert read_body(execution.result.output) == ["2: 汉🙂"]
    assert header["source_bytes"] == path.stat().st_size
    assert (
        len(execution.result.output)
        <= tool.guard_context.configured_limits.max_output_characters
    )
    assert records and records[0].executor_attempts == 1


@pytest.mark.parametrize(
    ("filename", "content", "arguments", "status"),
    [
        ("empty.txt", b"", {"path": "empty.txt"}, "empty"),
        ("eof.txt", b"one\n", {"path": "eof.txt", "line_count": 2}, "eof"),
        ("binary.bin", b"\x00plain", {"path": "binary.bin"}, "binary"),
        ("invalid.txt", b"\xff\xfe", {"path": "invalid.txt"}, "invalid_text"),
    ],
)
def test_read_distinguishes_empty_eof_binary_and_invalid_text(
    tmp_path: Path,
    filename: str,
    content: bytes,
    arguments: dict[str, object],
    status: str,
) -> None:
    (tmp_path / filename).write_bytes(content)
    tool = create_workspace_read_tool(make_workspace(tmp_path))

    execution, _records = invoke(tool, arguments)

    assert execution.result.outcome is ToolOutcome.SUCCESS
    assert read_header(execution.result.output)["status"] == status
    if status == "eof":
        assert read_header(execution.result.output)["returned_lines"] == 1
        assert read_body(execution.result.output) == ["1: one"]
    else:
        assert read_body(execution.result.output) == []


def test_read_reports_missing_without_exposing_denied_content(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    tool = create_workspace_read_tool(workspace)

    execution, records = invoke(tool, {"path": "missing.txt"})

    assert execution.result.outcome is ToolOutcome.SUCCESS
    assert read_header(execution.result.output)["status"] == "missing"
    assert records and records[0].executor_attempts == 1

    nested, nested_records = invoke(
        create_workspace_read_tool(make_workspace(tmp_path)),
        {"path": "missing/child.txt"},
        run_id="run-t6-nested-missing",
    )
    assert nested.result.outcome is ToolOutcome.SUCCESS
    assert read_header(nested.result.output)["status"] == "missing"
    assert nested_records and nested_records[0].executor_attempts == 1


def test_read_source_and_character_limits_are_distinct(tmp_path: Path) -> None:
    (tmp_path / "bytes.txt").write_bytes("éé\n".encode())
    source_limited = CodingToolLimits(
        max_read_source_bytes=3,
        max_read_line_characters=32,
        max_read_output_characters=1_000,
    )
    execution, _records = invoke(
        create_workspace_read_tool(make_workspace(tmp_path), limits=source_limited),
        {"path": "bytes.txt"},
    )
    header = read_header(execution.result.output)
    assert header["status"] == "source_limit"
    assert header["source_bytes"] == 3
    assert header["source_limit"] is True
    assert read_body(execution.result.output) == ["1: é"]

    (tmp_path / "long.txt").write_text("abcdefgh\n", encoding="utf-8")
    line_limited = CodingToolLimits(
        max_read_line_characters=4,
        max_read_output_characters=1_000,
    )
    execution, _records = invoke(
        create_workspace_read_tool(make_workspace(tmp_path), limits=line_limited),
        {"path": "long.txt"},
    )
    header = read_header(execution.result.output)
    assert header["status"] == "line_limit"
    assert header["line_limit"] is True
    assert len(read_body(execution.result.output)[0].split(": ", 1)[1]) <= 4
    assert "abcdefgh" not in execution.result.output


def test_read_output_is_bounded_and_marks_output_limit(tmp_path: Path) -> None:
    (tmp_path / "many.txt").write_text("line-1\nline-2\nline-3\n", encoding="utf-8")
    limits = CodingToolLimits(max_read_output_characters=220)
    tool = create_workspace_read_tool(make_workspace(tmp_path), limits=limits)

    execution, _records = invoke(tool, {"path": "many.txt", "line_count": 3})

    assert len(execution.result.output) <= limits.max_read_output_characters
    assert read_header(execution.result.output)["status"] == "output_limit"
    assert read_header(execution.result.output)["output_limit"] is True


def test_final_output_sanitization_updates_header_after_redaction_expands_output(
    tmp_path: Path,
) -> None:
    line = "hit " + ("QQ " * 48)
    content = "\n".join(line for _ in range(12))
    (tmp_path / "search.txt").write_text(content, encoding="utf-8")
    (tmp_path / "read.txt").write_text(content, encoding="utf-8")

    search_limits = CodingToolLimits(
        max_search_matches=20,
        max_search_output_characters=2_200,
    )
    search_execution, _records = invoke(
        create_workspace_search_tool(
            make_workspace(tmp_path),
            limits=search_limits,
            secret_values=("QQ",),
        ),
        {"query": "hit", "path": "search.txt"},
    )
    search_header = read_header(search_execution.result.output)
    assert len(search_execution.result.output) <= search_limits.max_search_output_characters
    assert search_header["status"] == "output_limit"
    assert search_header["output_limit"] is True
    assert "QQ" not in search_execution.result.output

    read_limits = CodingToolLimits(max_read_output_characters=2_200)
    read_execution, _records = invoke(
        create_workspace_read_tool(
            make_workspace(tmp_path),
            limits=read_limits,
            secret_values=("QQ",),
        ),
        {"path": "read.txt", "line_count": 12},
        run_id="run-t6-redaction-read",
    )
    read_header_value = read_header(read_execution.result.output)
    assert len(read_execution.result.output) <= read_limits.max_read_output_characters
    assert read_header_value["status"] == "output_limit"
    assert read_header_value["output_limit"] is True
    assert "QQ" not in read_execution.result.output


def test_structured_output_sanitization_preserves_metadata_and_redacts_values(
    tmp_path: Path,
) -> None:
    (tmp_path / "ok.txt").write_text("hit DATA_SECRET", encoding="utf-8")
    (tmp_path / "match.txt").write_text("hit DATA_SECRET hit", encoding="utf-8")
    (tmp_path / "source.txt").write_text("hit DATA_SECRET", encoding="utf-8")
    (tmp_path / "output.txt").write_text(
        "hit DATA_SECRET " + ("x" * 1_500), encoding="utf-8"
    )
    visited = tmp_path / "visited"
    visited.mkdir()
    (visited / "one.txt").write_text("absent DATA_SECRET", encoding="utf-8")
    (visited / "two.txt").write_text("absent DATA_SECRET", encoding="utf-8")
    (tmp_path / "read.txt").write_text("body DATA_SECRET", encoding="utf-8")

    def assert_search_header(output: str, expected_status: str) -> None:
        header = read_header(output)
        assert header["status"] == expected_status
        assert {
            "elapsed_limit",
            "line_projection_limit",
            "match_limit",
            "output_limit",
            "source_limit",
            "status",
            "visited_files_limit",
        }.issubset(header)
        for name, value in header.items():
            if name != "status" and isinstance(value, str):
                assert "DATA_SECRET" not in value
        assert all("DATA_SECRET" not in line for line in read_body(output))

    workspace = make_workspace(tmp_path)
    secret_values = ("DATA_SECRET",)
    cases = (
        ("ok.txt", CodingToolLimits(max_search_output_characters=2_000), {}, "ok"),
        (
            "match.txt",
            CodingToolLimits(max_search_output_characters=2_000),
            {"max_matches": 1},
            "match_limit",
        ),
        (
            "source.txt",
            CodingToolLimits(max_search_source_bytes=1, max_search_output_characters=2_000),
            {},
            "source_limit",
        ),
        (
            "output.txt",
            CodingToolLimits(max_search_output_characters=1_000),
            {},
            "output_limit",
        ),
    )
    for index, (path, limits, extra, status) in enumerate(cases):
        arguments = {"query": "hit", "path": path, **extra}
        execution, _records = invoke(
            create_workspace_search_tool(
                workspace,
                limits=limits,
                secret_values=secret_values,
            ),
            arguments,
            run_id=f"run-t6-schema-{index}",
        )
        assert_search_header(execution.result.output, status)

    visited_execution, _records = invoke(
        create_workspace_search_tool(
            workspace,
            limits=CodingToolLimits(
                max_search_visited_files=1,
                max_search_output_characters=2_000,
            ),
            secret_values=secret_values,
        ),
        {"query": "absent", "path": "visited"},
        run_id="run-t6-schema-visited",
    )
    assert_search_header(visited_execution.result.output, "visited_files_limit")

    read_execution, _records = invoke(
        create_workspace_read_tool(
            workspace,
            secret_values=secret_values,
        ),
        {"path": "read.txt", "line_count": 1},
        run_id="run-t6-schema-read",
    )
    read_header_value = read_header(read_execution.result.output)
    assert read_header_value["status"] == "ok"
    assert {
        "elapsed_limit",
        "line_limit",
        "output_limit",
        "source_limit",
        "status",
    }.issubset(read_header_value)
    assert "DATA_SECRET" not in str(read_header_value["path"])
    assert all(
        "DATA_SECRET" not in line for line in read_body(read_execution.result.output)
    )


@pytest.mark.parametrize(
    "secret",
    ("ok", "eof", "workspace_search", "status", "false", "limit"),
)
@pytest.mark.parametrize(
    "factory",
    (create_workspace_read_tool, create_workspace_search_tool),
    ids=("read", "search"),
)
def test_metadata_secret_collisions_are_rejected_before_tool_exposure(
    tmp_path: Path,
    secret: str,
    factory: Any,
) -> None:
    workspace = make_workspace(tmp_path)

    with pytest.raises(ValueError, match="structured output metadata") as error:
        factory(workspace, secret_values=(secret,))

    message = str(error.value)
    assert secret not in message
    assert len(message) <= 160


@pytest.mark.parametrize("secret", ("1", "0", "12"))
@pytest.mark.parametrize(
    "factory",
    (create_workspace_read_tool, create_workspace_search_tool),
    ids=("read", "search"),
)
def test_numeric_metadata_secret_collisions_are_rejected_before_tool_exposure(
    tmp_path: Path,
    secret: str,
    factory: Any,
) -> None:
    workspace = make_workspace(tmp_path)

    with pytest.raises(ValueError, match="structured output metadata") as error:
        factory(workspace, secret_values=(secret,))

    message = str(error.value)
    assert secret not in message
    assert len(message) <= 160


@pytest.mark.parametrize("secret", ("{", "}", '"', ":", ","))
@pytest.mark.parametrize(
    "factory",
    (create_workspace_read_tool, create_workspace_search_tool),
    ids=("read", "search"),
)
def test_json_syntax_secret_collisions_are_rejected_before_tool_exposure(
    tmp_path: Path,
    secret: str,
    factory: Any,
) -> None:
    workspace = make_workspace(tmp_path)

    with pytest.raises(ValueError, match="structured output metadata") as error:
        factory(workspace, secret_values=(secret,))

    message = str(error.value)
    assert secret not in message
    assert len(message) <= 160


@pytest.mark.parametrize(
    ("factory", "limit_kwargs"),
    (
        (create_workspace_read_tool, {"max_read_output_characters": 1}),
        (create_workspace_search_tool, {"max_search_output_characters": 1}),
    ),
    ids=("read", "search"),
)
def test_tiny_output_ceilings_are_rejected_before_tool_exposure(
    tmp_path: Path,
    factory: Any,
    limit_kwargs: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="compact structured output header") as error:
        factory(make_workspace(tmp_path), limits=CodingToolLimits(**limit_kwargs))

    assert len(str(error.value)) <= 160


@pytest.mark.parametrize(
    ("factory", "limit_kwargs", "arguments", "maximum"),
    (
        (
            create_workspace_read_tool,
            {"max_read_output_characters": 69},
            {"path": "file.txt"},
            69,
        ),
        (
            create_workspace_search_tool,
            {"max_search_output_characters": 71},
            {"query": "hit", "path": "file.txt"},
            71,
        ),
    ),
    ids=("read", "search"),
)
def test_compact_output_header_boundary_is_parseable_and_bounded(
    tmp_path: Path,
    factory: Any,
    limit_kwargs: dict[str, int],
    arguments: dict[str, object],
    maximum: int,
) -> None:
    (tmp_path / "file.txt").write_text("hit\n", encoding="utf-8")
    execution, _records = invoke(
        factory(
            make_workspace(tmp_path),
            limits=CodingToolLimits(**limit_kwargs),
        ),
        arguments,
    )

    assert len(execution.result.output) <= maximum
    header = read_header(execution.result.output)
    assert header["status"] == "output_limit"
    assert header["output_limit"] is True


@pytest.mark.parametrize(
    ("query", "content"),
    [("\u00df", "SS"), ("ss", "\u00df")],
)
def test_search_casefold_expansion_preserves_source_columns(
    tmp_path: Path,
    query: str,
    content: str,
) -> None:
    (tmp_path / "unicode.txt").write_text(content + "\n", encoding="utf-8")

    insensitive, _records = invoke(
        create_workspace_search_tool(make_workspace(tmp_path)),
        {"query": query, "path": "unicode.txt", "case_sensitive": False},
        run_id=f"run-t6-casefold-{query}",
    )
    assert read_header(insensitive.result.output)["matches"] == 1
    assert read_body(insensitive.result.output) == [f"unicode.txt:1:1: {content}"]

    sensitive, _records = invoke(
        create_workspace_search_tool(make_workspace(tmp_path)),
        {"query": query, "path": "unicode.txt", "case_sensitive": True},
        run_id=f"run-t6-case-sensitive-{query}",
    )
    assert read_header(sensitive.result.output)["status"] == "no_matches"
    assert read_header(sensitive.result.output)["matches"] == 0


def test_search_renderer_does_not_materialize_full_body_before_output_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "many.txt").write_text("\n".join("hit" for _ in range(100)), encoding="utf-8")
    original_renderer = coding_tools_module._render_bounded_output
    observed: list[bool] = []

    def observe_renderer(header, lines, maximum):
        observed.append(isinstance(lines, list))
        return original_renderer(header, lines, maximum)

    monkeypatch.setattr(coding_tools_module, "_render_bounded_output", observe_renderer)
    limits = CodingToolLimits(
        max_search_matches=100,
        max_search_output_characters=100,
    )
    execution, _records = invoke(
        create_workspace_search_tool(make_workspace(tmp_path), limits=limits),
        {"query": "hit", "path": "many.txt"},
    )

    assert observed
    assert all(not was_list for was_list in observed)
    assert len(execution.result.output) <= limits.max_search_output_characters
    assert read_header(execution.result.output)["status"] == "output_limit"
    assert read_header(execution.result.output)["output_limit"] is True


def test_search_is_literal_case_aware_and_deterministically_ordered(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_text("needle\nneedle", encoding="utf-8")
    (tmp_path / "a.txt").write_text("Needle needle\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "m.TxT").write_text("NEEDLE", encoding="utf-8")
    tool = create_workspace_search_tool(make_workspace(tmp_path))

    execution, _records = invoke(
        tool,
        {
            "query": "needle",
            "glob": "*.txt",
            "case_sensitive": False,
            "max_matches": 10,
        },
    )

    assert execution.result.outcome is ToolOutcome.SUCCESS
    assert read_header(execution.result.output)["status"] == "ok"
    assert read_body(execution.result.output) == [
        "a.txt:1:1: Needle needle",
        "a.txt:1:8: Needle needle",
        "sub/m.TxT:1:1: NEEDLE",
        "z.txt:1:1: needle",
        "z.txt:2:1: needle",
    ]


def test_search_uses_literal_not_regex_and_zero_match_is_success(tmp_path: Path) -> None:
    (tmp_path / "literal.txt").write_text("a.b\naxb\n", encoding="utf-8")
    tool = create_workspace_search_tool(make_workspace(tmp_path))

    literal, _records = invoke(tool, {"query": "."})
    no_match, _records = invoke(
        create_workspace_search_tool(make_workspace(tmp_path)),
        {"query": "absent"},
        run_id="run-t6-no-match",
    )

    assert read_body(literal.result.output) == ["literal.txt:1:2: a.b"]
    assert literal.result.outcome is ToolOutcome.SUCCESS
    assert no_match.result.outcome is ToolOutcome.SUCCESS
    assert read_header(no_match.result.output)["status"] == "no_matches"
    assert read_header(no_match.result.output)["matches"] == 0


def test_search_supports_contained_file_target_and_relative_glob(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("needle", encoding="utf-8")
    (tmp_path / "two.txt").write_text("needle", encoding="utf-8")
    tool = create_workspace_search_tool(make_workspace(tmp_path))

    execution, _records = invoke(
        tool,
        {"query": "needle", "path": "one.py", "glob": "*.py"},
    )

    assert read_body(execution.result.output) == ["one.py:1:1: needle"]
    assert read_header(execution.result.output)["status"] == "ok"


def test_search_reports_each_safe_omission_without_following_subtrees(tmp_path: Path) -> None:
    (tmp_path / "public.txt").write_text("needle public", encoding="utf-8")
    (tmp_path / "large.txt").write_text("needle " + ("x" * 100), encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\x00needle")
    (tmp_path / ".env").write_text("needle SECRET-CONTENT", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("needle PROTECTED-CONTENT", encoding="utf-8")
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "token.txt").write_text("needle PRIVATE-CONTENT", encoding="utf-8")
    target = tmp_path.parent / f"{tmp_path.name}-linked-target"
    target.mkdir()
    (target / "hidden.txt").write_text("needle LINK-CONTENT", encoding="utf-8")
    make_symlink(tmp_path / "link", target, directory=True)
    workspace = make_workspace(tmp_path, secret_paths=(PurePosixPath("private"),))
    limits = CodingToolLimits(max_search_file_bytes=32, max_search_output_characters=2_000)

    execution, _records = invoke(
        create_workspace_search_tool(workspace, limits=limits),
        {"query": "needle"},
    )

    header = read_header(execution.result.output)
    rendered = execution.result.output
    assert header["status"] == "ok"
    assert header["omissions"]["protected"] >= 1
    assert header["omissions"]["secret"] >= 1
    assert header["omissions"]["link"] >= 1
    assert header["omissions"]["binary"] >= 1
    assert header["omissions"]["oversized"] >= 1
    assert "public.txt:1:1: needle public" in rendered
    assert all(value not in rendered for value in ("SECRET-CONTENT", "PROTECTED-CONTENT"))
    assert str(tmp_path) not in rendered
    assert "hidden.txt" not in rendered


def test_search_drops_matches_when_file_drift_is_detected(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "drift.txt").write_text("needle", encoding="utf-8")
    workspace = make_workspace(tmp_path)
    original = workspace.revalidate
    calls = 0

    def drift_after_read(resolved):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise WorkspaceDriftError(
                reason_code=WorkspaceReason.DRIFT,
                purpose=resolved.purpose,
                workspace_id=workspace.scope.workspace_id,
            )
        return original(resolved)

    monkeypatch.setattr(workspace, "revalidate", drift_after_read)
    execution, _records = invoke(
        create_workspace_search_tool(workspace),
        {"query": "needle"},
    )

    header = read_header(execution.result.output)
    assert header["status"] == "no_matches"
    assert header["matches"] == 0
    assert header["omissions"]["drift"] == 1
    assert "needle" not in "\n".join(read_body(execution.result.output))
    assert "path" not in header


def test_search_limits_visited_source_matches_line_and_output(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("needle needle\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("needle\n", encoding="utf-8")

    visited_limits = CodingToolLimits(
        max_search_visited_files=1,
        max_search_output_characters=2_000,
    )
    execution, _records = invoke(
        create_workspace_search_tool(make_workspace(tmp_path), limits=visited_limits),
        {"query": "absent"},
    )
    header = read_header(execution.result.output)
    assert header["status"] == "visited_files_limit"
    assert header["visited_files"] <= 1
    assert header["omissions"]["visited_files_limit"] >= 1

    exact = tmp_path / "exact"
    exact.mkdir()
    (exact / "only.txt").write_text("different", encoding="utf-8")
    execution, _records = invoke(
        create_workspace_search_tool(make_workspace(tmp_path), limits=visited_limits),
        {"query": "absent", "path": "exact"},
        run_id="run-t6-exact-visited",
    )
    header = read_header(execution.result.output)
    assert header["status"] == "no_matches"
    assert header["visited_files"] == 1
    assert header["omissions"]["visited_files_limit"] == 0

    source_limits = CodingToolLimits(max_search_source_bytes=4, max_search_output_characters=2_000)
    execution, _records = invoke(
        create_workspace_search_tool(make_workspace(tmp_path), limits=source_limits),
        {"query": "needle", "path": "b.txt"},
        run_id="run-t6-source",
    )
    header = read_header(execution.result.output)
    assert header["status"] == "source_limit"
    assert header["source_bytes"] == 4
    assert header["matches"] == 0

    exact_source = tmp_path / "exact-source.txt"
    exact_source.write_bytes(b"abcd")
    exact_limits = CodingToolLimits(
        max_search_source_bytes=4,
        max_search_output_characters=2_000,
    )
    execution, _records = invoke(
        create_workspace_search_tool(make_workspace(tmp_path), limits=exact_limits),
        {"query": "absent", "path": "exact-source.txt"},
        run_id="run-t6-exact-source",
    )
    header = read_header(execution.result.output)
    assert header["status"] == "no_matches"
    assert header["source_bytes"] == 4
    assert header["source_limit"] is False

    match_limits = CodingToolLimits(max_search_output_characters=2_000)
    execution, _records = invoke(
        create_workspace_search_tool(make_workspace(tmp_path), limits=match_limits),
        {"query": "needle", "path": "a.txt", "max_matches": 1},
        run_id="run-t6-match",
    )
    header = read_header(execution.result.output)
    assert header["status"] == "match_limit"
    assert header["matches"] == 1
    assert read_body(execution.result.output) == ["a.txt:1:1: needle needle"]

    line_limits = CodingToolLimits(max_search_line_characters=4, max_search_output_characters=2_000)
    execution, _records = invoke(
        create_workspace_search_tool(make_workspace(tmp_path), limits=line_limits),
        {"query": "needle", "path": "a.txt", "max_matches": 10},
        run_id="run-t6-line",
    )
    header = read_header(execution.result.output)
    assert header["status"] == "line_limit"
    assert header["line_projection_limit"] is True
    assert len(read_body(execution.result.output)[0].split(": ", 3)[-1]) <= 4

    output_limits = CodingToolLimits(max_search_output_characters=220)
    execution, _records = invoke(
        create_workspace_search_tool(make_workspace(tmp_path), limits=output_limits),
        {"query": "needle", "max_matches": 10},
        run_id="run-t6-output",
    )
    assert len(execution.result.output) <= output_limits.max_search_output_characters
    assert read_header(execution.result.output)["status"] == "output_limit"


def test_read_and_search_elapsed_limits_are_model_visible(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("needle", encoding="utf-8")
    ticks = 0

    def elapsed_after_first_check() -> float:
        nonlocal ticks
        ticks += 1
        return 0.0 if ticks < 2 else 1.0

    monkeypatch.setattr(coding_tools_module.time, "monotonic", elapsed_after_first_check)
    limits = CodingToolLimits(
        max_read_elapsed_seconds=0.5,
        max_search_elapsed_seconds=0.5,
        max_read_output_characters=2_000,
        max_search_output_characters=2_000,
    )
    read_execution, _records = invoke(
        create_workspace_read_tool(make_workspace(tmp_path), limits=limits),
        {"path": "file.txt"},
    )
    assert read_header(read_execution.result.output)["status"] == "elapsed_limit"

    ticks = 0
    search_execution, _records = invoke(
        create_workspace_search_tool(make_workspace(tmp_path), limits=limits),
        {"query": "needle"},
        run_id="run-t6-elapsed-search",
    )
    assert read_header(search_execution.result.output)["status"] == "elapsed_limit"


def test_cancellation_propagates_and_retains_governed_record(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "file.txt").write_text("needle", encoding="utf-8")
    workspace = make_workspace(tmp_path)
    context = RunContext(run_id="run-t6-cancel")
    original = coding_tools_module._search_file

    def cancel_before_scan(*args, **kwargs):
        context.cancel("test cancellation")
        return original(*args, **kwargs)

    monkeypatch.setattr(coding_tools_module, "_search_file", cancel_before_scan)
    collector = _RunActionRecordCollector(context.run_id, 1)
    execution_context = ToolExecutionContext(context, record_collector=collector)
    tool = create_workspace_search_tool(workspace)

    with pytest.raises(RunCancelledError):
        ToolRegistry((tool,)).execute_detailed(
            ToolCall("call-cancel", tool.definition.name, '{"query":"needle"}'),
            context,
            execution_context=execution_context,
        )

    assert collector.records
    assert collector.records[0].executor_attempts == 1


def test_raw_arguments_and_records_events_are_bounded_and_sanitized(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    tiny = CodingToolLimits(max_argument_bytes=32)
    events: list[tuple[str, object]] = []
    execution, records = invoke(
        create_workspace_read_tool(workspace, limits=tiny),
        {"path": "x" * 100},
        events=events,
    )
    assert execution.result.error_code is ToolErrorCode.ARGUMENT_TOO_LARGE
    assert records == ()
    assert any(stage == RunEventType.ACTION_OBSERVED.value for stage, _ in events)

    (tmp_path / "file.txt").write_text("needle", encoding="utf-8")
    events = []
    execution, records = invoke(
        create_workspace_search_tool(workspace),
        {"query": "needle"},
        run_id="run-t6-events",
        events=events,
    )
    assert execution.result.outcome is ToolOutcome.SUCCESS
    assert records and execution.action_record is records[0]
    stages = [stage for stage, _attributes in events]
    assert stages == [
        RunEventType.ACTION_PREPARED.value,
        RunEventType.ACTION_GUARDS_EVALUATED.value,
        RunEventType.ACTION_POLICY_DECIDED.value,
        RunEventType.ACTION_REVALIDATED.value,
        RunEventType.ACTION_PRE_HOOKS_COMPLETED.value,
        RunEventType.ACTION_EFFECT_REVALIDATED.value,
        RunEventType.ACTION_EXECUTOR_STARTED.value,
        RunEventType.ACTION_EXECUTOR_COMPLETED.value,
        RunEventType.ACTION_POST_HOOKS_COMPLETED.value,
        RunEventType.ACTION_OBSERVED.value,
    ]
    assert str(tmp_path) not in repr(events)
    assert str(tmp_path) not in repr(records[0])
