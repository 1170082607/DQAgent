from __future__ import annotations

import builtins
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dqagent import memory_cli
from dqagent.memory import (
    MemoryCandidate,
    MemoryConfidence,
    MemoryKind,
    MemoryLifecycleStatus,
    MemoryProvenance,
    MemoryScope,
    MemoryScopeKind,
    MemorySensitivity,
    MemorySourceType,
)
from dqagent.memory_policy import DefaultMemoryPolicy
from dqagent.memory_service import MemoryService
from dqagent.memory_store import SqliteMemoryStore

USER_SCOPE = MemoryScope(MemoryScopeKind.USER, "cli-user")


def _service(database: Path) -> MemoryService:
    return MemoryService(SqliteMemoryStore(database), DefaultMemoryPolicy())


def _candidate_args(
    database: Path,
    command: str,
    *,
    content: str = "The user prefers concise answers.",
    topic: str = "response.style",
) -> list[str]:
    return [
        command,
        "--database",
        str(database),
        "--scope-kind",
        "user",
        "--scope-id",
        USER_SCOPE.scope_id,
        "--kind",
        "preference",
        "--topic",
        topic,
        "--content",
        content,
    ]


def _confirm_remember(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    content: str = "The user prefers concise answers.",
) -> None:
    monkeypatch.setattr(builtins, "input", lambda prompt: "yes")
    assert memory_cli.main(_candidate_args(database, "remember", content=content)) == 0


def test_remember_preview_is_transient_and_rejection_does_not_create_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "memory.sqlite3"
    monkeypatch.setattr(builtins, "input", lambda prompt: "no")

    exit_code = memory_cli.main(_candidate_args(database, "remember"))

    captured = capsys.readouterr()
    assert exit_code == memory_cli.EXIT_DECLINED
    assert "candidate_type=transient" in captured.out
    assert "candidate_digest=" in captured.out
    assert "confirmation=declined reason=explicit_rejection" in captured.out
    assert captured.err == ""
    assert not database.exists()


def test_eof_rejection_keeps_an_existing_sqlite_scope_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "memory.sqlite3"
    _confirm_remember(database, monkeypatch)
    before = _service(database).list(USER_SCOPE)

    monkeypatch.setattr(builtins, "input", lambda prompt: (_ for _ in ()).throw(EOFError))
    exit_code = memory_cli.main(
        _candidate_args(
            database,
            "correct",
            content="The user prefers detailed answers.",
        )
        + ["--memory-id", before.records[0].memory_id]
    )

    after = _service(database).list(USER_SCOPE)
    assert exit_code == memory_cli.EXIT_DECLINED
    assert after == before


def test_confirmed_remember_is_visible_after_reconstructing_sqlite_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "memory.sqlite3"
    monkeypatch.setenv("DQAGENT_MODEL", "must-not-be-read")
    monkeypatch.setenv("DQAGENT_PROVIDER", "must-not-be-read")
    monkeypatch.setattr(builtins, "input", lambda prompt: "confirm")

    exit_code = memory_cli.main(
        _candidate_args(database, "remember")
        + [
            "--source-item-digest",
            "a" * 64,
            "--extractor-identity",
            "manual-cli-test",
        ]
    )

    captured = capsys.readouterr()
    record = _service(database).list(USER_SCOPE).records[0]
    assert exit_code == memory_cli.EXIT_OK
    assert "status=confirmed" in captured.out
    assert "outcome=added" in captured.out
    assert "The user prefers concise answers." in captured.out
    assert record.provenance.source_item_digest == "a" * 64
    assert record.provenance.extractor_identity == "manual-cli-test"

    capsys.readouterr()
    assert memory_cli.main(
        [
            "list",
            "--database",
            str(database),
            "--scope-kind",
            "user",
            "--scope-id",
            USER_SCOPE.scope_id,
        ]
    ) == memory_cli.EXIT_OK
    listed = capsys.readouterr()
    assert "memory_list=true" in listed.out
    assert "lifecycle.status=active" in listed.out
    assert "revision=1" in listed.out
    assert "provenance.source_type=user_draft" in listed.out
    assert "confirmation.candidate_digest=" in listed.out
    assert "The user prefers concise answers." in listed.out


@pytest.mark.parametrize("sensitivity", ["sensitive", "secret"])
def test_denied_sensitive_candidate_never_echoes_payload_or_writes(
    tmp_path: Path,
    sensitivity: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "memory.sqlite3"
    payload = "DO-NOT-ECHO-SENSITIVE-MEMORY"

    exit_code = memory_cli.main(
        _candidate_args(database, "remember", content=payload)
        + ["--sensitivity", sensitivity]
    )

    captured = capsys.readouterr()
    assert exit_code == memory_cli.EXIT_POLICY_DENIED
    assert payload not in captured.out
    assert payload not in captured.err
    assert "error: policy_denied" in captured.err
    expected_reason = (
        "secret_content_not_allowed"
        if sensitivity == "secret"
        else "sensitive_content_not_allowed"
    )
    assert expected_reason in captured.err
    assert not database.exists()


def test_credential_like_candidate_cannot_bypass_policy_with_non_sensitive_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "memory.sqlite3"
    payload = "AWS_SECRET_ACCESS_KEY=example"
    monkeypatch.setattr(builtins, "input", lambda prompt: "yes")

    exit_code = memory_cli.main(
        _candidate_args(database, "remember", content=payload)
        + ["--sensitivity", "non_sensitive"]
    )

    captured = capsys.readouterr()
    assert exit_code == memory_cli.EXIT_POLICY_DENIED
    assert payload not in captured.out
    assert payload not in captured.err
    assert "reason=secret_content_not_allowed" in captured.err
    assert not database.exists()


@pytest.mark.parametrize(
    "payload",
    [
        "OPENAI_API_KEY=example",
        "GITHUB_TOKEN=example",
        "DATABASE_PASSWORD=example",
    ],
)
def test_prefixed_credential_like_candidate_cannot_bypass_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: str,
) -> None:
    database = tmp_path / "memory.sqlite3"
    monkeypatch.setattr(builtins, "input", lambda prompt: "yes")

    exit_code = memory_cli.main(
        _candidate_args(database, "remember", content=payload)
        + ["--sensitivity", "non_sensitive"]
    )

    captured = capsys.readouterr()
    assert exit_code == memory_cli.EXIT_POLICY_DENIED
    assert payload not in captured.out
    assert payload not in captured.err
    assert "reason=secret_content_not_allowed" in captured.err
    assert not database.exists()


def test_remember_preview_escapes_control_characters_in_payload_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "memory.sqlite3"
    content = "safe\nstatus=confirmed\n\x1b[2J"
    extractor_identity = "manual\nforged=true\x1b[31m"
    monkeypatch.setattr(builtins, "input", lambda prompt: "no")

    exit_code = memory_cli.main(
        _candidate_args(database, "remember", content=content)
        + ["--extractor-identity", extractor_identity]
    )

    captured = capsys.readouterr()
    assert exit_code == memory_cli.EXIT_DECLINED
    assert "content=safe\\x0astatus=confirmed\\x0a\\x1b[2J" in captured.out
    assert "provenance.extractor_identity=manual\\x0aforged=true\\x1b[31m" in captured.out
    assert "content=safe\nstatus=confirmed\n" not in captured.out
    assert "\x1b" not in captured.out
    assert not database.exists()


def test_list_escapes_control_characters_in_persisted_payload_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "memory.sqlite3"
    content = "safe\r\nforged=true\x1b[2J"
    extractor_identity = "manual\r\nprovenance=forged\x1b[31m"
    monkeypatch.setattr(builtins, "input", lambda prompt: "yes")

    assert (
        memory_cli.main(
            _candidate_args(database, "remember", content=content)
            + ["--extractor-identity", extractor_identity]
        )
        == memory_cli.EXIT_OK
    )
    capsys.readouterr()

    assert (
        memory_cli.main(
            [
                "list",
                "--database",
                str(database),
                "--scope-kind",
                "user",
                "--scope-id",
                USER_SCOPE.scope_id,
            ]
        )
        == memory_cli.EXIT_OK
    )
    captured = capsys.readouterr()
    assert "content=safe\\x0d\\x0aforged=true\\x1b[2J" in captured.out
    assert "provenance.extractor_identity=manual\\x0d\\x0aprovenance=forged\\x1b[31m" in (
        captured.out
    )
    assert "content=safe\r\nforged=true" not in captured.out
    assert "\x1b" not in captured.out


def test_show_and_list_are_explicit_memory_projections_with_provenance_and_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "memory.sqlite3"
    _confirm_remember(database, monkeypatch)
    record = _service(database).list(USER_SCOPE).records[0]

    assert memory_cli.main(
        [
            "show",
            record.memory_id,
            "--database",
            str(database),
            "--scope-kind",
            "user",
            "--scope-id",
            USER_SCOPE.scope_id,
        ]
    ) == memory_cli.EXIT_OK
    shown = capsys.readouterr()
    assert "memory_show=true" in shown.out
    assert "memory_record=true" in shown.out
    assert "lifecycle.status=active" in shown.out
    assert "revision=1" in shown.out
    assert "provenance.extractor_identity=dqagent-memory-cli-v1" in shown.out
    assert "citation" not in shown.out.lower()
    assert "transcript" not in shown.out.lower()


def test_correct_requires_confirmation_then_atomically_supersedes_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "memory.sqlite3"
    _confirm_remember(database, monkeypatch)
    original = _service(database).list(USER_SCOPE).records[0]

    monkeypatch.setattr(builtins, "input", lambda prompt: "no")
    rejected = memory_cli.main(
        _candidate_args(
            database,
            "correct",
            content="The user prefers detailed answers.",
        )
        + ["--memory-id", original.memory_id]
    )
    capsys.readouterr()
    unchanged = _service(database).list(USER_SCOPE)
    assert rejected == memory_cli.EXIT_DECLINED
    assert unchanged.revision == 1
    assert unchanged.records == (original,)

    monkeypatch.setattr(builtins, "input", lambda prompt: "yes")
    confirmed = memory_cli.main(
        _candidate_args(
            database,
            "correct",
            content="The user prefers detailed answers.",
        )
        + ["--memory-id", original.memory_id]
    )
    output = capsys.readouterr().out
    records = _service(database).list(USER_SCOPE).records

    assert confirmed == memory_cli.EXIT_OK
    assert "status=corrected" in output
    assert "superseded_memory_id=" + original.memory_id in output
    assert len(records) == 2
    superseded = next(record for record in records if record.memory_id == original.memory_id)
    replacement = next(record for record in records if record.memory_id != original.memory_id)
    assert superseded.status is MemoryLifecycleStatus.SUPERSEDED
    assert replacement.status is MemoryLifecycleStatus.ACTIVE
    assert replacement.supersedes_id == original.memory_id
    assert replacement.content == "The user prefers detailed answers."


def test_forget_rejection_is_zero_write_and_confirmation_leaves_content_free_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "memory.sqlite3"
    _confirm_remember(database, monkeypatch)
    original = _service(database).list(USER_SCOPE).records[0]
    before = _service(database).list(USER_SCOPE)

    monkeypatch.setattr(builtins, "input", lambda prompt: "no")
    rejected = memory_cli.main(
        [
            "forget",
            "--database",
            str(database),
            "--scope-kind",
            "user",
            "--scope-id",
            USER_SCOPE.scope_id,
            "--memory-id",
            original.memory_id,
        ]
    )
    capsys.readouterr()
    after_rejection = _service(database).list(USER_SCOPE)
    assert rejected == memory_cli.EXIT_DECLINED
    assert after_rejection == before

    monkeypatch.setattr(builtins, "input", lambda prompt: "yes")
    forgotten = memory_cli.main(
        [
            "forget",
            "--database",
            str(database),
            "--scope-kind",
            "user",
            "--scope-id",
            USER_SCOPE.scope_id,
            "--memory-id",
            original.memory_id,
        ]
    )
    output = capsys.readouterr().out
    listed = _service(database).list(USER_SCOPE)

    assert forgotten == memory_cli.EXIT_OK
    assert "forget_target=memory" in output
    assert "status=forgotten" in output
    assert listed.records == ()
    assert listed.tombstones[0].memory_id == original.memory_id
    assert original.content not in repr(listed.tombstones[0])


def test_forget_rejection_does_not_materialize_expiry_before_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "memory.sqlite3"
    created_at = datetime(2026, 8, 11, 8, 0, tzinfo=UTC)
    candidate = MemoryCandidate(
        scope=USER_SCOPE,
        kind=MemoryKind.PREFERENCE,
        topic="expiry.preference",
        content="This candidate has an expiry.",
        confidence=MemoryConfidence(1.0),
        sensitivity=MemorySensitivity.NON_SENSITIVE,
        provenance=MemoryProvenance(
            source_type=MemorySourceType.USER_DRAFT,
            source_item_digest=hashlib.sha256(b"expiry-source").hexdigest(),
            extractor_identity="cli-expiry-test",
            extracted_at=created_at,
        ),
        valid_from=created_at,
        expires_at=created_at + timedelta(seconds=1),
    )
    setup = MemoryService(
        SqliteMemoryStore(database),
        DefaultMemoryPolicy(),
        clock=lambda: created_at,
        id_factory=lambda: "memory-expiring",
    )
    original = setup.confirm(candidate, candidate.digest, scope=USER_SCOPE).record
    before = SqliteMemoryStore(database).load(USER_SCOPE)

    monkeypatch.setattr(builtins, "input", lambda prompt: "no")
    exit_code = memory_cli.main(
        [
            "forget",
            "--database",
            str(database),
            "--scope-kind",
            "user",
            "--scope-id",
            USER_SCOPE.scope_id,
            "--memory-id",
            original.memory_id,
        ]
    )

    after = SqliteMemoryStore(database).load(USER_SCOPE)
    assert exit_code == memory_cli.EXIT_DECLINED
    assert after == before
    assert after.records[0].status is MemoryLifecycleStatus.ACTIVE


def test_missing_target_has_stable_error_without_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "memory.sqlite3"
    exit_code = memory_cli.main(
        [
            "show",
            "missing-memory",
            "--database",
            str(database),
            "--scope-kind",
            "user",
            "--scope-id",
            USER_SCOPE.scope_id,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == memory_cli.EXIT_ERROR
    assert captured.out == ""
    assert captured.err.startswith("error: not_found operation=show")


def test_scope_is_required_and_no_weak_yes_flag_exists() -> None:
    with pytest.raises(SystemExit) as missing_scope:
        memory_cli.main(["list"])
    assert missing_scope.value.code == memory_cli.EXIT_USAGE

    with pytest.raises(SystemExit) as weak_confirmation:
        memory_cli.main(
            [
                "forget",
                "--scope-kind",
                "user",
                "--scope-id",
                "cli-user",
                "--memory-id",
                "memory-1",
                "--yes",
            ]
        )
    assert weak_confirmation.value.code == memory_cli.EXIT_USAGE
