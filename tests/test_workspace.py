from __future__ import annotations

import os
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path, PurePosixPath
from threading import Event

import pytest

import dqagent.workspace as workspace_module
from dqagent.workspace import (
    PathKind,
    Sanitizer,
    Workspace,
    WorkspaceAccessError,
    WorkspaceBlindSpotReason,
    WorkspaceChangeKind,
    WorkspaceDriftError,
    WorkspaceEntryKind,
    WorkspaceError,
    WorkspaceLimits,
    WorkspaceObserver,
    WorkspacePurpose,
    WorkspaceScope,
    sanitize_text,
)


def make_workspace(tmp_path: Path, **kwargs: object) -> Workspace:
    scope = WorkspaceScope("fixture", tmp_path, **kwargs)
    return Workspace(scope)


def make_observer(tmp_path: Path, **kwargs: object) -> WorkspaceObserver:
    return WorkspaceObserver(make_workspace(tmp_path, **kwargs))


def make_symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation requires unavailable developer privileges")
        raise


def make_directory_reparse(link: Path, target: Path) -> None:
    if os.name != "nt":
        make_symlink(link, target, directory=True)
        return
    result = subprocess.run(
        ["cmd.exe", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("Windows directory junction creation is unavailable")


def test_scope_is_immutable_and_root_is_canonical(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)

    assert workspace.root == tmp_path.resolve()
    with pytest.raises((AttributeError, TypeError)):
        workspace.scope.workspace_id = "changed"  # type: ignore[misc]


def test_scope_canonicalizes_a_symlink_root(tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    make_symlink(alias, tmp_path, directory=True)

    assert make_workspace(alias).root == tmp_path.resolve()


@pytest.mark.parametrize(
    "scope_kwargs",
    [
        {"workspace_id": ""},
        {"workspace_id": "host/path"},
        {"workspace_id": "host\\path"},
        {"root": Path("missing-root")},
        {"protected_paths": (PurePosixPath("../outside"),)},
        {"secret_paths": (PurePosixPath("bad\\path"),)},
    ],
)
def test_scope_rejects_invalid_trusted_configuration(
    tmp_path: Path, scope_kwargs: dict[str, object]
) -> None:
    kwargs: dict[str, object] = {"workspace_id": "fixture", "root": tmp_path}
    kwargs.update(scope_kwargs)

    with pytest.raises(WorkspaceError):
        WorkspaceScope(**kwargs)  # type: ignore[arg-type]


def test_workspace_rule_iterables_are_bounded_before_materialization(tmp_path: Path) -> None:
    consumed = 0
    maximum = workspace_module._MAX_RULE_ITEMS

    def secret_paths():
        nonlocal consumed
        for index in range(10_000):
            consumed += 1
            yield PurePosixPath(f"secret-{index}")

    with pytest.raises(WorkspaceError):
        WorkspaceScope("fixture", tmp_path, secret_paths=secret_paths())

    assert consumed == maximum + 1


def test_sanitizer_iterables_are_bounded_before_materialization() -> None:
    secret_consumed = 0
    host_path_consumed = 0
    secret_maximum = workspace_module._MAX_SANITIZER_ITEMS
    host_path_maximum = workspace_module._MAX_SANITIZER_HOST_PATH_ITEMS

    def secrets():
        nonlocal secret_consumed
        for index in range(10_000):
            secret_consumed += 1
            yield f"secret-{index}"

    def host_paths():
        nonlocal host_path_consumed
        for index in range(10_000):
            host_path_consumed += 1
            yield f"C:/workspace-{index}"

    with pytest.raises(ValueError, match="secrets.*item bound"):
        Sanitizer(secrets=secrets())
    with pytest.raises(ValueError, match="host_paths.*item bound"):
        Sanitizer(host_paths=host_paths())

    assert secret_consumed == secret_maximum + 1
    assert host_path_consumed == host_path_maximum + 1


def test_limits_are_immutable_and_do_not_include_other_owner_limits() -> None:
    limits = WorkspaceLimits(max_path_characters=12, max_diff_characters=30)

    assert limits.max_logical_path_characters == 12
    assert limits.max_rendered_diff_characters == 30
    assert not hasattr(limits, "max_tool_output_characters")
    with pytest.raises((AttributeError, TypeError)):
        limits.max_snapshot_entries = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_logical_path_characters": 0},
        {"max_logical_path_segments": 0},
        {"max_snapshot_entries": 0},
        {"max_snapshot_bytes": 0},
        {"max_snapshot_elapsed_seconds": 0},
        {"max_rendered_diff_characters": 0},
        {"max_logical_path_characters": True},
    ],
)
def test_limits_reject_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        WorkspaceLimits(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("", "path_empty"),
        ("a\x00b", "path_nul"),
        ("/tmp/file", "path_absolute"),
        ("C:/tmp/file", "path_drive"),
        ("\\\\server\\share", "path_unc"),
        ("dir\\file", "path_backslash"),
        ("../file", "path_parent"),
        ("dir/../file", "path_parent"),
        ("dir//file", "path_ambiguous"),
        ("./file", "path_ambiguous"),
        ("file/", "path_ambiguous"),
    ],
)
def test_logical_path_forms_fail_closed(tmp_path: Path, raw: str, reason: str) -> None:
    workspace = make_workspace(tmp_path)

    with pytest.raises(WorkspaceError) as raised:
        workspace.resolve(raw, purpose=WorkspacePurpose.READ)

    assert raised.value.reason_code == reason
    assert str(tmp_path) not in str(raised.value)
    assert "C:/tmp" not in str(raised.value)


def test_path_limits_are_applied_before_filesystem_access(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path, limits=WorkspaceLimits(max_path_characters=4))

    with pytest.raises(WorkspaceError) as raised:
        workspace.resolve("long-name", purpose=WorkspacePurpose.READ)

    assert raised.value.reason_code == "path_too_long"


def test_purpose_specific_kind_and_missing_contracts(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("content", encoding="utf-8")
    (tmp_path / "directory").mkdir()
    workspace = make_workspace(tmp_path)

    assert workspace.resolve("file.txt", purpose="read").kind is PathKind.REGULAR_FILE
    assert workspace.resolve("directory", purpose="search").kind is PathKind.DIRECTORY
    assert workspace.resolve("directory", purpose="command_cwd").kind is PathKind.DIRECTORY
    created = workspace.resolve("new.txt", purpose=WorkspacePurpose.PATCH)
    assert created.is_missing
    assert created.authority_parent == PurePosixPath(".")
    assert created.revalidation.target_must_remain_missing is True
    assert workspace.revalidate(created).is_missing
    assert workspace.revalidate(
        workspace.resolve("file.txt", purpose=WorkspacePurpose.PATCH)
    ).is_file

    with pytest.raises(WorkspaceError, match="target_kind"):
        workspace.resolve("directory", purpose=WorkspacePurpose.READ)
    with pytest.raises(WorkspaceError, match="target_missing"):
        workspace.resolve("missing.txt", purpose=WorkspacePurpose.READ)
    with pytest.raises(WorkspaceError, match="parent_missing"):
        workspace.resolve("missing/child.txt", purpose=WorkspacePurpose.PATCH)
    with pytest.raises(WorkspaceError, match="target_kind"):
        workspace.resolve("file.txt", purpose=WorkspacePurpose.COMMAND_CWD)


def test_case_behavior_follows_the_filesystem(tmp_path: Path) -> None:
    (tmp_path / "Case.txt").write_text("content", encoding="utf-8")
    workspace = make_workspace(tmp_path)

    if (tmp_path / "case.txt").exists():
        assert workspace.resolve("case.txt", purpose=WorkspacePurpose.READ).is_file
    else:
        with pytest.raises(WorkspaceError) as raised:
            workspace.resolve("case.txt", purpose=WorkspacePurpose.READ)
        assert raised.value.reason_code == "target_missing"


def test_protected_and_secret_rules_are_checked_without_reading_content(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("do not read", encoding="utf-8")
    (tmp_path / "credentials").mkdir()
    (tmp_path / "credentials" / "token.txt").write_text("secret", encoding="utf-8")
    (tmp_path / ".env.local").write_text("TOKEN=secret", encoding="utf-8")
    workspace = make_workspace(tmp_path, secret_paths=(PurePosixPath("credentials"),))

    for logical in (".git/config", ".env.local", "credentials/token.txt"):
        with pytest.raises(WorkspaceAccessError) as raised:
            workspace.resolve(logical, purpose=WorkspacePurpose.READ)
        assert raised.value.reason_code in {"protected", "secret"}
        assert "TOKEN=secret" not in str(raised.value)
        assert "do not read" not in str(raised.value)
    assert str(tmp_path) not in str(raised.value)


@pytest.mark.skipif(os.name != "nt", reason="Windows path normalization is required")
@pytest.mark.parametrize(
    ("logical_path", "reason"),
    [
        (".git./new.txt", "protected"),
        (".env./new.txt", "secret"),
    ],
)
def test_windows_missing_patch_targets_recheck_canonical_parent_rules(
    tmp_path: Path, logical_path: str, reason: str
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".env").mkdir()
    workspace = make_workspace(tmp_path)

    with pytest.raises(WorkspaceAccessError) as raised:
        workspace.resolve(logical_path, purpose=WorkspacePurpose.PATCH)

    assert raised.value.reason_code == reason


def test_missing_patch_targets_recheck_linked_protected_and_secret_parents(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "private").mkdir()
    make_directory_reparse(tmp_path / "git-alias", tmp_path / ".git")
    make_directory_reparse(tmp_path / "private-alias", tmp_path / "private")
    workspace = make_workspace(tmp_path, secret_paths=(PurePosixPath("private"),))

    for logical_path, reason in (
        ("git-alias/new.txt", "protected"),
        ("private-alias/new.txt", "secret"),
    ):
        with pytest.raises(WorkspaceAccessError) as raised:
            workspace.resolve(logical_path, purpose=WorkspacePurpose.PATCH)
        assert raised.value.reason_code == reason


@pytest.mark.skipif(os.name != "nt", reason="NTFS alternate data streams are Windows-only")
def test_windows_alternate_data_stream_paths_are_rejected(tmp_path: Path) -> None:
    secret_file = tmp_path / ".env"
    secret_file.write_text("TOKEN=secret", encoding="utf-8")
    try:
        Path(f"{secret_file}:hidden").write_text("secret stream", encoding="utf-8")
    except OSError:
        pytest.skip("NTFS alternate data streams are unavailable")

    workspace = make_workspace(tmp_path)

    with pytest.raises(WorkspaceError) as raised:
        workspace.resolve(".env:hidden", purpose=WorkspacePurpose.READ)

    assert raised.value.reason_code == "path_drive"


def test_resolve_root_rejects_replaced_reparse_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = make_workspace(root)
    resolved = workspace.resolve_root()

    root.rmdir()
    make_directory_reparse(root, outside)

    with pytest.raises(WorkspaceDriftError) as raised:
        workspace.resolve_root()
    assert raised.value.reason_code == "drift"

    with pytest.raises(WorkspaceDriftError):
        workspace.revalidate(resolved)


def test_custom_secret_exact_and_subtree_rules(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "token.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "nested.txt").write_text("secret", encoding="utf-8")
    workspace = make_workspace(
        tmp_path,
        secret_paths=(PurePosixPath("config/token.txt"), PurePosixPath("private")),
    )

    for logical in ("config/token.txt", "private/nested.txt"):
        with pytest.raises(WorkspaceAccessError) as raised:
            workspace.resolve(logical, purpose=WorkspacePurpose.READ)
        assert raised.value.reason_code == "secret"


def test_in_root_symlink_is_allowed_but_escape_is_denied(tmp_path: Path) -> None:
    (tmp_path / "inside.txt").write_text("inside", encoding="utf-8")
    make_symlink(tmp_path / "inside-link", tmp_path / "inside.txt")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside", encoding="utf-8")
    make_symlink(tmp_path / "outside-link", outside)
    workspace = make_workspace(tmp_path)

    inside = workspace.resolve("inside-link", purpose=WorkspacePurpose.READ)
    assert inside.is_file
    assert inside.followed_link is True
    with pytest.raises(WorkspaceError) as raised:
        workspace.resolve("outside-link", purpose=WorkspacePurpose.READ)
    assert raised.value.reason_code in {"link_escape", "containment"}

    inside_replacement = tmp_path / "inside-replacement.txt"
    inside_replacement.write_text("replacement", encoding="utf-8")
    link = tmp_path / "inside-link"
    link.unlink()
    make_symlink(link, inside_replacement)
    with pytest.raises(WorkspaceDriftError):
        workspace.revalidate(inside)


def test_revalidation_detects_missing_target_creation_and_parent_drift(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    missing = workspace.resolve("new.txt", purpose=WorkspacePurpose.PATCH)
    (tmp_path / "new.txt").write_text("created", encoding="utf-8")

    with pytest.raises(WorkspaceDriftError) as raised:
        workspace.revalidate(missing)
    assert raised.value.reason_code == "drift"

    sibling_missing = workspace.resolve("sibling.txt", purpose=WorkspacePurpose.PATCH)
    (tmp_path / "unrelated.txt").write_text("does not replace the parent", encoding="utf-8")
    assert workspace.revalidate(sibling_missing).is_missing

    other = tmp_path / "other"
    other.mkdir()
    nested = workspace.resolve("other/new.txt", purpose=WorkspacePurpose.PATCH)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    try:
        other.rename(tmp_path / "old")
        replacement.rename(other)
    except OSError:
        pytest.skip("platform cannot replace the authority parent in this fixture")
    with pytest.raises(WorkspaceDriftError) as raised:
        workspace.revalidate(nested)
    assert raised.value.reason_code == "drift"


def test_root_revalidation_reports_safe_drift(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    resolved = workspace.resolve_root()
    renamed = tmp_path.with_name(f"{tmp_path.name}-renamed")
    try:
        tmp_path.rename(renamed)
    except OSError:
        pytest.skip("platform cannot rename the active fixture root")

    with pytest.raises(WorkspaceDriftError) as raised:
        workspace.revalidate(resolved)
    assert raised.value.reason_code == "drift"


def test_existing_target_replacement_is_detected(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("one", encoding="utf-8")
    workspace = make_workspace(tmp_path)
    resolved = workspace.resolve("target.txt", purpose=WorkspacePurpose.PATCH)
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("two", encoding="utf-8")
    replacement.replace(target)

    with pytest.raises(WorkspaceDriftError):
        workspace.revalidate(resolved)


def test_safe_errors_expose_reason_not_host_path_or_raw_os_error(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)

    with pytest.raises(WorkspaceError) as raised:
        workspace.resolve("../not-safe", purpose=WorkspacePurpose.READ)

    error = raised.value
    assert error.reason_code == "path_parent"
    assert error.logical_path is None
    assert "not-safe" not in str(error)
    assert "No such file" not in str(error)
    assert dict(error.event_attributes) == {
        "reason_code": "path_parent",
        "purpose": "read",
        "workspace_id": "fixture",
    }


def test_sanitizer_redacts_before_truncating() -> None:
    sanitizer = Sanitizer(secrets=("TOP-SECRET",), host_paths=(Path("C:/workspace"),))
    output = sanitizer.sanitize_with_evidence(
        "prefix TOP-SECRET at C:/workspace and trailing data", max_characters=34
    )

    assert "TOP-SECRET" not in output.text
    assert "C:/workspace" not in output.text
    assert output.redacted is True
    assert output.truncated is True
    assert len(output.text) <= 34

    assert "TOP-SECRET" not in sanitize_text(
        "TOP-SECRET trailing", secrets=("TOP-SECRET",), max_characters=8
    )


@pytest.mark.parametrize(
    ("secret", "replacement"),
    [
        ("TOP-SECRET", "prefix TOP-SECRET"),
        ("[REDACTED]suffix", "[REDACTED]"),
        ("[REDACTED]", "[REDACTED]"),
    ],
)
def test_sanitizer_rejects_a_replacement_that_overlaps_a_configured_secret(
    secret: str, replacement: str
) -> None:
    with pytest.raises(ValueError):
        Sanitizer(secrets=(secret,), replacement=replacement)


def test_sanitizer_rejects_output_that_reintroduces_a_secret_at_a_boundary() -> None:
    sanitizer = Sanitizer(secrets=("abc",), replacement="Xab")

    with pytest.raises(ValueError, match="configured secret"):
        sanitizer.sanitize("abcc")


def test_workspace_sanitizer_redacts_canonical_root_variants(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    rendered = workspace.sanitize(
        f"failed at {tmp_path.as_posix()}\\file.txt", max_characters=200
    )

    assert str(tmp_path) not in rendered
    assert tmp_path.as_posix() not in rendered
    assert "[REDACTED]" in rendered


def test_missing_target_root_drift_is_typed_and_does_not_expose_host_path(
    tmp_path: Path,
) -> None:
    workspace = make_workspace(tmp_path)
    missing = workspace.resolve("new.txt", purpose=WorkspacePurpose.PATCH)

    tmp_path.rmdir()

    with pytest.raises(WorkspaceDriftError) as raised:
        workspace.revalidate(missing)

    assert raised.value.reason_code == "drift"
    assert str(tmp_path) not in str(raised.value)


def test_observer_records_untracked_modify_delete_and_type_change_in_stable_order(
    tmp_path: Path,
) -> None:
    (tmp_path / "same.txt").write_text("one", encoding="utf-8")
    (tmp_path / "deleted.txt").write_text("gone", encoding="utf-8")
    (tmp_path / "type.txt").write_text("file", encoding="utf-8")
    observer = make_observer(tmp_path)

    baseline = observer.capture(target_paths=("same.txt", "type.txt"))
    (tmp_path / "same.txt").write_text("two", encoding="utf-8")
    (tmp_path / "deleted.txt").unlink()
    (tmp_path / "type.txt").unlink()
    (tmp_path / "type.txt").mkdir()
    (tmp_path / "new.txt").write_text("new", encoding="utf-8")

    final = observer.capture(target_paths=("same.txt", "type.txt"))
    diff = observer.diff(baseline, final)

    assert [item.logical_path.as_posix() for item in diff.changes] == [
        "deleted.txt",
        "new.txt",
        "same.txt",
        "type.txt",
    ]
    assert {item.kind for item in diff.changes} == {
        WorkspaceChangeKind.CREATE,
        WorkspaceChangeKind.MODIFY,
        WorkspaceChangeKind.DELETE,
        WorkspaceChangeKind.TYPE_CHANGE,
    }
    assert diff.untracked == diff.creates
    assert diff.completeness.global_complete is True
    assert diff.completeness.target_complete is True


def test_observer_uses_full_digest_for_same_size_changes(tmp_path: Path) -> None:
    target = tmp_path / "same.txt"
    target.write_bytes(b"abc")
    observer = make_observer(tmp_path)
    baseline = observer.capture()

    target.write_bytes(b"xyz")
    final = observer.capture()
    diff = observer.diff(baseline, final)

    assert baseline.entries[0].size == final.entries[0].size == 3
    assert baseline.entries[0].digest != final.entries[0].digest
    assert [item.kind for item in diff.changes] == [WorkspaceChangeKind.MODIFY]
    assert diff.changes[0].comparison_complete is True


def test_observer_does_not_capture_secret_ignored_or_volatile_content(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env.local").write_text("TOKEN=do-not-capture", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("private", encoding="utf-8")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "cache.txt").write_text("cache", encoding="utf-8")
    (tmp_path / "volatile").mkdir()
    (tmp_path / "volatile" / "runtime.txt").write_text("runtime", encoding="utf-8")
    (tmp_path / "target.txt").write_text("target", encoding="utf-8")
    observer = make_observer(
        tmp_path,
        ignored_paths=(PurePosixPath("ignored"),),
        volatile_paths=(PurePosixPath("volatile"),),
    )

    snapshot = observer.capture(target_paths=("target.txt",))
    paths = {item.logical_path.as_posix() for item in snapshot.entries}
    reasons = {(item.logical_path.as_posix(), item.reason_code) for item in snapshot.omissions}

    assert ".env.local" not in paths
    assert ".git" not in paths
    assert "ignored" not in paths
    assert "volatile" not in paths
    assert (".env.local", WorkspaceBlindSpotReason.SECRET.value) in reasons
    assert (".git", WorkspaceBlindSpotReason.PROTECTED.value) in reasons
    assert ("ignored", WorkspaceBlindSpotReason.IGNORED.value) in reasons
    assert ("volatile", WorkspaceBlindSpotReason.VOLATILE_EXCLUSION.value) in reasons
    assert snapshot.completeness.global_complete is False
    assert snapshot.completeness.target_complete is True
    assert all("do-not-capture" not in repr(item) for item in snapshot.entries)


def test_observer_records_links_without_following_them(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("inside", encoding="utf-8")
    link = tmp_path / "target-link"
    make_symlink(link, target)
    observer = make_observer(tmp_path)

    snapshot = observer.capture()
    entries = {item.logical_path.as_posix(): item for item in snapshot.entries}

    assert entries["target-link"].kind is WorkspaceEntryKind.LINK
    assert entries["target-link"].digest is None
    assert entries["target-link"].text is None
    assert any(
        item.logical_path == PurePosixPath("target-link")
        and item.reason_code == WorkspaceBlindSpotReason.LINK_NOT_FOLLOWED.value
        for item in snapshot.omissions
    )


def test_directory_link_blind_spot_covers_nested_targets(tmp_path: Path) -> None:
    target_directory = tmp_path / "target-directory"
    target_directory.mkdir()
    (target_directory / "child.txt").write_text("inside", encoding="utf-8")
    link = tmp_path / "directory-link"
    make_symlink(link, target_directory, directory=True)
    observer = make_observer(tmp_path)

    snapshot = observer.capture(target_paths=("directory-link/child.txt",))

    assert not any(
        item.logical_path.as_posix().startswith("directory-link/")
        for item in snapshot.entries
    )
    assert snapshot.completeness.target_complete is False
    assert any(
        item.logical_path == PurePosixPath("directory-link") and item.subtree
        for item in snapshot.omissions
    )


def test_observation_records_are_immutable(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("before", encoding="utf-8")
    observer = make_observer(tmp_path)
    baseline = observer.capture()

    target.write_text("after", encoding="utf-8")
    final = observer.capture()
    diff = observer.diff(baseline, final)

    with pytest.raises(FrozenInstanceError):
        baseline.entries = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        baseline.entries[0].size = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        baseline.completeness.global_complete = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        diff.changes = ()  # type: ignore[misc]


def test_link_blind_spot_does_not_prove_its_target_content(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("link-target-secret", encoding="utf-8")
    link = tmp_path / "secret-link"
    make_symlink(link, secret)
    observer = make_observer(tmp_path, secret_paths=(PurePosixPath("secret.txt"),))

    baseline = observer.capture(target_paths=("secret-link",))
    secret.write_text("changed-link-target-secret", encoding="utf-8")
    final = observer.capture(target_paths=("secret-link",))
    diff = observer.diff(baseline, final, target_paths=("secret-link",))

    link_entry = next(
        item for item in baseline.entries if item.logical_path == PurePosixPath("secret-link")
    )
    assert link_entry.digest is None
    assert link_entry.text is None
    assert "link-target-secret" not in repr(baseline)
    assert not any(
        item.logical_path == PurePosixPath("secret-link") for item in diff.changes
    )
    assert diff.completeness.target_complete is False
    assert diff.completeness.global_complete is False


def test_target_and_forbidden_completeness_are_scoped_independently(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_text("before", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")
    observer = make_observer(tmp_path)
    baseline = observer.capture(target_paths=("target.txt",))

    (tmp_path / "target.txt").write_text("after!", encoding="utf-8")
    final = observer.capture(target_paths=("target.txt",))
    diff = observer.diff(
        baseline,
        final,
        target_paths=("target.txt",),
        forbidden_paths=(".env",),
    )

    assert diff.completeness.target_complete is True
    assert diff.completeness.forbidden_complete is False
    assert diff.completeness.global_complete is False
    assert diff.completeness.observation_complete is False
    assert any(item.reason_code == WorkspaceBlindSpotReason.SECRET.value for item in diff.omissions)


@pytest.mark.parametrize(
    ("limits_kwargs", "reason"),
    [
        (
            {"max_snapshot_entries": 1},
            WorkspaceBlindSpotReason.ENTRIES_LIMIT,
        ),
        (
            {"max_snapshot_file_bytes": 2},
            WorkspaceBlindSpotReason.PER_FILE_BYTES_LIMIT,
        ),
        (
            {"max_snapshot_bytes": 2, "max_snapshot_file_bytes": 20},
            WorkspaceBlindSpotReason.AGGREGATE_BYTES_LIMIT,
        ),
    ],
)
def test_observer_records_each_snapshot_resource_limit(
    tmp_path: Path,
    limits_kwargs: dict[str, object],
    reason: WorkspaceBlindSpotReason,
) -> None:
    (tmp_path / "a.txt").write_text("1234", encoding="utf-8")
    (tmp_path / "b.txt").write_text("5678", encoding="utf-8")
    observer = make_observer(tmp_path, limits=WorkspaceLimits(**limits_kwargs))

    snapshot = observer.capture()

    assert any(item.reason_code == reason.value for item in snapshot.omissions)
    assert snapshot.completeness.global_complete is False
    if reason is WorkspaceBlindSpotReason.PER_FILE_BYTES_LIMIT:
        entry = next(
            item
            for item in snapshot.entries
            if item.logical_path == PurePosixPath("a.txt")
        )
        assert entry.digest is None
        assert entry.text is None


def test_observer_bounds_entry_limit_blind_spots_for_wide_nested_directories(
    tmp_path: Path,
) -> None:
    for index in range(128):
        (tmp_path / f"root-{index:03d}.txt").write_text("root", encoding="utf-8")

    nested = tmp_path / "nested"
    nested.mkdir()
    for index in range(128):
        (nested / f"child-{index:03d}.txt").write_text("child", encoding="utf-8")

    observer = make_observer(
        tmp_path,
        limits=WorkspaceLimits(max_snapshot_entries=1),
    )

    snapshot = observer.capture()

    entry_limit_spots = [
        item
        for item in snapshot.omissions
        if item.reason_code == WorkspaceBlindSpotReason.ENTRIES_LIMIT.value
    ]
    assert len(snapshot.entries) <= 1
    assert [item.logical_path for item in snapshot.entries] == [PurePosixPath("nested")]
    assert snapshot.completeness.observed_entries == len(snapshot.entries)
    assert snapshot.completeness.global_complete is False
    assert len(entry_limit_spots) <= 2
    assert {item.logical_path for item in entry_limit_spots} == {
        PurePosixPath("."),
        PurePosixPath("nested"),
    }
    assert all(item.subtree for item in entry_limit_spots)


def test_observer_bounds_explicit_exclusion_blind_spots_without_poisoning_target_scope(
    tmp_path: Path,
) -> None:
    for index in range(128):
        (tmp_path / f"secret-{index:03d}.pem").write_text(
            "do-not-capture", encoding="utf-8"
        )
    (tmp_path / "target.txt").write_text("target", encoding="utf-8")
    observer = make_observer(
        tmp_path,
        limits=WorkspaceLimits(max_snapshot_entries=1),
    )

    snapshot = observer.capture(target_paths=("target.txt",))

    secret_spots = [
        item
        for item in snapshot.omissions
        if item.reason_code == WorkspaceBlindSpotReason.SECRET.value
    ]
    assert [item.logical_path for item in snapshot.entries] == [PurePosixPath("target.txt")]
    assert len(secret_spots) <= 2
    assert any(item.aggregate for item in secret_spots)
    assert snapshot.completeness.target_complete is True
    assert snapshot.completeness.global_complete is False
    assert "do-not-capture" not in repr(snapshot)


def test_observer_marks_an_explicitly_excluded_target_incomplete_after_aggregation(
    tmp_path: Path,
) -> None:
    for index in range(16):
        (tmp_path / f"secret-{index:03d}.pem").write_text(
            "do-not-capture", encoding="utf-8"
        )
    observer = make_observer(
        tmp_path,
        limits=WorkspaceLimits(max_snapshot_entries=1),
    )

    snapshot = observer.capture(target_paths=("secret-015.pem",))

    assert snapshot.completeness.target_complete is False
    assert snapshot.completeness.global_complete is False


def test_observer_checks_cancellation_while_ordering_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(16):
        (tmp_path / f"file-{index:03d}.txt").write_text("content", encoding="utf-8")
    cancelled = Event()
    original_heappop = workspace_module.heapq.heappop

    def cancel_after_pop(heap: object) -> object:
        item = original_heappop(heap)  # type: ignore[arg-type]
        cancelled.set()
        return item

    monkeypatch.setattr(workspace_module.heapq, "heappop", cancel_after_pop)

    snapshot = make_observer(tmp_path).capture(cancel=cancelled)

    assert any(
        item.reason_code == WorkspaceBlindSpotReason.CANCELLED.value
        for item in snapshot.omissions
    )
    assert snapshot.completeness.global_complete is False


def test_observer_records_elapsed_limit_and_cancellation() -> None:
    class AdvancingClock:
        def __init__(self) -> None:
            self.value = 0.0

        def __call__(self) -> float:
            current = self.value
            self.value += 1.0
            return current

    import tempfile

    with tempfile.TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        (root / "file.txt").write_text("content", encoding="utf-8")
        elapsed_observer = WorkspaceObserver(
            make_workspace(
                root,
                limits=WorkspaceLimits(max_snapshot_elapsed_seconds=0.5),
            ),
            monotonic=AdvancingClock(),
        )
        elapsed = elapsed_observer.capture()
        assert any(
            item.reason_code == WorkspaceBlindSpotReason.ELAPSED_LIMIT.value
            for item in elapsed.omissions
        )
        assert elapsed.completeness.global_complete is False

        cancelled = Event()
        cancelled.set()
        cancelled_snapshot = make_observer(root).capture(cancel=cancelled)
        assert any(
            item.reason_code == WorkspaceBlindSpotReason.CANCELLED.value
            for item in cancelled_snapshot.omissions
        )
        assert cancelled_snapshot.completeness.global_complete is False


def test_observer_uses_safe_binary_and_oversized_metadata(tmp_path: Path) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "large.txt").write_text("12345", encoding="utf-8")
    observer = make_observer(
        tmp_path,
        limits=WorkspaceLimits(max_snapshot_file_bytes=4),
    )

    snapshot = observer.capture()
    entries = {item.logical_path.as_posix(): item for item in snapshot.entries}

    assert entries["binary.bin"].binary is True
    assert entries["binary.bin"].text is None
    assert entries["binary.bin"].digest is not None
    assert entries["large.txt"].text is None
    assert entries["large.txt"].digest is None
    assert "12345" not in repr(entries["large.txt"])


def test_observer_normalizes_line_endings_only_in_rendered_projection(tmp_path: Path) -> None:
    target = tmp_path / "lines.txt"
    target.write_bytes(b"one\r\ntwo\r\n")
    observer = make_observer(tmp_path)
    baseline = observer.capture()

    target.write_bytes(b"one\ntwo\n")
    final = observer.capture()
    diff = observer.diff(baseline, final)

    assert diff.changes[0].kind is WorkspaceChangeKind.MODIFY
    assert "metadata differs; text projection is equal" in diff.rendered_diff
    assert "\r" not in diff.rendered_diff


def test_observer_propagates_incomplete_same_size_content_comparison(tmp_path: Path) -> None:
    target = tmp_path / "large.txt"
    target.write_bytes(b"aaaa")
    observer = make_observer(
        tmp_path,
        limits=WorkspaceLimits(max_snapshot_file_bytes=3),
    )
    baseline = observer.capture(target_paths=("large.txt",))

    target.write_bytes(b"bbbb")
    final = observer.capture(target_paths=("large.txt",))
    diff = observer.diff(baseline, final, target_paths=("large.txt",))

    assert diff.changes == ()
    assert diff.completeness.target_complete is False
    assert diff.completeness.global_complete is False


def test_observer_bounds_rendered_diff_and_keeps_structured_change(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("before", encoding="utf-8")
    observer = make_observer(
        tmp_path,
        limits=WorkspaceLimits(max_rendered_diff_characters=8),
    )
    baseline = observer.capture()

    target.write_text("after content", encoding="utf-8")
    final = observer.capture()
    diff = observer.diff(baseline, final)

    assert len(diff.rendered_diff) <= 8
    assert diff.completeness.rendered_diff_complete is False
    assert (
        diff.completeness.rendered_diff_omission_reason
        == WorkspaceBlindSpotReason.RENDERED_DIFF_LIMIT
    )
    assert len(diff.changes) == 1
