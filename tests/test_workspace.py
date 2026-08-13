from __future__ import annotations

import os
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from dqagent.workspace import (
    PathKind,
    Sanitizer,
    Workspace,
    WorkspaceAccessError,
    WorkspaceDriftError,
    WorkspaceError,
    WorkspaceLimits,
    WorkspacePurpose,
    WorkspaceScope,
    sanitize_text,
)


def make_workspace(tmp_path: Path, **kwargs: object) -> Workspace:
    scope = WorkspaceScope("fixture", tmp_path, **kwargs)
    return Workspace(scope)


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
