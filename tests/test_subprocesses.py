import os
import subprocess
import sys
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Timer

import pytest

from dqagent.execution import RunContext
from dqagent.subprocesses import (
    IsolationCapability,
    LocalSubprocessRunner,
    OutputSanitizer,
    SubprocessRequest,
    SubprocessStatus,
    build_minimal_environment,
    normalize_isolation_capabilities,
    validate_isolation_capabilities,
)
from dqagent.workspace import Sanitizer


def make_runner(
    *,
    sanitizer: OutputSanitizer | None = None,
    cleanup_timeout_seconds: float = 1.0,
    poll_interval_seconds: float = 0.005,
    stream_chunk_bytes: int = 16 * 1024,
) -> LocalSubprocessRunner:
    return LocalSubprocessRunner(
        sanitizer=sanitizer or Sanitizer(),
        cleanup_timeout_seconds=cleanup_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        stream_chunk_bytes=stream_chunk_bytes,
    )


class _RaisingSanitizer:
    def sanitize(self, value: str, *, max_characters: int | None = None) -> str:
        raise RuntimeError("sanitizer unavailable")


class _MalformedSanitizer:
    def sanitize(self, value: str, *, max_characters: int | None = None) -> str:
        return None  # type: ignore[return-value]


class _TerminateIgnoringProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminate_called = False
        self.kill_called = False

    def terminate(self) -> None:
        self.terminate_called = True

    def kill(self) -> None:
        self.kill_called = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode


def make_request(
    cwd: Path,
    code: str,
    *,
    environment: dict[str, str] | None = None,
    timeout_seconds: float = 2.0,
    stdout_limit_bytes: int = 4_096,
    stderr_limit_bytes: int = 4_096,
    required_capabilities: tuple[IsolationCapability, ...] = (),
    arguments: tuple[str, ...] = (),
) -> SubprocessRequest:
    return SubprocessRequest(
        argv=(sys.executable, "-c", code, *arguments),
        cwd=cwd,
        environment=environment or {},
        timeout_seconds=timeout_seconds,
        stdout_limit_bytes=stdout_limit_bytes,
        stderr_limit_bytes=stderr_limit_bytes,
        required_capabilities=required_capabilities,
    )


def test_isolation_capabilities_are_technical_and_immutable() -> None:
    capabilities = normalize_isolation_capabilities(
        (
            IsolationCapability.DIRECT_ARGV,
            IsolationCapability.BOUNDED_OUTPUT,
            IsolationCapability.DIRECT_ARGV,
        )
    )

    assert capabilities == frozenset(
        {IsolationCapability.DIRECT_ARGV, IsolationCapability.BOUNDED_OUTPUT}
    )
    assert validate_isolation_capabilities(capabilities) == capabilities
    assert all(
        term not in capability.value
        for capability in IsolationCapability
        for term in ("approval", "deny", "policy")
    )


@pytest.mark.parametrize("value", ["direct_argv", ("direct_argv",), (None,)])
def test_isolation_capability_validation_rejects_malformed_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_isolation_capabilities(value)  # type: ignore[arg-type]


def test_request_freezes_direct_argv_cwd_environment_and_limits(tmp_path: Path) -> None:
    environment = {"SAFE": "before"}
    request = SubprocessRequest(
        argv=(sys.executable, "-c", "pass"),
        cwd=tmp_path,
        env=environment,
        timeout_seconds=1,
        stdout_limit=128,
        stderr_max_bytes=256,
        required_capabilities=(IsolationCapability.DIRECT_ARGV,),
    )

    environment["SAFE"] = "after"
    assert request.cwd == tmp_path.resolve()
    assert request.argv[0] == sys.executable
    assert request.environment == {"SAFE": "before"}
    assert request.stdout_limit_bytes == 128
    assert request.stderr_limit_bytes == 256
    assert request.required_capabilities == frozenset({IsolationCapability.DIRECT_ARGV})
    with pytest.raises(TypeError):
        request.environment["NEW"] = "value"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        request.timeout_seconds = 2  # type: ignore[misc]


def test_minimal_environment_uses_only_allowlist_and_omits_secrets() -> None:
    source = {
        "SAFE_VALUE": "safe",
        "SECRET_TOKEN": "secret-token",
        "OTHER_VALUE": "contains-secret-token",
        "NOT_SELECTED": "not selected",
    }

    environment = build_minimal_environment(
        ("SAFE_VALUE", "SECRET_TOKEN", "OTHER_VALUE"),
        source=source,
        secret_values=("secret-token",),
    )

    assert dict(environment) == {"SAFE_VALUE": "safe"}
    with pytest.raises(TypeError):
        environment["NEW"] = "value"  # type: ignore[index]


def test_local_runner_executes_direct_argv_with_cwd_env_and_no_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DQAGENT_HOST_ONLY", "must-not-cross-boundary")
    code = (
        "import os,sys; "
        "os.write(1, (os.environ.get('SAFE','missing') + '\\n').encode()); "
        "os.write(1, (os.environ.get('DQAGENT_HOST_ONLY','missing') + '\\n').encode()); "
        "os.write(1, b'stdin=' + (b'closed' if sys.stdin.read(1) == '' else b'open'))"
    )
    request = make_request(
        tmp_path,
        code,
        environment=dict(build_minimal_environment({"SAFE": "allowed"})),
    )

    result = make_runner().run(request)

    assert result.status is SubprocessStatus.NORMAL
    assert result.returncode == 0
    assert result.stdout == "allowed\nmissing\nstdin=closed"
    assert result.stderr == ""
    assert result.cleanup_succeeded
    assert result.succeeded
    assert IsolationCapability.NO_STDIN in result.capabilities
    assert "host_filesystem_isolation" in make_runner().unavailable_guarantees


def test_local_runner_drains_both_streams_and_bounds_output(tmp_path: Path) -> None:
    code = "import os; os.write(1, b'A' * 100000); os.write(2, b'B' * 100000)"
    result = make_runner().run(
        make_request(
            tmp_path,
            code,
            stdout_limit_bytes=257,
            stderr_limit_bytes=193,
        )
    )

    assert result.status is SubprocessStatus.NORMAL
    assert result.returncode == 0
    assert len(result.stdout.encode()) <= 257
    assert len(result.stderr.encode()) <= 193
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True
    assert result.cleanup_succeeded


def test_local_runner_reassembles_split_utf8_without_replacement(tmp_path: Path) -> None:
    code = (
        "import os,time; "
        "os.write(1, b'\\xe2'); time.sleep(0.01); "
        "os.write(1, b'\\x82'); time.sleep(0.01); os.write(1, b'\\xac')"
    )
    result = make_runner(stream_chunk_bytes=1).run(
        make_request(tmp_path, code, stdout_limit_bytes=64)
    )

    assert result.status is SubprocessStatus.NORMAL
    assert result.stdout == "\u20ac"
    assert result.stdout_decode_replacements == 0
    assert result.stdout_truncated is False


def test_local_runner_does_not_count_a_valid_replacement_character(tmp_path: Path) -> None:
    code = "import os; os.write(1, 'valid-\\ufffd-character'.encode('utf-8'))"
    result = make_runner().run(make_request(tmp_path, code))

    assert result.stdout == "valid-" + chr(0xFFFD) + "-character"
    assert result.stdout_decode_replacements == 0


def test_local_runner_records_decode_replacement_and_sanitizes_output(tmp_path: Path) -> None:
    code = "import os; os.write(1, b'prefix-secret\\xff-suffix')"
    result = make_runner(
        sanitizer=Sanitizer(secrets=("secret",)),
        stream_chunk_bytes=1,
    ).run(make_request(tmp_path, code))

    assert result.status is SubprocessStatus.NORMAL
    assert "secret" not in result.stdout
    assert result.stdout_decode_replacements == 1
    assert result.decode_replacements


def test_local_runner_requires_output_sanitizer_before_spawn() -> None:
    with pytest.raises(ValueError, match="output sanitizer"):
        LocalSubprocessRunner()


@pytest.mark.parametrize("sanitizer", [_RaisingSanitizer(), _MalformedSanitizer()])
def test_local_runner_fails_closed_when_sanitizer_is_unusable(
    tmp_path: Path, sanitizer: OutputSanitizer
) -> None:
    result = LocalSubprocessRunner(sanitizer=sanitizer).run(
        make_request(
            tmp_path,
            "import os; os.write(1, b'contains-secret')",
        )
    )

    assert result.status is SubprocessStatus.OUTPUT_SANITIZATION_ERROR
    assert result.succeeded is False
    assert result.stdout == ""
    assert "contains-secret" not in result.stdout
    assert "output sanitization failed" in result.diagnostic


def test_local_runner_combines_minimal_environment_with_result_sanitization(
    tmp_path: Path,
) -> None:
    secret = "secret-token"
    environment = build_minimal_environment(
        ("SAFE_VALUE", "SECRET_TOKEN"),
        source={"SAFE_VALUE": "safe", "SECRET_TOKEN": secret},
        secret_values=(secret,),
    )
    result = make_runner(sanitizer=Sanitizer(secrets=(secret,))).run(
        make_request(
            tmp_path,
            "import os; print(os.environ.get('SECRET_TOKEN', 'missing')); print('secret-token')",
            environment=dict(environment),
        )
    )

    assert result.status is SubprocessStatus.NORMAL
    assert result.succeeded
    assert "missing" in result.stdout
    assert secret not in result.stdout


def test_local_runner_reports_nonzero_without_raising(tmp_path: Path) -> None:
    code = "import os,sys; os.write(1,b'out'); os.write(2,b'err'); sys.exit(7)"
    result = make_runner().run(make_request(tmp_path, code))

    assert result.status is SubprocessStatus.NONZERO
    assert result.returncode == 7
    assert result.stdout == "out"
    assert result.stderr == "err"
    assert result.cleanup_succeeded


def test_local_runner_reports_missing_executable_without_spawn(tmp_path: Path) -> None:
    request = SubprocessRequest(
        argv=(str(tmp_path / "missing-executable"),),
        cwd=tmp_path,
        environment={},
    )

    result = make_runner().run(request)

    assert result.status is SubprocessStatus.SPAWN_ERROR
    assert result.spawned is False
    assert result.spawn_error in {"FileNotFoundError", "PermissionError"}
    assert result.cleanup.status.value == "not_attempted"


def test_missing_required_capability_is_denied_before_spawn(tmp_path: Path) -> None:
    request = make_request(
        tmp_path,
        "raise SystemExit('must not spawn')",
        required_capabilities=(IsolationCapability.PROCESS_GROUP_TERMINATION,),
    )

    result = make_runner().run(request)

    assert result.status is SubprocessStatus.CAPABILITY_DENIED
    assert result.spawned is False
    assert result.missing_capabilities == (IsolationCapability.PROCESS_GROUP_TERMINATION,)


def test_timeout_terminates_reaps_direct_child_and_excludes_late_write(tmp_path: Path) -> None:
    code = (
        "import sys,time; print('before', flush=True); time.sleep(10); "
        "print('late', flush=True)"
    )
    result = make_runner(cleanup_timeout_seconds=1).run(
        make_request(tmp_path, code, timeout_seconds=0.05)
    )

    assert result.status is SubprocessStatus.TIMEOUT
    assert result.spawned
    assert result.cleanup.termination_requested
    assert result.cleanup.reaped
    assert result.cleanup_succeeded
    assert "before" in result.stdout
    assert "late" not in result.stdout


def test_cleanup_force_kills_when_terminate_is_ignored() -> None:
    process = _TerminateIgnoringProcess()
    cleanup = make_runner()._terminate_and_reap(process, time.monotonic() + 0.1)

    assert process.terminate_called
    assert process.kill_called
    assert cleanup.status.value == "terminated_and_reaped"
    assert cleanup.reaped


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM disposition is POSIX-specific")
def test_posix_timeout_force_kills_sigterm_ignoring_direct_child(tmp_path: Path) -> None:
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(10)"
    result = make_runner(cleanup_timeout_seconds=0.2).run(
        make_request(tmp_path, code, timeout_seconds=0.05)
    )

    assert result.status is SubprocessStatus.TIMEOUT
    assert result.cleanup.terminated
    assert result.cleanup.reaped
    assert result.cleanup_succeeded
    assert "direct-child kill fallback used" in result.cleanup.diagnostic


def test_cancellation_terminates_direct_child(tmp_path: Path) -> None:
    context = RunContext(run_id="subprocess-cancel")
    timer = Timer(0.05, context.cancel, args=("test cancellation",))
    timer.start()
    try:
        result = make_runner().run(
            make_request(tmp_path, "import time; time.sleep(10)", timeout_seconds=2),
            context,
        )
    finally:
        timer.join()

    assert result.status is SubprocessStatus.CANCELLED
    assert result.cleanup.reaped
    assert result.cleanup_succeeded


def test_run_deadline_wins_over_request_timeout(tmp_path: Path) -> None:
    result = make_runner().run(
        make_request(tmp_path, "import time; time.sleep(10)", timeout_seconds=2),
        RunContext(run_id="subprocess-deadline", timeout_seconds=0.05),
    )

    assert result.status is SubprocessStatus.DEADLINE_EXCEEDED
    assert result.cleanup.reaped
    assert result.cleanup_succeeded


def test_descendant_behavior_is_not_claimed_as_descendant_cleanup(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-marker.txt"
    descendant_code = (
        "import pathlib,sys,time; time.sleep(0.15); "
        "pathlib.Path(sys.argv[1]).write_text('descendant', encoding='utf-8')"
    )
    parent_code = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]], "
        "stdout=sys.stdout, stderr=sys.stderr, stdin=subprocess.DEVNULL); "
        "print('parent', flush=True)"
    )
    runner = make_runner(cleanup_timeout_seconds=0.05)
    result = runner.run(
        make_request(
            tmp_path,
            parent_code,
            timeout_seconds=1,
            arguments=(descendant_code, str(marker)),
        )
    )

    assert IsolationCapability.DESCENDANT_TREE_TERMINATION not in runner.capabilities
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.01)
    assert marker.read_text(encoding="utf-8") == "descendant"
    assert result.status is SubprocessStatus.NORMAL


@pytest.mark.skipif(
    os.name != "nt",
    reason="the cleanup-failure fixture relies on Windows inherited pipe handles",
)
def test_windows_descendant_pipe_can_make_cleanup_incomplete(tmp_path: Path) -> None:
    descendant_code = "import time; time.sleep(0.2)"
    parent_code = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[1]], "
        "stdout=sys.stdout, stderr=sys.stderr, stdin=subprocess.DEVNULL)"
    )
    started = time.monotonic()
    result = make_runner(cleanup_timeout_seconds=0.02).run(
        make_request(
            tmp_path,
            parent_code,
            timeout_seconds=1,
            arguments=(descendant_code,),
        )
    )

    assert result.status is SubprocessStatus.NORMAL
    assert result.cleanup.reaped
    assert result.cleanup_succeeded is False
    assert time.monotonic() - started < 0.5
    time.sleep(0.25)
