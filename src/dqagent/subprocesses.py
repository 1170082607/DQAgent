"""Bounded direct subprocess contracts and the initial local backend.

The local backend owns only the direct child it starts.  It deliberately does
not claim host filesystem, network, credential, syscall, workspace, process
group, or descendant-tree isolation.
"""

from __future__ import annotations

import math
import os
import subprocess
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import IO, Final, Protocol

from dqagent.errors import RunCancelledError, RunDeadlineExceededError
from dqagent.execution import RunContext

__all__ = [
    "CleanupResult",
    "CleanupStatus",
    "IsolationCapability",
    "LOCAL_SUBPROCESS_CAPABILITIES",
    "LOCAL_UNAVAILABLE_GUARANTEES",
    "LocalSubprocessRunner",
    "OutputSanitizer",
    "SubprocessCleanup",
    "SubprocessOutcome",
    "SubprocessRequest",
    "SubprocessResult",
    "SubprocessRunner",
    "SubprocessStatus",
    "build_minimal_environment",
    "construct_minimal_environment",
    "normalize_isolation_capabilities",
    "validate_isolation_capabilities",
]


class IsolationCapability(StrEnum):
    """A technical subprocess guarantee that a backend can declare and test."""

    DIRECT_ARGV = "direct_argv"
    WORKING_DIRECTORY_CONTROL = "working_directory_control"
    ALLOWLISTED_ENVIRONMENT = "allowlisted_environment"
    NO_STDIN = "no_stdin"
    WALL_TIME_LIMIT = "wall_time_limit"
    BOUNDED_OUTPUT = "bounded_output"
    DIRECT_CHILD_TERMINATION = "direct_child_termination"
    DIRECT_CHILD_REAP = "direct_child_reap"
    PROCESS_GROUP_TERMINATION = "process_group_termination"
    DESCENDANT_TREE_TERMINATION = "descendant_tree_termination"


def normalize_isolation_capabilities(
    capabilities: Iterable[IsolationCapability],
) -> frozenset[IsolationCapability]:
    """Validate and freeze a backend capability declaration."""

    if isinstance(capabilities, (str, bytes)):
        raise TypeError("isolation capabilities must be an iterable of IsolationCapability")
    try:
        values = tuple(capabilities)
    except TypeError as error:
        raise TypeError(
            "isolation capabilities must be an iterable of IsolationCapability"
        ) from error
    if any(not isinstance(value, IsolationCapability) for value in values):
        raise ValueError("isolation capabilities must contain only IsolationCapability values")
    return frozenset(values)


def validate_isolation_capabilities(
    capabilities: Iterable[IsolationCapability],
) -> frozenset[IsolationCapability]:
    """Compatibility-named validator used by action and backend contracts."""

    return normalize_isolation_capabilities(capabilities)


_MAX_ARGV_ITEMS: Final = 128
_MAX_ARGV_CHARACTERS: Final = 32_000
_MAX_ENV_ITEMS: Final = 128
_MAX_ENV_CHARACTERS: Final = 32_000
_MAX_STREAM_BYTES: Final = 4 * 1024 * 1024
_MAX_DIAGNOSTIC_CHARACTERS: Final = 256
_MAX_STREAM_CHUNK_BYTES: Final = 16 * 1024
_DEFAULT_TIMEOUT_SECONDS: Final = 30.0
_DEFAULT_OUTPUT_LIMIT_BYTES: Final = 32_000
_DEFAULT_CLEANUP_TIMEOUT_SECONDS: Final = 1.0
_DEFAULT_POLL_INTERVAL_SECONDS: Final = 0.005

_SENSITIVE_ENV_MARKERS: Final[tuple[str, ...]] = (
    "API_KEY",
    "APIKEY",
    "AUTH",
    "CREDENTIAL",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "PRIVATEKEY",
    "SECRET",
    "TOKEN",
)

LOCAL_SUBPROCESS_CAPABILITIES: Final[frozenset[IsolationCapability]] = frozenset(
    {
        IsolationCapability.DIRECT_ARGV,
        IsolationCapability.WORKING_DIRECTORY_CONTROL,
        IsolationCapability.ALLOWLISTED_ENVIRONMENT,
        IsolationCapability.NO_STDIN,
        IsolationCapability.WALL_TIME_LIMIT,
        IsolationCapability.BOUNDED_OUTPUT,
        IsolationCapability.DIRECT_CHILD_TERMINATION,
        IsolationCapability.DIRECT_CHILD_REAP,
    }
)

LOCAL_UNAVAILABLE_GUARANTEES: Final[tuple[str, ...]] = (
    "process_group_isolation",
    "descendant_tree_isolation",
    "host_filesystem_isolation",
    "network_isolation",
    "credential_isolation",
    "syscall_isolation",
    "workspace_only_isolation",
)


class OutputSanitizer(Protocol):
    """Structural adapter for the T1 literal secret/path sanitizer."""

    def sanitize(self, value: str, *, max_characters: int | None = None) -> str:
        """Return sanitized text without changing the sanitizer owner."""


def _normalize_bounded_argv(argv: Iterable[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)):
        raise TypeError("argv must be an iterable of argument strings")
    try:
        iterator = iter(argv)
    except TypeError as error:
        raise TypeError("argv must be an iterable of argument strings") from error

    values: list[str] = []
    characters = 0
    for index in range(_MAX_ARGV_ITEMS + 1):
        try:
            value = next(iterator)
        except StopIteration:
            break
        if index >= _MAX_ARGV_ITEMS:
            raise ValueError("argv exceeds its item bound")
        if not isinstance(value, str):
            raise TypeError("argv must contain only strings")
        if not value or "\x00" in value:
            raise ValueError("argv arguments must be non-empty and NUL-free")
        characters += len(value)
        if characters > _MAX_ARGV_CHARACTERS:
            raise ValueError("argv exceeds its character bound")
        values.append(value)
    if not values:
        raise ValueError("argv must not be empty")
    return tuple(values)


def _normalize_environment(environment: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(environment, Mapping):
        raise TypeError("environment must be a mapping of strings")
    try:
        if len(environment) > _MAX_ENV_ITEMS:
            raise ValueError("environment exceeds its item bound")
    except TypeError:
        pass

    values: dict[str, str] = {}
    characters = 0
    for index, (name, value) in enumerate(environment.items()):
        if index >= _MAX_ENV_ITEMS:
            raise ValueError("environment exceeds its item bound")
        if not isinstance(name, str) or not name or "=" in name or "\x00" in name:
            raise ValueError("environment names must be non-empty and not contain '=' or NUL")
        if not isinstance(value, str) or "\x00" in value:
            raise TypeError("environment values must be NUL-free strings")
        characters += len(name) + len(value)
        if characters > _MAX_ENV_CHARACTERS:
            raise ValueError("environment exceeds its character bound")
        values[name] = value
    return MappingProxyType(dict(sorted(values.items())))


def _normalize_names(values: Iterable[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be an iterable of names")
    try:
        iterator = iter(values)
    except TypeError as error:
        raise TypeError(f"{label} must be an iterable of names") from error

    normalized: list[str] = []
    for index in range(_MAX_ENV_ITEMS + 1):
        try:
            value = next(iterator)
        except StopIteration:
            break
        if index >= _MAX_ENV_ITEMS:
            raise ValueError(f"{label} exceeds its item bound")
        if not isinstance(value, str) or not value or "=" in value or "\x00" in value:
            raise ValueError(f"{label} contains an invalid name")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _normalize_secret_values(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("secret values must be an iterable of strings")
    try:
        iterator = iter(values)
    except TypeError as error:
        raise TypeError("secret values must be an iterable of strings") from error

    normalized: list[str] = []
    for index in range(_MAX_ENV_ITEMS + 1):
        try:
            value = next(iterator)
        except StopIteration:
            break
        if index >= _MAX_ENV_ITEMS:
            raise ValueError("secret values exceed their item bound")
        if not isinstance(value, str) or not value:
            raise ValueError("secret values must be non-empty strings")
        if "\x00" in value:
            raise ValueError("secret values must be NUL-free")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _looks_sensitive_environment_name(name: str) -> bool:
    upper_name = name.upper()
    return any(marker in upper_name for marker in _SENSITIVE_ENV_MARKERS)


def build_minimal_environment(
    allowlist: Mapping[str, str] | Iterable[str],
    *,
    source: Mapping[str, str] | None = None,
    secret_names: Iterable[str] = (),
    secret_values: Iterable[str] = (),
) -> Mapping[str, str]:
    """Construct a frozen environment from trusted inputs only.

    An iterable allowlist selects names from ``source`` (``os.environ`` only
    when the trusted caller explicitly omits ``source``).  A mapping is already
    a trusted name/value allowlist.  Unlisted host variables are never copied.
    Configured secret names/values and conservatively obvious secret names are
    omitted instead of being redacted after the child receives them.
    """

    configured_secret_names = frozenset(
        name.casefold() for name in _normalize_names(secret_names, "secret names")
    )
    configured_secret_values = _normalize_secret_values(secret_values)

    if isinstance(allowlist, Mapping):
        candidates = allowlist
    else:
        if isinstance(allowlist, (str, bytes)):
            raise TypeError("environment allowlist must be a mapping or iterable of names")
        names = _normalize_names(allowlist, "environment allowlist")
        selected_source = os.environ if source is None else source
        if not isinstance(selected_source, Mapping):
            raise TypeError("environment source must be a mapping of strings")
        candidates = {
            name: selected_source[name]
            for name in names
            if name in selected_source
        }

    selected: dict[str, str] = {}
    for index, (name, value) in enumerate(candidates.items()):
        if index >= _MAX_ENV_ITEMS:
            raise ValueError("environment allowlist exceeds its item bound")
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("environment allowlist must contain only string names and values")
        folded_name = name.casefold()
        if (
            not name
            or "=" in name
            or "\x00" in name
            or "\x00" in value
            or folded_name in configured_secret_names
            or _looks_sensitive_environment_name(name)
            or any(secret in value for secret in configured_secret_values)
        ):
            continue
        selected[name] = value
    return _normalize_environment(selected)


def construct_minimal_environment(
    allowlist: Mapping[str, str] | Iterable[str],
    *,
    source: Mapping[str, str] | None = None,
    secret_names: Iterable[str] = (),
    secret_values: Iterable[str] = (),
) -> Mapping[str, str]:
    """Compatibility-named wrapper for :func:`build_minimal_environment`."""

    return build_minimal_environment(
        allowlist,
        source=source,
        secret_names=secret_names,
        secret_values=secret_values,
    )


def _normalize_capability_iterable(
    capabilities: Iterable[IsolationCapability],
) -> frozenset[IsolationCapability]:
    if isinstance(capabilities, (str, bytes)):
        raise TypeError("required capabilities must be an iterable of IsolationCapability")
    try:
        iterator = iter(capabilities)
    except TypeError as error:
        raise TypeError(
            "required capabilities must be an iterable of IsolationCapability"
        ) from error
    values: list[IsolationCapability] = []
    for index in range(len(IsolationCapability) + 1):
        try:
            value = next(iterator)
        except StopIteration:
            break
        if index >= len(IsolationCapability):
            raise ValueError("required capabilities exceed their bound")
        values.append(value)
    return normalize_isolation_capabilities(values)


@dataclass(frozen=True, slots=True, init=False)
class SubprocessRequest:
    """Immutable, direct-argv request prepared by trusted composition."""

    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    timeout_seconds: float
    stdout_limit_bytes: int
    stderr_limit_bytes: int
    required_capabilities: frozenset[IsolationCapability]

    def __init__(
        self,
        argv: Iterable[str],
        cwd: Path,
        environment: Mapping[str, str] | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        stdout_limit_bytes: int = _DEFAULT_OUTPUT_LIMIT_BYTES,
        stderr_limit_bytes: int = _DEFAULT_OUTPUT_LIMIT_BYTES,
        required_capabilities: Iterable[IsolationCapability] = (),
        *,
        env: Mapping[str, str] | None = None,
        stdout_limit: int | None = None,
        stderr_limit: int | None = None,
        stdout_max_bytes: int | None = None,
        stderr_max_bytes: int | None = None,
    ) -> None:
        if environment is not None and env is not None:
            raise TypeError("provide environment or env, not both")
        selected_environment = environment if environment is not None else env
        if selected_environment is None:
            selected_environment = {}

        selected_stdout_limit = _resolve_alias_limit(
            "stdout", stdout_limit_bytes, stdout_limit, stdout_max_bytes
        )
        selected_stderr_limit = _resolve_alias_limit(
            "stderr", stderr_limit_bytes, stderr_limit, stderr_max_bytes
        )
        if not isinstance(cwd, Path):
            raise TypeError("cwd must be a pathlib.Path")
        try:
            canonical_cwd = cwd.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError("cwd must resolve to an existing directory") from error
        if not canonical_cwd.is_dir():
            raise ValueError("cwd must resolve to an existing directory")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout must be a finite number greater than zero")

        object.__setattr__(self, "argv", _normalize_bounded_argv(argv))
        object.__setattr__(self, "cwd", canonical_cwd)
        object.__setattr__(self, "environment", _normalize_environment(selected_environment))
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(
            self,
            "stdout_limit_bytes",
            _validate_stream_limit("stdout limit", selected_stdout_limit),
        )
        object.__setattr__(
            self,
            "stderr_limit_bytes",
            _validate_stream_limit("stderr limit", selected_stderr_limit),
        )
        object.__setattr__(
            self,
            "required_capabilities",
            _normalize_capability_iterable(required_capabilities),
        )

    @property
    def env(self) -> Mapping[str, str]:
        """Alias matching ``subprocess.Popen`` terminology."""

        return self.environment

    @property
    def stdout_limit(self) -> int:
        return self.stdout_limit_bytes

    @property
    def stderr_limit(self) -> int:
        return self.stderr_limit_bytes

    @property
    def stdout_max_bytes(self) -> int:
        return self.stdout_limit_bytes

    @property
    def stderr_max_bytes(self) -> int:
        return self.stderr_limit_bytes


def _resolve_alias_limit(
    label: str,
    canonical: int,
    short_alias: int | None,
    long_alias: int | None,
) -> int:
    values = [value for value in (short_alias, long_alias) if value is not None]
    if len(values) == 2 and values[0] != values[1]:
        raise TypeError(f"{label} limit aliases disagree")
    if values:
        return values[0]
    return canonical


def _validate_stream_limit(label: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0 or value > _MAX_STREAM_BYTES:
        raise ValueError(f"{label} is outside its bound")
    return value


class CleanupStatus(StrEnum):
    """Bounded cleanup evidence for the direct child and its output pipes."""

    NOT_ATTEMPTED = "not_attempted"
    REAPED = "reaped"
    TERMINATED_AND_REAPED = "terminated_and_reaped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CleanupResult:
    """Immutable lifecycle evidence with no unbounded exception text."""

    status: CleanupStatus = CleanupStatus.NOT_ATTEMPTED
    termination_requested: bool = False
    terminated: bool = False
    reaped: bool = False
    streams_drained: bool = False
    duration_seconds: float = 0.0
    diagnostic: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, CleanupStatus):
            raise TypeError("cleanup status must be a CleanupStatus")
        if not isinstance(self.duration_seconds, (int, float)) or isinstance(
            self.duration_seconds, bool
        ) or not math.isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise ValueError("cleanup duration must be a finite non-negative number")
        if (
            not isinstance(self.diagnostic, str)
            or len(self.diagnostic) > _MAX_DIAGNOSTIC_CHARACTERS
        ):
            raise ValueError("cleanup diagnostic exceeds its bound")

    @property
    def succeeded(self) -> bool:
        return self.status in {
            CleanupStatus.REAPED,
            CleanupStatus.TERMINATED_AND_REAPED,
        } and self.reaped and self.streams_drained


SubprocessCleanup = CleanupResult


class SubprocessStatus(StrEnum):
    """Externally meaningful outcomes of one bounded process attempt."""

    NORMAL = "normal"
    NONZERO = "nonzero"
    OUTPUT_SANITIZATION_ERROR = "output_sanitization_error"
    SPAWN_ERROR = "spawn_error"
    CAPABILITY_DENIED = "capability_denied"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"


SubprocessOutcome = SubprocessStatus


@dataclass(frozen=True, slots=True)
class SubprocessResult:
    """Bounded, immutable process output and lifecycle evidence."""

    status: SubprocessStatus
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_decode_replacements: int = 0
    stderr_decode_replacements: int = 0
    spawned: bool = False
    backend_identity: str = "unavailable"
    backend_capabilities: tuple[IsolationCapability, ...] = ()
    cleanup: CleanupResult = field(default_factory=CleanupResult)
    spawn_error: str | None = None
    diagnostic: str = ""
    missing_capabilities: tuple[IsolationCapability, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, SubprocessStatus):
            raise TypeError("subprocess status must be a SubprocessStatus")
        if self.returncode is not None and (
            isinstance(self.returncode, bool) or not isinstance(self.returncode, int)
        ):
            raise TypeError("return code must be an integer or None")
        for label, value in (("stdout", self.stdout), ("stderr", self.stderr)):
            if not isinstance(value, str):
                raise TypeError(f"{label} must be text")
            try:
                encoded_size = len(value.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError(f"{label} contains an invalid Unicode value") from error
            if encoded_size > _MAX_STREAM_BYTES:
                raise ValueError(f"{label} exceeds the result bound")
        if not isinstance(self.duration_seconds, (int, float)) or isinstance(
            self.duration_seconds, bool
        ) or not math.isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise ValueError("duration must be a finite non-negative number")
        for replacement_label, replacement_count in (
            ("stdout decode replacements", self.stdout_decode_replacements),
            ("stderr decode replacements", self.stderr_decode_replacements),
        ):
            if (
                isinstance(replacement_count, bool)
                or not isinstance(replacement_count, int)
                or replacement_count < 0
            ):
                raise ValueError(f"{replacement_label} must be a non-negative integer")
        if not isinstance(self.spawned, bool):
            raise TypeError("spawned must be a boolean")
        if (
            not isinstance(self.backend_identity, str)
            or not self.backend_identity.strip()
            or len(self.backend_identity) > _MAX_DIAGNOSTIC_CHARACTERS
        ):
            raise ValueError("backend identity must be non-empty text")
        if not isinstance(self.cleanup, CleanupResult):
            raise TypeError("cleanup must be a CleanupResult")
        if self.spawn_error is not None and (
            not isinstance(self.spawn_error, str)
            or len(self.spawn_error) > _MAX_DIAGNOSTIC_CHARACTERS
        ):
            raise ValueError("spawn error exceeds its bound")
        if (
            not isinstance(self.diagnostic, str)
            or len(self.diagnostic) > _MAX_DIAGNOSTIC_CHARACTERS
        ):
            raise ValueError("diagnostic exceeds its bound")
        object.__setattr__(
            self,
            "backend_capabilities",
            tuple(
                sorted(
                    normalize_isolation_capabilities(self.backend_capabilities),
                    key=lambda capability: capability.value,
                )
            ),
        )
        object.__setattr__(
            self,
            "missing_capabilities",
            tuple(
                sorted(
                    normalize_isolation_capabilities(self.missing_capabilities),
                    key=lambda capability: capability.value,
                )
            ),
        )

    @property
    def outcome(self) -> SubprocessStatus:
        return self.status

    @property
    def exit_code(self) -> int | None:
        return self.returncode

    @property
    def capabilities(self) -> tuple[IsolationCapability, ...]:
        return self.backend_capabilities

    @property
    def truncated(self) -> bool:
        return self.stdout_truncated or self.stderr_truncated

    @property
    def decode_replacements(self) -> bool:
        return bool(self.stdout_decode_replacements or self.stderr_decode_replacements)

    @property
    def cleanup_succeeded(self) -> bool:
        return self.cleanup.succeeded

    @property
    def succeeded(self) -> bool:
        return self.status is SubprocessStatus.NORMAL and self.returncode == 0


class SubprocessRunner(Protocol):
    """Provider-neutral port for one bounded direct process attempt."""

    @property
    def backend_identity(self) -> str:
        """Stable non-secret backend identity."""

    @property
    def capabilities(self) -> frozenset[IsolationCapability]:
        """Technical guarantees enforced by this backend."""

    @property
    def unavailable_guarantees(self) -> tuple[str, ...]:
        """Isolation guarantees deliberately not claimed by this backend."""

    def run(
        self,
        request: SubprocessRequest,
        context: RunContext | None = None,
    ) -> SubprocessResult:
        """Run one request without implicit shell or inherited stdin."""


class _BoundedByteCollector:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._data = bytearray()
        self.truncated = False

    def append(self, chunk: bytes) -> None:
        remaining = self._limit - len(self._data)
        if remaining > 0:
            self._data.extend(chunk[:remaining])
        if len(chunk) > max(remaining, 0):
            self.truncated = True

    @property
    def data(self) -> bytes:
        return bytes(self._data)


def _drain_stream(
    stream: IO[bytes],
    collector: _BoundedByteCollector,
    errors: list[str],
    chunk_size: int,
) -> None:
    try:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                return
            collector.append(chunk)
    except Exception as error:  # pragma: no cover - platform pipe behavior varies
        errors.append(type(error).__name__)


def _truncate_utf8(value: str, limit: int) -> tuple[str, bool]:
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        return "", True
    if len(encoded) <= limit:
        return value, False
    if limit == 0:
        return "", True
    prefix = encoded[:limit]
    while prefix:
        try:
            return prefix.decode("utf-8"), True
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return "", True


def _render_stream(
    collector: _BoundedByteCollector,
    limit: int,
    sanitizer: OutputSanitizer,
) -> tuple[str, int, bool, bool]:
    decoded_with_surrogates = collector.data.decode("utf-8", errors="surrogateescape")
    replacements = sum(
        0xDC80 <= ord(character) <= 0xDCFF for character in decoded_with_surrogates
    )
    decoded = (
        decoded_with_surrogates.encode("utf-8", errors="surrogateescape").decode(
            "utf-8", errors="replace"
        )
        if replacements
        else decoded_with_surrogates
    )
    sanitization_failed = False
    rendered = decoded
    try:
        rendered = sanitizer.sanitize(decoded)
        if not isinstance(rendered, str):
            raise TypeError("sanitizer returned non-text output")
    except Exception:  # pragma: no cover - sanitizer owner controls behavior
        rendered = ""
        sanitization_failed = True
    rendered, post_sanitize_truncation = _truncate_utf8(rendered, limit)
    return (
        rendered,
        replacements,
        collector.truncated or post_sanitize_truncation,
        sanitization_failed,
    )


def _control_status(context: RunContext | None) -> SubprocessStatus | None:
    if context is None:
        return None
    try:
        context.check_active()
    except RunCancelledError:
        return SubprocessStatus.CANCELLED
    except RunDeadlineExceededError:
        return SubprocessStatus.DEADLINE_EXCEEDED
    return None


class LocalSubprocessRunner:
    """Bounded local runner that owns and reaps only its direct child."""

    def __init__(
        self,
        *,
        sanitizer: OutputSanitizer | None = None,
        cleanup_timeout_seconds: float = _DEFAULT_CLEANUP_TIMEOUT_SECONDS,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        stream_chunk_bytes: int = _MAX_STREAM_CHUNK_BYTES,
    ) -> None:
        if (
            not math.isfinite(cleanup_timeout_seconds)
            or cleanup_timeout_seconds <= 0
        ):
            raise ValueError("cleanup timeout must be a finite number greater than zero")
        if not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0:
            raise ValueError("poll interval must be a finite number greater than zero")
        if (
            isinstance(stream_chunk_bytes, bool)
            or not isinstance(stream_chunk_bytes, int)
            or stream_chunk_bytes <= 0
            or stream_chunk_bytes > _MAX_STREAM_CHUNK_BYTES
        ):
            raise ValueError("stream chunk size is outside its bound")
        if sanitizer is None:
            raise ValueError("output sanitizer is required")
        if not callable(getattr(sanitizer, "sanitize", None)):
            raise TypeError("output sanitizer must provide a callable sanitize method")
        self._sanitizer = sanitizer
        self._cleanup_timeout_seconds = cleanup_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._stream_chunk_bytes = stream_chunk_bytes
        platform_name = "windows" if os.name == "nt" else "posix"
        self._backend_identity = f"local-{platform_name}-direct-child"

    @property
    def backend_identity(self) -> str:
        return self._backend_identity

    @property
    def capabilities(self) -> frozenset[IsolationCapability]:
        return LOCAL_SUBPROCESS_CAPABILITIES

    @property
    def unavailable_guarantees(self) -> tuple[str, ...]:
        return LOCAL_UNAVAILABLE_GUARANTEES

    def execute(
        self,
        request: SubprocessRequest,
        context: RunContext | None = None,
    ) -> SubprocessResult:
        """Alias for callers that use execute terminology."""

        return self.run(request, context)

    def run(
        self,
        request: SubprocessRequest,
        context: RunContext | None = None,
    ) -> SubprocessResult:
        if not isinstance(request, SubprocessRequest):
            raise TypeError("request must be a SubprocessRequest")
        started = time.monotonic()
        backend_capabilities = tuple(sorted(self.capabilities, key=lambda value: value.value))
        missing = request.required_capabilities.difference(self.capabilities)
        if missing:
            return self._result(
                started,
                SubprocessStatus.CAPABILITY_DENIED,
                backend_capabilities=backend_capabilities,
                missing_capabilities=tuple(missing),
                diagnostic="required subprocess capability unavailable",
            )

        initial_control = _control_status(context)
        if initial_control is not None:
            return self._result(
                started,
                initial_control,
                backend_capabilities=backend_capabilities,
                diagnostic="run control stopped process before spawn",
            )

        try:
            process = subprocess.Popen(
                request.argv,
                cwd=str(request.cwd),
                env=dict(request.environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                close_fds=True,
            )
        except (OSError, ValueError) as error:
            return self._result(
                started,
                SubprocessStatus.SPAWN_ERROR,
                backend_capabilities=backend_capabilities,
                spawn_error=type(error).__name__,
                diagnostic="process spawn failed",
            )

        assert process.stdout is not None
        assert process.stderr is not None
        stdout_collector = _BoundedByteCollector(request.stdout_limit_bytes)
        stderr_collector = _BoundedByteCollector(request.stderr_limit_bytes)
        stream_errors: list[str] = []
        stdout_thread = threading.Thread(
            target=_drain_stream,
            args=(process.stdout, stdout_collector, stream_errors, self._stream_chunk_bytes),
            name="dqagent-subprocess-stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_drain_stream,
            args=(process.stderr, stderr_collector, stream_errors, self._stream_chunk_bytes),
            name="dqagent-subprocess-stderr",
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        process_status: SubprocessStatus | None = None
        returncode: int | None = None
        process_deadline = started + request.timeout_seconds
        while True:
            control = _control_status(context)
            if control is not None:
                process_status = control
                break
            polled = process.poll()
            if polled is not None:
                late_control = _control_status(context)
                if late_control is not None:
                    process_status = late_control
                else:
                    returncode = polled
                    process_status = (
                        SubprocessStatus.NORMAL if polled == 0 else SubprocessStatus.NONZERO
                    )
                break
            remaining = process_deadline - time.monotonic()
            if remaining <= 0:
                process_status = SubprocessStatus.TIMEOUT
                break
            time.sleep(min(self._poll_interval_seconds, remaining))

        cleanup_started = time.monotonic()
        cleanup_deadline = cleanup_started + self._cleanup_timeout_seconds
        if process_status in {
            SubprocessStatus.TIMEOUT,
            SubprocessStatus.CANCELLED,
            SubprocessStatus.DEADLINE_EXCEEDED,
        }:
            cleanup = self._terminate_and_reap(process, cleanup_deadline)
        else:
            reaped, reap_error = self._reap(process, cleanup_deadline)
            cleanup = CleanupResult(
                status=CleanupStatus.REAPED if reaped else CleanupStatus.FAILED,
                reaped=reaped,
                diagnostic="reap failed" if reap_error else "",
            )
            if returncode is None:
                returncode = process.returncode

        streams_drained = self._join_streams(
            (stdout_thread, stderr_thread),
            (process.stdout, process.stderr),
            cleanup_deadline,
        )
        cleanup = self._finish_cleanup(cleanup, streams_drained, cleanup_started)
        stdout, stdout_replacements, stdout_truncated, stdout_sanitize_failed = _render_stream(
            stdout_collector,
            request.stdout_limit_bytes,
            self._sanitizer,
        )
        stderr, stderr_replacements, stderr_truncated, stderr_sanitize_failed = _render_stream(
            stderr_collector,
            request.stderr_limit_bytes,
            self._sanitizer,
        )
        diagnostic_parts: list[str] = []
        if stream_errors:
            diagnostic_parts.append("stream drain failed")
        if stdout_sanitize_failed or stderr_sanitize_failed:
            diagnostic_parts.append("output sanitization failed")
            process_status = SubprocessStatus.OUTPUT_SANITIZATION_ERROR
        if cleanup.status is CleanupStatus.FAILED:
            diagnostic_parts.append("cleanup incomplete")
        return self._result(
            started,
            process_status or SubprocessStatus.SPAWN_ERROR,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            stdout_decode_replacements=stdout_replacements,
            stderr_decode_replacements=stderr_replacements,
            spawned=True,
            backend_capabilities=backend_capabilities,
            cleanup=cleanup,
            diagnostic="; ".join(diagnostic_parts),
        )

    def _result(
        self,
        started: float,
        status: SubprocessStatus,
        *,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        stdout_truncated: bool = False,
        stderr_truncated: bool = False,
        stdout_decode_replacements: int = 0,
        stderr_decode_replacements: int = 0,
        spawned: bool = False,
        backend_capabilities: tuple[IsolationCapability, ...] = (),
        cleanup: CleanupResult | None = None,
        spawn_error: str | None = None,
        diagnostic: str = "",
        missing_capabilities: tuple[IsolationCapability, ...] = (),
    ) -> SubprocessResult:
        return SubprocessResult(
            status=status,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=max(0.0, time.monotonic() - started),
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            stdout_decode_replacements=stdout_decode_replacements,
            stderr_decode_replacements=stderr_decode_replacements,
            spawned=spawned,
            backend_identity=self.backend_identity,
            backend_capabilities=backend_capabilities,
            cleanup=cleanup or CleanupResult(),
            spawn_error=spawn_error,
            diagnostic=diagnostic,
            missing_capabilities=missing_capabilities,
        )

    def _terminate_and_reap(
        self,
        process: subprocess.Popen[bytes],
        deadline: float,
    ) -> CleanupResult:
        termination_error: str | None = None
        terminated = False
        try:
            process.terminate()
            terminated = True
        except (OSError, ValueError) as error:
            termination_error = type(error).__name__

        now = time.monotonic()
        remaining = max(0.0, deadline - now)
        grace_deadline = now + remaining / 2
        reaped, reap_error = self._reap(process, grace_deadline)
        force_kill = False
        if not reaped:
            try:
                process.kill()
                force_kill = True
            except (OSError, ValueError) as error:
                termination_error = termination_error or type(error).__name__
            reaped, final_reap_error = self._reap(process, deadline)
            reap_error = reap_error or final_reap_error

        diagnostic = ""
        if termination_error or reap_error:
            diagnostic = "direct-child cleanup failed"
        elif force_kill:
            diagnostic = "direct-child kill fallback used"
        return CleanupResult(
            status=(
                CleanupStatus.TERMINATED_AND_REAPED if reaped else CleanupStatus.FAILED
            ),
            termination_requested=True,
            terminated=terminated or force_kill,
            reaped=reaped,
            diagnostic=diagnostic,
        )

    @staticmethod
    def _reap(
        process: subprocess.Popen[bytes],
        deadline: float,
    ) -> tuple[bool, bool]:
        while time.monotonic() < deadline:
            try:
                process.wait(timeout=min(0.02, max(0.001, deadline - time.monotonic())))
                return True, False
            except subprocess.TimeoutExpired:
                continue
            except (OSError, ValueError):
                return False, True
        return process.poll() is not None, False

    @staticmethod
    def _join_streams(
        threads: tuple[threading.Thread, threading.Thread],
        streams: tuple[IO[bytes], IO[bytes]],
        deadline: float,
    ) -> bool:
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                thread.join(remaining)
        drained = all(not thread.is_alive() for thread in threads)
        if not drained:
            # A descendant may still own the inherited pipe handle.  Closing a
            # Windows pipe can itself block until that descendant exits, so do
            # not make a second synchronous close/wait attempt after the bound.
            # The daemon drainers retain the pipe only as residual cleanup
            # evidence; a finite descendant can release it after this result.
            drained = False
        return drained

    @staticmethod
    def _finish_cleanup(
        cleanup: CleanupResult,
        streams_drained: bool,
        started: float,
    ) -> CleanupResult:
        status = cleanup.status
        if not cleanup.reaped or not streams_drained:
            status = CleanupStatus.FAILED
        return CleanupResult(
            status=status,
            termination_requested=cleanup.termination_requested,
            terminated=cleanup.terminated,
            reaped=cleanup.reaped,
            streams_drained=streams_drained,
            duration_seconds=max(0.0, time.monotonic() - started),
            diagnostic=cleanup.diagnostic,
        )
