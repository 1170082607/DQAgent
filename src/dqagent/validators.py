"""Trusted, sequential validator execution for the coding harness.

Validators are application-owned checks.  They are deliberately not model
tools: the model cannot select, configure, approve, reorder, or retry them.
The runner reuses the bounded subprocess boundary and returns evidence that is
safe to retain in a coding result.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final

from dqagent.errors import RunCancelledError, RunDeadlineExceededError
from dqagent.execution import RunContext
from dqagent.subprocesses import (
    CleanupResult,
    IsolationCapability,
    LocalSubprocessRunner,
    SubprocessRequest,
    SubprocessResult,
    SubprocessRunner,
    SubprocessStatus,
    build_minimal_environment,
    normalize_isolation_capabilities,
)
from dqagent.workspace import Workspace, WorkspaceError, WorkspacePurpose

__all__ = [
    "TaskVerdict",
    "ValidatorDefinition",
    "ValidatorResult",
    "ValidatorRunner",
    "ValidatorStatus",
    "compute_task_verdict",
    "derive_task_verdict",
    "evaluate_validator_verdict",
    "run_validators",
]


_MAX_VALIDATORS: Final[int] = 128
_MAX_ARGV_ITEMS: Final[int] = 128
_MAX_ARGV_CHARACTERS: Final[int] = 32_000
_MAX_ARGUMENT_CHARACTERS: Final[int] = 8_192
_MAX_TIMEOUT_SECONDS: Final[float] = 86_400.0
_MAX_STREAM_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_RESULT_CHARACTERS: Final[int] = 128_000
_MAX_DIAGNOSTIC_CHARACTERS: Final[int] = 256
_MAX_DIAGNOSTICS: Final[int] = 8
_MAX_EXIT_CODES: Final[int] = 32


def _validate_identity(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or "\x00" in value
        or any(ord(character) < 32 for character in value)
        or "/" in value
        or "\\" in value
        or ":" in value
    ):
        raise ValueError(f"{label} must be a non-empty opaque identity")
    return value


def _normalize_argv(argv: Iterable[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)):
        raise TypeError("validator argv must be an iterable of strings")
    try:
        values = tuple(argv)
    except TypeError as error:
        raise TypeError("validator argv must be an iterable of strings") from error
    if not values or len(values) > _MAX_ARGV_ITEMS:
        raise ValueError("validator argv must be non-empty and bounded")
    if any(
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or len(value) > _MAX_ARGUMENT_CHARACTERS
        for value in values
    ):
        raise ValueError("validator argv arguments must be non-empty NUL-free strings")
    if sum(len(value) for value in values) > _MAX_ARGV_CHARACTERS:
        raise ValueError("validator argv exceeds its character bound")
    return values


def _normalize_cwd(value: str | PurePosixPath) -> PurePosixPath:
    raw = str(value) if isinstance(value, PurePosixPath) else value
    if not isinstance(raw, str) or not raw:
        raise ValueError("validator cwd must be non-empty text")
    if raw == ".":
        return PurePosixPath(".")
    if (
        "\x00" in raw
        or "\\" in raw
        or raw.startswith("/")
        or raw.startswith("//")
        or (len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha())
    ):
        raise ValueError("validator cwd must be a relative POSIX path")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("validator cwd contains an ambiguous or parent component")
    return PurePosixPath(*parts)


def _normalize_environment(value: Mapping[str, str] | None) -> Mapping[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("validator environment must be a mapping")
    normalized: dict[str, str] = {}
    for name, item in value.items():
        if (
            not isinstance(name, str)
            or not name
            or "=" in name
            or "\x00" in name
            or not isinstance(item, str)
            or "\x00" in item
        ):
            raise ValueError("validator environment must contain NUL-free strings")
        normalized[name] = item
    return MappingProxyType(dict(sorted(normalized.items())))


def _normalize_ignored_paths(
    values: Iterable[str | PurePosixPath],
) -> tuple[PurePosixPath, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("validator ignored paths must be an iterable")
    normalized: list[PurePosixPath] = []
    for value in values:
        path = _normalize_cwd(value)
        if path not in normalized:
            normalized.append(path)
    return tuple(normalized)


def _resolve_alias(
    canonical: int,
    aliases: tuple[int | None, ...],
    label: str,
) -> int:
    selected = tuple(value for value in aliases if value is not None)
    if len(selected) > 1 and len(set(selected)) != 1:
        raise TypeError(f"{label} aliases disagree")
    return selected[0] if selected else canonical


@dataclass(frozen=True, slots=True, init=False)
class ValidatorDefinition:
    """Trusted definition for one direct-argv validator."""

    validator_id: str
    argv: tuple[str, ...]
    cwd: PurePosixPath
    timeout_seconds: float
    stdout_limit_bytes: int
    stderr_limit_bytes: int
    accepted_exit_codes: frozenset[int]
    required_capabilities: frozenset[IsolationCapability]
    environment: Mapping[str, str]
    trusted_ignored_paths: tuple[PurePosixPath, ...]

    def __init__(
        self,
        validator_id: str | None = None,
        argv: Iterable[str] = (),
        cwd: str | PurePosixPath = ".",
        timeout_seconds: float = 30.0,
        stdout_limit_bytes: int = 32_000,
        stderr_limit_bytes: int = 32_000,
        accepted_exit_codes: Iterable[int] = (0,),
        required_capabilities: Iterable[IsolationCapability] = (),
        environment: Mapping[str, str] | None = None,
        trusted_ignored_paths: Iterable[str | PurePosixPath] = (),
        *,
        id: str | None = None,
        name: str | None = None,
        env: Mapping[str, str] | None = None,
        stdout_limit: int | None = None,
        stderr_limit: int | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        ignored_paths: Iterable[str | PurePosixPath] | None = None,
    ) -> None:
        selected_id = validator_id
        aliases = tuple(value for value in (id, name) if value is not None)
        if selected_id is not None and aliases and any(value != selected_id for value in aliases):
            raise TypeError("validator identity aliases disagree")
        if selected_id is None:
            if not aliases:
                raise TypeError("validator_id is required")
            selected_id = aliases[0]
        if environment is not None and env is not None:
            raise TypeError("provide environment or env, not both")
        selected_environment = environment if environment is not None else env
        selected_stdout = _resolve_alias(
            stdout_limit_bytes,
            (stdout_limit, max_stdout_bytes),
            "stdout limit",
        )
        selected_stderr = _resolve_alias(
            stderr_limit_bytes,
            (stderr_limit, max_stderr_bytes),
            "stderr limit",
        )
        selected_ignored = (
            trusted_ignored_paths
            if ignored_paths is None
            else ignored_paths
        )
        normalized_id = _validate_identity(selected_id, "validator ID")
        normalized_argv = _normalize_argv(argv)
        normalized_cwd = _normalize_cwd(cwd)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
            or float(timeout_seconds) > _MAX_TIMEOUT_SECONDS
        ):
            raise ValueError("validator timeout is outside its bound")
        for label, value in (
            ("stdout limit", selected_stdout),
            ("stderr limit", selected_stderr),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= _MAX_STREAM_BYTES
            ):
                raise ValueError(f"{label} is outside its bound")
        try:
            exit_codes = frozenset(accepted_exit_codes)
        except TypeError as error:
            raise TypeError("accepted exit codes must be an iterable of integers") from error
        if (
            not exit_codes
            or len(exit_codes) > _MAX_EXIT_CODES
            or any(isinstance(code, bool) or not isinstance(code, int) for code in exit_codes)
        ):
            raise ValueError("accepted exit codes are malformed or unbounded")
        normalized_capabilities = normalize_isolation_capabilities(required_capabilities)
        normalized_environment = _normalize_environment(selected_environment)
        normalized_ignored = _normalize_ignored_paths(selected_ignored)
        object.__setattr__(self, "validator_id", normalized_id)
        object.__setattr__(self, "argv", normalized_argv)
        object.__setattr__(self, "cwd", normalized_cwd)
        object.__setattr__(self, "timeout_seconds", float(timeout_seconds))
        object.__setattr__(self, "stdout_limit_bytes", selected_stdout)
        object.__setattr__(self, "stderr_limit_bytes", selected_stderr)
        object.__setattr__(self, "accepted_exit_codes", exit_codes)
        object.__setattr__(self, "required_capabilities", normalized_capabilities)
        object.__setattr__(self, "environment", normalized_environment)
        object.__setattr__(self, "trusted_ignored_paths", normalized_ignored)

    @property
    def id(self) -> str:
        return self.validator_id

    @property
    def name(self) -> str:
        return self.validator_id

    @property
    def env(self) -> Mapping[str, str]:
        return self.environment

    @property
    def stdout_limit(self) -> int:
        return self.stdout_limit_bytes

    @property
    def stderr_limit(self) -> int:
        return self.stderr_limit_bytes

    @property
    def max_stdout_bytes(self) -> int:
        return self.stdout_limit_bytes

    @property
    def max_stderr_bytes(self) -> int:
        return self.stderr_limit_bytes

    @property
    def ignored_paths(self) -> tuple[PurePosixPath, ...]:
        return self.trusted_ignored_paths


class ValidatorStatus(StrEnum):
    PASSED = "passed"
    PASS = "passed"
    FAILED = "failed"
    FAIL = "failed"
    UNAVAILABLE = "unavailable"
    TIMED_OUT = "timed_out"
    TIMEOUT = "timed_out"
    CANCELLED = "cancelled"
    CAPABILITY_MISSING = "capability_missing"
    ERROR = "error"


def _bounded_result_text(value: str, maximum: int = _MAX_RESULT_CHARACTERS) -> tuple[str, bool]:
    if len(value) <= maximum:
        return value, False
    return value[:maximum], True


@dataclass(frozen=True, slots=True)
class ValidatorResult:
    """Bounded evidence for one configured validator attempt."""

    validator_id: str
    status: ValidatorStatus
    argv: tuple[str, ...]
    cwd: PurePosixPath
    returncode: int | None = None
    duration_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_decode_replacements: int = 0
    stderr_decode_replacements: int = 0
    backend_identity: str = "unavailable"
    cleanup: CleanupResult = field(default_factory=CleanupResult)
    spawned: bool = False
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_identity(self.validator_id, "validator ID")
        if not isinstance(self.status, ValidatorStatus):
            raise TypeError("validator status must be a ValidatorStatus")
        if not isinstance(self.argv, tuple) or not all(isinstance(item, str) for item in self.argv):
            raise TypeError("validator result argv must be a tuple of strings")
        if not isinstance(self.cwd, PurePosixPath):
            raise TypeError("validator result cwd must be a PurePosixPath")
        if self.returncode is not None and (
            isinstance(self.returncode, bool) or not isinstance(self.returncode, int)
        ):
            raise TypeError("validator result returncode must be an integer or None")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(float(self.duration_seconds))
            or self.duration_seconds < 0
        ):
            raise ValueError("validator result duration is malformed")
        for label, text_value in (("stdout", self.stdout), ("stderr", self.stderr)):
            if not isinstance(text_value, str):
                raise TypeError(f"validator result {label} must be text")
            if len(text_value) > _MAX_RESULT_CHARACTERS:
                raise ValueError(f"validator result {label} exceeds its bound")
        boolean_fields: tuple[tuple[str, object], ...] = (
            ("stdout_truncated", self.stdout_truncated),
            ("stderr_truncated", self.stderr_truncated),
            ("spawned", self.spawned),
        )
        for label, boolean_value in boolean_fields:
            if not isinstance(boolean_value, bool):
                raise TypeError(f"validator result {label} must be a boolean")
        integer_fields: tuple[tuple[str, object], ...] = (
            ("stdout_decode_replacements", self.stdout_decode_replacements),
            ("stderr_decode_replacements", self.stderr_decode_replacements),
        )
        for label, integer_value in integer_fields:
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value < 0
            ):
                raise ValueError(f"validator result {label} is malformed")
        _validate_identity(self.backend_identity, "validator backend identity")
        if not isinstance(self.cleanup, CleanupResult):
            raise TypeError("validator result cleanup must be a CleanupResult")
        if not isinstance(self.diagnostics, tuple) or len(self.diagnostics) > _MAX_DIAGNOSTICS:
            raise ValueError("validator result diagnostics are unbounded")
        if any(
            not isinstance(value, str) or not value or len(value) > _MAX_DIAGNOSTIC_CHARACTERS
            for value in self.diagnostics
        ):
            raise ValueError("validator result diagnostics are malformed")

    @property
    def id(self) -> str:
        return self.validator_id

    @property
    def argv_identity(self) -> tuple[str, ...]:
        return self.argv

    @property
    def exit_code(self) -> int | None:
        return self.returncode

    @property
    def passed(self) -> bool:
        return self.status is ValidatorStatus.PASSED

    @property
    def failed(self) -> bool:
        return self.status is ValidatorStatus.FAILED

    @property
    def available(self) -> bool:
        return self.status not in {
            ValidatorStatus.UNAVAILABLE,
            ValidatorStatus.CAPABILITY_MISSING,
            ValidatorStatus.ERROR,
        }

    @property
    def cleanup_succeeded(self) -> bool:
        return self.cleanup.succeeded


def _safe_argv_identity(workspace: Workspace, argv: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for index, value in enumerate(argv):
        candidate = Path(value).name if index == 0 and Path(value).is_absolute() else value
        values.append(workspace.sanitize(candidate, max_characters=256))
    return tuple(values)


def _is_trusted_ignored_path(workspace: Workspace, path: PurePosixPath) -> bool:
    return any(
        path == rule or rule == PurePosixPath(".") or rule in path.parents
        for rule in workspace.scope.ignored_paths
    )


class ValidatorRunner:
    """Run trusted validator definitions sequentially on one subprocess port."""

    def __init__(
        self,
        workspace: Workspace,
        subprocess_runner: SubprocessRunner | None = None,
        *,
        secret_names: Iterable[str] = (),
        secret_values: Iterable[str] = (),
        max_validators: int = _MAX_VALIDATORS,
    ) -> None:
        if not isinstance(workspace, Workspace):
            raise TypeError("validator runner requires a Workspace")
        if (
            isinstance(max_validators, bool)
            or not isinstance(max_validators, int)
            or max_validators < 1
        ):
            raise ValueError("max_validators must be a positive integer")
        secrets = tuple(secret_values)
        self._workspace = workspace
        self._secret_names = tuple(secret_names)
        self._secret_values = secrets
        self._max_validators = max_validators
        self._runner = subprocess_runner or LocalSubprocessRunner(
            sanitizer=workspace.sanitizer(secrets=secrets),
        )
        if not callable(getattr(self._runner, "run", None)):
            raise TypeError("validator subprocess runner must provide run")
        self._capabilities = normalize_isolation_capabilities(
            getattr(self._runner, "capabilities", ())
        )
        identity = getattr(self._runner, "backend_identity", "unavailable")
        self._backend_identity = (
            identity if isinstance(identity, str) and identity.strip() else "unavailable"
        )

    @property
    def subprocess_runner(self) -> SubprocessRunner:
        return self._runner

    @property
    def backend_identity(self) -> str:
        return self._backend_identity

    @property
    def capabilities(self) -> frozenset[IsolationCapability]:
        return self._capabilities

    def run(
        self,
        definitions: Iterable[ValidatorDefinition],
        context: RunContext | None = None,
    ) -> tuple[ValidatorResult, ...]:
        if isinstance(definitions, (str, bytes)):
            raise TypeError("validator definitions must be an iterable")
        try:
            values = tuple(definitions)
        except TypeError as error:
            raise TypeError("validator definitions must be an iterable") from error
        if len(values) > self._max_validators:
            raise ValueError("validator definitions exceed the configured bound")
        if any(not isinstance(value, ValidatorDefinition) for value in values):
            raise TypeError("validator definitions must contain ValidatorDefinition values")

        results: list[ValidatorResult] = []
        for index, definition in enumerate(values):
            if context is not None:
                try:
                    context.check_active()
                except RunCancelledError:
                    results.extend(
                        self._not_started_results(values[index:], ValidatorStatus.CANCELLED)
                    )
                    break
                except RunDeadlineExceededError:
                    results.extend(
                        self._not_started_results(values[index:], ValidatorStatus.TIMED_OUT)
                    )
                    break
            results.append(self.run_one(definition, context=context))
        return tuple(results)

    def run_all(
        self,
        definitions: Iterable[ValidatorDefinition],
        context: RunContext | None = None,
    ) -> tuple[ValidatorResult, ...]:
        return self.run(definitions, context)

    def execute(
        self,
        definitions: Iterable[ValidatorDefinition],
        context: RunContext | None = None,
    ) -> tuple[ValidatorResult, ...]:
        return self.run(definitions, context)

    def run_one(
        self,
        definition: ValidatorDefinition,
        *,
        context: RunContext | None = None,
    ) -> ValidatorResult:
        if not isinstance(definition, ValidatorDefinition):
            raise TypeError("validator definition must be a ValidatorDefinition")
        for path in definition.trusted_ignored_paths:
            if not _is_trusted_ignored_path(self._workspace, path):
                return self._result(
                    definition,
                    ValidatorStatus.UNAVAILABLE,
                    diagnostics=("validator_artifact_path_not_ignored",),
                )
        if context is not None:
            try:
                context.check_active()
            except RunCancelledError:
                return self._result(definition, ValidatorStatus.CANCELLED)
            except RunDeadlineExceededError:
                return self._result(definition, ValidatorStatus.TIMED_OUT)

        missing = definition.required_capabilities.difference(self._capabilities)
        if missing:
            return self._result(
                definition,
                ValidatorStatus.CAPABILITY_MISSING,
                diagnostics=(
                    "required_capability_missing",
                    *tuple(
                        f"missing={item.value}"
                        for item in sorted(missing, key=lambda item: item.value)
                    ),
                ),
            )

        try:
            if definition.cwd == PurePosixPath("."):
                resolved_cwd = self._workspace.resolve_root(purpose=WorkspacePurpose.COMMAND_CWD)
            else:
                resolved_cwd = self._workspace.resolve(
                    definition.cwd,
                    purpose=WorkspacePurpose.COMMAND_CWD,
                )
            if not resolved_cwd.is_directory or resolved_cwd.followed_link:
                return self._result(
                    definition,
                    ValidatorStatus.UNAVAILABLE,
                    diagnostics=("validator_cwd_unavailable",),
                )
            environment = build_minimal_environment(
                definition.environment,
                secret_names=self._secret_names,
                secret_values=self._secret_values,
            )
            request = SubprocessRequest(
                argv=definition.argv,
                cwd=resolved_cwd.path,
                environment=environment,
                timeout_seconds=definition.timeout_seconds,
                stdout_limit_bytes=definition.stdout_limit_bytes,
                stderr_limit_bytes=definition.stderr_limit_bytes,
                required_capabilities=definition.required_capabilities,
            )
            result = self._runner.run(request, context)
        except (WorkspaceError, OSError, ValueError, TypeError):
            return self._result(
                definition,
                ValidatorStatus.UNAVAILABLE,
                diagnostics=("validator_spawn_configuration_unavailable",),
            )
        except Exception:
            return self._result(
                definition,
                ValidatorStatus.ERROR,
                diagnostics=("validator_runner_failure",),
            )
        return self._from_subprocess(definition, result)

    def _from_subprocess(
        self,
        definition: ValidatorDefinition,
        result: SubprocessResult,
    ) -> ValidatorResult:
        if result.status in {SubprocessStatus.NORMAL, SubprocessStatus.NONZERO}:
            status = (
                ValidatorStatus.PASSED
                if result.returncode in definition.accepted_exit_codes
                and result.cleanup_succeeded
                else ValidatorStatus.FAILED
            )
        elif result.status is SubprocessStatus.SPAWN_ERROR:
            status = ValidatorStatus.UNAVAILABLE
        elif result.status is SubprocessStatus.CAPABILITY_DENIED:
            status = ValidatorStatus.CAPABILITY_MISSING
        elif result.status is SubprocessStatus.TIMEOUT:
            status = ValidatorStatus.TIMED_OUT
        elif result.status is SubprocessStatus.CANCELLED:
            status = ValidatorStatus.CANCELLED
        elif result.status is SubprocessStatus.DEADLINE_EXCEEDED:
            status = ValidatorStatus.TIMED_OUT
        else:
            status = ValidatorStatus.ERROR
        diagnostics = [f"subprocess_status={result.status.value}"]
        if result.diagnostic:
            diagnostics.append(result.diagnostic)
        if result.spawn_error:
            diagnostics.append(f"spawn_error={result.spawn_error}")
        if result.missing_capabilities:
            diagnostics.extend(
                f"missing={item.value}" for item in result.missing_capabilities
            )
        return self._result(
            definition,
            status,
            subprocess_result=result,
            diagnostics=tuple(diagnostics[:_MAX_DIAGNOSTICS]),
        )

    def _result(
        self,
        definition: ValidatorDefinition,
        status: ValidatorStatus,
        *,
        subprocess_result: SubprocessResult | None = None,
        diagnostics: Iterable[str] = (),
    ) -> ValidatorResult:
        result = subprocess_result
        safe_stdout = "" if result is None else result.stdout
        safe_stderr = "" if result is None else result.stderr
        stdout_truncated = False if result is None else result.stdout_truncated
        stderr_truncated = False if result is None else result.stderr_truncated
        if result is not None:
            stdout_value, stdout_limited = _bounded_result_text(safe_stdout)
            stderr_value, stderr_limited = _bounded_result_text(safe_stderr)
            safe_stdout = stdout_value
            safe_stderr = stderr_value
            stdout_truncated = stdout_truncated or stdout_limited
            stderr_truncated = stderr_truncated or stderr_limited
        safe_diagnostics = tuple(
            self._workspace.sanitize(
                value,
                secrets=self._secret_values,
                max_characters=_MAX_DIAGNOSTIC_CHARACTERS,
            )
            for value in tuple(diagnostics)[:_MAX_DIAGNOSTICS]
            if isinstance(value, str) and value
        )
        return ValidatorResult(
            validator_id=definition.validator_id,
            status=status,
            argv=_safe_argv_identity(self._workspace, definition.argv),
            cwd=definition.cwd,
            returncode=None if result is None else result.returncode,
            duration_seconds=0.0 if result is None else result.duration_seconds,
            stdout=self._workspace.sanitize(
                safe_stdout,
                secrets=self._secret_values,
                max_characters=_MAX_RESULT_CHARACTERS,
            ),
            stderr=self._workspace.sanitize(
                safe_stderr,
                secrets=self._secret_values,
                max_characters=_MAX_RESULT_CHARACTERS,
            ),
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            stdout_decode_replacements=(
                0 if result is None else result.stdout_decode_replacements
            ),
            stderr_decode_replacements=(
                0 if result is None else result.stderr_decode_replacements
            ),
            backend_identity=(
                self._backend_identity if result is None else result.backend_identity
            ),
            cleanup=CleanupResult() if result is None else result.cleanup,
            spawned=False if result is None else result.spawned,
            diagnostics=safe_diagnostics,
        )

    def _not_started_results(
        self,
        definitions: Sequence[ValidatorDefinition],
        status: ValidatorStatus,
    ) -> tuple[ValidatorResult, ...]:
        return tuple(self._result(definition, status) for definition in definitions)


class TaskVerdict(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"
    NOT_VALIDATED = "not_validated"


def derive_task_verdict(
    validator_results: Sequence[ValidatorResult],
    *,
    target_observation_complete: bool = True,
    workspace_effect_unknown: bool = False,
    model_success_claim: bool | None = None,
) -> TaskVerdict:
    """Derive the harness verdict from evidence, ignoring model claims."""

    del model_success_claim
    if any(not isinstance(result, ValidatorResult) for result in validator_results):
        raise TypeError("validator results must contain ValidatorResult values")
    if not isinstance(target_observation_complete, bool) or not isinstance(
        workspace_effect_unknown, bool
    ):
        raise TypeError("verdict evidence flags must be booleans")
    if any(result.status is ValidatorStatus.FAILED for result in validator_results):
        return TaskVerdict.FAILED
    if (
        not target_observation_complete
        or workspace_effect_unknown
        or any(
            result.status
            in {
                ValidatorStatus.UNAVAILABLE,
                ValidatorStatus.CAPABILITY_MISSING,
                ValidatorStatus.TIMED_OUT,
                ValidatorStatus.CANCELLED,
                ValidatorStatus.ERROR,
            }
            for result in validator_results
        )
    ):
        return TaskVerdict.INDETERMINATE
    if not validator_results:
        return TaskVerdict.NOT_VALIDATED
    if all(result.status is ValidatorStatus.PASSED for result in validator_results):
        return TaskVerdict.PASSED
    return TaskVerdict.INDETERMINATE


def compute_task_verdict(
    validator_results: Sequence[ValidatorResult],
    *,
    target_observation_complete: bool = True,
    workspace_effect_unknown: bool = False,
    model_success_claim: bool | None = None,
) -> TaskVerdict:
    return derive_task_verdict(
        validator_results,
        target_observation_complete=target_observation_complete,
        workspace_effect_unknown=workspace_effect_unknown,
        model_success_claim=model_success_claim,
    )


def evaluate_validator_verdict(
    validator_results: Sequence[ValidatorResult],
    *,
    target_observation_complete: bool = True,
    workspace_effect_unknown: bool = False,
    model_success_claim: bool | None = None,
) -> TaskVerdict:
    return derive_task_verdict(
        validator_results,
        target_observation_complete=target_observation_complete,
        workspace_effect_unknown=workspace_effect_unknown,
        model_success_claim=model_success_claim,
    )


def run_validators(
    workspace: Workspace,
    definitions: Iterable[ValidatorDefinition],
    *,
    subprocess_runner: SubprocessRunner | None = None,
    context: RunContext | None = None,
    secret_names: Iterable[str] = (),
    secret_values: Iterable[str] = (),
) -> tuple[ValidatorResult, ...]:
    """Convenience entry point for a harness-owned validator sequence."""

    return ValidatorRunner(
        workspace,
        subprocess_runner,
        secret_names=secret_names,
        secret_values=secret_values,
    ).run(definitions, context)
