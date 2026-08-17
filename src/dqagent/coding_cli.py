"""Foreground ``dqagent-code`` command-line composition."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv

from dqagent.coding import (
    CodingFailureEvidence,
    CodingRequest,
    CodingRunResult,
    ForegroundApprovalProvider,
    create_coding_agent_application,
)
from dqagent.config import ModelProvider, Settings
from dqagent.errors import DQAgentError
from dqagent.providers import create_llm_client
from dqagent.tool_governance import (
    NonInteractiveApprovalProvider as SafeNonInteractiveApprovalProvider,
)
from dqagent.validators import ValidatorDefinition
from dqagent.workspace import Workspace, WorkspaceBlindSpot, WorkspaceScope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dqagent-code",
        description="Run one foreground, workspace-scoped coding-agent task.",
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("-m", "--message", required=True)
    parser.add_argument("--target", dest="targets", action="append", required=True)
    parser.add_argument("--skill", dest="skills", action="append", default=[])
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in ModelProvider],
        help="Override DQAGENT_PROVIDER.",
    )
    parser.add_argument("--model", help="Override DQAGENT_MODEL.")
    parser.add_argument("--base-url", help="Override the selected provider base URL.")
    parser.add_argument("--timeout", type=float, help="Override DQAGENT_TIMEOUT_SECONDS.")
    parser.add_argument(
        "--run-timeout",
        type=float,
        help="Override DQAGENT_RUN_TIMEOUT_SECONDS.",
    )
    parser.add_argument(
        "--max-model-attempts",
        type=int,
        help="Override DQAGENT_MAX_MODEL_ATTEMPTS.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail closed if an action requires foreground approval.",
    )
    parser.add_argument(
        "--allow-executable",
        action="append",
        default=[],
        metavar="IDENTITY=PATH",
        help="Trusted direct executable mapping for workspace_command; repeatable.",
    )
    parser.add_argument(
        "--validator",
        action="append",
        default=[],
        metavar="ID=JSON_ARGV",
        help="Trusted validator, for example check=[\"python\",\"-m\",\"pytest\"].",
    )
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    environ = dict(os.environ)
    if args.provider:
        environ["DQAGENT_PROVIDER"] = args.provider
    if args.model:
        environ["DQAGENT_MODEL"] = args.model
    if args.base_url:
        environ["DQAGENT_BASE_URL"] = args.base_url
    if args.timeout is not None:
        environ["DQAGENT_TIMEOUT_SECONDS"] = str(args.timeout)
    if args.run_timeout is not None:
        environ["DQAGENT_RUN_TIMEOUT_SECONDS"] = str(args.run_timeout)
    if args.max_model_attempts is not None:
        environ["DQAGENT_MAX_MODEL_ATTEMPTS"] = str(args.max_model_attempts)
    return Settings.from_env(environ)


def _parse_executable_allowlist(values: Sequence[str]) -> Mapping[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        identity, separator, path = raw.partition("=")
        if not separator or not identity.strip() or not path.strip():
            raise ValueError("--allow-executable must use IDENTITY=PATH")
        if identity in result:
            raise ValueError(f"duplicate executable identity: {identity}")
        result[identity] = Path(path)
    return result


def _parse_validators(values: Sequence[str]) -> tuple[ValidatorDefinition, ...]:
    definitions: list[ValidatorDefinition] = []
    for raw in values:
        identity, separator, encoded_argv = raw.partition("=")
        if not separator or not identity.strip() or not encoded_argv.strip():
            raise ValueError("--validator must use ID=JSON_ARGV")
        try:
            argv = json.loads(encoded_argv)
        except json.JSONDecodeError as error:
            raise ValueError("--validator JSON_ARGV is invalid JSON") from error
        if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
            raise ValueError("--validator JSON_ARGV must be a JSON string array")
        definitions.append(ValidatorDefinition(identity, tuple(argv)))
    return tuple(definitions)


def _safe_text(
    workspace: Workspace,
    value: str,
    *,
    maximum: int,
    secret_values: Sequence[str] = (),
) -> str:
    return workspace.sanitize(value, secrets=secret_values, max_characters=maximum)


def _render_result(
    result: CodingRunResult,
    *,
    workspace: Workspace,
    secret_values: Sequence[str],
    output: TextIO,
) -> None:
    diff = _safe_text(
        workspace,
        result.diff.rendered_diff,
        maximum=workspace.scope.limits.max_rendered_diff_characters,
        secret_values=secret_values,
    )
    output.write(f"run_id: {result.run_id}\n")
    output.write(f"verdict: {result.verdict.value}\n")
    output.write("diff:\n")
    output.write(f"{diff if diff else '(no observed task changes)'}\n")
    output.write("validators:\n")
    if not result.validator_results:
        output.write("  (none configured)\n")
    for validator in result.validator_results:
        output.write(
            f"  {validator.validator_id}: {validator.status.value} "
            f"(exit={validator.returncode if validator.returncode is not None else 'none'})\n"
        )
        if validator.stdout:
            safe_stdout = _safe_text(
                workspace,
                validator.stdout,
                maximum=4_096,
                secret_values=secret_values,
            )
            output.write(f"    stdout: {safe_stdout}\n")
        if validator.stderr:
            safe_stderr = _safe_text(
                workspace,
                validator.stderr,
                maximum=4_096,
                secret_values=secret_values,
            )
            output.write(f"    stderr: {safe_stderr}\n")
    output.write("blind_spots:\n")
    if not result.blind_spots:
        output.write("  (none reported)\n")
    for blind_spot in result.blind_spots:
        if isinstance(blind_spot, WorkspaceBlindSpot):
            output.write(
                f"  workspace:{blind_spot.logical_path.as_posix()}:{blind_spot.reason_code}\n"
            )
        else:
            output.write(
                f"  {_safe_text(workspace, blind_spot, maximum=256, secret_values=secret_values)}\n"
            )


def _render_failure(
    error: DQAgentError,
    *,
    workspace: Workspace | None,
    secret_values: Sequence[str] = (),
    output: TextIO,
) -> None:
    evidence = getattr(error, "coding_failure_evidence", None)
    output.write(f"error_type: {type(error).__name__}\n")
    output.write(f"error_category: {error.category.value}\n")
    if workspace is not None:
        safe_error = _safe_text(
            workspace,
            str(error),
            maximum=512,
            secret_values=secret_values,
        )
        output.write(
            f"error: {safe_error}\n"
        )
    else:
        output.write("error: [unavailable before workspace composition]\n")
    if isinstance(evidence, CodingFailureEvidence):
        output.write("failure_evidence: bounded\n")
        output.write(f"action_record_count: {len(evidence.action_records)}\n")
        output.write(f"final_snapshot_observed: {evidence.final is not None}\n")
        output.write(f"diff_observed: {evidence.diff is not None}\n")
        output.write("rollback_claimed: false\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace: Workspace | None = None
    secret_values: tuple[str, ...] = ()
    try:
        load_dotenv()
        settings = _settings_from_args(args)
        workspace = Workspace(WorkspaceScope("cli-workspace", args.workspace))
        executable_allowlist = _parse_executable_allowlist(args.allow_executable)
        validators = _parse_validators(args.validator)
        non_interactive = args.non_interactive or not sys.stdin.isatty()
        approval_provider = (
            SafeNonInteractiveApprovalProvider()
            if non_interactive
            else ForegroundApprovalProvider(output=sys.stderr)
        )
        secret_values = () if settings.api_key == "local" else (settings.api_key,)
        application = create_coding_agent_application(
            workspace,
            create_llm_client(settings),
            approval_provider=approval_provider,
            executable_allowlist=executable_allowlist,
            validators=validators,
            secret_values=secret_values,
        )
        result = application.run(
            CodingRequest(
                args.message,
                tuple(args.targets),
                tuple(args.skills),
            )
        )
        _render_result(
            result,
            workspace=workspace,
            secret_values=secret_values,
            output=sys.stdout,
        )
        return 0 if result.verdict.value == "passed" else 1
    except DQAgentError as error:
        _render_failure(
            error,
            workspace=workspace,
            secret_values=secret_values,
            output=sys.stderr,
        )
        return 1
    except (OSError, TypeError, ValueError) as error:
        if workspace is not None:
            print(
                "error: "
                f"{_safe_text(workspace, str(error), maximum=512, secret_values=secret_values)}",
                file=sys.stderr,
            )
        else:
            print(f"error_type: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
