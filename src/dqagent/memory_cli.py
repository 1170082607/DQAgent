"""Independent, model-free command-line management for explicit memory."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from dqagent.errors import (
    MemoryAdmissionDeniedError,
    MemoryConflictError,
    MemoryDigestMismatchError,
    MemoryNotFoundError,
    MemoryServiceError,
    MemoryValidationError,
)
from dqagent.errors import MemoryError as DQMemoryError
from dqagent.memory import (
    AdmissionAction,
    MemoryCandidate,
    MemoryConfidence,
    MemoryKind,
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
    MemoryScopeKind,
    MemorySensitivity,
    MemorySourceType,
    MemoryTombstone,
)
from dqagent.memory_policy import DefaultMemoryPolicy
from dqagent.memory_service import (
    MemoryConfirmResult,
    MemoryCorrectionResult,
    MemoryForgetResult,
    MemoryListResult,
    MemoryProposal,
    MemoryService,
    MemoryShowResult,
)
from dqagent.memory_store import InMemoryMemoryStore, SqliteMemoryStore

DEFAULT_DATABASE_PATH = Path(".local/memory.sqlite3")
DEFAULT_EXTRACTOR_IDENTITY = "dqagent-memory-cli-v1"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_DECLINED = 3
EXIT_POLICY_DENIED = 4

_CONFIRM_WORDS = frozenset({"confirm", "yes"})

__all__ = [
    "DEFAULT_DATABASE_PATH",
    "EXIT_DECLINED",
    "EXIT_ERROR",
    "EXIT_OK",
    "EXIT_POLICY_DENIED",
    "EXIT_USAGE",
    "build_parser",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone parser without loading settings or a model provider."""

    parser = argparse.ArgumentParser(
        prog="dqagent-memory",
        description="Inspect and explicitly manage durable DQAgent memory.",
    )
    parser.add_argument(
        "--database",
        "--db",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="SQLite database path (default: .local/memory.sqlite3).",
    )
    _add_scope_arguments(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    remember = subparsers.add_parser(
        "remember",
        help="Preview and explicitly confirm one new memory candidate.",
    )
    _add_command_options(remember)
    _add_candidate_arguments(remember)

    list_command = subparsers.add_parser(
        "list",
        help="List all lifecycle records and content-free tombstones in one scope.",
    )
    _add_command_options(list_command)

    show = subparsers.add_parser(
        "show",
        help="Show one memory record by exact ID and scope.",
    )
    _add_command_options(show)
    _add_memory_id_argument(show)

    correct = subparsers.add_parser(
        "correct",
        help="Preview and explicitly confirm a correction for one exact memory ID.",
    )
    _add_command_options(correct)
    _add_memory_id_argument(correct)
    _add_candidate_arguments(correct)

    forget = subparsers.add_parser(
        "forget",
        help="Show and explicitly confirm forgetting one exact memory ID.",
    )
    _add_command_options(forget)
    _add_memory_id_argument(forget)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit memory management operation and return a stable exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    scope = _scope_from_args(parser, args)
    database = _database_from_args(args)
    operation = str(args.command)

    try:
        if operation == "remember":
            return _remember(args, scope, database)
        if operation == "list":
            return _list(scope, database)
        if operation == "show":
            return _show(args, scope, database, parser)
        if operation == "correct":
            return _correct(args, scope, database, parser)
        if operation == "forget":
            return _forget(args, scope, database, parser)
    except DQMemoryError as error:
        return _report_error(error, operation=operation, scope=scope)
    except Exception:
        # Do not expose dependency messages: a store or parser dependency may accidentally
        # include a memory payload in its exception text.
        return _report_internal_error(operation=operation, scope=scope)

    return _report_internal_error(operation=operation, scope=scope)


def _add_command_options(parser: argparse.ArgumentParser) -> None:
    # Suppressed defaults let the same explicit options work before or after the subcommand.
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        type=Path,
        default=argparse.SUPPRESS,
        help="SQLite database path.",
    )
    _add_scope_arguments(parser, suppress_default=True)


def _add_scope_arguments(
    parser: argparse.ArgumentParser,
    *,
    suppress_default: bool = False,
) -> None:
    default: object = argparse.SUPPRESS if suppress_default else None
    parser.add_argument(
        "--scope-kind",
        choices=[kind.value for kind in MemoryScopeKind],
        default=default,
        help="Explicit memory owner kind: user or project.",
    )
    parser.add_argument(
        "--scope-id",
        default=default,
        help="Explicit memory owner ID; never inferred from a session or payload.",
    )


def _add_memory_id_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "memory_id",
        nargs="?",
        help="Exact target memory ID.",
    )
    parser.add_argument(
        "--memory-id",
        dest="memory_id_option",
        help="Exact target memory ID.",
    )


def _add_candidate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--kind",
        required=True,
        choices=[kind.value for kind in MemoryKind],
        help="Memory proposition kind.",
    )
    parser.add_argument("--topic", required=True, help="Stable proposition topic.")
    parser.add_argument("--content", required=True, help="Exact candidate content.")
    parser.add_argument(
        "--confidence",
        type=float,
        default=1.0,
        help="Extractor confidence metadata (default: 1.0).",
    )
    parser.add_argument(
        "--sensitivity",
        choices=[sensitivity.value for sensitivity in MemorySensitivity],
        default=MemorySensitivity.NON_SENSITIVE.value,
        help="Candidate sensitivity classification; policy cannot be bypassed.",
    )
    parser.add_argument(
        "--source-type",
        choices=[source_type.value for source_type in MemorySourceType],
        default=MemorySourceType.USER_DRAFT.value,
        help="Provenance source type (default: user_draft).",
    )
    parser.add_argument(
        "--source-item-digest",
        help="SHA-256 digest of the referenced source item; defaults to the draft digest.",
    )
    parser.add_argument(
        "--extractor-identity",
        default=DEFAULT_EXTRACTOR_IDENTITY,
        help="Provenance extractor identity.",
    )
    parser.add_argument(
        "--extracted-at",
        type=_parse_datetime,
        help="Timezone-aware ISO-8601 extraction timestamp (default: now).",
    )
    parser.add_argument("--source-id", help="Source reference ID for committed provenance.")
    parser.add_argument(
        "--source-revision",
        type=_parse_positive_int,
        help="Source revision for committed session-turn provenance.",
    )
    parser.add_argument("--run-id", help="Optional extraction run reference ID.")
    parser.add_argument(
        "--valid-from",
        type=_parse_datetime,
        help="Timezone-aware ISO-8601 validity start (default: now).",
    )
    parser.add_argument(
        "--expires-at",
        type=_parse_datetime,
        help="Optional timezone-aware ISO-8601 expiry timestamp.",
    )


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an ISO-8601 datetime") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("datetime must include a timezone")
    return parsed


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _scope_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> MemoryScope:
    scope_kind = getattr(args, "scope_kind", None)
    scope_id = getattr(args, "scope_id", None)
    if scope_kind is None or scope_id is None:
        parser.error("--scope-kind and --scope-id are required for every command")
    try:
        return MemoryScope(MemoryScopeKind(scope_kind), scope_id)
    except (MemoryValidationError, ValueError):
        parser.error("invalid explicit memory scope")


def _database_from_args(args: argparse.Namespace) -> Path:
    database = getattr(args, "database", DEFAULT_DATABASE_PATH)
    if not isinstance(database, Path):
        raise MemoryValidationError("memory database path must be a Path")
    return database


def _preview_service() -> MemoryService:
    return MemoryService(InMemoryMemoryStore(), DefaultMemoryPolicy())


def _persistent_service(database: Path) -> MemoryService:
    return MemoryService(SqliteMemoryStore(database), DefaultMemoryPolicy())


def _remember(args: argparse.Namespace, scope: MemoryScope, database: Path) -> int:
    candidate = _candidate_from_args(args, scope)
    preview = _preview_service().preview(candidate, scope=scope)
    if preview.decision.action is AdmissionAction.DENY:
        return _report_policy_denial(preview, operation="remember")

    _print_candidate_preview(preview, operation="remember")
    if not _request_confirmation("confirm remember? type 'yes' or 'confirm': "):
        return EXIT_DECLINED

    result = _persistent_service(database).confirm(
        candidate,
        preview.candidate_digest,
        scope=scope,
    )
    _print_confirmed(result)
    return EXIT_OK


def _list(scope: MemoryScope, database: Path) -> int:
    result = _persistent_service(database).list(scope)
    _print_list(result)
    return EXIT_OK


def _show(
    args: argparse.Namespace,
    scope: MemoryScope,
    database: Path,
    parser: argparse.ArgumentParser,
) -> int:
    memory_id = _target_id(args, parser)
    result = _persistent_service(database).show(scope, memory_id)
    _print_show(result)
    return EXIT_OK


def _correct(
    args: argparse.Namespace,
    scope: MemoryScope,
    database: Path,
    parser: argparse.ArgumentParser,
) -> int:
    memory_id = _target_id(args, parser)
    candidate = _candidate_from_args(args, scope)
    preview = _preview_service().preview(candidate, scope=scope)
    if preview.decision.action is AdmissionAction.DENY:
        return _report_policy_denial(preview, operation="correct", memory_id=memory_id)

    print(f"target_memory_id={_safe_text(memory_id)}")
    _print_candidate_preview(preview, operation="correct")
    if not _request_confirmation("confirm correct? type 'yes' or 'confirm': "):
        return EXIT_DECLINED

    result = _persistent_service(database).correct(
        scope,
        memory_id,
        candidate,
        candidate_digest=preview.candidate_digest,
    )
    _print_corrected(result)
    return EXIT_OK


def _forget(
    args: argparse.Namespace,
    scope: MemoryScope,
    database: Path,
    parser: argparse.ArgumentParser,
) -> int:
    memory_id = _target_id(args, parser)
    service = _persistent_service(database)
    target = service.show(scope, memory_id, materialize_expiry=False)
    print("forget_target=memory")
    _print_record(target.record)
    if not _request_confirmation("confirm forget? type 'yes' or 'confirm': "):
        return EXIT_DECLINED

    result = service.forget(scope, memory_id)
    _print_forgotten(result)
    return EXIT_OK


def _candidate_from_args(args: argparse.Namespace, scope: MemoryScope) -> MemoryCandidate:
    now = datetime.now(UTC)
    content = str(args.content)
    source_item_digest = args.source_item_digest or hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()
    provenance = MemoryProvenance(
        source_type=MemorySourceType(args.source_type),
        source_item_digest=source_item_digest,
        extractor_identity=args.extractor_identity,
        extracted_at=args.extracted_at or now,
        source_id=args.source_id,
        source_revision=args.source_revision,
        run_id=args.run_id,
    )
    return MemoryCandidate(
        scope=scope,
        kind=MemoryKind(args.kind),
        topic=args.topic,
        content=content,
        confidence=MemoryConfidence(args.confidence),
        sensitivity=MemorySensitivity(args.sensitivity),
        provenance=provenance,
        valid_from=args.valid_from or now,
        expires_at=args.expires_at,
    )


def _target_id(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    positional = getattr(args, "memory_id", None)
    option = getattr(args, "memory_id_option", None)
    if positional and option and positional != option:
        parser.error("positional memory ID and --memory-id must match")
    memory_id = option or positional
    if not isinstance(memory_id, str) or not memory_id:
        parser.error("an exact memory ID is required")
    return memory_id


def _request_confirmation(prompt: str) -> bool:
    print("confirmation_required=true")
    try:
        response = input(prompt)
    except EOFError:
        print("confirmation=declined reason=eof")
        return False
    except KeyboardInterrupt:
        print("confirmation=declined reason=interrupt")
        return False
    if response.strip().casefold() in _CONFIRM_WORDS:
        print("confirmation=accepted")
        return True
    print("confirmation=declined reason=explicit_rejection")
    return False


def _print_candidate_preview(preview: MemoryProposal, *, operation: str) -> None:
    candidate = preview.candidate
    provenance = candidate.provenance
    print(f"candidate_operation={operation}")
    print("candidate_type=transient")
    print(f"candidate_schema_version={candidate.schema_version}")
    print(f"scope_kind={candidate.scope.kind.value}")
    print(f"scope_id={_safe_text(candidate.scope.scope_id)}")
    print(f"kind={_safe_text(candidate.kind.value)}")
    print(f"topic={_safe_text(candidate.topic)}")
    print(f"content={_safe_text(candidate.content)}")
    print(f"confidence={candidate.confidence.value}")
    print(f"sensitivity={candidate.sensitivity.value}")
    print(f"valid_from={candidate.valid_from.isoformat()}")
    print(f"expires_at={_optional_datetime(candidate.expires_at)}")
    print(f"provenance.source_type={provenance.source_type.value}")
    print(f"provenance.source_item_digest={provenance.source_item_digest}")
    print(f"provenance.extractor_identity={_safe_text(provenance.extractor_identity)}")
    print(f"provenance.extracted_at={provenance.extracted_at.isoformat()}")
    print(f"provenance.source_id={_optional_text(provenance.source_id)}")
    print(f"provenance.source_revision={_optional_value(provenance.source_revision)}")
    print(f"provenance.run_id={_optional_text(provenance.run_id)}")
    print(f"provenance.model_identity={_optional_text(provenance.model_identity)}")
    print(f"provenance.response_identity={_optional_text(provenance.response_identity)}")
    print(f"candidate_digest={preview.candidate_digest}")
    print(f"policy.action={preview.decision.action.value}")
    print(f"policy.reason={preview.decision.reason.value}")


def _print_record(record: MemoryRecord) -> None:
    provenance = record.provenance
    print("memory_record=true")
    print(f"memory_id={_safe_text(record.memory_id)}")
    print(f"scope_kind={_safe_text(record.scope.kind.value)}")
    print(f"scope_id={_safe_text(record.scope.scope_id)}")
    print(f"kind={_safe_text(record.kind.value)}")
    print(f"topic={_safe_text(record.topic)}")
    print(f"content={_safe_text(record.content)}")
    print(f"confidence={record.confidence.value}")
    print(f"sensitivity={record.sensitivity.value}")
    print(f"lifecycle.status={record.status.value}")
    print(f"revision={record.revision}")
    print(f"supersedes_id={_optional_text(record.supersedes_id)}")
    print(f"valid_from={record.valid_from.isoformat()}")
    print(f"expires_at={_optional_datetime(record.expires_at)}")
    print(f"created_at={record.created_at.isoformat()}")
    print(f"updated_at={record.updated_at.isoformat()}")
    print(f"provenance.source_type={provenance.source_type.value}")
    print(f"provenance.source_item_digest={provenance.source_item_digest}")
    print(f"provenance.extractor_identity={_safe_text(provenance.extractor_identity)}")
    print(f"provenance.extracted_at={provenance.extracted_at.isoformat()}")
    print(f"provenance.source_id={_optional_text(provenance.source_id)}")
    print(f"provenance.source_revision={_optional_value(provenance.source_revision)}")
    print(f"provenance.run_id={_optional_text(provenance.run_id)}")
    print(f"provenance.model_identity={_optional_text(provenance.model_identity)}")
    print(f"provenance.response_identity={_optional_text(provenance.response_identity)}")
    print(f"confirmation.candidate_digest={record.confirmation.candidate_digest}")
    print(f"confirmation.confirmed_at={record.confirmation.confirmed_at.isoformat()}")


def _print_list(result: MemoryListResult) -> None:
    print("memory_list=true")
    print(f"scope_kind={result.scope.kind.value}")
    print(f"scope_id={result.scope.scope_id}")
    print(f"scope_revision={result.revision}")
    print(f"record_count={len(result.records)}")
    print(f"tombstone_count={len(result.tombstones)}")
    for record in result.records:
        _print_record(record)
    for tombstone in result.tombstones:
        _print_tombstone(tombstone)


def _print_show(result: MemoryShowResult) -> None:
    print("memory_show=true")
    _print_record(result.record)


def _print_tombstone(tombstone: MemoryTombstone) -> None:
    print("memory_tombstone=true")
    print(f"memory_id={_safe_text(tombstone.memory_id)}")
    print(f"scope_kind={_safe_text(tombstone.scope.kind.value)}")
    print(f"scope_id={_safe_text(tombstone.scope.scope_id)}")
    print(f"revision={tombstone.revision}")
    print(f"forgotten_at={tombstone.forgotten_at.isoformat()}")
    print(f"forget_reason={tombstone.reason.value}")


def _print_confirmed(result: MemoryConfirmResult) -> None:
    print("status=confirmed")
    print(f"outcome={result.outcome.value}")
    print(f"memory_id={_safe_text(result.record.memory_id)}")
    print(f"scope_revision={result.revision}")
    print(f"candidate_digest={result.metadata.candidate_digest}")


def _print_corrected(result: MemoryCorrectionResult) -> None:
    print("status=corrected")
    print(f"superseded_memory_id={_safe_text(result.superseded.memory_id)}")
    print(f"replacement_memory_id={_safe_text(result.replacement.memory_id)}")
    print(f"scope_revision={result.revision}")
    print(f"candidate_digest={result.metadata.candidate_digest}")


def _print_forgotten(result: MemoryForgetResult) -> None:
    print("status=forgotten")
    print(f"memory_id={_safe_text(result.memory_id)}")
    print(f"scope_revision={result.revision}")
    print(f"forgotten_at={result.tombstone.forgotten_at.isoformat()}")


def _report_policy_denial(
    preview: MemoryProposal,
    *,
    operation: str,
    memory_id: str | None = None,
) -> int:
    print(
        "error: policy_denied"
        f" operation={operation}"
        f" scope_kind={_safe_text(preview.candidate.scope.kind.value)}"
        f" scope_id={_safe_text(preview.candidate.scope.scope_id)}"
        f" reason={preview.decision.reason.value}"
        f" candidate_digest={preview.candidate_digest}"
        + (f" memory_id={_safe_text(memory_id)}" if memory_id is not None else ""),
        file=sys.stderr,
    )
    return EXIT_POLICY_DENIED


def _report_error(
    error: DQMemoryError,
    *,
    operation: str,
    scope: MemoryScope,
) -> int:
    if isinstance(error, MemoryAdmissionDeniedError):
        code = "policy_denied"
        exit_code = EXIT_POLICY_DENIED
    elif isinstance(error, MemoryDigestMismatchError):
        code = "digest_mismatch"
        exit_code = EXIT_ERROR
    elif isinstance(error, MemoryNotFoundError):
        code = "not_found"
        exit_code = EXIT_ERROR
    elif isinstance(error, MemoryConflictError):
        code = "conflict"
        exit_code = EXIT_ERROR
    elif isinstance(error, MemoryServiceError):
        code = error.reason or "memory_service_failure"
        exit_code = EXIT_ERROR
    elif isinstance(error, MemoryValidationError):
        code = "invalid_input"
        exit_code = EXIT_USAGE
    else:
        code = "memory_failure"
        exit_code = EXIT_ERROR
    reason = getattr(error, "reason", None)
    reason_suffix = f" reason={reason}" if isinstance(reason, str) else ""
    print(
        f"error: {code} operation={operation}"
        f" scope_kind={scope.kind.value} scope_id={_safe_text(scope.scope_id)}{reason_suffix}",
        file=sys.stderr,
    )
    return exit_code


def _report_internal_error(*, operation: str, scope: MemoryScope) -> int:
    print(
        f"error: internal_failure operation={operation}"
        f" scope_kind={scope.kind.value} scope_id={_safe_text(scope.scope_id)}",
        file=sys.stderr,
    )
    return EXIT_ERROR


def _optional_datetime(value: datetime | None) -> str:
    return value.isoformat() if value is not None else "<none>"


def _safe_text(value: str) -> str:
    escaped: list[str] = []
    for character in value:
        if character.isprintable():
            escaped.append(character)
            continue
        code_point = ord(character)
        if code_point <= 0xFF:
            escaped.append(f"\\x{code_point:02x}")
        elif code_point <= 0xFFFF:
            escaped.append(f"\\u{code_point:04x}")
        else:
            escaped.append(f"\\U{code_point:08x}")
    return "".join(escaped)


def _optional_text(value: str | None) -> str:
    return _safe_text(value) if value is not None else "<none>"


def _optional_value(value: object | None) -> str:
    return _safe_text(str(value)) if value is not None else "<none>"


if __name__ == "__main__":
    raise SystemExit(main())
