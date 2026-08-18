"""Command-line entry point for the credential-free coding evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from dqagent.coding_evaluation import (
    CodingEvaluationMode,
    CodingEvaluationRunner,
    load_coding_evaluation_suite,
)
from dqagent.errors import DQAgentError

DEFAULT_SUITE = Path("evaluations/cases/phase-9-coding-smoke-v1.json")
DEFAULT_OUTPUT = Path(".local/evaluations/phase-9-coding-deterministic-report.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate DQAgent coding behavior in disposable fixture repositories."
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in CodingEvaluationMode],
        default=CodingEvaluationMode.DETERMINISTIC.value,
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Write the bounded report under .local/evaluations by default.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        suite = load_coding_evaluation_suite(args.suite)
        report = CodingEvaluationRunner(CodingEvaluationMode(args.mode)).run(suite)
        rendered = json.dumps(report.to_dict(), indent=2, ensure_ascii=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    except (DQAgentError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"report: {args.output}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
