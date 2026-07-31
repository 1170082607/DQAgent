"""Command-line entry point for deterministic context regression evaluation."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from dqagent.context_evaluation import (
    ContextEvaluationRunner,
    load_context_evaluation_suite,
)
from dqagent.errors import DQAgentError

DEFAULT_SUITE = Path("evaluations/cases/phase-6-context-v1.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DQAgent context construction.")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = ContextEvaluationRunner().run(load_context_evaluation_suite(args.suite))
    except (DQAgentError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report.to_dict(), indent=2, ensure_ascii=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
