"""Command-line entry point for deterministic and live behavioral evaluation."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from dqagent.config import Settings
from dqagent.errors import DQAgentError
from dqagent.evaluation import EvaluationMode, EvaluationRunner, load_evaluation_suite
from dqagent.providers import create_llm_client

DEFAULT_SUITE = Path("evaluations/cases/phase-3-baseline-v1.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DQAgent behavior.")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in EvaluationMode],
        default=EvaluationMode.DETERMINISTIC.value,
        help="Use scripted fixtures by default; live mode requires provider credentials.",
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = EvaluationMode(args.mode)
    try:
        suite = load_evaluation_suite(args.suite)
        if mode is EvaluationMode.LIVE:
            load_dotenv()
            settings = Settings.from_env()
            runner = EvaluationRunner(
                mode,
                live_client=create_llm_client(settings),
                run_timeout_seconds=settings.run_timeout_seconds,
                max_model_attempts=settings.max_model_attempts,
            )
        else:
            runner = EvaluationRunner(mode, max_model_attempts=1)
        report = runner.run(suite)
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
