"""Start or resume a durable two-step workflow."""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from dqagent.checkpoint import JsonFileCheckpointStore
from dqagent.execution import RunContext
from dqagent.workflow import (
    END,
    NextTransition,
    NodeResult,
    WorkflowDefinition,
    WorkflowNode,
    WorkflowRunner,
)


def prepare(state: Mapping[str, object], context: RunContext) -> NodeResult:
    context.check_active()
    return NodeResult.interrupt(
        "review the prepared request before applying it",
        {"prepared": True},
    )


def apply(state: Mapping[str, object], context: RunContext) -> NodeResult:
    context.check_active()
    return NodeResult(
        {
            "applied": True,
            "idempotency_key": context.metadata["idempotency_key"],
        }
    )


def definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        "durable-demo",
        "1",
        "prepare",
        (
            WorkflowNode("prepare", prepare, NextTransition("apply")),
            WorkflowNode("apply", apply, END),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "resume"))
    parser.add_argument("workflow_id")
    args = parser.parse_args(argv)

    runner = WorkflowRunner(JsonFileCheckpointStore(Path(".local/checkpoints")))
    if args.action == "start":
        result = runner.start(
            definition(),
            {"request": "phase-5-demo"},
            workflow_id=args.workflow_id,
        )
    else:
        result = runner.resume(definition(), args.workflow_id)

    print(
        json.dumps(
            {"state": result.state.value, "data": dict(result.data)},
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
