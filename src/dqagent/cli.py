"""Command-line interface for the tool-using agent."""

import argparse
import os
import sys
from collections.abc import Sequence

from dqagent.application import AgentApplication, ChatApplication
from dqagent.builtin_tools import create_builtin_tool_registry
from dqagent.config import Settings
from dqagent.errors import DQAgentError
from dqagent.providers.openai import OpenAIResponsesClient
from dqagent.runtime import AgentRuntime, RetryPolicy

EXIT_COMMANDS = {"/exit", "/quit"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tool-using OpenAI agent.")
    parser.add_argument("-m", "--message", help="Send one message and exit.")
    parser.add_argument("--model", help="Override DQAGENT_MODEL.")
    parser.add_argument("--system", help="Optional system prompt for this session.")
    parser.add_argument("--base-url", help="Override OPENAI_BASE_URL.")
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
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    environ = dict(os.environ)
    if args.model:
        environ["DQAGENT_MODEL"] = args.model
    if args.base_url:
        environ["OPENAI_BASE_URL"] = args.base_url
    if args.timeout is not None:
        environ["DQAGENT_TIMEOUT_SECONDS"] = str(args.timeout)
    if args.run_timeout is not None:
        environ["DQAGENT_RUN_TIMEOUT_SECONDS"] = str(args.run_timeout)
    if args.max_model_attempts is not None:
        environ["DQAGENT_MAX_MODEL_ATTEMPTS"] = str(args.max_model_attempts)
    return Settings.from_env(environ)


def _interactive_chat(app: ChatApplication | AgentApplication) -> int:
    print("DQAgent chat. Use /reset to clear the conversation or /exit to quit.")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            return 0
        if user_input.lower() == "/reset":
            app.reset()
            print("Conversation reset.")
            continue

        response = app.send(user_input)
        print(f"assistant> {response.content}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = _settings_from_args(args)
        runtime = AgentRuntime(
            OpenAIResponsesClient(settings),
            create_builtin_tool_registry(),
            default_timeout_seconds=settings.run_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=settings.max_model_attempts),
        )
        app = AgentApplication(
            runtime,
            system_prompt=args.system,
        )

        if args.message:
            print(app.send(args.message).content)
            return 0
        return _interactive_chat(app)
    except DQAgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
