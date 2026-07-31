"""Command-line interface for the tool-using agent."""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from dqagent.application import AgentApplication, ChatApplication, SessionAgentApplication
from dqagent.builtin_tools import create_builtin_tool_registry
from dqagent.config import ModelProvider, Settings
from dqagent.context import ContextBudget, ContextBuilder, PromptAssembler, PromptSection
from dqagent.errors import DQAgentError
from dqagent.providers import create_llm_client
from dqagent.runtime import AgentRuntime, RetryPolicy
from dqagent.session import JsonFileSessionStore

EXIT_COMMANDS = {"/exit", "/quit"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tool-using agent.")
    parser.add_argument("-m", "--message", help="Send one message and exit.")
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in ModelProvider],
        help="Override DQAGENT_PROVIDER.",
    )
    parser.add_argument("--model", help="Override DQAGENT_MODEL.")
    parser.add_argument("--system", help="Optional system prompt for this session.")
    parser.add_argument(
        "--session-id",
        help="Create or resume this durable session ID.",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=Path(".local/sessions"),
        help="Directory for durable session JSON files.",
    )
    parser.add_argument(
        "--context-max-characters",
        type=int,
        default=32_000,
        help="Provider-neutral active-context character budget.",
    )
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


def _interactive_chat(
    app: ChatApplication | AgentApplication | SessionAgentApplication,
) -> int:
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
            if isinstance(app, SessionAgentApplication):
                print("Durable sessions cannot be reset; start a new session ID.")
                continue
            app.reset()
            print("Conversation reset.")
            continue

        response = app.send(user_input)
        print(f"assistant> {response.content}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_dotenv()
        settings = _settings_from_args(args)
        runtime = AgentRuntime(
            create_llm_client(settings),
            create_builtin_tool_registry(),
            default_timeout_seconds=settings.run_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=settings.max_model_attempts),
        )
        if args.session_id:
            store = JsonFileSessionStore(args.session_dir)
            sections = (
                (PromptSection("behavior", args.system),) if args.system else ()
            )
            builder = ContextBuilder(
                PromptAssembler(sections),
                ContextBudget(max_characters=args.context_max_characters),
            )
            app: AgentApplication | SessionAgentApplication
            if store.load(args.session_id) is None:
                app = SessionAgentApplication.create(
                    runtime,
                    store,
                    builder,
                    session_id=args.session_id,
                )
            else:
                app = SessionAgentApplication.resume(
                    runtime,
                    store,
                    builder,
                    args.session_id,
                )
        else:
            app = AgentApplication(runtime, system_prompt=args.system)

        if args.message:
            print(app.send(args.message).content)
            return 0
        return _interactive_chat(app)
    except (DQAgentError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
