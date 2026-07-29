from collections.abc import Iterator

import pytest

from dqagent import cli
from dqagent.models import Message, Role


class StubChatApplication:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.reset_count = 0

    def send(self, user_input: str) -> Message:
        self.sent.append(user_input)
        return Message(Role.ASSISTANT, f"reply:{user_input}")

    def reset(self) -> None:
        self.reset_count += 1


def test_interactive_chat_handles_messages_reset_and_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs: Iterator[str] = iter(["hello", "/reset", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    app = StubChatApplication()

    exit_code = cli._interactive_chat(app)  # type: ignore[arg-type]

    assert exit_code == 0
    assert app.sent == ["hello"]
    assert app.reset_count == 1
    output = capsys.readouterr().out
    assert "assistant> reply:hello" in output
    assert "Conversation reset." in output


def test_main_reports_missing_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DQAGENT_MODEL", raising=False)

    exit_code = cli.main(["--message", "hello"])

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "OPENAI_API_KEY" in error
    assert "DQAGENT_MODEL" in error


def test_cli_arguments_select_llama_cpp_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = cli.build_parser().parse_args(
        [
            "--provider",
            "llama_cpp",
            "--model",
            "local-model",
            "--base-url",
            "http://localhost:9000/v1",
        ]
    )

    settings = cli._settings_from_args(args)

    assert settings.provider.value == "llama_cpp"
    assert settings.base_url == "http://localhost:9000/v1"
