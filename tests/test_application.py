from collections.abc import Sequence

import pytest

from dqagent.application import ChatApplication
from dqagent.models import Completion, Message, Role


class StubLLM:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = iter(responses)
        self.requests: list[tuple[Message, ...]] = []

    def complete(self, messages: Sequence[Message]) -> Completion:
        self.requests.append(tuple(messages))
        return Completion(next(self._responses))


class FailingLLM:
    def complete(self, messages: Sequence[Message]) -> Completion:
        raise RuntimeError("provider unavailable")


def test_send_builds_and_commits_conversation_history() -> None:
    llm = StubLLM(["Hello", "Still here"])
    app = ChatApplication(llm, system_prompt="Be concise.")

    first = app.send("Hi")
    second = app.send("Are you there?")

    assert first == Message(Role.ASSISTANT, "Hello")
    assert second == Message(Role.ASSISTANT, "Still here")
    assert llm.requests == [
        (
            Message(Role.SYSTEM, "Be concise."),
            Message(Role.USER, "Hi"),
        ),
        (
            Message(Role.SYSTEM, "Be concise."),
            Message(Role.USER, "Hi"),
            Message(Role.ASSISTANT, "Hello"),
            Message(Role.USER, "Are you there?"),
        ),
    ]


def test_send_does_not_commit_partial_history_when_provider_fails() -> None:
    app = ChatApplication(FailingLLM(), system_prompt="Be concise.")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        app.send("Hi")

    assert app.messages == (Message(Role.SYSTEM, "Be concise."),)


def test_reset_preserves_only_the_system_message() -> None:
    app = ChatApplication(StubLLM(["Hello"]), system_prompt="Be concise.")
    app.send("Hi")

    app.reset()

    assert app.messages == (Message(Role.SYSTEM, "Be concise."),)


def test_send_rejects_blank_user_input() -> None:
    app = ChatApplication(StubLLM([]))

    with pytest.raises(ValueError, match="must not be empty"):
        app.send("   ")
