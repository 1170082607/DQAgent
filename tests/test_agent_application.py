from collections.abc import Mapping, Sequence

import pytest

from dqagent.application import AgentApplication
from dqagent.errors import AgentLoopError
from dqagent.models import (
    Completion,
    ConversationItem,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
    ToolErrorCode,
    ToolResult,
)
from dqagent.tools import Tool, ToolRegistry


class StubLLM:
    def __init__(self, completions: Sequence[Completion]) -> None:
        self._completions = iter(completions)
        self.requests: list[
            tuple[tuple[ConversationItem, ...], tuple[ToolDefinition, ...]]
        ] = []

    def complete(
        self,
        messages: Sequence[ConversationItem],
        tools: Sequence[ToolDefinition] = (),
    ) -> Completion:
        self.requests.append((tuple(messages), tuple(tools)))
        return next(self._completions)


def make_registry(calls: list[str] | None = None) -> ToolRegistry:
    def greet(arguments: Mapping[str, object]) -> str:
        name = str(arguments["name"])
        if calls is not None:
            calls.append(name)
        return f"hello {name}"

    return ToolRegistry(
        (
            Tool(
                ToolDefinition(
                    "greet",
                    "Greet a person.",
                    {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                ),
                greet,
            ),
        )
    )


def test_agent_executes_tool_and_commits_complete_history() -> None:
    call = ToolCall("call-1", "greet", '{"name":"Ada"}')
    llm = StubLLM([Completion(tool_calls=(call,)), Completion("Done")])
    app = AgentApplication(llm, make_registry(), system_prompt="Use tools.")

    response = app.send("Say hello")

    assert response == Message(Role.ASSISTANT, "Done")
    assert len(llm.requests) == 2
    second_request = llm.requests[1][0]
    assert second_request[:3] == (
        Message(Role.SYSTEM, "Use tools."),
        Message(Role.USER, "Say hello"),
        call,
    )
    assert isinstance(second_request[3], ToolResult)
    assert second_request[3].output == "hello Ada"
    assert app.messages == (*second_request, Message(Role.ASSISTANT, "Done"))


def test_agent_rejects_repeated_semantic_call_without_executing_it_again() -> None:
    executions: list[str] = []
    first = ToolCall("call-1", "greet", '{"name":"Ada"}')
    repeated = ToolCall("call-2", "greet", '{ "name": "Ada" }')
    llm = StubLLM(
        [
            Completion(tool_calls=(first,)),
            Completion(tool_calls=(repeated,)),
            Completion("Done"),
        ]
    )
    app = AgentApplication(llm, make_registry(executions))

    app.send("Say hello")

    assert executions == ["Ada"]
    repeated_result = llm.requests[2][0][-1]
    assert isinstance(repeated_result, ToolResult)
    assert repeated_result.error_code is ToolErrorCode.REPEATED_CALL


def test_agent_returns_tool_errors_to_model_for_recovery() -> None:
    unknown = ToolCall("call-1", "missing", "{}")
    llm = StubLLM([Completion(tool_calls=(unknown,)), Completion("Cannot use that tool")])
    app = AgentApplication(llm, make_registry())

    app.send("Do something")

    observation = llm.requests[1][0][-1]
    assert isinstance(observation, ToolResult)
    assert observation.error_code is ToolErrorCode.UNKNOWN_TOOL


def test_agent_does_not_commit_when_iteration_limit_is_reached() -> None:
    calls = [
        ToolCall("call-1", "greet", '{"name":"Ada"}'),
        ToolCall("call-2", "greet", '{"name":"Grace"}'),
    ]
    llm = StubLLM([Completion(tool_calls=(call,)) for call in calls])
    app = AgentApplication(llm, make_registry(), system_prompt="Use tools.", max_iterations=2)

    with pytest.raises(AgentLoopError, match="within 2 model calls"):
        app.send("Keep going")

    assert app.messages == (Message(Role.SYSTEM, "Use tools."),)
