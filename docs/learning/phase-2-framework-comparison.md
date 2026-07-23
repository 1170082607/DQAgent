# Phase 2 Comparison: DQAgent, OpenAI Agents SDK, and EINO

## Scope

This note compares DQAgent's minimal Phase 2 model/tool loop with two mature implementations. It is
a source-reading comparison, not a recommendation to replace the local implementation.

Sources were read on 2026-07-23 at OpenAI Agents Python commit
`1e8d506a32ea7b84f3a5a811e101378c0b1bc137` and EINO commit
`ca0441ac0bceed8945dcf7d5a18c237c924c6aa8`.

## Shared Core

All three systems implement the same essential state machine:

```text
model request -> final text -> stop
              -> tool calls -> execute -> append observations -> model request
```

This is closer to a bounded backend work loop than to a planner. The model chooses the next action,
while application code owns available operations, validation, execution, state accumulation, and
termination policy.

## Comparison

| Concern | DQAgent Phase 2 | OpenAI Agents SDK | EINO |
| --- | --- | --- | --- |
| Loop | Explicit synchronous `for` loop | `Runner` supports async, sync, and streaming runs | `ChatModelAgent` builds a ReAct graph around ChatModel and ToolsNode |
| Bound | 8 model calls by default | `max_turns`, with `MaxTurnsExceeded` | `MaxIterations`, default 20, with `ErrExceedMaxIterations` |
| Tool contract | Explicit name, description, JSON Schema, handler | `@function_tool` infers schema from signatures/docstrings through inspect, griffe, and Pydantic; explicit `FunctionTool` is also available | `BaseTool.Info` separates metadata; invokable, streamable, and enhanced interfaces separate execution forms |
| Validation | Draft 2020-12 validation before every handler call | Generated Pydantic schema and validation for function tools | Constructors can decode JSON into typed Go inputs; raw interfaces also expose JSON strings |
| Failure recovery | All expected tool errors become observations | Behavior is configurable; unknown tools default to a run error but can be returned to the model, and async tool timeouts can be result or exception | Tool and graph errors flow through Go errors and agent events; context and middleware provide wider control |
| Parallel calls | Accepted but executed sequentially | Local function-call concurrency is configurable | ToolsNode and streaming composition support concurrent/streamed execution |
| Runtime services | None | Rich run items, tracing, guardrails, approvals, handoffs, sessions, resume state | Events, callbacks, context cancellation, middleware, checkpoints, interrupt/resume, workflows |

## Design Lessons

DQAgent intentionally keeps tool schema and handler registration explicit. This makes the boundary
visible: model output is untrusted input, JSON Schema is the contract, and a tool result is an
observation rather than an ordinary assistant message. OpenAI's decorator and EINO's typed helper
constructors are ergonomic layers over the same mechanism.

The mature systems also show why Phase 3 is a runtime phase. Once streaming, concurrent calls,
approvals, cancellation, retries, tracing, or resume are required, a plain loop no longer owns enough
lifecycle state. OpenAI exposes that state through Runner results and run configuration. EINO embeds
the loop in its graph/event runtime and threads `context.Context` through components.

DQAgent's soft timeout is the most important current limitation. It can stop waiting and return an
error observation, but it cannot terminate a Python worker thread. OpenAI's documented timeout is
for async function tools, where cancellation is cooperative. EINO's context-based execution also
depends on components honoring cancellation. Strong termination needs a process or remote-worker
boundary in every design.

Semantic repeated-call rejection is a DQAgent-specific safety policy, not a universal Agent rule. It
is useful for a minimal stateless tool set, but it would break legitimate polling or state-changing
retries. A future runtime should replace it with per-tool idempotency and retry policy.

## Sources

- [OpenAI Agents SDK: Running agents](https://github.com/openai/openai-agents-python/blob/main/docs/running_agents.md)
- [OpenAI Agents SDK: Tools](https://github.com/openai/openai-agents-python/blob/main/docs/tools.md)
- [OpenAI Agents SDK: Results](https://github.com/openai/openai-agents-python/blob/main/docs/results.md)
- [EINO README](https://github.com/cloudwego/eino/blob/main/README.md)
- [EINO tool interfaces](https://github.com/cloudwego/eino/blob/main/components/tool/interface.go)
- [EINO ReAct implementation](https://github.com/cloudwego/eino/blob/main/adk/react.go)
- [EINO ChatModelAgent configuration](https://github.com/cloudwego/eino/blob/main/adk/chatmodel.go)
- [EINO Runner](https://github.com/cloudwego/eino/blob/main/adk/runner.go)
