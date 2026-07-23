# DQAgent Architecture

## Status

This document describes the implemented Phase 2 architecture. Planned runtime capabilities belong
in the [roadmap](roadmap.md) until code and accepted decisions make them part of the system.

## System Context

DQAgent is a local command-line agent. It maintains in-memory conversation state, lets an OpenAI
model select application-owned tools, executes those tools locally, and repeats until the model
returns final text or the loop reaches its bound.

```text
User -> CLI -> AgentApplication -> LLMClient -> OpenAIResponsesClient -> Responses API
                     |
                     v
                ToolRegistry -> local tool handler
```

The CLI is the composition root. It wires `OpenAIResponsesClient`, `AgentApplication`, and the
built-in tool registry. `ChatApplication` remains available as a direct, text-only model use case.

## Provider-Neutral Conversation

`dqagent.models` defines the values that can cross application boundaries:

- `Message`: system, user, or assistant text.
- `ToolDefinition`: name, description, and JSON Schema shown to a model.
- `ToolCall`: provider-neutral call ID, tool name, and raw JSON arguments.
- `ToolResult`: correlated output with success/error outcome and an optional stable error code.
- `Completion`: final/intermediate text, tool calls, or both.

`LLMClient` accepts an ordered sequence of conversation items plus available tool definitions. The
OpenAI adapter is solely responsible for translating them to `function` tools, `function_call`
items, and `function_call_output` items. OpenAI SDK types do not cross that adapter.

## Tool Boundary

`dqagent.tools.ToolRegistry` owns registration and execution. Registration rejects duplicate names,
non-object input schemas, and invalid Draft 2020-12 schemas. Execution follows one path:

```text
lookup -> parse JSON -> validate schema -> invoke with timeout -> classify result
```

Expected failures are data rather than exceptions escaping the loop:

| Condition | Error code | Agent behavior |
| --- | --- | --- |
| Tool name is absent | `unknown_tool` | Append an error observation |
| JSON is malformed or violates schema | `invalid_arguments` | Append an error observation |
| Handler exceeds its configured wait | `timeout` | Append an error observation |
| Handler raises or returns invalid output | `execution_error` | Append an error observation |
| Call ID or normalized call repeats | `repeated_call` | Do not execute; append an error observation |

The CLI currently registers one `current_time` tool. Its explicit schema both guides the model and
guards the handler; provider-side schema adherence is not treated as a security boundary.

## Agent Loop

`AgentApplication.send` constructs a pending history and then performs at most eight model calls by
default:

1. Request a completion with the pending conversation and registered tool definitions.
2. If there are no tool calls, commit the pending history and return the final assistant message.
3. Append each tool call, execute it or reject it as repeated, and append its correlated result.
4. Use the expanded history for the next model request.
5. Raise `AgentLoopError` if no final answer is produced within the bound.

Multiple calls from one model response are supported but executed sequentially. Parallel execution
needs cancellation and lifecycle semantics and is deferred to the runtime phase.

## Consistency and Side Effects

Conversation state uses commit-after-success semantics. Provider errors and loop exhaustion do not
commit the user message or intermediate items. This preserves a coherent transcript, but it is not
a distributed transaction: external effects from a tool that already ran cannot be rolled back.
Mutating tools must provide their own idempotency and transactional guarantees.

Repeated-call tracking is scoped to one `send` run. Calls are compared by call ID and by tool name
plus canonicalized JSON arguments. This prevents simple infinite tool loops, at the cost of rejecting
polling-like repetition; no current built-in tool requires polling.

## Timeouts

Synchronous handlers run in a worker thread. On timeout, DQAgent stops waiting and returns an error
observation. Python cannot safely terminate that thread, so a handler may continue in the background.
This is a soft response-time bound, not hard cancellation. Phase 3 must introduce an execution
context and an isolation strategy before tools with untrusted or unbounded work are appropriate.

## Dependency Rules

```text
CLI -> AgentApplication -> LLM protocol + neutral models
CLI -> ToolRegistry -> tool handlers
OpenAI adapter -> neutral models + OpenAI SDK
ToolRegistry -> neutral models + jsonschema
```

- Application and tool modules must not import provider SDKs.
- Provider adapters translate wire data and provider failures.
- The registry owns generic execution mechanics; handlers own domain behavior.
- Runtime, workflow, persistence, and multi-agent modules are not introduced before their phases.

## Testing Strategy

- Agent tests use an in-memory LLM fake and assert the complete call/result history.
- Tool tests cover successful execution, schema failures, unknown names, exceptions, timeouts, and
  duplicate registration.
- Provider tests verify both directions of Responses API function-call mapping without network use.
- CLI and built-in tool tests cover composition behavior and the real tool surface.
- CI runs Ruff, strict mypy, and pytest with at least 85% coverage.

No live model test runs in CI because credentials, cost, and network nondeterminism would make it an
unsuitable correctness gate.

## Current Limitations

- Conversation history is process-local, unbounded, and non-streaming.
- Tool execution is synchronous and sequential, with only soft timeouts.
- There are no approvals, hard cancellation, retries, lifecycle events, persistence, or tracing.
- Only OpenAI Responses is implemented as a provider adapter.
- Tool results are text-only.

## Related Material

- [ADR-0001: Provider-Neutral LLM Boundary](adr/0001-provider-neutral-llm-boundary.md)
- [ADR-0002: Explicit Tool Boundary and Bounded Agent Loop](adr/0002-explicit-tool-boundary-and-bounded-loop.md)
- [Phase 2 Framework Comparison](learning/phase-2-framework-comparison.md)
