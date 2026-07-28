# DQAgent Architecture

## Status

This document describes the implemented Phase 3 architecture. Workflow, persistence, hard execution
isolation, and other future capabilities remain in the [roadmap](roadmap.md).

## System Context

DQAgent is a local command-line agent. It maintains in-memory conversation state and delegates each
user turn to an observable runtime that calls an OpenAI model, executes application-owned tools, and
continues until it reaches a terminal state.

```text
User -> CLI -> AgentApplication -> AgentRuntime -> LLMClient -> OpenAI Responses API
                                      |
                                      +---------> ToolRegistry -> tool handler
                                      |
RunContext ---------------------------+
                                      |
                                      +---------> EventSink adapters
```

The CLI is the composition root. It creates the provider adapter, built-in tool registry, retry
policy, runtime, and stateful application. `ChatApplication` remains a direct text-only use case and
does not use the agent runtime.

## Responsibility Boundaries

`AgentApplication` owns a conversation. It adds a user message to pending history, calls the runtime,
and commits the returned conversation only when the run completes successfully.

`AgentRuntime` owns one execution. It controls model/tool iteration, repeated-call protection,
provider retry, lifecycle transitions, and event emission. It does not persist conversation state.

`RunContext` carries data and control signals shared by all work in one run: run ID, start time,
deadline, cancellation, and read-only metadata.

`ToolRegistry` validates untrusted model arguments and invokes context-aware handlers. The OpenAI
adapter translates provider-neutral values and classifies SDK failures.

This split is similar to separating a stateful application service from a request-scoped middleware
and execution engine. The analogy is incomplete because the model can request more work, so the
runtime owns a bounded state machine rather than a single RPC call.

## Run Lifecycle

`AgentApplication.run` creates pending history and delegates it to `AgentRuntime.run`. The runtime
uses this state machine:

```text
RUNNING -> model request -> final text --------------------> COMPLETED
             | failure          |
             |                  +-> tool calls -> execute -> next model request
             |
             +-> retryable and attempts remain -> backoff -> retry
             +-> non-retryable / attempts exhausted ------> FAILED

RUNNING -> cancellation ----------------------------------> CANCELLED
RUNNING -> deadline exhausted -----------------------------> TIMED_OUT
RUNNING -> iteration limit / unexpected failure ----------> FAILED
```

Terminal failure, cancellation, and timeout do not commit pending conversation items. External tool
side effects are not rolled back; mutating handlers still need their own transaction and idempotency
guarantees.

## Execution Context

`RunContext` generates a UUID unless the caller supplies a run ID. Deadlines are exposed as UTC wall
time for events but enforced with a monotonic clock so system-clock changes cannot extend or shorten
a run. Metadata is copied into a read-only mapping for correlation values such as tenant, request,
or experiment identifiers.

Cancellation is thread-safe and idempotent. `check_active()` raises a stable cancellation or
deadline exception, while `wait()` makes retry backoff responsive to both signals. The context is
passed through `LLMClient.complete` and every tool handler.

`wait()` uses a monotonic target and rechecks it after each operating-system wait. This prevents an
early timer wake-up from shortening retry backoff or allowing one more attempt just before a run
deadline. Caller-supplied run IDs must be non-empty; run and tool timeout values must be finite and
positive.

Cancellation is cooperative, not preemptive:

- The OpenAI request timeout is capped by the remaining run deadline.
- Runtime backoff and tool waiting check the context repeatedly.
- Long-running handlers must call `context.check_active()` or `context.wait()` themselves.
- An SDK call or Python thread that ignores cancellation may continue until its own timeout or exit.

Hard termination requires a process, container, or remote-worker boundary. Phase 3 deliberately does
not claim that a Python worker thread can be safely killed.

## Structured Events

Every event contains a run ID, monotonically increasing sequence, typed event name, lifecycle state,
UTC timestamp, elapsed seconds, and structured attributes. Current event types cover:

- Run start and completed, failed, cancelled, or timed-out terminal states.
- Model request start, completion, and failure for each attempt.
- Scheduled retry and backoff duration.
- Tool-call processing start and completion.

`AgentRunResult.events` returns the complete successful event sequence. `EventSink` adapters receive
events as they happen, including terminal failure events, and can translate them into traces,
metrics, or audit records.

Every started model request is closed by either a completed or failed event. Errors escaping the
model boundary are associated with the current run ID before the attempt and terminal events are
emitted, so cancellation, timeout, provider failure, and unexpected adapter failure remain
correlatable.

Sink failures are logged and isolated from run semantics. This best-effort contract is appropriate
for local telemetry; durable or compliance audit delivery needs a future fail-closed persistence
boundary. Event sinks are internal operational channels and must protect diagnostic data.

## Error Classification and Retry

All user-visible runtime exceptions inherit `DQAgentError` and expose a stable `ErrorCategory`, a
`retryable` flag, and an optional run ID. Categories distinguish provider failure, rate limiting,
unavailability, request timeout, cancellation, deadline exhaustion, loop limit, configuration, and
unexpected internal failure.

The OpenAI adapter marks timeout, rate-limit, connection, and server errors retryable. Other SDK
errors remain non-retryable. The SDK's implicit retries are disabled so `AgentRuntime` is the single
owner of attempt count, exponential backoff, deadline checks, and retry events.

If an OpenAI timeout arrives after the end-to-end run deadline has elapsed, the run deadline takes
precedence and the runtime terminates as `TIMED_OUT`; an active run still classifies an individual
request timeout as a retryable provider failure.

Only model requests are retried. Tools are never retried automatically because the runtime cannot
infer whether a handler is idempotent or whether a prior attempt produced an external side effect.
Retry policy defaults to three attempts with bounded exponential delay.

## Tool Boundary

Tool registration still rejects duplicate names, non-object input schemas, and invalid Draft
2020-12 schemas. Tool timeouts must be finite and greater than zero. Execution follows one path:

```text
context check -> lookup -> parse JSON -> validate schema -> invoke -> classify result
```

Expected failures remain model-visible observations with stable codes: `unknown_tool`,
`invalid_arguments`, `timeout`, `execution_error`, and `repeated_call`. Handler exception details are
not copied into `ToolResult`; `ToolExecution` carries them separately so internal tool events can
retain diagnostic type and message without leaking infrastructure details to the model.

Handlers run in a worker thread. The registry polls at short intervals so cancellation and the run
deadline can interrupt the caller's wait. A tool-specific timeout remains an error observation and
allows the model to recover. In all three cases, an already-running non-cooperative thread may still
continue in the background.

Multiple calls from one model response remain sequential. Parallel execution requires an explicit
concurrency limit, sibling-cancellation policy, result ordering, and side-effect policy; those are
not implied merely by having a runtime context.

## Provider-Neutral Conversation

`dqagent.models` retains the Phase 2 neutral values: `Message`, `ToolDefinition`, `ToolCall`,
`ToolResult`, and `Completion`. `LLMClient` now also accepts an optional `RunContext`. The OpenAI
adapter alone maps these values to Responses API messages, function calls, function outputs, request
timeouts, and SDK errors.

Provider SDK types never enter the runtime, application, tool registry, or their tests.

## Dependency Rules

```text
CLI -> AgentApplication -> AgentRuntime
AgentRuntime -> RunContext + LLMClient + ToolRegistry + neutral models
ToolRegistry -> RunContext + neutral models + jsonschema
OpenAI adapter -> RunContext + neutral models + OpenAI SDK
Event adapters -> RunEvent (future concrete integrations)
```

- Session state is owned above the runtime; one run cannot commit partial history.
- Runtime and tool modules must not import provider SDKs.
- Provider adapters translate wire data and classify infrastructure failures.
- Event sinks observe execution but cannot mutate runtime state.
- Workflow, persistence, and multi-agent modules are not introduced before their phases.

## Configuration

`DQAGENT_TIMEOUT_SECONDS` bounds an individual OpenAI request. `DQAGENT_RUN_TIMEOUT_SECONDS` bounds
one end-to-end agent run and defaults to 120 seconds. `DQAGENT_MAX_MODEL_ATTEMPTS` configures the
runtime's model-attempt budget and defaults to three. All values are validated before composition.

## Testing Strategy

- Runtime tests assert event order, run-ID propagation, retries, cancellation, deadlines, terminal
  classification, tool diagnostic isolation, and sink failure isolation.
- Application tests assert complete-history commit and rollback on failed runs.
- Tool tests cover validation, expected failures, soft timeout, and context-aware handlers.
- Provider tests verify wire mapping, deadline-derived request timeout, and error classification.
- CI runs Ruff, strict mypy, and pytest with at least 85% coverage.

No live model test runs in CI because credentials, cost, and network nondeterminism make it an
unsuitable correctness gate.

## Current Limitations

- The runtime is synchronous; cancellation cannot preempt a blocking SDK call or Python thread.
- `AgentApplication` conversation state is not safe for concurrent callers.
- Tool calls are sequential and tool retries are intentionally unsupported.
- Event sinks are best-effort and no concrete durable telemetry adapter is included.
- Conversation history is process-local, unbounded, and non-streaming.
- There are no approvals, checkpoints, persistence, or workflow orchestration.
- Only OpenAI Responses is implemented as a provider adapter.

## Related Material

- [ADR-0001: Provider-Neutral LLM Boundary](adr/0001-provider-neutral-llm-boundary.md)
- [ADR-0002: Explicit Tool Boundary and Bounded Agent Loop](adr/0002-explicit-tool-boundary-and-bounded-loop.md)
- [ADR-0003: Observable Runtime and Cooperative Cancellation](adr/0003-observable-runtime-and-cooperative-cancellation.md)
- [Phase 2 Framework Comparison](learning/phase-2-framework-comparison.md)
