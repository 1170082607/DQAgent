# DQAgent Architecture

## Status

This document describes the implemented Phase 5 architecture. Durable sessions, context engineering,
hard execution isolation, and other future capabilities remain in the [roadmap](roadmap.md).

## System Context

DQAgent is a local command-line agent. It maintains in-memory conversation state and delegates each
user turn to an observable runtime that calls a configured model, executes application-owned tools,
and continues until it reaches a terminal state.

```text
User -> CLI -> AgentApplication -> AgentRuntime -> LLMClient -> provider adapter
                                                            -> OpenAI Responses API
                                                            -> llama-server Chat Completions API
                                      |
                                      +---------> ToolRegistry -> tool handler
                                      |
RunContext ---------------------------+
                                      |
                                      +---------> EventSink adapters

Versioned case -> EvaluationRunner -> AgentRuntime -> EvaluationReport
                       |                   |
                       +-> scripted LLM    +-> production tools and events
                       +-> live LLM

Workflow input -> WorkflowRunner -> validated node graph -> WorkflowRunResult
                        |                   |
                        +-> CheckpointStore +-> sequential / conditional / parallel nodes
                        +-> RunContext + shared lifecycle events
```

The CLI is the composition root. It creates the provider adapter, built-in tool registry, retry
policy, runtime, and stateful application. `ChatApplication` remains a direct text-only use case and
does not use the agent runtime.

The evaluation CLI is a separate composition root. It loads versioned cases, selects either a
scripted or live `LLMClient`, creates an isolated production runtime per case, and writes a structured
report. Evaluation does not add a second agent loop.

Workflow definitions are application composition. `WorkflowRunner` executes known control flow and
persists progress through a `CheckpointStore`; an agent runtime may be called inside a node when a
step genuinely needs model decisions, but workflow transitions remain deterministic.

## Responsibility Boundaries

`AgentApplication` owns a conversation. It adds a user message to pending history, calls the runtime,
and commits the returned conversation only when the run completes successfully.

`AgentRuntime` owns one execution. It controls model/tool iteration, repeated-call protection,
provider retry, lifecycle transitions, and event emission. It does not persist conversation state.

`RunContext` carries data and control signals shared by all work in one run: run ID, start time,
deadline, cancellation, and read-only metadata.

`ToolRegistry` validates untrusted model arguments and invokes context-aware handlers. Provider
adapters translate provider-neutral values and classify transport failures.

The built-in registry exposes `current_time` and `get_weather`. `current_time` reads the local clock
for a validated UTC offset. `get_weather` is a deterministic tool-calling demonstration: it validates
a non-empty city and RFC 3339 full-date string, then returns structured fixed sunny data with an
explicit demo marker and no network access. It is not a weather-provider adapter and its result is
not a real forecast.

`EvaluationRunner` owns case isolation and behavioral judgment. It consumes final output,
conversation items, and runtime events; it does not alter runtime control flow or conversation state.

`WorkflowRunner` owns graph traversal, node-boundary commits, conditional selection, bounded branch
execution, interruption, recovery, and workflow events. `CheckpointStore` owns compare-and-swap
persistence but does not execute nodes or infer retry safety.

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
`ToolResult`, and `Completion`. `LLMClient` also accepts an optional `RunContext`.

`OpenAIResponsesClient` maps these values to Responses API input items. `LlamaCppChatClient` maps
them to llama-server's OpenAI-compatible Chat Completions messages. Chat Completions represents an
assistant tool request and its observations differently, so the llama.cpp adapter groups adjacent
assistant text and `ToolCall` items into one assistant `tool_calls` message, then emits each
`ToolResult` as a tool message.

Provider SDK types never enter the runtime, application, tool registry, or their tests. Both adapters
use the OpenAI Python SDK for transport, but API compatibility is not treated as identical execution
semantics: request and response mapping remains separate.

When a provider exposes token counts, the adapter maps them to provider-neutral `TokenUsage` on the
`Completion`. The runtime copies those counts into the corresponding `MODEL_REQUEST_COMPLETED`
event. Absence remains `None`, so reports do not confuse unknown usage with zero usage.

## Evaluation Harness

Evaluation suites are JSON documents with a schema version, suite identity, and isolated cases. Each
case defines input, eligible modes, expected answer predicates, exact ordered tool behavior, trace
constraints, and deterministic completion fixtures when applicable.

Deterministic mode replaces only the LLM boundary. It still runs the production `AgentRuntime`, tool
registry, validation, recovery observations, and event emitter. CI uses this mode to catch regressions
without credentials, network access, model drift, or cost. All scripted completions must be consumed,
which makes premature termination visible.

Live mode is explicit and uses `OpenAIResponsesClient`. Cases can opt out when their fixture tests a
harness failure path that a real model should not be asked to reproduce. A live pass rate is a sample,
not a deterministic build invariant; meaningful comparisons require repeated runs and fixed model,
prompt, case, and environment identities.

Version 1 evaluates:

- Exact, non-empty, and case-insensitive final-answer properties.
- Exact tool-call count and order, structural JSON arguments, outcomes, and stable error codes.
- Required event-type subsequences and total model attempts, plus mode-specific elapsed-time and
  token ceilings.
- Raw per-case latency, attempts, tool calls, and provider-reported token counts in JSON reports.

LLM-as-judge remains excluded because all current qualities have direct predicates. The harness is
intentionally product-specific rather than a general benchmark framework.

## Workflow and Durable Execution

`WorkflowDefinition` is an acyclic graph of explicit `WorkflowNode` values. Every node has one end,
next, conditional, or parallel transition. Construction rejects duplicate IDs, missing targets,
cycles, unreachable nodes, and branch nodes that can be entered outside their parallel owner.

Node handlers receive a defensive JSON state snapshot and a child `RunContext`. They return
`NodeResult` updates rather than mutating shared state. A sequential node's updates and selected
transition commit in one checkpoint. `NodeResult.interrupt` commits those updates, records the next
node, and terminates the current run attempt as `interrupted`.

Parallel branches are terminal leaf nodes and run from the same post-coordinator snapshot. The runner
caps concurrency, cancels siblings cooperatively after a failure, and emits results in declared branch
order. Updates are merged only after every branch succeeds. Two branches writing the same key fail
the group; no coordinator or branch update is checkpointed on partial failure. External side effects
that completed before failure cannot be rolled back.

`WorkflowCheckpoint` stores schema version, workflow and definition identity, original input,
committed state, current node, completed nodes, lifecycle status, revision, and last failure. Resume
first claims the loaded revision through compare-and-swap, then reruns the last uncompleted node.
Replay starts a new workflow ID from the original input. A definition ID or version mismatch is
rejected rather than applying old state to changed code.

Every node receives a stable idempotency key derived from workflow ID, definition ID/version, and node
ID. Resume reuses it; replay changes it. This establishes an at-least-once boundary analogous to a job
consumer passing a deduplication key to a transactional downstream service. The local checkpoint
alone cannot provide exactly-once effects.

## Dependency Rules

```text
CLI -> AgentApplication -> AgentRuntime
AgentRuntime -> RunContext + LLMClient + ToolRegistry + neutral models
ToolRegistry -> RunContext + neutral models + jsonschema
OpenAI adapter -> RunContext + neutral models + OpenAI SDK
llama.cpp adapter -> RunContext + neutral models + OpenAI SDK transport
Event adapters -> RunEvent (future concrete integrations)
EvaluationRunner -> AgentRuntime + neutral models + RunEvent
WorkflowRunner -> WorkflowDefinition + CheckpointStore + RunContext + RunEvent
JSON checkpoint store -> WorkflowCheckpoint + local filesystem
```

- Session state is owned above the runtime; one run cannot commit partial history.
- Runtime and tool modules must not import provider SDKs.
- Provider adapters translate wire data and classify infrastructure failures.
- Event sinks observe execution but cannot mutate runtime state.
- Evaluation observes production-owned results and events; it cannot implement alternate execution.
- Workflow nodes depend on shared execution contracts, not provider SDKs or CLI state.
- Checkpoint stores persist scheduler state but do not own workflow transitions or external effects.
- Session/context and multi-agent modules are not introduced before their phases.

## Configuration

`DQAGENT_PROVIDER` selects `openai` or `llama_cpp`. `DQAGENT_MODEL` is required for both. OpenAI
requires `OPENAI_API_KEY`; llama.cpp defaults to `http://127.0.0.1:8080/v1` and uses an optional
`LLAMA_CPP_API_KEY`. `DQAGENT_BASE_URL` overrides either provider-specific URL.

`DQAGENT_TIMEOUT_SECONDS` bounds an individual provider request. `DQAGENT_RUN_TIMEOUT_SECONDS`
bounds one end-to-end agent run and defaults to 120 seconds. `DQAGENT_MAX_MODEL_ATTEMPTS` configures
the runtime's model-attempt budget and defaults to three. All values are validated before composition.

## Testing Strategy

- Runtime tests assert event order, run-ID propagation, retries, cancellation, deadlines, terminal
  classification, tool diagnostic isolation, and sink failure isolation.
- Application tests assert complete-history commit and rollback on failed runs.
- Tool tests cover validation, expected failures, soft timeout, context-aware handlers, and the
  externally visible contracts of both built-in tools.
- Provider tests verify Responses and Chat Completions wire mapping, deadline-derived request timeout,
  tool-history translation, usage extraction, and error classification.
- Evaluation tests assert schema validation, mode isolation, predicates, metrics, report serialization,
  and the committed Phase 3 deterministic case set.
- Workflow tests assert graph validation, conditional routing, deterministic parallel merge, sibling
  cancellation, checkpoint conflicts, interruption, resume, replay, and idempotency-key stability.
- CI runs Ruff, strict mypy, and pytest with at least 85% coverage.
- CI also runs the credential-free deterministic behavioral suite after implementation tests.

No live model evaluation runs in CI because credentials, cost, provider drift, and network
nondeterminism make it an unsuitable correctness gate.

## Current Limitations

- The runtime is synchronous; cancellation cannot preempt a blocking SDK call or Python thread.
- `AgentApplication` conversation state is not safe for concurrent callers.
- Agent-requested tool calls are sequential and tool retries are intentionally unsupported.
- Event sinks are best-effort and no concrete durable telemetry adapter is included.
- Conversation history is process-local, unbounded, and non-streaming.
- There is no approval policy or durable agent conversation/session storage.
- Agent conversation state is still in memory; workflow checkpoints are not session storage.
- JSON file checkpoints retain only the latest revision and coordinate one process. There is no
  distributed lease, checkpoint history, migration framework, or multi-worker recovery.
- Workflows reject cycles, nested parallel subgraphs, dynamic graph mutation, and conflicting branch
  writes. Cooperative cancellation cannot force-stop a blocking node.
- llama.cpp compatibility depends on the selected model's chat template and tool-calling support;
  the adapter does not attempt to infer or rewrite incompatible templates.
- The evaluation corpus is intentionally small, live mode runs once per case, and reports do not yet
  aggregate repeated samples or compare against a prior report automatically.
- Answer predicates are deterministic; there is no semantic or LLM-based judge.

## Related Material

- [ADR-0001: Provider-Neutral LLM Boundary](adr/0001-provider-neutral-llm-boundary.md)
- [ADR-0002: Explicit Tool Boundary and Bounded Agent Loop](adr/0002-explicit-tool-boundary-and-bounded-loop.md)
- [ADR-0003: Observable Runtime and Cooperative Cancellation](adr/0003-observable-runtime-and-cooperative-cancellation.md)
- [ADR-0004: Separate Evaluation Harness from Agent Execution](adr/0004-evaluation-harness-boundary.md)
- [ADR-0005: Checkpoint Deterministic Workflow Progress at Node Boundaries](adr/0005-checkpointed-deterministic-workflow.md)
- [Phase 2 Framework Comparison](learning/phase-2-framework-comparison.md)
- [Roadmap Reassessment After Phase 3](learning/roadmap-reassessment-after-phase-3.md)
- [Phase 4 BFCL and GAIA Comparison](learning/phase-4-bfcl-gaia-comparison.md)
- [Phase 5 LangGraph and EINO Workflow Comparison](learning/phase-5-langgraph-eino-workflow-comparison.md)
