# ADR-0003: Observable Runtime and Cooperative Cancellation

- Status: Accepted
- Date: 2026-07-23
- Lifecycle ownership amended by: [ADR-0008](0008-coordinate-end-to-end-run-above-agent-runtime.md)
- Subprocess isolation extended by: [ADR-0012](0012-separate-workspace-containment-from-process-isolation.md)

## Context

The Phase 2 loop can bound model-call count and convert expected tool failures into observations,
but it has no identity or lifecycle outside an in-memory `send` call. Provider errors are too broad
for retry policy, tool exceptions lose their internal diagnostics, and a model request or tool call
cannot share one end-to-end deadline.

Adding these concerns directly to `AgentApplication` would mix durable conversation ownership with
the lifecycle of one execution. Provider SDK retries would also make retry count and events opaque
to the application.

## Decision

Introduce `RunContext` as the provider-neutral execution context for one run. It owns a UUID by
default, an optional monotonic deadline, thread-safe cooperative cancellation, and read-only
metadata. The same context is passed to the LLM boundary and every tool handler.

Move the model/tool loop into `AgentRuntime`. `AgentApplication` continues to own conversation state
and commits only a successful `AgentRunResult`. The runtime owns iteration bounds, repeated-call
protection, provider retry policy, and model/tool stage events. ADR-0008 later moves the widened
end-to-end lifecycle and ordered event stream into an independent `RunCoordinator` so pre-model
application stages can participate without sharing terminal ownership with the runtime.

Runtime events are immutable records with a run ID, sequence, timestamp, lifecycle state, and typed
event name. Event sinks are best-effort hooks: sink failures are logged and do not change run
semantics. Sinks can adapt the same event stream to tracing, metrics, or audit storage. Tool events
may contain internal diagnostic text, while the model receives only the existing sanitized
`ToolResult`. Every started model request emits a completed or failed event, and runtime errors are
bound to the current run ID before attempt and terminal events are emitted.

Only `LLMProviderError` values explicitly marked retryable are retried. The OpenAI adapter classifies
timeouts, rate limits, connection failures, and server failures as retryable and disables the SDK's
implicit retry layer. Tool calls are not automatically retried because the runtime cannot infer
idempotency or side-effect safety.

Cancellation is cooperative. Runtime waits and context-aware handlers respond promptly, and the
OpenAI request timeout is capped by the remaining run deadline. A synchronous SDK call or Python
thread that ignores the context cannot be force-stopped. Monotonic waits recheck their target after
timer wake-ups, and an exhausted run deadline takes precedence over an individual provider timeout.

## Consequences

- Every model attempt and tool call can be correlated by one run ID and ordered event sequence.
- Tests and adapters can observe lifecycle behavior without importing provider SDK types.
- Retry count and backoff are explicit and deadline-aware instead of hidden in the OpenAI client.
- Conversation history still has commit-after-success semantics on cancellation, timeout, or error.
- Tool handlers now accept `RunContext` and are responsible for checking it during long work.
- Best-effort sinks are suitable for development telemetry, but compliance audit delivery requires
  a future durable, fail-closed boundary.
- Hard termination requires process or remote-worker isolation and remains out of scope.
- The synchronous runtime is not safe for concurrent calls sharing one `AgentApplication` instance.

## Alternatives Considered

### Keep the loop in `AgentApplication`

Rejected because session state and per-run lifecycle have different ownership and evolution. A
workflow engine needs a reusable runtime contract, not instrumentation embedded in one chat use case.

### Rely on OpenAI SDK retries and timeouts

Rejected because SDK retries are provider-specific and invisible to runtime events. They also risk
stacking with application retries and obscuring the actual attempt budget.

### Automatically retry tools

Rejected because retries can duplicate external side effects. Tool-specific idempotency keys and
retry declarations must exist before the runtime can make that decision safely.

### Force-stop timed-out threads

Rejected because Python provides no safe thread termination mechanism. Pretending otherwise would
make the deadline contract stronger on paper than in execution.
