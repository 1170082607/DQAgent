# ADR-0008: Coordinate the End-to-End Run Above AgentRuntime

- Status: Accepted
- Date: 2026-08-04

## Context

ADR-0003 assigned lifecycle transitions to `AgentRuntime` when one observable run was the
model/tool loop. Phase 7 widened the useful run boundary: retrieval and bounded context construction
happen before the model loop but must share its run ID, deadline, ordered events, cancellation, and
terminal failure semantics.

Putting retrieval or session storage inside `AgentRuntime` would reverse the intended dependency
direction. The first Phase 7 implementation instead exposed a runtime-created `RunExecution` to
`SessionAgentApplication`. That object prevented callers from forging arbitrary states, but the
application still invoked `start_execution`, emitted lifecycle-related stage events, and called
`fail` on pre-model errors. Lifecycle policy was encapsulated while lifecycle ownership remained
split across application and runtime control flow.

## Decision

Introduce an independent `RunCoordinator` above the application stages and `AgentRuntime`.
`RunCoordinator.execute` owns the complete lifecycle boundary: it creates one event stream, emits
exactly one start event, invokes a supplied operation, binds escaped errors to the run, classifies
failure/cancellation/timeout, and emits exactly one terminal event.

The operation receives a capability-limited `RunScope`. A scope exposes the shared `RunContext` plus
`emit` and `emit_error` for non-terminal stage events. It has no public start, complete, or fail
operation and rejects lifecycle event types. Retaining a scope after the coordinated operation ends
does not permit further event emission.

`AgentRuntime.execute` owns only the bounded model/tool stage: model attempts, provider retry,
tool-call processing, repeated-call protection, and their stage events. `AgentRuntime.run` remains a
convenience entry point for callers with no pre-model stages; it delegates lifecycle ownership to its
default coordinator and calls `AgentRuntime.execute` inside that boundary.

`SessionAgentApplication` invokes a coordinator around retrieval, context construction, and
`AgentRuntime.execute`. It may use the runtime's default coordinator or receive a separately composed
coordinator. The session compare-and-swap remains outside the run lifecycle and occurs only after a
successful coordinated run.

This amends the lifecycle ownership part of ADR-0003. `AgentRuntime` continues to own its bounded
model/tool state machine and retry policy; `RunCoordinator` owns the end-to-end run lifecycle and
ordered event stream.

## Consequences

- Application and runtime stages cannot independently start or terminate a run through their public
  capability.
- Retrieval, context, model, and tool events retain one run ID and sequence without making the agent
  runtime depend on retrieval or session modules.
- A successful model/tool stage is represented by `AgentExecutionResult`; lifecycle metadata is
  attached afterward as `AgentRunResult` from the coordinator's immutable `RunRecord`.
- Unexpected failures escaping any coordinated stage receive one common terminal fallback, while
  stage owners may still translate raw dependency failures into a more specific `DQAgentError`.
- Session commit failure can still follow `RUN_COMPLETED`; the run record describes computation, not
  the surrounding durable session transaction.
- Workflow lifecycle remains owned by `WorkflowRunner`, whose interruption, resume, and checkpoint
  state machine differs from an agent run.

## Alternatives Considered

### Keep the runtime-created RunExecution handle

Rejected because restricting terminal methods does not fix split ownership while application code
still decides start and failure transitions through a runtime-owned object.

### Move retrieval and context construction into AgentRuntime

Rejected because retrieval policy, prompt projection, and durable session state are application
responsibilities. Injecting them into the model/tool engine would create a stronger dependency
inversion than the lifecycle problem being fixed.

### Exclude pre-model stages from the run lifecycle

Rejected for Phase 7 because retrieval failure, cancellation, and timeout need the same end-to-end
identity and terminal observability as model and tool failures.

### Expose a general-purpose event emitter

Rejected because an emitter constructs records but cannot enforce legal state transitions, terminal
uniqueness, error binding, or closure after completion.
