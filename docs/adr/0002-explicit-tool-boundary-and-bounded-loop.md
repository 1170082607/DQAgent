# ADR-0002: Explicit Tool Boundary and Bounded Agent Loop

- Status: Accepted
- Date: 2026-07-23

Phase 3's ADR-0003 supersedes only the decision that `AgentApplication` owns the loop. The tool
contract, loop semantics, and commit-after-success behavior remain accepted.

## Context

Phase 2 needs model-selected application actions without adopting an agent framework. The design
must keep OpenAI wire types out of application state, validate untrusted model arguments, and make
loop termination and tool failure behavior inspectable.

## Decision

Represent tool metadata, model tool calls, and execution results as provider-neutral values. A
`ToolRegistry` owns name lookup, Draft 2020-12 JSON Schema validation, handler invocation, and result
classification. `jsonschema` is used because partial handwritten validation would create a false
standards-compliance boundary.

`AgentApplication` owns the synchronous ReAct-style loop. Each model output either ends the run with
text or appends tool calls and observations before the next model request. A run permits at most
eight model calls by default. It rejects duplicate call IDs and duplicate normalized
`(tool name, arguments)` pairs within one run.

Expected tool failures become structured, model-visible observations. Provider failures and loop
exhaustion fail the run. Conversation history is committed only after a final answer is available.

## Consequences

- The application loop and its tests do not import the OpenAI SDK.
- Tool schemas are explicit and reviewable instead of inferred from Python signatures.
- Unknown tools, invalid arguments, handler exceptions, and timeouts use stable error codes.
- A timed-out synchronous handler runs in a worker thread that cannot be safely killed. The caller
  returns a timeout observation, but the handler may continue in the background. Phase 3 adds
  cooperative cancellation and a run deadline; hard cancellation still requires a process,
  container, or remote-worker isolation boundary.
- Conversation rollback cannot undo external side effects from tools that already ran. Tools that
  mutate state still need idempotency keys or transactional safeguards at their own boundary.
- Semantic repeated-call rejection is intentionally conservative. Polling and other legitimate
  repeated actions require a future policy rather than silently weakening the loop guard.

## Alternatives Considered

### Use OpenAI SDK request and response types throughout the loop

Rejected because it would make application state and tool execution provider-specific, reversing
the dependency established by ADR-0001.

### Infer schemas from handler signatures

Rejected for this phase because it hides the metadata-to-validation mechanism that the project is
intended to teach. Mature frameworks provide this convenience once the underlying boundary is
understood.

### Adopt an agent framework

Rejected because the current capability is small enough to implement and test directly. Framework
features such as streaming, tracing, approvals, handoffs, checkpointing, and cancellation belong to
later roadmap phases.

### Force-stop timed-out handlers

Rejected because Python threads cannot be safely terminated. Per-call processes would provide a
stronger boundary but impose serialization and lifecycle constraints beyond the local synchronous
runtime introduced in Phase 3.
