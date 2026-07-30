# ADR-0005: Checkpoint Deterministic Workflow Progress at Node Boundaries

- Status: Accepted
- Date: 2026-07-29

## Context

Phase 5 needs deterministic multi-step control flow, interruption, and recovery without replacing
the Phase 3 agent loop or adopting a graph framework. Persistence must make the recovery point
explicit, but a local checkpoint cannot make arbitrary external side effects transactional.

Parallel execution adds another ambiguity: branches may finish in different orders, write the same
state keys, or partially complete before a sibling fails. Leaving merge and retry behavior implicit
would make replay nondeterministic and could duplicate effects.

## Decision

Define an acyclic `WorkflowDefinition` containing explicit nodes and end, next, conditional, or
parallel transitions. A node receives a defensive JSON state snapshot plus a child `RunContext` and
returns JSON-compatible updates in `NodeResult`. Conditional selectors run after node updates.

Sequential nodes commit after each node. A parallel transition runs two or more terminal branch
nodes from the same snapshot with configured maximum concurrency. Branch updates merge in declaration
order, and two branches writing the same key fail the group. Any branch failure cooperatively cancels
siblings; no branch update or coordinator update is checkpointed on partial failure.

`WorkflowCheckpoint` records the definition identity and version, original input, committed state,
next node, completed nodes, lifecycle state, revision, and last error. Stores implement compare-and-
swap `save`. Resume claims a new revision before executing, continues from the last uncompleted node,
and rejects a changed definition. Replay starts a new workflow ID from the original input.

Each node receives a stable `idempotency_key` in `RunContext.metadata`. Resume reuses the same key for
an uncommitted node; replay uses a new key because it is a new logical execution. Side-effecting
handlers must pass that key to an external system that supports deduplication. Checkpointing provides
at-least-once recovery, not exactly-once effects.

Workflow execution extends the existing `RunState`, `RunEvent`, `RunEventType`, event sinks, and
`RunContext`. It does not create a second lifecycle abstraction. The JSON file store uses a hashed
workflow filename and atomic replacement; its revision lock is process-local.

## Consequences

- Invalid targets, cycles, unreachable nodes, and invalid parallel ownership fail before execution.
- Interruption is durable only after a successful node and requires a next node to resume.
- State and checkpoint evidence are deterministic even when branch completion timing is not.
- A failed node is retried from its last checkpoint and may execute more than once.
- Parallel branch side effects cannot be rolled back; all-or-nothing applies only to workflow state.
- Nested parallel graphs, cycles, dynamic graph mutation, distributed leases, and exactly-once effect
  delivery remain unsupported.
- JSON state is deliberately less flexible than arbitrary Python objects but is inspectable and
  portable across processes.

## Alternatives Considered

Adopting LangGraph or EINO was rejected because Phase 5 is intended to expose the execution model,
and DQAgent does not yet need their broader component ecosystems. Event sourcing every state mutation
was rejected because replay reducers, schema migration, and compaction would dominate this phase.
Persisting only final workflow output was rejected because it cannot resume. Last-writer-wins branch
merge was rejected because timing or declaration details could silently overwrite valid work.
