# Phase 5 Comparison: DQAgent, LangGraph, and EINO Workflow Execution

## Scope and Evidence

This comparison asks which persistence and graph semantics belong in DQAgent Phase 5. It does not
compare framework ergonomics or recommend replacing the local implementation.

Repository HEADs were resolved on 2026-07-29 as LangGraph commit
[`4134145`](https://github.com/langchain-ai/langgraph/tree/41341457342327166d72fc11952ab28fb61ec0bf)
and EINO commit
[`ca0441a`](https://github.com/cloudwego/eino/tree/ca0441ac0bceed8945dcf7d5a18c237c924c6aa8).
The environment could resolve Git refs but could not retrieve GitHub raw/API pages, so LangGraph
details below are limited to its stable public persistence and interrupt contracts. EINO conclusions
also reuse the source evidence recorded for the same commit in the Phase 2 comparison. This note does
not claim coverage of unverified internal changes at either HEAD.

## Shared Execution Model

All three systems separate deterministic graph control from work performed inside a node:

```text
committed state -> selected node(s) -> state updates -> transition -> checkpoint
```

The graph owns known ordering, branching, stopping, and recovery. An agent node may still ask a model
what to do internally, but that does not make graph transitions model-controlled.

## Comparison

| Concern | DQAgent Phase 5 | LangGraph | EINO |
| --- | --- | --- | --- |
| State | JSON object plus update mappings | State channels with reducers and snapshots | Typed graph input/output and component values |
| Scheduling | Synchronous DAG; bounded terminal leaf fan-out | Pregel-style supersteps with pending task/write state | Compiled graph execution with Go context propagation |
| Persistence | One latest checkpoint with CAS revision | Checkpoint history keyed by thread/configuration namespaces | Checkpoint and interrupt support integrated with graph execution |
| Interrupt | Node commits updates and pauses before its successor | Interrupt/resume values are part of graph task state | Interrupt/resume is represented through graph execution state |
| Parallel merge | Unique keys, declaration-order result events | Channel reducers define how concurrent writes combine | Graph topology and aggregation determine joins |
| Recovery | At-least-once from last uncompleted node | Resume from persisted superstep/task state | Resume through compiled graph/checkpoint facilities |

## Reusable Lessons

### Persistence is scheduler state, not only business data

A resumable graph must store where execution continues, not merely the current values. DQAgent
therefore stores `current_node`, completed nodes, definition version, status, and revision beside the
business state. This resembles persisting a job scheduler cursor together with its payload.

### Concurrent writes need an algebra

LangGraph channels/reducers illustrate that parallel state updates require explicit merge semantics.
DQAgent does not need a reducer registry yet, so Phase 5 chooses the smallest safe rule: disjoint keys
only. Silent last-writer-wins behavior would hide conflicts, while a general reducer abstraction would
be speculative with only one workflow implementation.

### Interrupt and resume are not exception retry

An interrupt is an expected terminal outcome for one run attempt. Its checkpoint points at the next
node. A failure points at the uncompleted current node and may rerun it. Treating both as exceptions
would erase an important operator-visible state distinction.

### A checkpoint cannot guarantee exactly-once side effects

Framework checkpointing can restore scheduler progress, but a process can fail after an external
effect and before its checkpoint commits. DQAgent exposes a stable node idempotency key instead of
claiming exactly-once behavior. The external service must enforce deduplication or transactions.

## Deliberate Non-Adoptions

- No general state-channel or reducer system: conflicting parallel keys fail explicitly.
- No cycles or dynamic graph mutation: Phase 5 workflows are finite DAGs.
- No nested parallel subgraphs: parallel branches are terminal leaf nodes followed by one continuation.
- No checkpoint history/query API: the local stores retain only the latest revision.
- No distributed lease: the JSON store is atomic and process-local, not a multi-worker scheduler.
- No framework dependency: current requirements fit a small provider-neutral implementation.

## Sources

- [LangGraph repository at `4134145`](https://github.com/langchain-ai/langgraph/tree/41341457342327166d72fc11952ab28fb61ec0bf)
- [LangGraph persistence guide](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangGraph interrupts guide](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [EINO repository at `ca0441a`](https://github.com/cloudwego/eino/tree/ca0441ac0bceed8945dcf7d5a18c237c924c6aa8)
- [Phase 2 framework comparison](phase-2-framework-comparison.md)
