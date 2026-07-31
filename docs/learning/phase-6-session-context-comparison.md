# Phase 6 Comparison: DQAgent Session and Context Boundaries

## Scope and Evidence

This comparison asks which session and context semantics belong in DQAgent Phase 6. It does not
recommend adopting a framework or claim that framework session APIs define the architecture.

The local source-reading baseline is the OpenAI Agents Python material previously inspected at
commit `1e8d506a32ea7b84f3a5a811e101378c0b1bc137` for the Phase 2 comparison and LangGraph persistence
material inspected at commit `41341457342327166d72fc11952ab28fb61ec0bf` for Phase 5. On 2026-07-30,
LangGraph HEAD still resolved to `4134145`, but this environment could not retrieve GitHub raw/API
content or resolve OpenAI Agents SDK HEAD reliably because of TLS failures. Conclusions below are
therefore limited to the already verified public contracts; no claim is made about newer internals.

## Shared Model

All useful designs need two responsibilities even when one framework API hides the split:

```text
durable history/state -> context selection/compaction -> bounded model request
                    \-> recovery, inspection, and later re-projection
```

The storage side resembles a session or checkpoint service. Context construction resembles a query
projection over that durable record. Treating the projection as the source record confuses recovery
with prompt optimization.

## Comparison

| Concern | DQAgent Phase 6 | OpenAI Agents SDK contract | LangGraph contract |
| --- | --- | --- | --- |
| Durable identity | Explicit session ID and CAS revision | Runner can integrate session-backed run history | Thread/configuration identity selects persisted graph state |
| Stored value | Full neutral conversation transcript | Session history is added around runner input/output | Checkpoint contains graph/channel state, often including messages |
| Active context | Separate `ContextBuilder` projection | Runner/session integration supplies history to a run | Graph nodes or middleware transform message state before model calls |
| Trimming unit | Complete user turns, including tool pairs | Not inferred from the previously inspected runner contract | Reducers/message operations are framework state mechanics, not a DQAgent policy |
| Summary evidence | Method, digest, source counts, provider identity | No newer compaction internals were verified in this reading | Checkpoint metadata and history are richer than DQAgent's latest-only session file |
| Concurrency | Process-local serialization plus CAS failure | Backend-specific session semantics | Checkpointer/backend and graph execution semantics determine conflicts |

## Reusable Lessons

### Session state belongs above one agent run

The runtime should not decide what a user session means. DQAgent keeps one run transactional: it
returns newly generated items, and the application commits them only after success. This preserves
the Phase 3 invariant while allowing multiple storage implementations.

### Checkpointing and conversation history are related but not interchangeable

LangGraph persistence shows that scheduler state needs execution position and pending work, not only
messages. DQAgent workflow checkpoints therefore remain separate from session transcripts. Reusing
the workflow checkpoint model for chat would couple graph recovery semantics to conversation data.

### Context reduction is an application policy

A generic framework can expose message transforms, reducers, or session hooks, but it cannot know
which project constraints are critical. DQAgent starts with deterministic whole-turn selection and
measures known loss. Model summarization is optional because it adds latency, cost, and another
probabilistic failure mode.

### Persistence does not solve concurrent ownership by itself

A store that can load and save history still needs conflict behavior. DQAgent rejects stale revisions.
This is analogous to optimistic locking in a backend service: it prevents lost updates, but without a
lease it cannot prevent duplicate model/tool work before the losing transaction commits.

## Deliberate Non-Adoptions

- No framework session dependency: the current neutral transcript and two local stores are sufficient.
- No graph checkpoint reuse: sessions do not own node position or workflow replay.
- No automatic memory extraction: selected cross-session memory belongs to Phase 8.
- No retrieval index: on-demand project knowledge is an allowlisted file read, not Phase 7 RAG.
- No claim of lossless summary: the Phase 6 evaluation intentionally records a bounded structural
  compaction case where a later old marker disappears.

## Sources

- [OpenAI Agents SDK comparison at pinned commit](phase-2-framework-comparison.md)
- [LangGraph persistence comparison at pinned commit](phase-5-langgraph-eino-workflow-comparison.md)
- [OpenAI Agents SDK repository](https://github.com/openai/openai-agents-python)
- [LangGraph repository at `4134145`](https://github.com/langchain-ai/langgraph/tree/41341457342327166d72fc11952ab28fb61ec0bf)
