# Phase 8 Comparison: DQAgent, LangGraph Store, and Letta

## Scope and Evidence

This note compares memory responsibilities that are relevant to DQAgent Phase 8. It does not treat
either project API as DQAgent architecture, and it does not use external source code as evidence
that DQAgent behavior is correct. Local tests and the committed Phase 8 deterministic report are the
behavioral evidence for DQAgent.

The source reading is pinned to the following revisions so that the observations remain reviewable:

- LangGraph: commit
  [`644815f9e5bc52ad8f7a5227a456227e9c3e639b`](https://github.com/langchain-ai/langgraph/tree/644815f9e5bc52ad8f7a5227a456227e9c3e639b).
  The relevant source is the `BaseStore` contract, `InMemoryStore`, and SQLite store adapter:
  [`store/base/__init__.py`](https://github.com/langchain-ai/langgraph/blob/644815f9e5bc52ad8f7a5227a456227e9c3e639b/libs/checkpoint/langgraph/store/base/__init__.py),
  [`store/memory/__init__.py`](https://github.com/langchain-ai/langgraph/blob/644815f9e5bc52ad8f7a5227a456227e9c3e639b/libs/checkpoint/langgraph/store/memory/__init__.py),
  and [`store/sqlite/base.py`](https://github.com/langchain-ai/langgraph/blob/644815f9e5bc52ad8f7a5227a456227e9c3e639b/libs/checkpoint-sqlite/langgraph/store/sqlite/base.py).
- Letta: commit
  [`ff19ffeafeb54bd2a7dc5d4a552f10191732a235`](https://github.com/letta-ai/letta/tree/ff19ffeafeb54bd2a7dc5d4a552f10191732a235),
  whose `pyproject.toml` declares version `0.16.8`. The relevant source is the memory schema,
  block schema/ORM, and block manager:
  [`schemas/memory.py`](https://github.com/letta-ai/letta/blob/ff19ffeafeb54bd2a7dc5d4a552f10191732a235/letta/schemas/memory.py),
  [`schemas/block.py`](https://github.com/letta-ai/letta/blob/ff19ffeafeb54bd2a7dc5d4a552f10191732a235/letta/schemas/block.py),
  [`orm/block.py`](https://github.com/letta-ai/letta/blob/ff19ffeafeb54bd2a7dc5d4a552f10191732a235/letta/orm/block.py),
  and [`services/block_manager.py`](https://github.com/letta-ai/letta/blob/ff19ffeafeb54bd2a7dc5d4a552f10191732a235/letta/services/block_manager.py).

No claim is made here about upstream code after these revisions. DQAgent does not add either
dependency.

## Shared Responsibility Model

The useful common decomposition is:

```text
owner/namespace -> durable memory representation -> bounded read projection -> model context
                                |                         |
                         update/delete/version       search/selection/limits
```

The three projects place different policies at these boundaries:

| Concern | DQAgent Phase 8 | LangGraph Store evidence | Letta evidence |
| --- | --- | --- | --- |
| Ownership | Explicit `user` or `project` `MemoryScope`; the caller supplies the exact scope | `BaseStore` uses hierarchical namespace tuples and keys; namespace is a generic storage boundary | Blocks carry labels and are associated with agents, organizations, and projects in the block model |
| Durable unit | Confirmed atomic `MemoryRecord` with provenance, lifecycle, confirmation digest, and revision | `Item` is a JSON-like value with key, namespace, and timestamps | `Block` is a bounded text section rendered as core memory; the ORM also stores a version and history pointer |
| Admission | Service-owned policy denies sensitive/secret content and requires preview plus exact digest confirmation | `put` stores or updates a value; optional indexing and TTL are store-operation options | `core_memory_append` and exact substring `core_memory_replace` expose mutation of a block |
| Read selection | Exact-scope eligibility before request-time embedding, deterministic ranking, and atomic record limits | `search` supports namespace prefix, filters, optional query, pagination, and adapter-dependent semantic search | The memory schema separates core blocks from archival/recall metadata in the context overview; block rendering places core memory in the prompt |
| Context authority | Selected memory is a separate lower-authority, untrusted user-data block; it cannot create citations or side effects | Store results are values/search items; authority and prompt-injection policy are application concerns | Block values are explicitly part of the in-context memory representation; this is a prompt composition mechanic, not proof of instruction safety |
| Update/delete | Duplicate refresh, explicit correction, expiry, and forget tombstone are separate service operations | `put` updates and `delete` removes a key; TTL may expire items where the adapter supports it | Block changes are persisted through a manager/ORM boundary, with block history and optimistic versioning in the inspected model |
| Indexing | No persistent memory vector index; vectors are computed per recall | `index` configuration can create vector representations for selected fields; SQLite stores vectors when configured | Archival/recall storage and search are broader product mechanisms than the inspected core block API |

The table shows why "memory store" is not one portable abstraction. A generic key-value store,
editable prompt blocks, and policy-governed propositions can all be called memory while providing
different guarantees.

## Reusable Principles

### Make ownership and namespace explicit

LangGraph's `BaseStore` documents namespace tuples and keys as the identity of stored values, and
its `search` contract scopes a query by namespace prefix. Letta's block ORM carries organization,
project, agent, and relationship boundaries around a labelled block. These are useful reminders that
memory ownership must be selected by the application boundary rather than inferred from arbitrary
model text.

DQAgent applies the narrower rule needed by its current contract: the composition root supplies one
`MemoryScope`, and `MemoryService` rejects candidates and recall results whose scope does not match.
The source evidence for this claim is the local `MemoryScope`/`MemoryService` contract and the
cross-scope tests in [`tests/test_memory_service.py`](../../tests/test_memory_service.py) and
[`tests/test_session_memory_recall.py`](../../tests/test_session_memory_recall.py). A namespace
API alone does not establish authorization or tenancy.

### Separate durable state from the model projection

Letta's `Memory` schema explicitly renders blocks into an in-context representation with character
limits and metadata. LangGraph's store returns `Item` or `SearchItem` values, leaving context
assembly to the caller. Both support a useful architectural split: a durable representation and a
model-facing projection need not be the same object.

DQAgent makes that split stricter. `ContextBuilder` accepts a completed `MemoryRecall` and creates a
bounded, separately delimited user-data block. It does not access storage or decide eligibility.
Memory records and recall evidence are not written into a session transcript. This is supported by
[`src/dqagent/context.py`](../../src/dqagent/context.py),
[`src/dqagent/application.py`](../../src/dqagent/application.py), and the context/session memory
tests. The reusable principle is projection ownership; the exact lower-authority marker is DQAgent
mechanics.

### Keep indexing and selection explicit

LangGraph's `BaseStore` treats semantic search as optional and configures which value fields are
indexed. Its in-memory implementation embeds configured text and ranks cosine similarity results;
its SQLite adapter has a separate vector table and search path. This supports a general principle:
index lifecycle and read selection should be visible configuration, not an accidental side effect of
serializing a value.

DQAgent deliberately chooses a smaller initial boundary. `MemorySelector` uses the existing neutral
embedding interface at request time, reports selector identity, applies deterministic tie-breaking,
and records why candidates were omitted. There is no dual-write index to repair after correction or
forgetting. The Phase 8 baseline therefore measures architecture behavior with lexical hashing, not
semantic search quality. A future persistent index would need a new lifecycle, migration, stale-data,
forgetting, and measured-scale contract.

### Version changes and concurrent ownership

The inspected LangGraph SQLite adapter groups store operations behind a connection lock and explicit
transactions. Letta's block ORM declares an optimistic version counter, and its block manager uses
the database session to flush updates while the caller controls the final commit. These are useful
distributed-systems lessons: a mutable memory boundary needs an ownership or conflict rule, and a
successful write should have a visible version/history story.

DQAgent's rule is more specific: `SqliteMemoryStore.apply` starts `BEGIN IMMEDIATE`, loads a scope
revision, compares the caller's expected revision, applies one validated change set, and commits or
rolls back atomically. `correct` uses one change set for supersession plus replacement; `forget`
removes payload and provenance and leaves a content-free tombstone. The local tests prove
cross-connection CAS behavior, but they do not prove distributed leases, exactly-once effects, or
tenant isolation. This is optimistic concurrency for a local memory store, not a complete ownership
protocol.

## Where the Analogies Fail

The backend analogy "memory is a database-backed state service" is helpful for CAS, schema, and
transaction reasoning, but it fails in several important ways:

- A LangGraph `Item` or Letta `Block` is not automatically a user-approved proposition. Storage
  success does not establish truth, consent, sensitivity classification, or long-term usefulness.
- Letta core memory is intentionally prompt-facing mutable text. DQAgent memory records are
  policy-governed propositions that enter context as lower-authority data. A block's presence in the
  system's context does not prove it is safe to follow as an instruction.
- LangGraph namespace-prefix search is broader and more flexible than DQAgent exact-scope recall.
  Prefix matching must not be treated as proof of tenant isolation for a privacy-sensitive product.
- TTL expiration or a normal delete is not the same as DQAgent's lifecycle-aware expiry and logical
  forgetting tombstone. Conversely, DQAgent's tombstone is not forensic erasure from database files,
  backups, or snapshots.
- A vector similarity score ranks relevance. It cannot authorize a write, prove a fact, or make a
  recalled record authoritative. The same limitation applies to DQAgent's hashing score and to an
  optional vector-enabled Store.
- Prompt delimiters, XML-like block rendering, and `untrusted_data` labels are model-facing
  conventions. They reduce authority confusion but are not a hard language-level security boundary.

## Deliberate Non-Adoptions

DQAgent does not adopt LangGraph Store or Letta as dependencies because the current implementation
needs to keep admission, confirmation, correction, forgetting, scope isolation, and failure
semantics visible in project-owned code. The comparison supports the following limited conclusions:

- Reusable: explicit ownership, a durable-versus-projection split, configurable indexing, bounded
  context sections, and visible version/conflict behavior.
- Framework mechanics: LangGraph operation envelopes, namespace-prefix matching, adapter-specific
  vector tables/TTL, Letta block labels/rendering, agent/project relationships, block-history
  APIs, and SQLAlchemy session details.
- Not established by source reading: encryption, forensic deletion, automatic-write safety,
  distributed tenancy, background consolidation quality, or model extraction truth.

The following remain explicitly deferred in DQAgent: encrypted sensitive-memory storage, forensic
erase, persistent memory vector indexing, unconfirmed or automatic writes, distributed tenancy or
leases, background consolidation, and any capability not supported by the Phase 8 deterministic
contract. The pinned comparisons motivate these boundaries; they do not close them.

## Local Evidence

The claims about DQAgent behavior are grounded in the implementation and tests below, not in the
framework comparison:

- [`src/dqagent/memory_service.py`](../../src/dqagent/memory_service.py): policy, digest, lifecycle,
  consolidation, correction, forgetting, and recall orchestration.
- [`src/dqagent/memory_store.py`](../../src/dqagent/memory_store.py): exact-scope snapshots,
  transaction/CAS, SQLite schema, records, and tombstones.
- [`src/dqagent/memory_recall.py`](../../src/dqagent/memory_recall.py): request-time selection,
  deterministic ranking, and atomic post-rank limits.
- [`src/dqagent/memory_extraction.py`](../../src/dqagent/memory_extraction.py): bounded committed
  source, transient candidates, strict model output, and preview-only pipeline.
- [`evaluations/cases/phase-8-memory-v1.json`](../../evaluations/cases/phase-8-memory-v1.json) and
  [`evaluations/baselines/phase-8-memory-deterministic-v1.json`](../../evaluations/baselines/phase-8-memory-deterministic-v1.json):
  the 13-case deterministic contract and its committed baseline.
