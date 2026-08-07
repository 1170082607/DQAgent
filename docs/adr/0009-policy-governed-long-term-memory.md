# ADR-0009: Policy-Governed Long-Term Memory

- Status: Proposed
- Date: 2026-08-07

## Context

Phase 6 deliberately separates a lossless session transcript from the bounded context sent to a
model. Phase 7 adds request-scoped retrieval of external, cited evidence. Phase 8 needs a third
kind of state: a small, cross-session set of user- or project-scoped facts, preferences, or
experience selected by policy. Saving the transcript as memory would retain noise and model
guesses; putting memory in the retrieval index would confuse personalization with external
evidence and make consent, correction, expiry, and forgetting implicit.

This ADR freezes the behavioral and ownership contract for the v1 design. It does not accept a
particular Python class layout, public constructor, or SQLite table schema; those details must be
validated by the later Phase 8 implementation tasks and tests.

## Decision

DQAgent will treat long-term memory as policy-governed selected state in a subsystem separate from
session transcripts, active context, project knowledge, and RAG retrieval. The application-level
memory service coordinates extraction, policy, selection, and storage. Storage and embedding
adapters remain policy-neutral.

### v1 scope and ownership

- A memory scope is explicit and supplied by the composition root. v1 supports `user` and `project`
  scopes. A session ID is not a scope, and neither model text nor a query may choose or broaden a
  scope.
- A durable memory record is an atomic, versioned proposition bound to one scope, with provenance,
  lifecycle, and confirmation state. It is not a transcript turn, context summary, or RAG passage.
- A memory candidate is a transient proposal produced from a committed source turn or an explicit
  user draft. It is not durable state and there is no persistent pending-candidate queue.
- Extractors, including optional model-assisted extractors, have no storage access or mutation
  authority. The model may suggest candidates but cannot write records or invoke memory mutation as
  a tool.
- Every durable write requires an exact preview and explicit user confirmation. Confirmation is
  bound to the candidate content and scope that were shown. Secret and sensitive content is denied
  in v1 because the local authoritative store is not encrypted; confidence is evidence about the
  extractor, never truth or consent.
- Correction creates a new record and atomically supersedes the selected old record. Ordinary
  confirmation cannot silently overwrite a conflicting active topic.

### Source of truth and ranking

An independent local SQLite database is the v1 source of truth for memory payload, provenance,
lifecycle, and scope revision. It is not a replacement for the session, workflow-checkpoint, or
RAG stores. The memory service owns policy decisions; the SQLite adapter only executes exact-scope
loads and atomic change sets with optimistic concurrency. A test adapter may implement the same
contract in memory.

v1 has no persistent memory vector index. Recall loads the exact scope, applies policy eligibility
filters, and computes request-time relevance over the remaining bounded records using the existing
provider-neutral embedding boundary. Scope, confirmation, lifecycle, expiry, and allowed-kind
filters happen before any embedding or ranking input is constructed. Ranking scores express query
relevance only; they do not authorize admission, prove truth, or establish consent. Post-ranking
limits include a score threshold, record count, allowed kinds, and a separate character budget.
Records are selected or omitted atomically, never truncated into partial records. An empty recall is
a successful, explicit no-result.

### Read-stage order and context authority

For a session run, the required order is:

```text
retrieval -> optional memory recall -> bounded context -> AgentRuntime.execute -> session CAS
```

Retrieval and optional memory recall are pre-context stages in the existing `RunCoordinator` run and
share its run ID, deadline, cancellation, ordering, and terminal semantics. Context construction
then receives their request-scoped projections. `AgentRuntime` remains responsible only for the
bounded model/tool stage. The session compare-and-swap occurs only after a successful coordinated
run and remains outside the run lifecycle transaction.

Memory enters context as a separately delimited, lower-authority, untrusted user-data block. It must
not be merged into system instructions, the durable transcript, generated summaries, or RAG
passages, and it cannot create citations or authorize side effects. System policy, the current user
request (including an explicit correction), required recent turns, and RAG evidence retain their
existing authority. Memory has an independent context budget and must not displace mandatory
sections or the active request. With memory disabled or no scope configured, Phase 6 and Phase 7
context items, events, budgets, results, and CLI behavior remain unchanged.

### Failure semantics

Chat recall is optional best effort: a declared, typed memory dependency failure may emit a
memory-recall-failed stage event and continue the run without memory. Best effort must not catch or
relabel cooperative cancellation, an exhausted deadline, or an unknown programming error. Those
errors escape to `RunCoordinator`, which emits the normal terminal failure/cancellation/timeout and
prevents a session commit. Explicit memory management and inspection operations are required
operations and fail closed on dependency errors; they must not turn failure into an empty result.

Memory read projections and recall evidence are never appended to the durable session transcript.
Memory writes are separate management operations after a source session revision has committed, so
extraction, confirmation, or memory CAS failure cannot roll back or mutate the successful chat
transaction. Model-assisted extraction, when introduced, uses its own coordinated run rather than
emitting into an already terminal chat run.

### Forgetting and deferred scope

`forget` is an application-level deletion guarantee. It must remove the record payload and raw
provenance from SQLite queries, inspection, and recall paths, and atomically leave only a minimal
content-free tombstone needed for conflict/audit semantics. v1 has no persistent vector index, so a
forgotten record cannot reappear through a stale derived index. SQLite `secure_delete` or equivalent
local sanitization may be enabled as best effort, but DQAgent does not promise forensic erasure from
filesystem blocks, WAL files, backups, snapshots, or other copies.

Deferred until a later phase and measured evidence justify them: encrypted sensitive-memory storage;
forensic deletion guarantees; persistent or managed vector indexes; unconfirmed or automatic model
writes; background consolidation; distributed tenancy/leases; bulk deletion; and any capability not
covered by the Phase 8 deterministic contract and evaluation evidence.

## Phase 0-7 invariants retained

Phase 8 does not change these existing dependency directions or observable behaviors:

- Provider-specific SDK and wire types stay behind the provider-neutral `LLMClient` boundary
  (ADR-0001). Memory does not import a provider SDK.
- `AgentRuntime` owns the bounded model/tool state machine, retry and repeated-call behavior, and
  its stage events; expected tool failures remain model-visible observations, provider failures and
  loop exhaustion remain run failures, and it does not depend on session, retrieval, memory, or
  workflow modules (ADR-0002/0003/0008).
- `RunCoordinator` owns one end-to-end run's start, ordered non-terminal stage events, error
  binding, and exactly one terminal transition. Pre-model stages use only its restricted scope
  (ADR-0008); memory cannot create a second lifecycle.
- `SessionAgentApplication` owns session orchestration. `SessionStore` remains lossless for
  successful transcript turns, uses revision compare-and-swap, and never persists context,
  retrieval passages, or memory projections (ADR-0006/0008).
- `ContextBuilder` remains the owner of bounded context projection, mandatory prompt sections,
  current-request retention, complete-turn handling, and overflow behavior. Memory is an optional
  lower-authority input, not a store query or a new truncation rule (ADR-0006).
- Retrieval remains an explicit request-scoped boundary with its own provenance, ranking, empty
  result semantics, untrusted-data handling, citation mapping, and failure events. A retrieval
  failure remains terminal, invokes no model, and does not commit the session. Memory is not a
  `RetrievalResult`, `TextChunk`, vector-store document, or citation (ADR-0007).
- Workflow checkpointing and resume remain owned by `WorkflowRunner` with their existing CAS,
  interruption, replay, and at-least-once side-effect semantics; memory is not workflow state
  (ADR-0005).
- Cooperative cancellation, deadline precedence, bounded model/tool calls, commit-after-success,
  and event-sink best-effort behavior remain unchanged (ADR-0002/0003).
- The evaluation harness remains above production execution and deterministic fixtures remain the
  CI gate; any model-assisted extraction or answer-utilization evaluation is separate from that
  gate (ADR-0004).

## Consequences

- Long-term state has explicit scope, consent, provenance, lifecycle, correction, and deletion
  boundaries without weakening transcript or retrieval contracts.
- Query-time ranking is simple and immediately reflects correction, expiry, and forgetting, but it
  is O(N) over eligible records and is suitable only for the initial small local set.
- SQLite introduces schema/version and local-file privacy responsibilities, while providing the
  multi-record transactions and cross-process concurrency that JSON whole-file replacement cannot
  guarantee for memory.
- Optional recall can improve personalization without reducing core chat availability, while the
  narrow catch boundary preserves cancellation, deadline, and programming-error visibility.
- The local store remains unencrypted and cannot satisfy compliance-grade deletion or sensitive-data
  requirements. Those limitations are intentional v1 scope, not production guarantees.

## Alternatives Considered

### Persist the transcript as memory

Rejected because a transcript records every successful turn, including noise, transient requests,
stale information, and assistant guesses. It has no admission, scope, consent, correction, or
forgetting policy.

### Reuse RAG documents or a vector store

Rejected because RAG ranks external evidence and owns document provenance/citations, not user
consent, memory lifecycle, conflict resolution, or application-level deletion. Treating a vector
store as authoritative would also make stale derived data able to resurrect forgotten content.

### Let the model write records directly

Rejected because probabilistic output would become durable-state authority and a prompt-injection
path to persistent mutation. It bypasses deterministic policy, user preview, and confirmation.

### Automatically write candidates above a confidence threshold

Rejected because extractor confidence is not truth, sensitivity classification, user consent, or
long-term usefulness. A numeric threshold cannot safely authorize persistence.

### Reuse the existing JSON file store

Rejected as the authoritative v1 memory store because correction/forget require multi-record atomic
transactions and cross-process scope revisions. The existing JSON session and vector stores remain
in place for their existing responsibilities.

### Add a persistent vector database immediately

Rejected because the initial memory set is expected to be small and request-time ranking is enough to
measure the contract. A persistent derived index would add dual-write, stale-index repair,
migration, and deployment costs before profiling demonstrates a need.
