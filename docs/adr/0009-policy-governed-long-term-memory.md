# ADR-0009: Policy-Governed Long-Term Memory

- Status: Accepted
- Date: 2026-08-07
- Accepted: 2026-08-13

## Context

Phase 6 deliberately separates a lossless session transcript from the bounded context sent to a
model. Phase 7 adds request-scoped retrieval of external, cited evidence. Phase 8 needs a third
kind of state: a small, cross-session set of user- or project-scoped facts, preferences, or
experience selected by policy. Saving the transcript as memory would retain noise and model
guesses; putting memory in the retrieval index would confuse personalization with external
evidence and make consent, correction, expiry, and forgetting implicit.

This ADR freezes the behavioral and ownership contract for the v1 design. Its implementation shape
is exercised by T5-T13. The ADR is Accepted for this bounded v1 contract; production encryption,
comprehensive PII classification, authorization proof, distributed tenancy, and scale remain
explicit limitations or deferred capabilities rather than implied guarantees.

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
- Every supported user-facing durable write requires an exact preview and explicit user confirmation.
  The core service enforces the candidate content/scope digest supplied after preview; the standalone
  CLI is the current implementation of the human confirmation step. A caller that invokes the service
  directly must provide its own authorization UX, so the service cannot prove human intent from a
  digest alone. Secret and sensitive content, plus a finite deterministic set of obvious PII patterns,
  is denied in v1 because the local authoritative store is not encrypted. This defense is not a
  complete PII classifier; confidence is evidence about the extractor, never truth or consent.
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

`forget` is a logical application-level deletion guarantee for the active store. It removes the
record payload and raw provenance from SQLite queries, inspection, and recall paths, and atomically
leaves only a minimal content-free tombstone needed for conflict/audit semantics. v1 has no
persistent vector index, so a forgotten record cannot reappear through a stale derived index.
SQLite `secure_delete` is enabled in the current adapter as best-effort local sanitization, but
DQAgent does not promise forensic erasure from filesystem blocks, WAL files, backups, snapshots, or
other copies. If the target is a corrected replacement, the prior superseded history record is a
separate lifecycle record and is not recursively erased; it remains inspection-only and is excluded
from recall.

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

## Implementation Evidence

T5-T13 validate the contract through the production paths rather than test-only doubles:

- `MemoryService` owns preview, exact digest revalidation, policy checks, deterministic
  duplicate-refresh/topic-conflict consolidation, correction, expiry, forgetting, and request-time
  recall. `tests/test_memory_service.py` covers stale previews, clock changes, policy denial, CAS
  conflicts, fail-closed required operations, correction, expiry, and tombstones.
- `SqliteMemoryStore` uses `BEGIN IMMEDIATE`, a scope revision compare-and-swap, one change-set
  transaction, a unique active `(scope, kind, topic)` constraint, and separate records/tombstones.
  `tests/test_memory_store.py` and the SQLite service smoke test cover cross-instance visibility,
  rollback/error behavior, and concurrency. This is cross-connection optimistic concurrency, not a
  distributed lease or tenant boundary.
- `MemorySelector` does no persistent indexing. `MemoryService.recall` filters exact scope,
  confirmation/lifecycle/expiry/sensitivity/kind eligibility before request-time hashing embeddings,
  then applies deterministic score, count, kind, and character limits. `tests/test_memory_recall.py`
  and `tests/test_session_memory_recall.py` cover ranking ties, no-result behavior, scope isolation,
  stale/forgotten exclusion, and atomic selection.
- `ContextBuilder` projects selected memory as a separate lower-authority untrusted user-data block
  with its own budget. Current-request, mandatory prompt, RAG, and recent-turn priority remain
  intact; memory payloads do not enter the durable transcript. `tests/test_context_memory.py` and
  `tests/test_session_memory_recall.py` cover authority, injection-shaped content, RAG separation,
  budget omission, event attributes, and the disabled-path checkpoint.
- `SessionAgentApplication` orders retrieval -> optional memory recall -> context -> runtime ->
  session CAS. Typed memory dependency errors fall back without memory; cancellation, deadlines,
  and unexpected failures escape to the coordinator. The session memory integration tests cover
  those failure paths and assert that a failed session CAS does not write memory.
- `MemoryExtractor` accepts one store-issued, committed, bounded session turn and returns transient
  candidates only. The model path has no tools or store access, derives provenance from the typed
  source, and sends candidates through preview; `tests/test_memory_extraction.py` covers malformed
  output, tool calls, hallucinated provenance, source injection, cancellation, deadline, and
  zero-write failure behavior. ADR-0010 remains the detailed extraction boundary.
- T13 closes the final audit findings with focused evidence: `MemoryEventMetadata` and
  `MemoryServiceError.metadata` expose `scope_id_digest` rather than a raw scope ID;
  `DefaultMemoryPolicy` denies a finite set of obvious SSN, telephone-number, and street-address
  patterns in addition to existing credential and sensitive-term checks; and the CI workflow runs
  and uploads `dqagent-memory-eval`. `tests/test_memory_service.py`,
  `tests/test_memory_policy.py`, and `tests/test_ci_workflow.py` cover these boundaries.
- `evaluations/cases/phase-8-memory-v1.json` contains 13 cases and
  `evaluations/baselines/phase-8-memory-deterministic-v1.json` records the production-path result:
  13/13 passed, false admission 0/3, mean `Recall@k` and `Precision@k` 1.0 over seven applicable
  cases, scope leakage 0/1, stale/forgotten recall 0/3, harmful over-retrieval 0/2, correction
  compliance 1/1, no-result correctness 12/12, and direct answer predicates 13/13. This evidence
  is deterministic fixture regression evidence, not an LLM quality or compliance certification.
- The T13 release run passed `ruff check .`, `mypy src`, `pytest --basetemp
  .local/pytest-phase8-t13` (`424 passed`, `89.06%` coverage), the Phase 3/6/7/8 deterministic
  evaluators, documentation/ADR consistency checks, and `git diff --check`. The release checks
  validate the v1 contract and repository hygiene; they do not close the explicitly deferred
  production capabilities below.

The implementation refines two v1 statements. First, "explicit confirmation" is enforced as an
exact digest at the service boundary and as interactive confirmation in `dqagent-memory`; arbitrary
callers are responsible for their own human authorization. Second, forgetting is logical deletion
from the application-visible store with a tombstone, not forensic erasure. These are limitations of
the current evidence, not capabilities to infer from the interface.

## Consequences

- Long-term state has explicit scope, consent, provenance, lifecycle, correction, and deletion
  boundaries without weakening transcript or retrieval contracts.
- Query-time ranking is simple and immediately reflects correction, expiry, and forgetting, but it
  is O(N) over eligible records with transient vectors and is suitable only for the initial small
  local set. The current hashing embedding is lexical feature hashing, not semantic memory quality.
- SQLite introduces schema/version and local-file privacy responsibilities. The adapter provides
  atomic change sets and cross-connection optimistic concurrency, but not distributed leases,
  tenant isolation, encryption, backup control, or forensic deletion.
- Optional recall can improve personalization without reducing core chat availability, while the
  narrow catch boundary preserves cancellation, deadline, and programming-error visibility.
- Memory context is untrusted lower-authority user data. Delimiting it protects the authority model
  in the harness, but it is not a hard prompt-injection boundary and the current answer checks are
  lexical predicates.
- The current extraction boundary prevents automatic writes, but it does not establish extraction
  truth, user intent, or a background consolidation service. Those remain explicit, deferred work.
- The local store remains unencrypted and cannot satisfy compliance-grade sensitive-data or deletion
  requirements; corrected superseded history is also retained as an inspection-only lifecycle
  record. These limitations are intentional v1 scope, not production guarantees.
- Deterministic content checks cover a finite set of obvious credentials, sensitive terms, SSNs,
  telephone numbers, and street addresses; they are defense-in-depth and not comprehensive PII
  classification.

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

### Treat a successful digest as proof of user authorization

Rejected as a system guarantee because a digest proves that the candidate supplied at confirmation
matches the candidate shown; it does not prove that a human approved it. The CLI currently supplies
the interactive confirmation boundary, while the service keeps the content-binding invariant.

## Explicitly Deferred

The current evidence does not support encrypted sensitive-memory storage, forensic erase,
persistent memory vector indexes, complete PII classification, unconfirmed or automatic writes,
distributed tenancy or leases, background consolidation, bulk deletion, durable audit delivery, or
any broader capability not covered by the Phase 8 deterministic contract. These are deferred rather
than implied by the v1 interfaces.
