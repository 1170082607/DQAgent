# DQAgent Architecture

## Status

This document describes the implemented Phase 8 T5-T13 architecture and the Phase 9 T1-T5
foundations. Memory management is available as a model-free explicit application service,
request-time policy-filtered recall, an optional durable session read stage, a bounded context
projection, a pure source-to-transient-candidate extraction boundary, and independent memory/session
CLI composition. Phase 9 currently has workspace authority/observation, prepared-action governance,
exact foreground approval, synchronous hook contracts, and a governed execution/runtime bridge;
coding adapters and subprocess backends remain later checkpoints.
The roadmap remains the source of truth for deferred capabilities.

## System Context

DQAgent is a local command-line agent. It can maintain in-memory conversation state or persist a
session transcript, project that transcript into a bounded active context, and coordinate each user
turn across retrieval, context construction, and an observable runtime. The runtime calls a configured
model, executes application-owned tools, and continues until it produces a result or fails.

```text
User -> CLI -> AgentApplication -> RunCoordinator -> AgentRuntime -> LLMClient
                                      |                |
                                      |                +-> ToolRegistry -> legacy handler or governed ActionTool
                                      |
                                      +-> RunContext + RunScope
                                      +-> EventSink adapters

Versioned case -> EvaluationRunner -> AgentRuntime -> EvaluationReport
                       |                   |
                       +-> scripted LLM    +-> production tools and events
                       +-> live LLM

Workflow input -> WorkflowRunner -> validated node graph -> WorkflowRunResult
                        |                   |
                        +-> CheckpointStore +-> sequential / conditional / parallel nodes
                        +-> RunContext + shared lifecycle events

Session ID -> SessionAgentApplication -> RunCoordinator -> Retriever
                       |                        +-> MemoryService -> exact MemoryScope
                       |                        +-> ContextBuilder
                       |                        +-> AgentRuntime
                       |
                       +-> SessionStore (full transcript + CAS revision after run success)

Context case -> ContextEvaluationRunner -> production ContextBuilder -> context report

Memory case -> MemoryEvaluationRunner -> MemoryService + MemorySelector + ContextBuilder
                                      -> SessionAgentApplication + scripted fixtures -> layered report

Memory management request -> dqagent-memory CLI -> MemoryService
                                      |             +-> MemoryPolicy + MemoryConsolidator + MemoryStore
                                      +-> explicit scope, confirmation, and output boundary

MemoryRecallRequest -> MemoryService -> exact-scope MemoryStore + MemoryPolicy
                                      +-> MemorySelector -> EmbeddingProvider
                                      +-> MemoryRecall (selected/omitted matches)

MemoryRecall -> ContextBuilder -> lower-authority memory projection

CommittedSessionTurn -> MemoryExtractor -> transient MemoryCandidate
                                      |
                                      +-> MemoryService.preview -> explicit digest confirmation
```

The main CLI is the composition root for model-backed chat. The independent `dqagent-memory` CLI is
another composition root: it creates only `DefaultMemoryPolicy`, `MemoryService`, and
`SqliteMemoryStore`; it does not load settings, provider credentials, models, agent tools, or chat
state. Every memory command receives `--scope-kind` and `--scope-id` explicitly. Its database
defaults to `.local/memory.sqlite3` and can be overridden with `--database`.

When `--session-id` is supplied, the CLI composes `JsonFileSessionStore`, `PromptAssembler`, and
`ContextBuilder` around `SessionAgentApplication`. Optional read-only memory recall is composed only
when `--memory-database`, `--memory-scope-kind`, and `--memory-scope-id` are all supplied; the
application receives one `MemoryService` and one exact `MemoryScope`. A later process using the same
ID resumes the stored transcript. Without `--session-id`, memory configuration is rejected and the
original `AgentApplication` behavior remains.

The evaluation CLI is a separate composition root. It loads versioned cases, selects either a
scripted or live `LLMClient`, creates an isolated production runtime per case, and writes a structured
report. Evaluation does not add a second agent loop.

Workflow definitions are application composition. `WorkflowRunner` executes known control flow and
persists progress through a `CheckpointStore`; an agent runtime may be called inside a node when a
step genuinely needs model decisions, but workflow transitions remain deterministic.

## Phase 8 Memory Data Flow

Memory has separate write and read paths. The write path is an explicit management operation; the
chat path only performs an optional read. Neither path treats the model as durable-state authority:

```text
committed session turn or user draft
    -> MemoryExtractor (optional, transient candidates only)
    -> MemoryService.preview / policy admission
    -> exact candidate + digest shown to caller
    -> explicit confirmation
    -> MemoryService rechecks scope, policy, clock, and digest
    -> MemoryConsolidator (add, duplicate refresh, or explicit conflict)
    -> MemoryStore exact-scope CAS transaction
    -> confirmed MemoryRecord / content-free tombstone

current request + explicit MemoryScope
    -> MemoryService.recall
    -> exact-scope snapshot
    -> policy/lifecycle/sensitivity/kind eligibility
    -> request-time embedding and deterministic ranking
    -> score/count/kind/character post-limits
    -> MemoryRecall selected or atomically omitted records
    -> ContextBuilder lower-authority untrusted user-data block
    -> AgentRuntime model/tool stage
    -> SessionStore CAS of user message and new agent items only
```

The extraction branch is not attached automatically to a successful chat turn. T10's
`CommittedSessionTurn` is store-issued, selects one complete bounded turn, and contains source
digest/revision metadata rather than a copy of the full transcript. `ModelMemoryExtractor` sends no
tools, rejects non-JSON/unknown/provenance-shaped output, and produces only transient candidates.
`MemoryExtractionPipeline` can preview those candidates but cannot confirm them.

The read branch is exact-scope and request-time. `MemoryService` loads the scope, rejects records
that are not confirmed durable active records or that are superseded, expired, not-yet-valid,
sensitive/secret, or outside the allowed kind set, then invokes `MemorySelector`. The v1 selector
uses the existing provider-neutral embedding boundary but persists no memory vectors. It ranks by
dot product with deterministic memory-ID ties and admits complete records until score, count, kind,
or character limits are reached. An empty recall is a successful no-result, not a storage failure.

`ContextBuilder` receives the completed `MemoryRecall`; it never queries the store. It places selected
memory in a separate `USER` block marked `untrusted_data=true` and `authority=lower-authority`.
Current-request text, mandatory prompt policy, RAG evidence, and required recent turns retain their
existing authority and priority. Memory has an independent character budget, records are omitted
atomically, and memory content is excluded from system instructions, summaries, RAG passages, and
the durable session transcript. This delimiter is an authority convention, not a hard prompt-
injection sandbox.

The current evidence supports the following transaction boundary:

- `MemoryService` decides the change set; `MemoryStore` does not infer policy or consolidation.
- `SqliteMemoryStore.apply` opens `BEGIN IMMEDIATE`, loads the exact scope revision, rejects a
  stale expected revision, applies one validated change set, rewrites records/tombstones for that
  scope, and commits or rolls back as one SQLite transaction.
- SQLite has a unique active `(scope_kind, scope_id, kind, topic)` index. Correction atomically
  supersedes the target and inserts the replacement. Forget atomically removes the record payload
  and raw provenance and inserts only a content-free tombstone.
- `PRAGMA secure_delete = ON` is enabled as best-effort local sanitization. It is not a forensic
  erase promise and does not cover backups, snapshots, WAL/filesystem remnants, or external copies.

The in-memory adapter is thread-safe for its process-local contract. SQLite supplies cross-connection
transactional CAS with a bounded busy timeout, not a distributed lease. A session owner can still
do model/tool work from a stale session revision and lose the final session CAS; the completed run
and any external tool effects are not rolled back. Distributed tenancy, leases, and exactly-once
side effects remain outside this architecture.

## Responsibility Boundaries

`AgentApplication` owns a conversation. It adds a user message to pending history, calls the runtime,
and commits the returned conversation only when the run completes successfully.

`SessionAgentApplication` owns one durable session transaction and the order of its retrieval,
context, and model/tool stages. It loads a snapshot, submits those stages to `RunCoordinator`, and
appends only the current user message and new model/tool items through compare-and-swap after run
success. `SessionStore` owns serialization and revision checks; it does not select model context.

`PromptAssembler` owns named system sections and explicitly requested project knowledge.
`ContextBuilder` owns budget estimation, complete-turn selection, compaction, summary provenance,
and the transient projection of an already-selected `MemoryRecall`. It does not query memory
storage, apply memory eligibility or ranking policy, or mutate the durable transcript.

`RunCoordinator` owns one end-to-end run across application and runtime stages. It controls the
single start and terminal transitions, error-to-run binding, ordered event stream, and default timeout.
It invokes application orchestration through a callback and exposes only a non-terminal `RunScope`.

`AgentRuntime` owns the bounded model/tool stage. It controls model/tool iteration, repeated-call
protection, provider retry, and model/tool stage events. It does not retrieve knowledge, construct
active context, persist conversation state, or own the surrounding run lifecycle.

`RunContext` carries data and control signals shared by all work in one run: run ID, start time,
deadline, cancellation, and read-only metadata.

`ToolRegistry` validates untrusted model arguments and invokes context-aware legacy handlers or
explicit `ActionTool` adapters. Governed adapters apply a pre-parse byte bound, fixed preparation,
guard, policy, approval, revalidation, hook, at-most-once executor, and bounded observation path.
The runtime passes only a provider-neutral `ToolExecutionContext` carrying `RunContext` and stage
event emission; it does not import governance domain types. Legacy handlers retain their existing
worker-thread timeout behavior. Provider adapters translate provider-neutral values and classify
transport failures.

The built-in registry exposes `current_time` and `get_weather`. `current_time` reads the local clock
for a validated UTC offset. `get_weather` is a deterministic tool-calling demonstration: it validates
a non-empty city and RFC 3339 full-date string, then returns structured fixed sunny data with an
explicit demo marker and no network access. It is not a weather-provider adapter and its result is
not a real forecast.

`EvaluationRunner` owns case isolation and behavioral judgment. It consumes final output,
conversation items, and runtime events; it does not alter runtime control flow or conversation state.

`MemoryEvaluationRunner` is an evaluation coordinator above production execution. It uses a temporary
SQLite database and production memory/session objects for each case, while replacing only extraction
and answer generation with deterministic fixtures. It reports admission, ranking, context projection,
and answer utilization independently. Its no-result metrics use `null` for zero denominators and a
separate no-result correctness value for empty-result semantics; no LLM judge is involved.

`WorkflowRunner` owns graph traversal, node-boundary commits, conditional selection, bounded branch
execution, interruption, recovery, and workflow events. `CheckpointStore` owns compare-and-swap
persistence but does not execute nodes or infer retry safety.

`ContextEvaluationRunner` measures the production context projection directly. It does not generate
answers or implement an alternate context algorithm.

`MemoryService` owns the explicit memory-management use cases. `propose`/`preview` evaluates a
transient candidate and never creates a pending queue. `confirm` receives the exact candidate and
digest again, obtains one fresh clock value, re-runs admission policy, and commits only through the
store's exact-scope revision CAS. `list` and `show` are inspection operations that materialize due
expiry into the record lifecycle. `correct` requires an explicit target ID and commits a superseded
old record plus a new record in one store change set. `forget` requires the exact scope and ID and
leaves only a content-free tombstone.

The service's digest check binds confirmation to the exact candidate shown; it does not by itself
prove that a human approved the operation. The current CLI supplies the interactive `yes`/`confirm`
step. Direct service callers own their authorization UX. Forgetting a corrected replacement removes
that target's payload and provenance, but does not recursively erase a prior superseded history
record; superseded records remain inspection-only and are excluded from recall.

`CommittedSessionTurn` is the extraction source boundary. It is created from one committed,
complete, bounded session turn and retains only that turn, its digest, revision, and bound metadata.
`MemoryExtractor` has no store access. `DeterministicMemoryExtractor` consumes explicit fixtures;
`ModelMemoryExtractor` calls only the neutral `LLMClient`, sends no tools, validates strict JSON, and
constructs trusted provenance from the source and completion. `MemoryExtractionPipeline` is the
explicit application bridge that previews each transient candidate through `MemoryService`; it does
not confirm or automatically write candidates.

`MemoryConsolidator` compares immutable scope data without importing or calling a store. It treats
the same kind/topic/content proposition as an exact duplicate that may refresh provenance and
expiry; a different content under an active kind/topic is a conflict. The service, rather than the
store, invokes this decision and maps it to add/refresh or an explicit correction requirement.

Memory result objects may contain records because they are the application payload returned to the
caller. Their `MemoryEventMetadata`, and the metadata attached to service errors, contain only
operation, outcome/reason, scope, revision, IDs, counts, and candidate digests. They never copy
memory content into event-ready or error metadata.

`MemoryService.recall` loads only the request's exact scope, then calls `MemoryPolicy.eligible` and
enforces the durable-record, active, validity, sensitivity, and kind boundary before invoking the
selector. Ineligible records never enter an embedding call. `MemorySelector` embeds eligible record
content and the query for each request, scores with a dot product, breaks ties by memory ID, and
reports its provider-qualified identity. It applies minimum score, maximum record count, kind
allowlist, and a character budget after ranking; records are selected or omitted atomically. The
recall character budget counts record content characters, and an empty recall is a successful,
explicit result. Scores describe query relevance only; they do not establish truth, confidence, or
consent. Eligibility, request-time embedding, and score computation are O(N) in the number of
eligible records with O(N) transient vector storage; the complete deterministic ordering adds the
usual O(N log N) comparison cost. There is no persistent vector index.

These boundaries are also the privacy boundary. The SQLite file is local and unencrypted; sensitive
and secret candidates are denied before storage under the default policy and service-owned hard
checks. The same defense rejects a finite set of obvious credential, sensitive-term, SSN,
telephone-number, and street-address patterns even when a candidate is labelled non-sensitive; it
is not a complete PII classifier. Scope IDs are explicit inputs and recall results must match the
requested scope. Memory payloads can appear in explicit management output and model context by
design, but not in run event attributes or sanitized service-error metadata. Event attributes and
service-error metadata use a scope digest; memory IDs and bounded counts identify selected results
without copying their content. The CLI escapes non-printable text and keeps dependency failures off
stderr payloads.

The memory CLI previews remember/correct candidates through a process-local, non-persistent service.
It prints the exact candidate fields and digest, then accepts only `yes` or `confirm`; a rejection or
EOF performs no SQLite operation. After confirmation, it constructs a persistent service and passes
that same immutable candidate and digest to `confirm` or `correct`, so the service rechecks policy,
clock, scope, and the store revision. `forget` shows the exact target before the same confirmation
step. There is no bulk-clear command or `--yes` bypass. Successful output is stdout; sanitized
errors are stderr. Exit code `0` means success, `1` an operational error, `2` usage/validation,
`3` explicit/EOF confirmation rejection, and `4` policy denial. Denied candidate output never prints
the candidate payload.

This split is similar to an application service invoking work inside a transaction coordinator while
delegating one stage to an execution engine. The analogy is incomplete because the lifecycle provides
ordered observations rather than database atomicity, and the model/tool stage is a bounded state
machine rather than a single RPC call.

## Run Lifecycle

`AgentApplication.run` creates pending history and calls the convenience `AgentRuntime.run`; that
method delegates lifecycle management to its default `RunCoordinator`. `SessionAgentApplication`
instead calls `RunCoordinator.execute` with an operation that performs retrieval, context construction,
and `AgentRuntime.execute`. This keeps the application responsible for stage order without giving it
start, complete, or fail operations.

The coordinator creates a `RunScope` containing the shared `RunContext` and non-terminal event
methods. Application and runtime stages may emit their own observations, but the scope rejects run
lifecycle event types. The coordinator alone maps normal return or escaped failure to exactly one
terminal transition:

```text
RUNNING -> retrieval -> context -> model request -> final text -> COMPLETED
             |           |           |
             |           |           +-> tool calls -> execute -> next model request
             |           |
             +-----------+---------------> stage failure ----------------> FAILED
                                     |
                                     +-> retryable -> backoff -> retry

RUNNING -> cancellation ----------------------------------> CANCELLED
RUNNING -> deadline exhausted -----------------------------> TIMED_OUT
RUNNING -> iteration limit / unexpected failure ----------> FAILED
```

Terminal failure, cancellation, and timeout do not commit pending conversation items. External tool
side effects are not rolled back; mutating handlers still need their own transaction and idempotency
guarantees.

`RUN_COMPLETED` means the coordinated computation, including request-scoped retrieval and context assembly,
completed. The session compare-and-swap is a surrounding application transaction performed after
that terminal event. A later session conflict therefore does not rewrite the completed agent event;
callers must observe the session error separately.

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
- Retrieval start, completion, and failure with query, result count, chunk IDs, scores, and error
  classification.
- Context assembly with budget, turn selection, knowledge keys, and summary provenance.
- Workflow node, transition, checkpoint, interruption, and resume activity.
- Memory extraction start, completion, and failure for its independent coordinator operation.

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
unavailability, request timeout, cancellation, deadline exhaustion, loop limit, context limit,
configuration, and unexpected internal failure.

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

## Sessions and Context Engineering

`SessionSnapshot` persists a schema version, session ID, validated provider-neutral transcript, CAS
revision, and timestamps. Only complete successful turns enter the transcript. System prompts, loaded
knowledge, and generated summaries are request-scoped projections and never become historical facts.

When a caller supplies a `RunContext`, the session application creates a child context that preserves
the run ID, deadline, cancellation, and caller metadata while setting the authoritative session ID.
This keeps summary and agent work correlated without accepting a conflicting caller-provided ID.

The in-memory store serializes access with a lock. The JSON store hashes session IDs into filenames,
uses atomic replacement, and checks the expected revision while holding one store-instance lock. This
prevents process-local lost updates. It is not a distributed lease: another process can perform model
or tool work from a stale revision and lose the final CAS race.

Prompt assembly keeps base behavior and project knowledge in named sections. Project files are loaded
only for explicitly requested keys from an allowlist, after verifying the resolved path remains under
the configured root. This is context selection, not retrieval ranking or long-term memory.

`ContextBudget` uses a deterministic serialized-character estimate because provider tokenizers differ.
`reserved_characters` accounts for tool definitions, expected output, and estimation error. The
builder rejects a budget that cannot fit prompt sections plus required recent turns. Otherwise it:

```text
validate transcript -> split complete user turns -> retain current + required completed turns
                    -> add newer old turns while budget remains
                    -> encode omitted turns as bounded, atomic JSONL records
                    -> optional model summary -> insert provenance-bearing system message
```

Trimming and structural compaction never split a turn, so tool calls and corresponding results remain
paired. Structural records that cannot fit are omitted and counted; no character prefix is emitted.
Summary drafts must satisfy their output contract and final serialized budget or are rejected/omitted
without mutation. Inserted summary text is labelled as untrusted historical data. Provenance records
the method, source digest and size, structural input size and turn loss, and model/response identity.
It separately reports turns admitted to structural input and turns available to the final summary.
The `CONTEXT_ASSEMBLED` event records selected/omitted turn counts, budget use, knowledge keys, and
summary identity.

The deterministic context suite measures old-constraint retention, recovery from an oversized full
history by trimming, and explicit whole-turn loss under a deliberately small structural input budget.
The loss case passes only when the omitted structural-turn count is observable; compaction remains
lossy but no longer creates partial semantic records.

When a `MemoryRecall` is supplied, the builder inserts selected records as one separately delimited
`USER` data block marked `untrusted_data=true` and `authority=lower-authority`. The current request,
mandatory prompt sections, retrieved RAG passages, and required recent turns stay ahead of older
transcript content; the current request is rendered after memory so an explicit correction has the
latest user authority. Memory has its own character cap, records are admitted atomically, and only
the remaining context budget is available to older turns or summaries. Memory content is excluded
from system messages, summaries, RAG passages, and durable session transcripts. `ContextWindow`
returns content-free projection evidence and `CONTEXT_ASSEMBLED` adds only memory IDs, kinds,
counts, scores, selector identity, and budget metrics to its event attributes.

For a durable session run, `SessionAgentApplication` owns the complete stage sequence:

```text
transcript + current request -> retrieval -> optional memory recall -> bounded context
                             -> AgentRuntime.execute -> successful session CAS
```

Memory recall events use the same `RunScope` as retrieval and runtime events. Started and completed
events expose only scope kind/digest, query digest/size, selector and bounded selection metrics;
failed events omit exception messages so memory scope and record content cannot leak through event
attributes. A `dqagent.errors.MemoryError` emits a failed stage event and continues without memory.
Cancellation, deadline exhaustion, and unknown exceptions escape to `RunCoordinator`, which emits
the terminal run event and prevents the session CAS. The application calls only `MemoryService.recall`;
there is no chat memory write path.

## Retrieval-Augmented Generation

`SourceDocument` defines an external source revision through document identity, content, citation
source, and metadata. `CharacterTextChunker` creates bounded whitespace-aware chunks with overlap,
source offsets, a content digest, and stable document-scoped chunk IDs. `DocumentIngestor` removes
exact duplicates within a document, embeds the remaining chunks, and replaces the complete indexed
view for that document. Updating a shorter document therefore removes stale old chunks; deletion is
an explicit lifecycle operation.

`EmbeddingProvider` and `VectorStore` are neutral protocols backed by concrete local implementations.
`EmbeddingProvider` exposes separate document and query embedding methods so a provider can apply
different task modes or prefixes. `HashingEmbeddingProvider` maps case-folded tokens through
deterministic feature hashing and L2 normalization. It needs lexical overlap and can have hash
collisions; it is a CI-capable architecture baseline, not a semantic embedding model. Each
`IndexedChunk` stores its embedding provider identity. The application-level `RetrievalResult`
reports an optional neutral retriever identity and optional candidate count instead of requiring a
local index size. The local vector retriever includes its embedding identity in the retriever
identity; vector-specific compatibility checks remain inside that implementation.

The in-memory store is thread-safe and test-oriented. `JsonFileVectorStore` holds a versioned complete
index and uses atomic replacement under a process-local lock. Each update rewrites and each query loads
the complete index, so it is intentionally limited to a small local corpus. It has no cross-process
coordination, approximate-nearest-neighbor structure, index migration, or history.

`VectorRetriever` embeds the query, ranks normalized vectors by dot product, filters by a positive
default score, applies deterministic ties, folds exact cross-document content duplicates by digest,
and returns rank-local citation IDs with full chunk provenance. An empty index or no score above the
threshold returns an explicit empty result instead of an error.

When configured, `SessionAgentApplication` retrieves from the current user input before context
construction and passes the same `RunContext` to the retriever:

```text
durable transcript + user query -> retrieve -> untrusted citation-labelled prompt passages
                               -> bounded ContextBuilder -> AgentRuntime -> answer + citation map
                               -> commit only user/agent conversation items
```

The trusted retrieval policy remains a system message. Each passage is a clearly delimited `USER`
message marked as untrusted data, so retrieved instructions do not receive system-message authority.
Delimiting data reduces accidental instruction following but is not a hard prompt-injection sandbox.
Retrieved passages, scores, and policies remain transient and do not enter the session transcript.
`SessionRunResult` retains the complete retrieval result and resolves answer IDs into cited sources,
retrieved-but-uncited IDs, and unknown IDs. Resolution is evidence for callers and evaluation; it does
not fail a completed model answer. Retrieval start/completion/failure and context assembly share the
same ordered run event stream. Retrieval failures emit a terminal event before the model request and
session commit.

The retrieval evaluation runner indexes fixture documents through the production pipeline and
measures `Recall@k`, reciprocal rank, and explicit no-result behavior before generation. Every case
uses the suite's configured score threshold. Recall and reciprocal rank are not applicable to
no-result cases and are excluded from ranking means. The fixture corpus includes multi-chunk
documents, lexical distractors, paraphrased and multi-relevant queries, and adversarial content.

A separate live-only answer evaluator runs the same session application path. Its deterministic
judge checks explicit answer fragments, forbidden injection outputs, insufficient-evidence behavior,
and claim-level citation coverage. A claim passes only when its sentence cites a retrieved chunk from
one allowed source and that chunk contains the same lexical claim. This tests citation locality and
source linkage, not semantic entailment.

## Dependency Rules

```text
CLI -> AgentApplication -> RunCoordinator + AgentRuntime
CLI -> SessionAgentApplication -> RunCoordinator + Retriever + MemoryService + ContextBuilder + SessionStore + AgentRuntime
RunCoordinator -> RunContext + RunEventEmitter + EventSink
AgentRuntime -> RunScope + LLMClient + ToolRegistry + neutral models
ToolRegistry -> RunContext + neutral models + jsonschema
OpenAI adapter -> RunContext + neutral models + OpenAI SDK
llama.cpp adapter -> RunContext + neutral models + OpenAI SDK transport
Event adapters -> RunEvent (future concrete integrations)
EvaluationRunner -> AgentRuntime + neutral models + RunEvent
WorkflowRunner -> WorkflowDefinition + CheckpointStore + RunContext + RunEvent
JSON checkpoint store -> WorkflowCheckpoint + local filesystem
ContextBuilder -> PromptAssembler + optional ConversationSummarizer + neutral models
JSON session store -> SessionSnapshot + local filesystem
ContextBuilder + SessionSnapshot -> transcript validator + MemoryRecall + neutral models
ContextEvaluationRunner -> ContextBuilder + neutral models
DocumentIngestor -> DocumentChunker + EmbeddingProvider + VectorStore
VectorRetriever -> EmbeddingProvider + VectorStore
SessionAgentApplication -> RunCoordinator + Retriever + MemoryService + explicit MemoryScope + ContextBuilder + SessionStore + AgentRuntime
RetrievalEvaluationRunner -> DocumentIngestor + VectorRetriever
MemoryService -> MemoryPolicy + MemoryConsolidator + MemoryStore + MemorySelector
MemorySelector -> EmbeddingProvider
MemoryExtractionPipeline -> MemoryExtractor + MemoryService
DeterministicMemoryExtractor -> committed bounded source + explicit fixture
ModelMemoryExtractor -> RunCoordinator + LLMClient + neutral models + jsonschema
SqliteMemoryStore -> sqlite3 + local filesystem
WorkspaceObserver -> Workspace + contained non-following filesystem traversal
WorkspaceDiff -> immutable WorkspaceSnapshot + bounded unified projection
  Governed action -> bounded parse/schema -> PreparedAction -> hard guards -> tri-state ActionPolicy
                 -> exact ApprovalRequest/ApprovalDecision -> revalidation -> ordered hooks
                 -> effect-boundary revalidation -> at-most-once executor -> bounded result/record
  PreparedAction -> Workspace + subprocesses.IsolationCapability
```

- Session state is owned above the runtime; one run cannot commit partial history.
- Session storage records what happened; context construction decides what the model sees now.
- Runtime and tool modules must not import provider SDKs.
- Provider adapters translate wire data and classify infrastructure failures.
- Event sinks observe execution but cannot mutate run state; stage-facing `RunScope` cannot emit
  lifecycle transitions.
- Evaluation observes production-owned results and events; it cannot implement alternate execution.
- Workflow nodes depend on shared execution contracts, not provider SDKs or CLI state.
- Checkpoint stores persist scheduler state but do not own workflow transitions or external effects.
- Retrieval content is request-scoped external data, not durable session state or long-term memory.
- MemoryService owns memory policy/consolidation orchestration; MemoryStore only loads exact scopes
  and applies validated change sets. The store has no remember/retrieve/sensitivity policy.
- MemoryExtractor owns no store or mutation capability and produces only transient candidates from an
  explicit bounded source. Model candidates are untrusted until `MemoryService.preview` and exact
  digest confirmation; extraction is an explicit operation, not a post-chat hook.
- Memory candidates are transient; there is no persistent pending-candidate queue. Recall is not
  connected to retrieval ranking or chat orchestration. ContextBuilder accepts only the completed
  `MemoryRecall` projection and does not perform store access, eligibility, or ranking.
- MemoryStore is policy-neutral and executes only validated exact-scope change sets. `SqliteMemoryStore`
  owns schema, transaction, CAS, and serialization concerns, not consent, sensitivity, ranking,
  or consolidation decisions.
- The current store boundary is local privacy and logical deletion. It is not encryption, forensic
  erasure, distributed tenancy, or durable audit delivery.
- Workspace observation is task-scoped evidence owned by `WorkspaceObserver`. It captures immutable
  baseline/final snapshots, compares regular-entry kind/size/full digest, and records deterministic
  create/modify/delete/type-change records without Git. Protected, secret, ignored, volatile, link,
  limit, cancellation, and filesystem omissions remain explicit blind spots; no secret content or
  secret fingerprint is retained. Target-scoped completeness may support a declared coding target,
  but global or forbidden predicates intersecting a blind spot remain indeterminate.
- Phase 9 T3 keeps action authorization separate from effect execution. `PreparedAction` contains
  normalized logical identity, effect preconditions, required technical capabilities, and effective
  limits. Its versioned sorted-key JSON/SHA-256 digest excludes absolute roots, secret values, Python
  representations, and display-only text. The fixed hard-guard order starts with the unified
  `max_governed_calls` ceiling and fails closed before policy.
- T3's default policy returns `allow` for read/search and `require_approval` for patch/command;
  hard-guard failure is always `deny`. `ActionRecord` stores bounded sanitized governance evidence,
  but T3 does not request approval, run hooks, call an executor, start a process, define subprocess
  request/results, or reserve validator capacity. A canonical digest proves action identity only.
- Phase 9 T4 projects approval requests and decisions through bounded immutable sanitized records.
  Approval is one foreground response bound to run/workspace/action/preconditions/capabilities;
  rejection, unavailable input, malformed data, identity mismatch, and drift fail closed, while
  cancellation remains a `RunContext` control error. The non-interactive provider never reads stdin
  and scripted approval responses are consumed once.
- T4 hook inputs contain only immutable sanitized action projections. Required pre-hook failure blocks
  before an effect, optional pre-hook failure is recorded while execution may continue, and ordered
  post-hook failure is visible without changing effect evidence. Hooks are synchronous trusted
  extensions; they are not `EventSink` instances and receive no workspace, subprocess, or executor
  capability.
- Phase 9 T5 keeps the governed path explicit beside legacy tools. Governed raw argument bytes are
  bounded before JSON parsing, reserved coding names cannot be registered as legacy handlers, and
  the runtime bridge carries only `RunContext` plus stage-event emission. A private synchronous
  run-scoped collector retains at most `max_governed_calls` bounded `ActionRecord` values for the
  exact active run; collector mismatch/failure is an observation failure, while event sinks remain
  best effort. Side-effecting executors are called at most once and are never retried by the
  runtime. Direct governed registry dispatch must receive that run's explicit
  `ToolExecutionContext`; omission fails closed instead of creating a per-call collector. This
  collector is not a public journal and provides no persistence, replay, or recovery.

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
- Session tests assert item round-trips, atomic JSON persistence, resume, rollback, CAS conflicts, and
  separation of transient summaries from durable history.
- Context tests assert prompt ownership, allowlisted on-demand knowledge, whole-turn trimming, tool
  pairing, structural/model summary provenance, overflow errors, deterministic context reports,
  lower-authority memory projection, atomic memory omission, RAG separation, and disabled-path
  regression snapshots.
- Retrieval tests assert exact offsets, update/delete behavior, duplicate folding, embedding identity,
  atomic JSON persistence, empty retrieval, citation propagation, prompt isolation, and ranking reports.
- Memory service tests assert transient proposals, policy rejection including obvious PII patterns,
  digest and clock revalidation, exact-scope operations, duplicate refresh, topic conflict, CAS
  concurrency, expiry, atomic correction/forgetting, digest-only content-free metadata, and an
  SQLite integration smoke path.
- Memory CLI tests assert explicit scope parsing, model-free composition, exact candidate display,
  confirmation rejection/EOF zero-write behavior, sensitive-payload suppression, cross-instance
  SQLite visibility, lifecycle/provenance output, correction, forgetting, tombstones, and stable
  exit/error streams.
- Session memory integration tests assert exact cross-session scope behavior, irrelevant queries,
  best-effort typed failures, cancellation/deadline/unknown failure handling, RAG ordering,
  lower-authority memory projection, disabled-path regression, CLI composition, and no memory write
  on session CAS failure.
- Workspace tests assert bounded non-following observation, untracked/type/link/secret/ignored
  handling, same-size digest changes, target versus global/forbidden completeness, each snapshot
  and rendered-diff limit, cancellation, stable ordering, normalized line endings, binary/oversized
  metadata, and incomplete evidence propagation.
- Governance tests assert golden canonicalization/digest sensitivity, immutable contracts, fixed
  guard ordering, hard-guard non-overridability, tri-state policy outcomes, fail-closed dependency
  behavior, unified call-capacity checks without reservation, executor non-invocation, and sanitized
  bounded records. T4 tests additionally cover every approval classification, active-run and drift
  revalidation, deadline capability honesty, sanitization, ordered hook modes, and post-effect failure.
- CI runs Ruff, strict mypy, and pytest with at least 85% coverage.
- CI also runs the credential-free deterministic behavioral, context, retrieval, and Phase 8 memory
  suites after implementation tests. The Phase 8 report uses the production memory/session path,
  temporary SQLite databases, deterministic hashing embeddings, scripted extraction, and scripted
  answers; it is a regression gate, not an LLM quality or compliance certification.

No live model evaluation runs in CI because credentials, cost, provider drift, and network
nondeterminism make it an unsuitable correctness gate.

## Current Limitations

- The runtime is synchronous; cancellation cannot preempt a blocking SDK call or Python thread.
- `AgentApplication` conversation state is not safe for concurrent callers.
- Agent-requested tool calls are sequential and tool retries are intentionally unsupported.
- Event sinks are best-effort and no concrete durable telemetry adapter is included.
- The legacy `AgentApplication` remains process-local and unbounded; durable behavior requires an
  explicit `SessionAgentApplication` or CLI `--session-id`.
- Session JSON stores retain only the latest revision and coordinate one process. They have no
  distributed lease, append-only history, migration framework, tenant isolation, or retention policy.
- Context budgets estimate characters rather than provider tokens. Structural and model summaries
  are lossy, and model summarization adds a provider call before the main agent run.
- Memory management, request-time recall, and bounded recall/context projection are explicit and
  cross-session through the service, stores, selector, ContextBuilder, and `dqagent-memory` CLI;
  extraction is also explicit through `MemoryExtractor` and is not integrated as automatic chat
  behavior. The store is unencrypted, logical forget is not forensic erase, corrected superseded
  history is not recursively erased, direct service calls do not prove human authorization, and the
  deterministic PII defense is not a complete classifier.
- Encrypted sensitive-memory storage, forensic erase, persistent memory vector indexing, unconfirmed
  or automatic writes, distributed tenancy/leases, background consolidation, bulk deletion, and
  unsupported capabilities remain deferred. No current test or evaluation justifies claiming them.
- The retrieval store is small, synchronous, single-process, and brute-force. Hashing embeddings are
  lexical, exact digest deduplication misses near-duplicates, and answer checks use explicit lexical
  claims rather than a semantic or LLM-based judge.
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
- [ADR-0006: Separate Durable Session Transcripts from Active Model Context](adr/0006-separate-session-transcript-from-active-context.md)
- [ADR-0007: Use an Explicit Retrieval Index and Citation Boundary](adr/0007-explicit-retrieval-index-and-citation-boundary.md)
- [Phase 2 Framework Comparison](learning/phase-2-framework-comparison.md)
- [Roadmap Reassessment After Phase 3](learning/roadmap-reassessment-after-phase-3.md)
- [Phase 4 BFCL and GAIA Comparison](learning/phase-4-bfcl-gaia-comparison.md)
- [Phase 5 LangGraph and EINO Workflow Comparison](learning/phase-5-langgraph-eino-workflow-comparison.md)
- [Phase 6 Session and Context Comparison](learning/phase-6-session-context-comparison.md)
- [Phase 7 Retrieval Framework Comparison](learning/phase-7-retrieval-framework-comparison.md)
- [Phase 8 Memory Framework Comparison](learning/phase-8-memory-framework-comparison.md)
- [ADR-0009: Policy-Governed Long-Term Memory](adr/0009-policy-governed-long-term-memory.md)
- [ADR-0010: Keep Memory Extraction Before Deterministic Admission](adr/0010-transient-memory-extraction-boundary.md)
