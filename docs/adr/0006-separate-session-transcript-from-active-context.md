# ADR-0006: Separate Durable Session Transcripts from Active Model Context

- Status: Accepted
- Date: 2026-07-30

## Context

Phase 6 must make long conversations recoverable without sending an unbounded transcript to every
model request. A transcript and a model context answer different questions: the transcript records
what successfully happened, while the context is a temporary projection of what the model should
see now. Persisting a trimmed or generated summary as if it were the original conversation would
destroy evidence and make later compaction impossible to audit.

Tool conversations add a structural constraint. Removing a tool call while retaining its result, or
the reverse, creates provider-invalid history. Concurrent session owners also need a defined outcome;
last-writer-wins would silently lose a completed turn.

## Decision

`SessionSnapshot` stores the complete provider-neutral transcript under a session ID and monotonically
increasing revision. Session stores use compare-and-swap. The in-memory and JSON implementations
serialize callers through a process-local lock; the JSON implementation hashes IDs into filenames and
uses atomic replacement. A stale owner receives `SessionConflictError` instead of overwriting data.

`SessionAgentApplication` loads one revision, asks `ContextBuilder` for a bounded model view, runs the
existing `AgentRuntime`, and appends only the current user message plus `AgentRunResult.new_items`.
The generated system summary and selected prompt knowledge are never committed to the transcript.
A failed run or failed CAS does not change session state.

`PromptAssembler` owns named system sections and explicitly requested knowledge documents.
`FileProjectKnowledgeSource` accepts only configured keys whose resolved paths remain under one root.
There is no automatic repository scan or bulk instruction injection.

`ContextBuilder` estimates serialized character size because the provider-neutral layer has no shared
tokenizer. The budget includes a caller-configured reserve for tool schemas, output, and tokenizer
error. It retains a configured number of recent complete turns, then adds older complete turns newest
first while space remains. Omitted turns are first rendered into a bounded structural form. That form
can be used directly or passed to an optional model summarizer.

Every inserted summary records method, source SHA-256, source item/character counts, structural input
size, and provider response/model identity when available. `CONTEXT_ASSEMBLED` exposes bounded context
metrics on the normal run event stream.

## Consequences

- Session storage remains lossless for successful turns even when active context is lossy.
- Tool-call/tool-result pairs are kept or omitted as part of a complete turn.
- Mandatory prompt sections plus recent turns fail with `ContextOverflowError` instead of silently
  truncating the current request.
- Structural and model summaries are inspectable projections, not trusted durable facts.
- A model summary sees bounded structural input rather than the entire raw transcript.
- A session race can waste a completed model call and external tool effects before CAS rejects the
  losing commit. Strong cross-process ownership would require leases or a transactional coordinator.
- Character estimates are portable and deterministic but cannot guarantee a provider token limit.
  Provider-aware token counting can be added behind a real tokenizer boundary later.
- JSON files contain the latest transcript revision only. There is no append-only event log, schema
  migration framework, distributed lock, tenant isolation, or retention/deletion policy yet.

## Alternatives Considered

Persisting only the compacted context was rejected because it discards source evidence. Sending the
complete transcript was rejected because cost and provider limits grow without bound. Trimming
individual messages was rejected because it can break tool protocol structure. Automatic loading of
all project files was rejected because it wastes context and expands the prompt-injection surface.
Exact token counting was deferred because OpenAI and llama.cpp do not share one tokenizer contract.
