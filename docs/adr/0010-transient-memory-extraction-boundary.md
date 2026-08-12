# ADR-0010: Keep Memory Extraction Before Deterministic Admission

- Status: Accepted
- Date: 2026-08-11

## Context

Long-term memory now has a durable transcript source, transient candidates, deterministic admission,
preview, and explicit confirmation. Adding a model extractor creates a new trust boundary: model text
must be useful for proposing facts without becoming storage authority. A chat run also owns one
terminal lifecycle, so extraction cannot append events to a completed chat run or make a successful
session commit depend on a later model call.

## Decision

DQAgent uses a provider-neutral `MemoryExtractor` protocol whose input is one explicit
`CommittedSessionTurn` and whose output is a `MemoryExtractionResult` containing transient
`MemoryCandidate` values only. The extractor has no `MemoryStore`, `SessionStore`, or mutation-tool
capability. Extraction is never automatically started by `SessionAgentApplication` after a chat turn.

`CommittedSessionTurn.from_snapshot` selects one complete turn from a store-loaded snapshot with a
positive committed revision and a timezone-aware commit timestamp. It stores only that turn, its
source digest, revision, and bound character count. The model prompt contains only this bounded turn;
it does not contain the full transcript.

The deterministic extractor uses explicit source-keyed fixtures. The model extractor sends a
provider-neutral system/user prompt with no tools and accepts only a strict JSON object. Schema and
domain validation enforce output character, candidate-count, individual-content, topic, confidence,
kind, and sensitivity limits. Tool calls, free text, malformed JSON, unknown fields, hallucinated
provenance, and ambiguous multi-claim candidates are rejected. Multiple accepted candidates remain
separate proposals so each can be reviewed independently.

Trusted provenance is constructed by the extractor, never copied from model JSON. It contains the
source digest and revision, extractor identity, optional provider model/response identities, a
timezone-aware extraction timestamp, and the independent extraction run ID. It does not contain the
source transcript payload.

Model extraction runs inside its own `RunCoordinator` operation and generated run ID. A parent chat
context may bound the operation's remaining time and is checked before and after the provider call,
but the chat run's `RunScope` is never retained or reused. Cancellation, deadline, provider, and
format failures are terminal extraction failures with no candidate write.

An explicit `MemoryExtractionPipeline` may pass successful candidates to `MemoryService.preview`.
Every candidate therefore still passes deterministic T2 policy and must be confirmed later with the
exact T5 digest. Extractor confidence is evidence about the extractor only; it never bypasses
sensitivity policy or user confirmation.

## Consequences

- Deterministic fixtures provide a credential-free core and evaluation path.
- Model output remains probabilistic and untrusted until deterministic policy and user confirmation.
- Extraction failures cannot mutate a transcript or memory store because the pure extractor has no
  store access and preview performs no durable write.
- Provider adapters remain behind `LLMClient`; the extraction parser depends on neutral
  `Completion` values and does not expose provider SDK types.
- Optional model/response identities round-trip through the existing v1 SQLite shape using a private
  extractor-identity envelope, preserving existing databases and schema contracts.
- The implementation remains synchronous, and a blocking provider call cannot be forcibly stopped;
  cooperative cancellation and the run deadline still govern the visible result.

## Alternatives Considered

### Run extraction automatically after every chat

Rejected because it adds an implicit model call and latency to the chat transaction, makes a later
failure appear coupled to a successful transcript commit, and encourages treating every turn as a
memory candidate.

### Let the model emit a memory mutation tool call

Rejected because it gives probabilistic output a durable side effect and creates a prompt-injection
path around deterministic policy, preview, and confirmation. Extraction calls the provider with no
tools.

### Trust model-provided provenance

Rejected because a model can hallucinate a source ID, revision, digest, or run identity. The parser
rejects provenance fields and derives all provenance from the typed source and completion boundary.

### Store a pending extraction queue

Rejected for T10 because pending candidates are transient proposals. A persistent queue would add
retention, deletion, replay, and authorization semantics before the candidate contract is measured.
