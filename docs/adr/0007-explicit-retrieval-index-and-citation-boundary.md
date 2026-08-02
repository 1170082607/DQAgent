# ADR-0007: Use an Explicit Retrieval Index and Citation Boundary

- Status: Accepted
- Date: 2026-08-01

## Context

Phase 7 must ground answers in external knowledge without confusing retrieval with session history,
prompt instructions, or long-term memory. The system needs observable document identity, chunking,
updates, deletion, embedding compatibility, ranking, and provenance. A vector database SDK exposed
throughout the application would collapse these responsibilities into one vendor API and make
failure behavior difficult to test.

Retrieved text is untrusted. It can be stale, duplicated, irrelevant, or contain prompt injection.
An empty result also has meaning: it says the configured index supplied no evidence above the
threshold, not that the proposition is false.

## Decision

`SourceDocument` is the authoritative ingestion input. `DocumentIngestor` chunks it, removes exact
within-document duplicate chunks, embeds the remaining text, and calls `VectorStore.replace_document`.
Replacement removes every prior chunk for that document before installing the new set; deletion is
explicit. `TextChunk` retains document ID, stable chunk ID, source, source offsets, metadata, and a
content SHA-256 digest.

`EmbeddingProvider` and `VectorStore` are provider-neutral protocols with concrete local
implementations. `HashingEmbeddingProvider` is a deterministic token feature-hashing baseline, not
a semantic embedding model. Every indexed chunk stores the provider identity, and retrieval rejects
an index created by a different identity instead of comparing incompatible vectors. The JSON store
uses a versioned schema and atomic file replacement; the in-memory store supports deterministic tests.

`VectorRetriever` applies cosine-equivalent dot-product ranking over normalized vectors, a minimum
score, deterministic tie-breaking, and exact-content deduplication across documents. It returns
rank-local citation IDs (`R1`, `R2`, ...), scores, and full chunk provenance. Empty retrieval is a
successful result.

Retrieval is enabled only on `SessionAgentApplication`, before bounded context construction. Each
passage becomes request-scoped system context with an explicit untrusted-data policy. Retrieved text
and generated context are never persisted in the durable session transcript. `SessionRunResult`
returns the citation mapping, while `CitationResolution` separates cited, uncited, and unknown answer
IDs. It observes citation behavior without converting a probabilistic answer-quality defect into a
runtime failure. `RETRIEVAL_COMPLETED` and `CONTEXT_ASSEMBLED` events expose result counts, IDs, and
scores. A retrieval failure happens before the model call and session commit.

Retrieval evaluation is separate from answer generation. Versioned cases measure `Recall@k` and
reciprocal rank through the production ingestion and retrieval implementations. A generated answer
cannot compensate for a missed relevant document.

## Consequences

- Updates cannot leave stale chunks from an older document revision in the index.
- Index files cannot be silently reused after changing embedding dimensions or implementations.
- Provenance survives retrieval, prompt assembly, application results, and structured events.
- Prompt injection risk is reduced through clear data/instruction delimiting but not eliminated;
  model compliance remains probabilistic and needs answer-level evaluation.
- Exact digest deduplication avoids repeated passages consuming top-k slots but does not detect
  near-duplicates.
- The JSON store rewrites the complete index and loads all vectors for each query. This is appropriate
  only for a small local learning corpus, not production scale or concurrent processes.
- Feature hashing needs lexical overlap and can collide; it establishes the architecture and a
  deterministic baseline, not acceptable semantic search quality for a production corpus.
- `R1` identifies a passage only within one retrieval result. Durable references use document ID,
  chunk ID, source, offsets, and digest.

## Alternatives Considered

Using a managed vector database first was rejected because it would add deployment and SDK concerns
before the ownership contracts were proven. Keyword substring search alone was rejected because it
would not validate an embedding boundary or vector lifecycle. Writing retrieved passages to the
session transcript was rejected because they are a reproducible context projection, not conversation
history. Letting the model call retrieval as a tool was deferred: always-on grounding for the current
use case is application policy, while agent-selected retrieval adds another behavior requiring its
own evaluation.
