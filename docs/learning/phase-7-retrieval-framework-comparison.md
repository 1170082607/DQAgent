# Phase 7 Comparison: DQAgent Retrieval, LangChain, and Chroma

## Scope and Evidence

This note compares reusable RAG responsibilities rather than package ergonomics. The intended
references are the public LangChain retrieval and Chroma collection contracts listed below. On
2026-08-01 this environment could not open an in-app browser and direct HTTPS retrieval failed during
TLS receive, so no claim is made about current framework internals or recently changed API details.
The comparison is limited to their established public responsibility model (documents/splitters,
embeddings, vector stores/retrievers, and collection lifecycle/query operations). These references do
not serve as evidence that DQAgent behavior is correct; local tests and evaluation cases do. DQAgent
does not import either project.

Sources:

- [LangChain retrieval documentation](https://docs.langchain.com/oss/python/langchain/retrieval)
- [LangChain vector store integrations](https://docs.langchain.com/oss/python/integrations/vectorstores)
- [Chroma collection API](https://docs.trychroma.com/reference/python/collection)
- [Chroma query and get](https://docs.trychroma.com/docs/querying-collections/query-and-get)

## Shared Architecture

The useful common decomposition is:

```text
source -> document identity -> chunking -> embedding -> indexed records
query  -> query embedding -> ranked retrieval -> provenance-bearing context -> answer
```

LangChain packages these as loaders, splitters, embeddings, vector stores, and retrievers. Chroma owns
the persistent collection and nearest-neighbor query operations. DQAgent uses the same responsibility
boundaries in a deliberately small implementation so lifecycle and failure semantics remain visible.

| Concern | DQAgent Phase 7 | LangChain / Chroma lesson |
| --- | --- | --- |
| Document model | Explicit ID, source, content, string metadata | Framework `Document` commonly carries page content and metadata |
| Chunking | Character bound, whitespace split, overlap, source offsets | Splitters provide many text-aware policies; choice affects recall and context quality |
| Embedding | Provider protocol plus deterministic hashing implementation | Embedding integrations isolate model-specific clients |
| Update | Replace all chunks by document ID | Collection APIs expose update/upsert, but applications still own source-to-record identity |
| Delete | Explicit document deletion | Vector stores expose delete; retention policy remains application responsibility |
| Query | Exact vector ranking, threshold, top-k, deterministic tie break | Retrievers normalize query interfaces over store-specific search features |
| Provenance | Citation ID plus document/chunk/source/offset/digest | Metadata must remain attached through retrieval if answers need auditable citations |
| Evaluation | Recall@k and MRR before generation | Retrieval quality must be measured independently from answer quality |

## Reusable Lessons

### The index lifecycle belongs to the application

`upsert` is not enough unless the application defines identity. If a source document shrinks from ten
chunks to four, updating only the four new IDs can leave six stale chunks retrievable. DQAgent makes
replace-by-document the store contract, analogous to replacing a backend aggregate within a clear
transaction boundary.

### Embedding identity is schema, not incidental configuration

Vectors from different models or dimensions do not share one coordinate space. DQAgent stores an
embedding identity with each record and fails closed on mismatch. A production migration would build
a new index version and switch readers after validation, similar to an online database migration.

### Retrieval and generation have different failure signals

An articulate answer cannot prove the relevant evidence was retrieved. Recall-oriented metrics answer
whether relevant documents entered top-k; answer predicates later determine whether the model used
them faithfully. Combining both into one score hides which subsystem regressed.

### Retrieved text is data crossing a trust boundary

Vector similarity does not make content authoritative. DQAgent labels passages as untrusted external
data and separates the instruction from each passage. This resembles parsing an external service
response into a typed value before using it, but the analogy breaks because model instruction/data
separation is probabilistic rather than a hard language security boundary.

## Deliberate Non-Adoptions

- No LangChain dependency: the current protocols and local implementations are small and explicit.
- No Chroma dependency: full-index JSON replacement is sufficient for the intended small corpus.
- No hybrid search, reranker, metadata filter, or query rewrite: the committed corpus has not shown a
  need that justifies those policies.
- No model-selected retrieval tool: the current product path always grounds session questions when
  an index is configured.
- No semantic-quality claim for hashing embeddings: lexical overlap and hash collisions constrain the
  baseline. A real embedding adapter is the next concrete implementation when a real corpus exists.
