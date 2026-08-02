import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dqagent.errors import RetrievalError
from dqagent.retrieval import (
    CharacterTextChunker,
    DocumentIngestor,
    HashingEmbeddingProvider,
    IndexedChunk,
    InMemoryVectorStore,
    JsonFileVectorStore,
    SourceDocument,
    VectorRetriever,
    resolve_answer_citations,
)


def make_pipeline(
    store: InMemoryVectorStore | JsonFileVectorStore,
    *,
    chunk_size: int = 80,
    overlap: int = 0,
) -> tuple[DocumentIngestor, VectorRetriever]:
    embeddings = HashingEmbeddingProvider(64)
    return (
        DocumentIngestor(
            CharacterTextChunker(
                max_characters=chunk_size,
                overlap_characters=overlap,
            ),
            embeddings,
            store,
        ),
        VectorRetriever(embeddings, store),
    )


def test_ingestion_update_replaces_stale_chunks_and_delete_removes_document() -> None:
    store = InMemoryVectorStore()
    ingestor, retriever = make_pipeline(store)

    first = ingestor.upsert(
        SourceDocument("policy", "refund policy allows thirty days", "docs/policy.md")
    )
    assert first.indexed_chunks == 1
    assert retriever.retrieve("refund thirty", limit=1).chunks[0].chunk.document_id == "policy"

    ingestor.upsert(SourceDocument("policy", "support hours are weekdays", "docs/policy.md"))

    chunks = store.all_chunks()
    assert len(chunks) == 1
    assert chunks[0].chunk.content == "support hours are weekdays"
    assert ingestor.delete("policy") == 1
    assert retriever.retrieve("support hours").chunks == ()
    assert ingestor.delete("policy") == 0


def test_chunker_preserves_offsets_and_deduplicates_repeated_content() -> None:
    content = "alpha beta\nalpha beta"
    document = SourceDocument("duplicate", content, "duplicate.txt")
    chunker = CharacterTextChunker(max_characters=10, overlap_characters=0)
    chunks = chunker.chunk(document)

    assert [content[chunk.start : chunk.end] for chunk in chunks] == [
        "alpha beta",
        "alpha beta",
    ]
    ingestor, _ = make_pipeline(InMemoryVectorStore(), chunk_size=10)
    result = ingestor.upsert(document)
    assert result.indexed_chunks == 1
    assert result.duplicate_chunks == 1


def test_retrieval_is_ranked_stable_and_deduplicates_across_documents() -> None:
    store = InMemoryVectorStore()
    ingestor, retriever = make_pipeline(store)
    ingestor.upsert(SourceDocument("a", "python retrieval ranking", "a.md"))
    ingestor.upsert(SourceDocument("b", "python retrieval ranking", "b.md"))
    ingestor.upsert(SourceDocument("c", "weather forecast", "c.md"))

    result = retriever.retrieve("python retrieval", limit=3)

    assert [item.citation_id for item in result.chunks] == ["R1"]
    assert result.chunks[0].chunk.document_id == "a"
    assert result.indexed_chunk_count == 3


def test_json_store_round_trips_and_rejects_corrupt_or_incompatible_index(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index.json"
    store = JsonFileVectorStore(path)
    ingestor, retriever = make_pipeline(store)
    ingestor.upsert(SourceDocument("one", "durable retrieval index", "one.md"))

    resumed = VectorRetriever(HashingEmbeddingProvider(64), JsonFileVectorStore(path))
    assert resumed.retrieve("durable retrieval", limit=1).chunks[0].chunk.source == "one.md"

    item = store.all_chunks()[0]
    incompatible = replace(item, embedding_provider="other:64", indexed_at=datetime.now(UTC))
    store.replace_document("one", (incompatible,))
    with pytest.raises(RetrievalError, match="identity does not match"):
        retriever.retrieve("durable")

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RetrievalError, match="cannot load retrieval index"):
        store.all_chunks()


def test_store_rejects_wrong_document_and_embedding_dimension_mismatch() -> None:
    store = InMemoryVectorStore()
    ingestor, _ = make_pipeline(store)
    ingestor.upsert(SourceDocument("one", "some searchable text", "one.md"))
    item = store.all_chunks()[0]

    with pytest.raises(ValueError, match="target document"):
        store.replace_document("other", (item,))

    bad = IndexedChunk(item.chunk, (1.0,), item.embedding_provider, item.indexed_at)
    store.replace_document("one", (bad,))
    with pytest.raises(RetrievalError, match="dimensions do not match"):
        VectorRetriever(HashingEmbeddingProvider(64), store).retrieve("text")


def test_json_index_is_structured_and_contains_no_provider_sdk_types(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    ingestor, _ = make_pipeline(JsonFileVectorStore(path))
    ingestor.upsert(
        SourceDocument("doc", "metadata stays portable", "doc.md", {"team": "platform"})
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["chunks"][0]["chunk"]["metadata"] == {"team": "platform"}


def test_answer_citations_resolve_valid_uncited_and_unknown_ids() -> None:
    store = InMemoryVectorStore()
    ingestor, retriever = make_pipeline(store)
    ingestor.upsert(SourceDocument("a", "alpha searchable", "a.md"))
    ingestor.upsert(SourceDocument("b", "alpha second", "b.md"))
    retrieval = retriever.retrieve("alpha", limit=2)

    resolved = resolve_answer_citations("Use [R2], then [R2]; ignore [R99].", retrieval)

    assert tuple(resolved.cited) == ("R2",)
    assert resolved.uncited_ids == ("R1",)
    assert resolved.unknown_ids == ("R99",)
