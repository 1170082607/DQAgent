"""Provider-neutral ingestion, embedding, indexing, and retrieval boundaries."""

import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, Protocol, cast
from uuid import uuid4

from dqagent.errors import RetrievalError
from dqagent.execution import RunContext

INDEX_SCHEMA_VERSION = 1
_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_CITATION_PATTERN = re.compile(r"\[(R[1-9][0-9]*)\]")
_JSON_INDEX_LOCKS: dict[Path, Lock] = {}
_JSON_INDEX_LOCKS_GUARD = Lock()


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One authoritative external document supplied to the ingestion pipeline."""

    document_id: str
    content: str
    source: str
    metadata: Mapping[str, str] = MappingProxyType({})

    def __post_init__(self) -> None:
        for label, value in (
            ("document ID", self.document_id),
            ("document content", self.content),
            ("document source", self.source),
        ):
            if not value.strip():
                raise ValueError(f"{label} must not be empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_id: str
    document_id: str
    content: str
    source: str
    start: int
    end: int
    content_digest: str
    metadata: Mapping[str, str] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not self.chunk_id.strip() or not self.document_id.strip():
            raise ValueError("chunk and document IDs must not be empty")
        if not self.content.strip() or not self.source.strip():
            raise ValueError("chunk content and source must not be empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("chunk offsets must define a non-empty range")
        if len(self.content_digest) != 64:
            raise ValueError("chunk content digest must be SHA-256")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class DocumentChunker(Protocol):
    def chunk(self, document: SourceDocument) -> tuple[TextChunk, ...]: ...


class CharacterTextChunker:
    """Splits text at whitespace when possible while preserving source offsets."""

    def __init__(self, *, max_characters: int = 800, overlap_characters: int = 100) -> None:
        if max_characters < 1:
            raise ValueError("maximum chunk characters must be positive")
        if overlap_characters < 0 or overlap_characters >= max_characters:
            raise ValueError("chunk overlap must be non-negative and below the maximum")
        self._max_characters = max_characters
        self._overlap_characters = overlap_characters

    def chunk(self, document: SourceDocument) -> tuple[TextChunk, ...]:
        content = document.content
        chunks: list[TextChunk] = []
        start = 0
        ordinal = 0
        while start < len(content):
            while start < len(content) and content[start].isspace():
                start += 1
            if start == len(content):
                break
            end = min(start + self._max_characters, len(content))
            if end < len(content):
                split = max(
                    content.rfind("\n", start + 1, end + 1),
                    content.rfind(" ", start + 1, end + 1),
                )
                if split > start:
                    end = split
            text = content[start:end].strip()
            if text:
                actual_start = start + len(content[start:end]) - len(content[start:end].lstrip())
                actual_end = end - (len(content[start:end]) - len(content[start:end].rstrip()))
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                chunk_id = f"{document.document_id}:{ordinal}:{digest[:12]}"
                chunks.append(
                    TextChunk(
                        chunk_id,
                        document.document_id,
                        text,
                        document.source,
                        actual_start,
                        actual_end,
                        digest,
                        document.metadata,
                    )
                )
                ordinal += 1
            if end == len(content):
                break
            next_start = end - self._overlap_characters
            start = next_start if next_start > start else end
        return tuple(chunks)


class EmbeddingProvider(Protocol):
    """Maps text into vectors without exposing a provider SDK to the application."""

    @property
    def identity(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


class HashingEmbeddingProvider:
    """Deterministic local feature hashing baseline; not a semantic embedding model."""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 8:
            raise ValueError("embedding dimensions must be at least eight")
        self._dimensions = dimensions

    @property
    def identity(self) -> str:
        return f"hashing-token-v1:{self._dimensions}"

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._embed_one(text) for text in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self._dimensions
        counts = Counter(token.casefold() for token in _TOKEN_PATTERN.findall(text))
        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self._dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            values[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in values))
        if norm:
            values = [value / norm for value in values]
        return tuple(values)


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    chunk: TextChunk
    embedding: tuple[float, ...]
    embedding_provider: str
    indexed_at: datetime

    def __post_init__(self) -> None:
        if not self.embedding or not all(math.isfinite(value) for value in self.embedding):
            raise ValueError("chunk embedding must contain finite values")
        if not self.embedding_provider.strip():
            raise ValueError("embedding provider identity must not be empty")
        if self.indexed_at.tzinfo is None:
            raise ValueError("index timestamp must be timezone-aware")


class VectorStore(Protocol):
    """Replace-by-document index with explicit delete and complete snapshot reads."""

    def replace_document(self, document_id: str, chunks: Sequence[IndexedChunk]) -> None: ...

    def delete_document(self, document_id: str) -> int: ...

    def all_chunks(self) -> tuple[IndexedChunk, ...]: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, IndexedChunk] = {}
        self._lock = Lock()

    def replace_document(self, document_id: str, chunks: Sequence[IndexedChunk]) -> None:
        _validate_replacement(document_id, chunks)
        with self._lock:
            self._chunks = {
                key: value
                for key, value in self._chunks.items()
                if value.chunk.document_id != document_id
            }
            self._chunks.update({item.chunk.chunk_id: item for item in chunks})

    def delete_document(self, document_id: str) -> int:
        if not document_id.strip():
            raise ValueError("document ID must not be empty")
        with self._lock:
            keys = [
                key
                for key, item in self._chunks.items()
                if item.chunk.document_id == document_id
            ]
            for key in keys:
                del self._chunks[key]
            return len(keys)

    def all_chunks(self) -> tuple[IndexedChunk, ...]:
        with self._lock:
            return tuple(self._chunks[key] for key in sorted(self._chunks))


class JsonFileVectorStore:
    """Single-process local index persisted through atomic JSON replacement."""

    def __init__(self, path: Path) -> None:
        self._path = path.resolve()
        with _JSON_INDEX_LOCKS_GUARD:
            self._lock = _JSON_INDEX_LOCKS.setdefault(self._path, Lock())

    def replace_document(self, document_id: str, chunks: Sequence[IndexedChunk]) -> None:
        _validate_replacement(document_id, chunks)
        with self._lock:
            current = self._load_unlocked()
            retained = [item for item in current if item.chunk.document_id != document_id]
            self._save_unlocked((*retained, *chunks))

    def delete_document(self, document_id: str) -> int:
        if not document_id.strip():
            raise ValueError("document ID must not be empty")
        with self._lock:
            current = self._load_unlocked()
            retained = tuple(item for item in current if item.chunk.document_id != document_id)
            deleted = len(current) - len(retained)
            if deleted:
                self._save_unlocked(retained)
            return deleted

    def all_chunks(self) -> tuple[IndexedChunk, ...]:
        with self._lock:
            return self._load_unlocked()

    def _load_unlocked(self) -> tuple[IndexedChunk, ...]:
        if not self._path.exists():
            return ()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("schema_version") != INDEX_SCHEMA_VERSION:
                raise ValueError("unsupported index schema version")
            raw_chunks = raw["chunks"]
            if not isinstance(raw_chunks, list):
                raise TypeError("chunks must be an array")
            return tuple(
                _indexed_chunk_from_dict(cast(dict[str, Any], item))
                for item in raw_chunks
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RetrievalError(f"cannot load retrieval index '{self._path}': {exc}") from exc

    def _save_unlocked(self, chunks: Sequence[IndexedChunk]) -> None:
        temporary = self._path.with_name(f".{self._path.name}.{uuid4().hex}.tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "chunks": [
                    _indexed_chunk_to_dict(item)
                    for item in sorted(chunks, key=lambda item: item.chunk.chunk_id)
                ],
            }
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self._path)
        except OSError as exc:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise RetrievalError(f"cannot save retrieval index '{self._path}': {exc}") from exc


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: str
    indexed_chunks: int
    duplicate_chunks: int
    content_digest: str


class DocumentIngestor:
    """Owns chunking, within-document deduplication, embedding, and atomic replacement."""

    def __init__(
        self,
        chunker: DocumentChunker,
        embeddings: EmbeddingProvider,
        store: VectorStore,
    ) -> None:
        self._chunker = chunker
        self._embeddings = embeddings
        self._store = store

    def upsert(self, document: SourceDocument) -> IngestionResult:
        candidates = self._chunker.chunk(document)
        unique: list[TextChunk] = []
        seen: set[str] = set()
        duplicates = 0
        for chunk in candidates:
            if chunk.content_digest in seen:
                duplicates += 1
                continue
            seen.add(chunk.content_digest)
            unique.append(chunk)
        vectors = self._embeddings.embed_documents([chunk.content for chunk in unique])
        if len(vectors) != len(unique):
            raise RetrievalError("embedding provider returned the wrong vector count")
        now = datetime.now(UTC)
        indexed = tuple(
            IndexedChunk(chunk, vector, self._embeddings.identity, now)
            for chunk, vector in zip(unique, vectors, strict=True)
        )
        self._store.replace_document(document.document_id, indexed)
        return IngestionResult(
            document.document_id,
            len(indexed),
            duplicates,
            hashlib.sha256(document.content.encode("utf-8")).hexdigest(),
        )

    def delete(self, document_id: str) -> int:
        return self._store.delete_document(document_id)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    citation_id: str
    chunk: TextChunk
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    chunks: tuple[RetrievedChunk, ...]
    retriever_identity: str | None = None
    candidate_count: int | None = None

    @property
    def citations(self) -> Mapping[str, TextChunk]:
        return MappingProxyType({item.citation_id: item.chunk for item in self.chunks})


@dataclass(frozen=True, slots=True)
class CitationResolution:
    """References observed in one answer, resolved against one retrieval result."""

    cited: Mapping[str, TextChunk]
    uncited_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cited", MappingProxyType(dict(self.cited)))


def resolve_answer_citations(answer: str, retrieval: RetrievalResult) -> CitationResolution:
    observed = tuple(dict.fromkeys(_CITATION_PATTERN.findall(answer)))
    available = retrieval.citations
    cited = {
        citation_id: available[citation_id]
        for citation_id in observed
        if citation_id in available
    }
    return CitationResolution(
        cited,
        tuple(citation_id for citation_id in available if citation_id not in observed),
        tuple(citation_id for citation_id in observed if citation_id not in available),
    )


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.05,
        context: RunContext | None = None,
    ) -> RetrievalResult: ...


class VectorRetriever:
    def __init__(self, embeddings: EmbeddingProvider, store: VectorStore) -> None:
        self._embeddings = embeddings
        self._store = store

    @property
    def identity(self) -> str:
        return f"vector-retriever-v1:{self._embeddings.identity}"

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.05,
        context: RunContext | None = None,
    ) -> RetrievalResult:
        if context is not None:
            context.check_active()
        if not query.strip():
            raise ValueError("retrieval query must not be empty")
        if limit < 1:
            raise ValueError("retrieval limit must be positive")
        if not math.isfinite(min_score):
            raise ValueError("minimum retrieval score must be finite")
        indexed = self._store.all_chunks()
        if context is not None:
            context.check_active()
        if not indexed:
            return RetrievalResult(query, (), self.identity, 0)
        incompatible = [
            item
            for item in indexed
            if item.embedding_provider != self._embeddings.identity
        ]
        if incompatible:
            raise RetrievalError("index embedding identity does not match the configured provider")
        query_vector = self._embeddings.embed_query(query)
        if not query_vector or not all(math.isfinite(value) for value in query_vector):
            raise RetrievalError("query embedding must contain finite values")
        if context is not None:
            context.check_active()
        scored = sorted(
            (
                (_dot(query_vector, item.embedding), item.chunk)
                for item in indexed
            ),
            key=lambda pair: (-pair[0], pair[1].document_id, pair[1].chunk_id),
        )
        unique: list[tuple[float, TextChunk]] = []
        seen_digests: set[str] = set()
        for score, chunk in scored:
            if score < min_score or chunk.content_digest in seen_digests:
                continue
            seen_digests.add(chunk.content_digest)
            unique.append((score, chunk))
            if len(unique) == limit:
                break
        selected = tuple(
            RetrievedChunk(f"R{rank}", chunk, score)
            for rank, (score, chunk) in enumerate(unique, start=1)
        )
        return RetrievalResult(query, selected, self.identity, len(indexed))


def _validate_replacement(document_id: str, chunks: Sequence[IndexedChunk]) -> None:
    if not document_id.strip():
        raise ValueError("document ID must not be empty")
    if any(item.chunk.document_id != document_id for item in chunks):
        raise ValueError("replacement chunks must belong to the target document")
    ids = [item.chunk.chunk_id for item in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("replacement chunk IDs must be unique")


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise RetrievalError("query and index embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _indexed_chunk_to_dict(item: IndexedChunk) -> dict[str, object]:
    chunk = item.chunk
    return {
        "chunk": {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "source": chunk.source,
            "start": chunk.start,
            "end": chunk.end,
            "content_digest": chunk.content_digest,
            "metadata": dict(chunk.metadata),
        },
        "embedding": list(item.embedding),
        "embedding_provider": item.embedding_provider,
        "indexed_at": item.indexed_at.isoformat(),
    }


def _indexed_chunk_from_dict(data: Mapping[str, Any]) -> IndexedChunk:
    raw = cast(dict[str, Any], data["chunk"])
    chunk = TextChunk(
        cast(str, raw["chunk_id"]),
        cast(str, raw["document_id"]),
        cast(str, raw["content"]),
        cast(str, raw["source"]),
        cast(int, raw["start"]),
        cast(int, raw["end"]),
        cast(str, raw["content_digest"]),
        cast(dict[str, str], raw.get("metadata", {})),
    )
    embedding = tuple(float(value) for value in cast(list[float], data["embedding"]))
    return IndexedChunk(
        chunk,
        embedding,
        cast(str, data["embedding_provider"]),
        datetime.fromisoformat(cast(str, data["indexed_at"])),
    )
