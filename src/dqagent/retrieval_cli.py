"""Manage and inspect the small local retrieval index."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from dqagent.errors import DQAgentError
from dqagent.retrieval import (
    CharacterTextChunker,
    DocumentIngestor,
    HashingEmbeddingProvider,
    JsonFileVectorStore,
    SourceDocument,
    VectorRetriever,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the DQAgent local retrieval index.")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(".local/retrieval/index.json"),
        help="Local JSON index path.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    upsert = subparsers.add_parser("upsert", help="Insert or replace one document.")
    upsert.add_argument("document_id")
    upsert.add_argument("path", type=Path)
    upsert.add_argument("--source", help="Citation source; defaults to the input path.")
    upsert.add_argument("--chunk-characters", type=int, default=800)
    upsert.add_argument("--chunk-overlap", type=int, default=100)

    delete = subparsers.add_parser("delete", help="Delete all chunks for one document.")
    delete.add_argument("document_id")

    query = subparsers.add_parser("query", help="Inspect ranked chunks without an LLM.")
    query.add_argument("text")
    query.add_argument("--limit", type=int, default=5)
    query.add_argument("--min-score", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        embeddings = HashingEmbeddingProvider()
        store = JsonFileVectorStore(args.index)
        if args.command == "upsert":
            content = args.path.read_text(encoding="utf-8")
            ingestor = DocumentIngestor(
                CharacterTextChunker(
                    max_characters=args.chunk_characters,
                    overlap_characters=args.chunk_overlap,
                ),
                embeddings,
                store,
            )
            ingestion = ingestor.upsert(
                SourceDocument(
                    args.document_id,
                    content,
                    args.source or str(args.path).replace("\\", "/"),
                )
            )
            payload: object = {
                "document_id": ingestion.document_id,
                "indexed_chunks": ingestion.indexed_chunks,
                "duplicate_chunks": ingestion.duplicate_chunks,
                "content_digest": ingestion.content_digest,
            }
        elif args.command == "delete":
            deleted = DocumentIngestor(
                CharacterTextChunker(), embeddings, store
            ).delete(args.document_id)
            payload = {"document_id": args.document_id, "deleted_chunks": deleted}
        else:
            retrieval = VectorRetriever(embeddings, store).retrieve(
                args.text,
                limit=args.limit,
                min_score=args.min_score,
            )
            payload = {
                "query": retrieval.query,
                "candidate_count": retrieval.candidate_count,
                "retriever_identity": retrieval.retriever_identity,
                "results": [
                    {
                        "citation_id": item.citation_id,
                        "document_id": item.chunk.document_id,
                        "chunk_id": item.chunk.chunk_id,
                        "source": item.chunk.source,
                        "score": item.score,
                        "content": item.chunk.content,
                    }
                    for item in retrieval.chunks
                ],
            }
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0
    except (DQAgentError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
