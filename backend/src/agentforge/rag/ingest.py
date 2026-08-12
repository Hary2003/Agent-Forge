"""
Document Ingestion Utility and CLI Script for AgentForge RAG.

Chunks input documents (fixed-size chunks with overlap), embeds them using
the configured EmbeddingProvider, and stores them in VectorStore with metadata.
"""
from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any, List, Optional, Union

from agentforge.rag.base import DocumentChunk, EmbeddingProvider, VectorStoreAdapter
from agentforge.rag.embeddings import get_embedding_provider
from agentforge.rag.vector_store import get_vector_store


class DocumentChunker:
    """Fixed-size text chunker with configurable overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Splits a single text block into fixed-size overlapping character chunks."""
        text = text.strip()
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        text_len = len(text)
        step = self.chunk_size - self.chunk_overlap

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_str = text[start:end]
            if chunk_str.strip():
                chunks.append(chunk_str.strip())
            if end >= text_len:
                break
            start += step

        return chunks


def ingest_text(
    text: str,
    source_name: str,
    vector_store: VectorStoreAdapter,
    embedding_provider: EmbeddingProvider,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> int:
    """Chunks, embeds, and stores a raw text string in the vector store."""
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = chunker.split_text(text)
    if not text_chunks:
        return 0

    doc_chunks: List[DocumentChunk] = []
    for idx, c_text in enumerate(text_chunks):
        meta = {
            "source": source_name,
            "chunk_index": idx,
            "total_chunks": len(text_chunks),
        }
        if extra_metadata:
            meta.update(extra_metadata)

        chunk_id = f"{source_name}_chunk_{idx}_{uuid.uuid4().hex[:6]}"
        doc_chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                text=c_text,
                metadata=meta,
            )
        )

    # Embed and store
    embeddings = embedding_provider.embed_texts([c.text for c in doc_chunks])
    vector_store.add_chunks(doc_chunks, embeddings)
    return len(doc_chunks)


def ingest_file_or_directory(
    input_path: Union[str, Path],
    vector_store: VectorStoreAdapter,
    embedding_provider: EmbeddingProvider,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    file_extensions: Tuple[str, ...] = (".txt", ".md", ".json", ".py", ".csv"),
) -> dict[str, int]:
    """Ingests a file or recursively processes a directory of text documents."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files_to_process: List[Path] = []
    if path.is_file():
        files_to_process.append(path)
    elif path.is_dir():
        for file in path.rglob("*"):
            if file.is_file() and file.suffix.lower() in file_extensions:
                files_to_process.append(file)

    summary = {}
    for file in files_to_process:
        try:
            content = file.read_text(encoding="utf-8", errors="ignore")
            num_chunks = ingest_text(
                text=content,
                source_name=file.name,
                vector_store=vector_store,
                embedding_provider=embedding_provider,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                extra_metadata={"path": str(file.resolve())},
            )
            summary[file.name] = num_chunks
        except Exception as err:
            print(f"Error ingesting {file.name}: {err}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="AgentForge RAG Document Ingestion CLI")
    parser.add_argument("--input", required=True, help="Path to document file or directory to ingest")
    parser.add_argument("--collection", default="agentforge_rag", help="Vector store collection name")
    parser.add_argument("--store-type", default="chroma", help="Vector store type (chroma, memory)")
    parser.add_argument("--persist-dir", default="./chroma_db", help="Chroma DB persistence directory")
    parser.add_argument("--embedding-provider", default="bge", help="Embedding provider (bge, openai, mock)")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=500, help="Fixed chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap size in characters")

    args = parser.parse_args()

    v_store = get_vector_store(
        store_type=args.store_type,
        collection_name=args.collection,
        persist_directory=args.persist_dir,
    )
    embedder = get_embedding_provider(
        provider_type=args.embedding_provider,
        model_name=args.embedding_model,
    )

    print(f"Starting ingestion for: {args.input}")
    print(f"Vector Store: {args.store_type} (Collection: '{args.collection}')")
    print(f"Embedding Provider: {args.embedding_provider} ({args.embedding_model})")
    print(f"Chunk Config: size={args.chunk_size}, overlap={args.chunk_overlap}")

    results = ingest_file_or_directory(
        input_path=args.input,
        vector_store=v_store,
        embedding_provider=embedder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    total_chunks = sum(results.values())
    print(f"\nIngestion Complete! Processed {len(results)} file(s), ingested {total_chunks} chunk(s) total.")
    for fname, count in results.items():
        print(f"  - {fname}: {count} chunk(s)")


if __name__ == "__main__":
    main()
