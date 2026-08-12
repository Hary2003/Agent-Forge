"""
Vector Store Adapters for AgentForge RAG.
Supports local ChromaDB, pure-Python InMemory fallback, and swappable store factory.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from agentforge.rag.base import DocumentChunk, VectorStoreAdapter


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore(VectorStoreAdapter):
    """Pure-Python in-memory vector store using cosine similarity (zero dependencies)."""

    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self._chunks: list[DocumentChunk] = []
        self._embeddings: list[list[float]] = []

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        for chunk, emb in zip(chunks, embeddings):
            self._chunks.append(chunk)
            self._embeddings.append(emb)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[DocumentChunk]:
        if not self._chunks:
            return []

        scored_chunks: list[tuple[float, DocumentChunk]] = []
        for chunk, emb in zip(self._chunks, self._embeddings):
            sim = _cosine_similarity(query_embedding, emb)
            chunk_copy = DocumentChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                score=sim,
            )
            scored_chunks.append((sim, chunk_copy))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]


class ChromaVectorStore(VectorStoreAdapter):
    """Local ChromaDB vector store adapter (supports persistent directory or ephemeral memory)."""

    def __init__(
        self,
        collection_name: str = "agentforge_rag",
        persist_directory: Optional[str] = "./chroma_db",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None

    def _init_chroma(self):
        if self._collection is None:
            try:
                import chromadb
            except ImportError as err:
                raise ImportError(
                    "chromadb package is required for ChromaVectorStore. "
                    "Install it via `pip install chromadb`."
                ) from err

            if self.persist_directory:
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        self._init_chroma()
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[DocumentChunk]:
        self._init_chroma()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[DocumentChunk] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return chunks

        ids = results["ids"][0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns L2/cosine distance; convert distance to similarity score
            score = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=meta or {},
                    score=score,
                )
            )

        return chunks


def get_vector_store(
    store_type: str = "chroma",
    collection_name: str = "agentforge_rag",
    persist_directory: Optional[str] = "./chroma_db",
    **kwargs: Any,
) -> VectorStoreAdapter:
    """Factory helper to instantiate a VectorStoreAdapter by type name."""
    store_type = store_type.lower()
    if store_type in ("chroma", "chromadb"):
        try:
            return ChromaVectorStore(
                collection_name=collection_name,
                persist_directory=persist_directory,
            )
        except ImportError:
            # Fall back gracefully to in-memory store if chromadb is not installed
            return InMemoryVectorStore(collection_name=collection_name)
    elif store_type in ("memory", "inmemory", "in_memory", "mock"):
        return InMemoryVectorStore(collection_name=collection_name)
    else:
        # Fall back to Chroma or InMemory
        try:
            return ChromaVectorStore(collection_name=collection_name, persist_directory=persist_directory)
        except Exception:
            return InMemoryVectorStore(collection_name=collection_name)
