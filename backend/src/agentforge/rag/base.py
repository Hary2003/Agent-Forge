"""
Base interfaces and data structures for AgentForge RAG module.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DocumentChunk:
    """Represents a chunk of document text with associated metadata."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    @property
    def chunk_index(self) -> int:
        return int(self.metadata.get("chunk_index", 0))


class EmbeddingProvider(ABC):
    """Abstract interface for embedding generation."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generates embedding vectors for a list of document texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generates embedding vector for a search query."""
        pass


class VectorStoreAdapter(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Stores document chunks along with pre-computed embeddings into the vector store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[DocumentChunk]:
        """Performs similarity search against stored embeddings and returns top_k matching chunks."""
        pass


class Reranker(ABC):
    """Abstract interface for cross-encoder reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int = 4,
    ) -> list[DocumentChunk]:
        """Reranks candidate chunks for a query and returns top_k most relevant chunks."""
        pass
