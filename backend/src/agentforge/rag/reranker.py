"""
Cross-Encoder Reranking for AgentForge RAG.
Supports ms-marco-MiniLM-L-6-v2 cross-encoder and PassThrough (toggle off) option.
"""
from __future__ import annotations

from typing import Optional

from agentforge.rag.base import DocumentChunk, Reranker


class PassThroughReranker(Reranker):
    """No-op reranker used when reranking is toggled off."""

    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int = 4,
    ) -> list[DocumentChunk]:
        return chunks[:top_k]


class CrossEncoderReranker(Reranker):
    """Reranker using Hugging Face CrossEncoder (default: cross-encoder/ms-marco-MiniLM-L-6-v2)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError as err:
                raise ImportError(
                    "sentence-transformers package is required for CrossEncoderReranker. "
                    "Install it via `pip install sentence-transformers`."
                ) from err

    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int = 4,
    ) -> list[DocumentChunk]:
        if not chunks:
            return []

        self._load_model()
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self._model.predict(pairs)

        rescored_chunks: list[DocumentChunk] = []
        for chunk, score in zip(chunks, scores):
            rescored_chunks.append(
                DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    score=float(score),
                )
            )

        rescored_chunks.sort(key=lambda c: c.score, reverse=True)
        return rescored_chunks[:top_k]


def get_reranker(
    enabled: bool = False,
    model_name: Optional[str] = None,
) -> Reranker:
    """Factory helper to obtain a Reranker instance based on settings."""
    if not enabled:
        return PassThroughReranker()
    try:
        return CrossEncoderReranker(model_name=model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    except ImportError:
        return PassThroughReranker()
