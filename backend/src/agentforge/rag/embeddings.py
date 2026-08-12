"""
Embedding providers for AgentForge RAG.
Supports BGE-small (via Hugging Face sentence-transformers), OpenAI, and a Mock fallback.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Optional

from agentforge.rag.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic, lightweight embedding provider for testing without external models."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_text(self, text: str) -> list[float]:
        # Generate pseudo-vector from text hash
        val = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
        vec = [(float((val >> (i % 32)) & 0xFF) / 255.0) - 0.5 for i in range(self.dimension)]
        # Normalize vector
        norm = (sum(x * x for x in vec) ** 0.5) or 1.0
        return [x / norm for x in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_text(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._hash_text(query)


class HuggingFaceEmbedding(EmbeddingProvider):
    """HuggingFace Sentence Transformers embedding model (default: BAAI/bge-small-en-v1.5)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError as err:
                raise ImportError(
                    "sentence-transformers package is required for HuggingFaceEmbedding. "
                    "Install it via `pip install sentence-transformers`."
                ) from err

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    def embed_query(self, query: str) -> list[float]:
        self._load_model()
        # BGE models benefit from instruction prefix for queries if applicable
        embedding = self._model.encode(query, normalize_embeddings=True)
        return embedding.tolist()


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding model provider (e.g. text-embedding-3-small)."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            import openai
        except ImportError as err:
            raise ImportError(
                "openai package is required for OpenAIEmbedding. "
                "Install it via `pip install openai`."
            ) from err

        client = openai.OpenAI(api_key=self.api_key)
        resp = client.embeddings.create(input=texts, model=self.model_name)
        return [item.embedding for item in resp.data]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


def get_embedding_provider(
    provider_type: str = "bge",
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Factory helper to instantiate an EmbeddingProvider by name."""
    provider_type = provider_type.lower()
    if provider_type in ("bge", "bge-small", "huggingface", "hf", "sentence-transformers"):
        return HuggingFaceEmbedding(model_name=model_name or "BAAI/bge-small-en-v1.5")
    elif provider_type in ("openai", "text-embedding-3-small"):
        return OpenAIEmbedding(model_name=model_name or "text-embedding-3-small", **kwargs)
    elif provider_type in ("mock", "test", "dummy"):
        return MockEmbeddingProvider(**kwargs)
    else:
        # Fall back to HuggingFace or Mock if unknown
        try:
            return HuggingFaceEmbedding(model_name=model_name or provider_type)
        except Exception:
            return MockEmbeddingProvider()
