"""
RetrievalNode implementation for AgentForge RAG.
Interactions with VectorStore, EmbeddingProvider, and optional Reranker to fetch context for downstream LLM nodes.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from agentforge.core.node import BaseNode
from agentforge.core.schema import AgentNodeConfig, NodeResult, NodeStatus, ToolCall
from agentforge.rag.base import EmbeddingProvider, Reranker, VectorStoreAdapter
from agentforge.rag.embeddings import get_embedding_provider
from agentforge.rag.reranker import get_reranker
from agentforge.rag.vector_store import get_vector_store


class RetrievalNode(BaseNode):
    """
    RetrievalNode performs similarity search (and optional reranking) against a vector store
    and formats retrieved document chunks as context for downstream LLM nodes in the DAG.
    """

    def __init__(
        self,
        config: AgentNodeConfig,
        vector_store: Optional[VectorStoreAdapter] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.config = config
        cfg = config.config or {}

        # Swappable components with defaults from node config
        store_type = cfg.get("vector_store_type", "chroma")
        collection_name = cfg.get("collection_name", "agentforge_rag")
        persist_directory = cfg.get("persist_directory", "./chroma_db")

        embed_type = cfg.get("embedding_provider", "bge")
        embed_model = cfg.get("embedding_model", "BAAI/bge-small-en-v1.5")

        self.top_k = cfg.get("top_k", 4)
        self.rerank_enabled = cfg.get("rerank", False)
        self.rerank_model = cfg.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.vector_store = vector_store or get_vector_store(
            store_type=store_type,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

        self.embedding_provider = embedding_provider or get_embedding_provider(
            provider_type=embed_type,
            model_name=embed_model,
        )

        self.reranker = reranker or get_reranker(
            enabled=self.rerank_enabled,
            model_name=self.rerank_model,
        )

    def _extract_query(self, task_prompt: str, upstream: dict[str, NodeResult]) -> str:
        """Determines the search query from task prompt or upstream node outputs."""
        query_source = self.config.config.get("query_source")
        if query_source and query_source in upstream:
            res = upstream[query_source]
            if res.text.strip():
                return res.text.strip()

        # Check all upstream nodes for non-empty text
        if upstream:
            for dep_id in self.config.inputs:
                if dep_id in upstream and upstream[dep_id].text.strip():
                    return upstream[dep_id].text.strip()

        return task_prompt

    async def run(
        self,
        task_prompt: str,
        upstream: dict[str, NodeResult],
        on_tool_call: Optional[Callable[[str, ToolCall], Any]] = None,
    ) -> NodeResult:
        result = NodeResult(node_id=self.config.id, status=NodeStatus.RUNNING)

        try:
            query = self._extract_query(task_prompt, upstream)
            # 1. Embed query
            query_emb = self.embedding_provider.embed_query(query)

            # 2. Vector search (fetch initial candidates)
            initial_k = self.top_k * 2 if self.rerank_enabled else self.top_k
            candidate_chunks = self.vector_store.search(query_emb, top_k=initial_k)

            # 3. Optional Reranking step
            final_chunks = self.reranker.rerank(
                query=query,
                chunks=candidate_chunks,
                top_k=self.top_k,
            )

            # 4. Format prompt context and structured data
            formatted_lines = [
                f"--- Retrieved Knowledge Context (Query: '{query}') ---"
            ]
            structured_chunks = []

            for idx, chunk in enumerate(final_chunks, 1):
                src = chunk.metadata.get("source", "unknown")
                c_idx = chunk.metadata.get("chunk_index", 0)
                score_str = f" (relevance score: {chunk.score:.4f})" if chunk.score else ""
                formatted_lines.append(
                    f"[{idx}] Source: {src} (chunk #{c_idx}){score_str}\n{chunk.text.strip()}"
                )

                structured_chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "score": chunk.score,
                })

            result.text = "\n\n".join(formatted_lines) if final_chunks else "No relevant context found in vector store."
            result.data = {
                "query": query,
                "chunks": structured_chunks,
                "count": len(final_chunks),
                "reranked": self.rerank_enabled,
            }
            result.status = NodeStatus.SUCCESS
            return result

        except Exception as exc:
            result.status = NodeStatus.FAILED
            result.error = f"RetrievalNode failed: {exc}"
            return result
