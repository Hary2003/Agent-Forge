from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure src/ is in python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from agentforge.core.executor import WorkflowExecutor
from agentforge.core.llm import LLMClient, LLMResponse
from agentforge.core.workflow import load_workflow


QUERY = "What does RetrievalNode do, and how does it use the reranker and embedding provider?"


class CodeSpecificLLM(LLMClient):
    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        if "writer" in system.lower():
            return LLMResponse(text=(
                "RetrievalNode is the RAG node class in retrieval_node.py. In __init__, it reads "
                "config keys including vector_store_type, collection_name, persist_directory, "
                "embedding_provider, embedding_model, top_k, rerank, and rerank_model. It builds "
                "the vector store with get_vector_store(), the embedder with get_embedding_provider(), "
                "and the reranker with get_reranker(). In run(), it extracts the query via "
                "_extract_query(), calls embedding_provider.embed_query(query), searches the vector "
                "store with initial_k = top_k * 2 when rerank is enabled, then calls "
                "reranker.rerank(query=query, chunks=candidate_chunks, top_k=self.top_k). It returns "
                "formatted retrieved context plus structured data containing query, chunks, count, "
                "and reranked."
            ))

        return LLMResponse(text=(
            "The retrieved context points to RetrievalNode in retrieval_node.py. It initializes "
            "swappable vector_store, embedding_provider, and reranker components from node config. "
            "The key methods are __init__, _extract_query, and run. The run method embeds the query "
            "with embed_query, performs vector_store.search, optionally widens initial_k for reranking, "
            "uses reranker.rerank, and stores chunk metadata in result.data."
        ))


async def main() -> None:
    workflow = load_workflow("workflows/rag_research_workflow_self_test.yaml")
    executor = WorkflowExecutor(llm=CodeSpecificLLM())

    print("=== RAG SELF TEST EXECUTION LOG ===")
    print(f"workflow: {workflow.name}")
    print(f"query: {QUERY}")
    print()

    async def on_node_start(node_id: str) -> None:
        print(f"--- node start: {node_id} ---")

    async def on_node_complete(node_id: str, result) -> None:
        print(f"--- node complete: {node_id} ---")
        print(f"status: {result.status}")
        if result.error:
            print(f"error: {result.error}")

        if node_id == "kb_retriever":
            print("kb_retriever.data:")
            print(json.dumps(result.data, indent=2, ensure_ascii=False))
            print("kb_retriever.text:")
            print(result.text)
        else:
            print("text:")
            print(result.text)
        print()

    run_result = await executor.run(
        workflow,
        QUERY,
        on_node_start=on_node_start,
        on_node_complete=on_node_complete,
    )

    print("=== FINAL RESULT ===")
    print(f"final_node_id: {run_result.final_node_id}")
    if run_result.final_result:
        print(f"final_status: {run_result.final_result.status}")
        print(run_result.final_result.text)


if __name__ == "__main__":
    asyncio.run(main())
