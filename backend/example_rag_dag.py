#!/usr/bin/env python3
"""
Example end-to-end demonstration of AgentForge RAG:
1. Ingesting documents into local Chroma vector store (or InMemory fallback) with fixed-size chunking.
2. Wiring a RetrievalNode into a multi-agent DAG alongside LLM nodes.
3. Executing the workflow.
"""
import asyncio
import sys

sys.path.insert(0, "src")

from agentforge.core.executor import WorkflowExecutor
from agentforge.core.llm import MockLLMClient
from agentforge.core.schema import AgentNodeConfig, WorkflowConfig
from agentforge.rag.embeddings import get_embedding_provider
from agentforge.rag.ingest import ingest_text
from agentforge.rag.vector_store import get_vector_store


async def main():
    print("=== AgentForge RAG & RetrievalNode Demo ===\n")

    # 1. Initialize vector store & embedding model (uses Chroma or InMemory fallback)
    v_store = get_vector_store(store_type="memory", collection_name="demo_kb")
    embedder = get_embedding_provider(provider_type="mock")

    # 2. Ingest sample documents into vector store with chunking & metadata
    doc_1 = (
        "AgentForge is a Python DAG-based multi-agent orchestration framework. "
        "It executes graph nodes in topological order using asyncio concurrency."
    )
    doc_2 = (
        "RetrievalNode allows RAG integration. It embeds queries, performs top-k similarity "
        "search against a local vector store (Chroma/Qdrant), and applies cross-encoder reranking."
    )

    ingest_text(doc_1, source_name="architecture.txt", vector_store=v_store, embedding_provider=embedder)
    ingest_text(doc_2, source_name="rag_spec.txt", vector_store=v_store, embedding_provider=embedder)
    print("Ingested sample documents into vector store.\n")

    # 3. Define DAG Workflow with RetrievalNode and LLM AgentNode
    wf = WorkflowConfig(
        name="rag_demo_workflow",
        description="DAG workflow with RAG retrieval node feeding into LLM agent",
        nodes=[
            AgentNodeConfig(
                id="kb_retriever",
                node_type="retrieval",
                inputs=[],
                config={
                    "top_k": 2,
                    "rerank": False,  # Toggle optional cross-encoder reranking
                },
            ),
            AgentNodeConfig(
                id="llm_agent",
                node_type="agent",
                role="technical_writer",
                goal="Explain the feature using the retrieved context provided in upstream outputs.",
                inputs=["kb_retriever"],
            ),
        ],
    )

    # 4. Inject instantiated vector store into RetrievalNode for demo run
    from agentforge.core import node as node_module
    from agentforge.rag.retrieval_node import RetrievalNode

    orig_create = node_module.create_node

    def demo_create_node(cfg, llm_client, registry=None):
        if cfg.id == "kb_retriever":
            return RetrievalNode(cfg, vector_store=v_store, embedding_provider=embedder)
        return orig_create(cfg, llm_client, registry)

    node_module.create_node = demo_create_node

    # 5. Execute DAG Workflow
    executor = WorkflowExecutor(llm=MockLLMClient())
    result = await executor.run(wf, task_prompt="How does RetrievalNode work in AgentForge?")

    print("--- Execution Results ---")
    for node_id, node_res in result.node_results.items():
        print(f"\nNode [{node_id}] ({node_res.status.value}):")
        print(node_res.text)


if __name__ == "__main__":
    asyncio.run(main())
