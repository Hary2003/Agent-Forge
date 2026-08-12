"""
Tests for RetrievalNode and DAG integration.
"""
import sys
sys.path.insert(0, "src")

import pytest

from agentforge.core.executor import WorkflowExecutor
from agentforge.core.llm import MockLLMClient
from agentforge.core.schema import AgentNodeConfig, NodeStatus, WorkflowConfig
from agentforge.rag.embeddings import MockEmbeddingProvider
from agentforge.rag.ingest import ingest_text
from agentforge.rag.retrieval_node import RetrievalNode
from agentforge.rag.vector_store import InMemoryVectorStore


@pytest.mark.asyncio
async def test_retrieval_node_standalone_execution():
    store = InMemoryVectorStore(collection_name="test_knowledge")
    embedder = MockEmbeddingProvider()

    ingest_text(
        text="Python 3.12 introduces enhanced performance and isolated subinterpreters.",
        source_name="python_features.txt",
        vector_store=store,
        embedding_provider=embedder,
        chunk_size=100,
        chunk_overlap=10,
    )

    node_cfg = AgentNodeConfig(
        id="retriever_node",
        node_type="retrieval",
        config={
            "top_k": 2,
            "rerank": False,
        },
    )

    node = RetrievalNode(
        config=node_cfg,
        vector_store=store,
        embedding_provider=embedder,
    )

    res = await node.run(task_prompt="What's new in Python 3.12?", upstream={})

    assert res.status == NodeStatus.SUCCESS
    assert "python_features.txt" in res.text
    assert len(res.data["chunks"]) > 0


@pytest.mark.asyncio
async def test_retrieval_node_in_dag_workflow():
    # Setup vector store knowledge base
    store = InMemoryVectorStore(collection_name="dag_kb")
    embedder = MockEmbeddingProvider()

    ingest_text(
        text="AgentForge RAG module uses swappable vector stores and embedding models.",
        source_name="rag_design.md",
        vector_store=store,
        embedding_provider=embedder,
        chunk_size=100,
        chunk_overlap=10,
    )

    # Instantiate RetrievalNode manually with injected mock components
    retrieval_cfg = AgentNodeConfig(
        id="kb_retriever",
        node_type="retrieval",
        config={"top_k": 1},
    )
    retrieval_node = RetrievalNode(config=retrieval_cfg, vector_store=store, embedding_provider=embedder)

    # Workflow DAG: kb_retriever (retrieval) -> llm_writer (agent)
    wf = WorkflowConfig(
        name="rag_workflow",
        nodes=[
            retrieval_cfg,
            AgentNodeConfig(
                id="llm_writer",
                node_type="agent",
                role="writer",
                goal="Write a summary using the provided context",
                inputs=["kb_retriever"],
            ),
        ],
    )

    llm = MockLLMClient(canned_text="Summary based on retrieved docs.")
    executor = WorkflowExecutor(llm=llm)

    # Monkeypatch node creation to pass our pre-populated vector store
    from agentforge.core import node as node_module
    orig_create = node_module.create_node

    def mock_create_node(cfg, llm_client, registry=None):
        if cfg.id == "kb_retriever":
            return retrieval_node
        return orig_create(cfg, llm_client, registry)

    node_module.create_node = mock_create_node
    try:
        result = await executor.run(wf, "Explain AgentForge RAG design")
    finally:
        node_module.create_node = orig_create

    assert result.node_results["kb_retriever"].status == NodeStatus.SUCCESS
    assert result.node_results["llm_writer"].status == NodeStatus.SUCCESS

    # Verify context was passed downstream to LLM writer prompt
    llm_prompt = llm.calls[-1]["messages"][0]["content"]
    assert "Output of 'kb_retriever'" in llm_prompt
    assert "rag_design.md" in llm_prompt
