import asyncio
import json
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from agentforge.core.executor import WorkflowExecutor
from agentforge.core.llm import MockLLMClient
from agentforge.core.schema import AgentNodeConfig, WorkflowConfig


async def test_global_warming_retrieval():
    wf = WorkflowConfig(
        name="gw_retrieval_test",
        nodes=[
            AgentNodeConfig(
                id="gw_retriever",
                node_type="retrieval",
                inputs=[],
                config={
                    "collection_name": "agentforge_rag",
                    "vector_store_type": "chroma",
                    "persist_directory": "./chroma_db",
                    "embedding_provider": "bge",
                    "embedding_model": "BAAI/bge-small-en-v1.5",
                    "top_k": 3,
                    "rerank": True,
                    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                },
            ),
        ],
    )

    executor = WorkflowExecutor(llm=MockLLMClient())
    result = await executor.run(wf, "What are the primary causes and consequences of global warming?")

    node_res = result.node_results["gw_retriever"]
    print("=== Global Warming Retrieval Results ===")
    print(f"Status: {node_res.status.value}")
    print(f"Retrieved {node_res.data.get('count')} chunks (reranked: {node_res.data.get('reranked')}):\n")
    print(node_res.text)


if __name__ == "__main__":
    asyncio.run(test_global_warming_retrieval())
