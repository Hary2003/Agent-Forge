"""
Tests for DocumentChunker and ingestion utility.
"""
import sys
sys.path.insert(0, "src")

import pytest

from agentforge.rag.embeddings import MockEmbeddingProvider
from agentforge.rag.ingest import DocumentChunker, ingest_file_or_directory, ingest_text
from agentforge.rag.vector_store import InMemoryVectorStore


def test_document_chunker_fixed_size_and_overlap():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.split_text(text)

    # Check chunks length and overlap
    assert len(chunks) > 1
    assert chunks[0] == "ABCDEFGHIJ"
    # Step size is 10 - 2 = 8, so next starts at index 8 ("IJ")
    assert chunks[1] == "IJKLMNOPQR"


def test_ingest_text_stores_chunks_in_vector_store():
    store = InMemoryVectorStore(collection_name="test_col")
    embedder = MockEmbeddingProvider()

    sample_doc = "AgentForge is a Python framework for multi-agent DAG execution. It supports node execution and RAG retrieval."
    num_chunks = ingest_text(
        text=sample_doc,
        source_name="guide.md",
        vector_store=store,
        embedding_provider=embedder,
        chunk_size=50,
        chunk_overlap=10,
    )

    assert num_chunks > 0
    # Search
    query_emb = embedder.embed_query("multi-agent DAG")
    results = store.search(query_emb, top_k=2)
    assert len(results) > 0
    assert "guide.md" in results[0].source
