"""
tests/test_hybrid_rag_chain.py - Unit tests for RAGChain with Knowledge Graph hybrid fusion.
"""

from unittest.mock import MagicMock
import pytest

from rag.chain import RAGChain


@pytest.fixture
def mock_chain_components():
    mock_vdb = MagicMock()
    mock_vdb.search.return_value = [
        ({"chunk_id": "c1", "filename": "policy.pdf", "page_number": 1, "text": "Travel reimbursement chunk text"}, 0.92)
    ]
    mock_vdb.chunks_metadata = [
        {"chunk_id": "c1", "filename": "policy.pdf", "page_number": 1, "text": "Travel reimbursement chunk text"}
    ]

    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [0.05] * 384

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "According to the Travel Policy [Source: policy.pdf, Page 1] and [Knowledge Graph], VP approval is required."
    mock_llm.generate_stream.return_value = iter(["According ", "to ", "the ", "Travel ", "Policy."])

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        ({"chunk_id": "c1", "filename": "policy.pdf", "page_number": 1, "text": "Travel reimbursement chunk text"}, 0.95)
    ]

    mock_graph_retriever = MagicMock()
    mock_graph_retriever.is_available.return_value = True
    mock_graph_retriever.retrieve_facts.return_value = (
        "• (Travel Policy [POLICY]) -[REQUIRES_APPROVAL_FROM]-> (VP Operations [ROLE])",
        ["c1"]
    )

    return mock_vdb, mock_emb, mock_llm, mock_reranker, mock_graph_retriever


def test_rag_chain_run_with_graph_fusion(mock_chain_components):
    mock_vdb, mock_emb, mock_llm, mock_reranker, mock_graph_retriever = mock_chain_components

    chain = RAGChain(
        vector_db=mock_vdb,
        embedding_manager=mock_emb,
        llm=mock_llm,
        reranker=mock_reranker,
        graph_retriever=mock_graph_retriever,
    )

    result = chain.run("What are the travel reimbursement approval rules?")

    assert result["answer"] is not None
    assert "graph_context" in result
    assert "Travel Policy" in result["graph_context"]
    assert "[Knowledge Graph]" in result["citations"]
    assert "policy.pdf (p. 1)" in result["citations"]

    # Verify that the LLM was called with the fused prompt containing the knowledge graph section
    called_prompt = mock_llm.generate.call_args[1]["prompt"]
    assert "KNOWLEDGE GRAPH CONTEXT (MULTI-HOP RELATIONS & GOVERNANCE):" in called_prompt
    assert "(Travel Policy [POLICY]) -[REQUIRES_APPROVAL_FROM]-> (VP Operations [ROLE])" in called_prompt
    assert "DOCUMENT CONTEXT:" in called_prompt


def test_rag_chain_run_without_graph_retriever(mock_chain_components):
    mock_vdb, mock_emb, mock_llm, mock_reranker, _ = mock_chain_components

    chain = RAGChain(
        vector_db=mock_vdb,
        embedding_manager=mock_emb,
        llm=mock_llm,
        reranker=mock_reranker,
        graph_retriever=None,
    )

    result = chain.run("What are the travel reimbursement approval rules?")

    assert result["graph_context"] == ""
    assert "[Knowledge Graph]" not in result["citations"]

    called_prompt = mock_llm.generate.call_args[1]["prompt"]
    assert "KNOWLEDGE GRAPH CONTEXT" not in called_prompt


def test_rag_chain_run_stream_with_graph_fusion(mock_chain_components):
    mock_vdb, mock_emb, mock_llm, mock_reranker, mock_graph_retriever = mock_chain_components

    chain = RAGChain(
        vector_db=mock_vdb,
        embedding_manager=mock_emb,
        llm=mock_llm,
        reranker=mock_reranker,
        graph_retriever=mock_graph_retriever,
    )

    events = list(chain.run_stream("What are the travel rules?"))

    event_types = [e[0] for e in events]
    assert "context" in event_types
    assert "token" in event_types
    assert "done" in event_types

    done_event = next(data for etype, data in events if etype == "done")
    assert "[Knowledge Graph]" in done_event["citations"]
    assert "Travel Policy" in done_event["graph_context"]
