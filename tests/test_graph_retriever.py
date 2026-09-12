"""
tests/test_graph_retriever.py - Unit tests for GraphRetriever entity resolution and traversal.
"""

from unittest.mock import MagicMock
import pytest

from rag.graph.retriever import GraphRetriever


def test_graph_retriever_offline():
    mock_neo4j = MagicMock()
    mock_neo4j.is_available.return_value = False

    retriever = GraphRetriever(neo4j_client=mock_neo4j)
    assert retriever.is_available() is False

    paths = retriever.retrieve_subgraph("What are the travel limits for HR?")
    assert paths == []

    facts, citations = retriever.retrieve_facts("What are the travel limits for HR?")
    assert facts == ""
    assert citations == []


def test_extract_query_entities():
    retriever = GraphRetriever(neo4j_client=None)

    # 1. Departments and Roles
    q1 = "Who does the VP of Engineering report to in the HR department?"
    entities1 = retriever.extract_query_entities(q1)
    assert "Vice President" in entities1
    assert "Engineering" in entities1
    assert "Human Resources" in entities1

    # 2. Policy Sections
    q2 = "What does Section 4.2 specify regarding reimbursement limits?"
    entities2 = retriever.extract_query_entities(q2)
    assert any("Section 4.2" in e for e in entities2)

    # 3. Currency and Metrics
    q3 = "Can I claim expenses exceeding $5,000 in FY24?"
    entities3 = retriever.extract_query_entities(q3)
    assert "$5,000" in entities3 or any("5,000" in e for e in entities3)
    assert any("FY24" in e for e in entities3)


def test_retrieve_subgraph_and_format_facts():
    mock_neo4j = MagicMock()
    mock_neo4j.is_available.return_value = True
    mock_neo4j.get_entity_neighborhood.return_value = [
        {
            "source": "Travel Policy",
            "source_type": "POLICY",
            "relation_type": "REQUIRES_APPROVAL_FROM",
            "description": "Approval required for > $5,000",
            "chunk_id": "chunk_pol_1",
            "target": "VP Operations",
            "target_type": "ROLE",
        },
        {
            "source": "VP Operations",
            "source_type": "ROLE",
            "relation_type": "REPORTS_TO",
            "description": "",
            "chunk_id": "chunk_org_2",
            "target": "Chief Operating Officer",
            "target_type": "ROLE",
        }
    ]

    retriever = GraphRetriever(neo4j_client=mock_neo4j)
    facts, citations = retriever.retrieve_facts("Who approves the Travel Policy?")

    assert "(Travel Policy [POLICY]) -[REQUIRES_APPROVAL_FROM]-> (VP Operations [ROLE])" in facts
    assert "Approval required for > $5,000" in facts
    assert "(VP Operations [ROLE]) -[REPORTS_TO]-> (Chief Operating Officer [ROLE])" in facts
    assert "chunk_pol_1" in citations
    assert "chunk_org_2" in citations
