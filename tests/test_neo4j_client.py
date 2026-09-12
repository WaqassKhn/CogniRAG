"""
tests/test_neo4j_client.py - Unit tests for Neo4jClient connection and operations.
"""

from unittest.mock import MagicMock, patch
import pytest

from rag.graph.neo4j_client import Neo4jClient


def test_neo4j_client_disabled():
    client = Neo4jClient(enabled=False)
    assert not client.is_available()
    assert client.get_driver() is None
    assert client.get_graph_statistics()["status"] == "disconnected"
    assert client.get_entity_neighborhood(["Acme"]) == []


def test_neo4j_client_offline_graceful():
    # Attempt connection to a non-existent port
    client = Neo4jClient(
        uri="bolt://127.0.0.1:9999",
        username="neo4j",
        password="badpassword",
        enabled=True,
    )
    # Must not raise an exception; must return False gracefully
    assert client.is_available(force_check=True) is False
    assert client.get_graph_statistics()["status"] == "disconnected"
    assert client.get_entity_neighborhood(["AnyEntity"]) == []


@patch("rag.graph.neo4j_client.NEO4J_AVAILABLE", True)
def test_neo4j_client_mocked_schema_and_batch_upsert():
    client = Neo4jClient(enabled=True)

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    # Mock ping for is_available
    mock_ping_record = {"ping": 1}
    mock_session.run.return_value.single.return_value = mock_ping_record

    client._driver = mock_driver
    client._connected = True

    # 1. Schema Init
    schema_ok = client.init_schema()
    assert schema_ok is True
    assert mock_session.run.call_count >= 1

    # 2. Batch Upsert Subgraph
    test_doc = {"filename": "policy.pdf", "total_chunks": 2, "doc_type": "pdf"}
    test_chunks = [
        {"filename": "policy.pdf", "chunk_id": "c1", "page_number": 1, "has_table": False, "text_preview": "preview 1"}
    ]
    test_entities = [
        {"name": "Travel Policy", "type": "POLICY", "description": "policy desc", "chunk_id": "c1"}
    ]
    test_relations = [
        {"source": "Travel Policy", "target": "VP Operations", "type": "REQUIRES_APPROVAL_FROM", "description": "desc", "confidence": 0.9, "chunk_id": "c1"}
    ]

    upsert_ok = client.batch_upsert_subgraph(
        document=test_doc,
        chunks=test_chunks,
        entities=test_entities,
        relations=test_relations,
    )
    assert upsert_ok is True

    # 3. Neighborhood Query
    mock_session.run.return_value = [
        MagicMock(data=lambda: {
            "source": "Travel Policy",
            "source_type": "POLICY",
            "relation_type": "REQUIRES_APPROVAL_FROM",
            "description": "Approval required",
            "chunk_id": "c1",
            "target": "VP Operations",
            "target_type": "ROLE"
        })
    ]

    neighborhood = client.get_entity_neighborhood(["Travel Policy"], max_hops=2)
    assert len(neighborhood) == 1
    assert neighborhood[0]["source"] == "Travel Policy"
    assert neighborhood[0]["target"] == "VP Operations"
