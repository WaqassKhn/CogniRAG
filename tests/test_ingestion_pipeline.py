"""
tests/test_ingestion_pipeline.py - Unit tests for DualIngestionPipeline.
"""

from unittest.mock import MagicMock, patch
import pytest

from pipeline.ingestion import DualIngestionPipeline


@pytest.fixture
def mock_pipeline_dependencies():
    mock_vdb = MagicMock()
    mock_emb = MagicMock()
    mock_emb.embed_texts.return_value = [[0.1] * 384, [0.2] * 384]

    mock_neo4j = MagicMock()
    mock_neo4j.is_available.return_value = True
    mock_neo4j.init_schema.return_value = True
    mock_neo4j.batch_upsert_subgraph.return_value = True
    mock_neo4j.delete_document_subgraph.return_value = True

    sample_entities = [{"name": "Travel Policy", "type": "POLICY", "description": "Travel rules"}]
    sample_relations = [{"source": "Travel Policy", "target": "VP Operations", "type": "REQUIRES_APPROVAL_FROM"}]
    mock_extractor = MagicMock()
    mock_extractor.extract_subgraph_from_chunk.return_value = {
        "entities": sample_entities,
        "relations": sample_relations,
    }
    mock_extractor.extract_subgraph_from_chunks.return_value = (sample_entities, sample_relations)
    mock_extractor.extract_subgraph_from_document.return_value = (sample_entities, sample_relations)

    return mock_vdb, mock_emb, mock_neo4j, mock_extractor


def test_dual_ingestion_neo4j_offline(mock_pipeline_dependencies):
    mock_vdb, mock_emb, mock_neo4j, mock_extractor = mock_pipeline_dependencies
    mock_neo4j.is_available.return_value = False

    pipeline = DualIngestionPipeline(
        vector_db=mock_vdb,
        embedding_manager=mock_emb,
        neo4j_client=mock_neo4j,
        graph_extractor=mock_extractor,
    )

    with patch("pipeline.ingestion.DocumentParser.parse_file") as mock_parse, \
         patch.object(pipeline.chunker, "chunk_parsed_document") as mock_chunk:
        mock_parse.return_value = {"pages": [{"text": "Sample text", "tables": []}]}
        mock_chunk.return_value = [
            {"chunk_id": "c1", "text": "Sample text chunk 1", "page_number": 1, "filename": "doc.pdf"},
        ]

        result = pipeline.ingest_file("doc.pdf")

        assert result["chunks"] == 1
        assert result["graph_synced"] is False
        assert mock_vdb.upsert_chunks.called
        assert not mock_neo4j.batch_upsert_subgraph.called


def test_dual_ingestion_success(mock_pipeline_dependencies):
    mock_vdb, mock_emb, mock_neo4j, mock_extractor = mock_pipeline_dependencies

    pipeline = DualIngestionPipeline(
        vector_db=mock_vdb,
        embedding_manager=mock_emb,
        neo4j_client=mock_neo4j,
        graph_extractor=mock_extractor,
    )

    with patch("pipeline.ingestion.DocumentParser.parse_file") as mock_parse, \
         patch.object(pipeline.chunker, "chunk_parsed_document") as mock_chunk:
        mock_parse.return_value = {"pages": [{"text": "Sample policy text", "tables": []}]}
        mock_chunk.return_value = [
            {"chunk_id": "c1", "text": "Sample policy chunk 1", "page_number": 1, "filename": "policy.pdf"},
            {"chunk_id": "c2", "text": "Sample policy chunk 2", "page_number": 2, "filename": "policy.pdf"},
        ]

        result = pipeline.ingest_file("policy.pdf")

        assert result["chunks"] == 2
        assert result["graph_synced"] is True
        assert result["entities"] >= 1
        assert result["relations"] >= 1
        assert mock_vdb.upsert_chunks.called
        assert mock_neo4j.batch_upsert_subgraph.called


def test_dual_ingestion_delete_document(mock_pipeline_dependencies):
    mock_vdb, mock_emb, mock_neo4j, mock_extractor = mock_pipeline_dependencies
    mock_vdb.delete_by_filename.return_value = 5

    pipeline = DualIngestionPipeline(
        vector_db=mock_vdb,
        embedding_manager=mock_emb,
        neo4j_client=mock_neo4j,
        graph_extractor=mock_extractor,
    )

    res = pipeline.delete_document("policy.pdf")
    assert res["deleted_chunks"] == 5
    assert res["graph_deleted"] is True
    mock_vdb.delete_by_filename.assert_called_once_with("policy.pdf")
    mock_neo4j.delete_document_subgraph.assert_called_once_with("policy.pdf")
