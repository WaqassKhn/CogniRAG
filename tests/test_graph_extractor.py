"""
tests/test_graph_extractor.py - Unit tests for EntityNormalizer and GraphExtractor.
"""

import pytest
from unittest.mock import MagicMock

from rag.graph.extractor import EntityNormalizer, GraphExtractor


def test_entity_normalizer():
    # Departments
    assert EntityNormalizer.normalize("hr", "DEPARTMENT") == "Human Resources"
    assert EntityNormalizer.normalize("Human Resource", "DEPARTMENT") == "Human Resources"
    assert EntityNormalizer.normalize("Engineering", "DEPARTMENT") == "Engineering"

    # Roles
    assert EntityNormalizer.normalize("vp", "ROLE") == "Vice President"
    assert EntityNormalizer.normalize("cfo", "ROLE") == "Chief Financial Officer"
    assert EntityNormalizer.normalize("Project Manager", "ROLE") == "Project Manager"

    # Fiscal Years
    assert EntityNormalizer.normalize("FY 2024") == "FY24"
    assert EntityNormalizer.normalize("FY-25") == "FY25"


def test_graph_extractor_policy_and_approval():
    extractor = GraphExtractor(llm=None, mode="local_rules")

    sample_policy_chunk = """
    Section 4.2 Travel Policy:
    All international business travel expenses exceeding $2,500 must receive prior written approval from the Vice President of Operations.
    Standard per diem reimbursement rate is capped at $150 per day under Section 4.2.
    """

    result = extractor.extract_subgraph_from_chunk(
        chunk_text=sample_policy_chunk,
        chunk_id="travel_policy_p4_c0",
        filename="travel_policy_2026.pdf",
        page_number=4,
    )

    entities = result["entities"]
    relations = result["relations"]

    entity_names = [e["name"] for e in entities]
    entity_types = {e["type"] for e in entities}

    # Policy section detected
    assert any("Section 4.2" in name for name in entity_names)
    # Role detected
    assert any("Vice President" in name for name in entity_names)
    # Metric detected
    assert any("$2,500" in name or "$150" in name for name in entity_names)

    # Relations detected
    assert len(relations) >= 1
    rel_types = {r["type"] for r in relations}
    assert "REQUIRES_APPROVAL_FROM" in rel_types or "DEFINES_LIMIT" in rel_types


def test_graph_extractor_financial_and_operational_report():
    extractor = GraphExtractor(llm=None, mode="local_rules")

    sample_report_chunk = """
    Q3 Operations Review:
    The Engineering Department delivered Project Titan with an SLA of 99.9% uptime.
    Total quarterly expenditure was reported at $4.8 Million, approved by Managing Director.
    """

    result = extractor.extract_subgraph_from_chunk(
        chunk_text=sample_report_chunk,
        chunk_id="ops_review_p2_c1",
        filename="q3_operations_review.pdf",
        page_number=2,
    )

    entities = result["entities"]
    entity_names = [e["name"] for e in entities]

    assert any("Engineering" in name for name in entity_names)
    assert any("Managing Director" in name for name in entity_names)
    assert any("99.9%" in name or "$4.8 Million" in name for name in entity_names)


def test_graph_extractor_with_mock_llm():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = """
    ```json
    {
      "entities": [
        {"name": "Project Titan", "type": "PROJECT", "description": "Cloud migration initiative"},
        {"name": "Engineering", "type": "DEPARTMENT", "description": "Core tech team"}
      ],
      "relations": [
        {"source": "Engineering", "target": "Project Titan", "type": "DELIVERS_PROJECT", "description": "Responsible for execution"}
      ]
    }
    ```
    """

    extractor = GraphExtractor(llm=mock_llm, mode="hybrid")
    sample_text = "Engineering is leading Project Titan for digital infrastructure."

    result = extractor.extract_subgraph_from_chunk(
        chunk_text=sample_text,
        chunk_id="titan_doc_c0",
        filename="project_titan.docx",
    )

    entities = result["entities"]
    relations = result["relations"]

    assert any(e["name"] == "Project Titan" for e in entities)
    assert any(r["type"] == "DELIVERS_PROJECT" for r in relations)


def test_extract_subgraph_from_document_gemini_batch():
    mock_gemini = MagicMock()
    mock_gemini.is_available.return_value = True
    mock_gemini.generate.return_value = """
    ```json
    {
      "entities": [
        {"name": "VP Operations", "type": "ROLE", "description": "Approving Authority"},
        {"name": "Travel Policy", "type": "POLICY", "description": "Corporate travel standard"}
      ],
      "relations": [
        {"source": "Travel Policy", "target": "VP Operations", "type": "REQUIRES_APPROVAL_FROM", "description": "Approves international flights"}
      ]
    }
    ```
    """

    extractor = GraphExtractor(gemini_llm=mock_gemini, mode="gemini_batch")
    doc_text = "Section 4.2 Travel Policy: All travel above $2,500 must be approved by VP Operations."
    chunks = [
        {"chunk_id": "travel_c0", "text": "Section 4.2 Travel Policy: All travel above $2,500 must be approved by VP Operations.", "page_number": 1}
    ]

    entities, relations = extractor.extract_subgraph_from_document(
        document_text=doc_text,
        filename="travel_policy.pdf",
        chunks=chunks,
    )

    # Verify Gemini was called exactly once in batch mode
    mock_gemini.generate.assert_called_once()
    assert any("Vice President" in e["name"] or "Travel Policy" in e["name"] for e in entities)
    assert any(r["type"] == "REQUIRES_APPROVAL_FROM" for r in relations)


def test_extract_subgraph_from_chunks_delegation():
    extractor = GraphExtractor(llm=None, mode="local_rules")
    chunks = [
        {"chunk_id": "chunk_1", "text": "Section 1.1 HR Policy applies to all staff.", "page_number": 1},
        {"chunk_id": "chunk_2", "text": "Managing Director approved $50,000 budget.", "page_number": 2},
    ]

    entities, relations = extractor.extract_subgraph_from_chunks(chunks, "company_rules.pdf")
    entity_names = [e["name"] for e in entities]

    assert any("Section 1.1" in name for name in entity_names)
    assert any("Managing Director" in name for name in entity_names)
    assert any("$50,000" in name for name in entity_names)

