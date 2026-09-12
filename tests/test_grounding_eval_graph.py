"""
tests/test_grounding_eval_graph.py - Unit tests for DocumentGroundingEvaluator with Knowledge Graph context.
"""

from unittest.mock import MagicMock
import pytest

from evaluation.grounding_eval import DocumentGroundingEvaluator


def test_grounding_eval_with_graph_context():
    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.generate.return_value = """
    {
      "claims": [
        {
          "claim": "VP of Operations approves travel expenses exceeding $5,000",
          "is_grounded": true,
          "explanation": "Directly supported by verified Knowledge Graph relation"
        }
      ],
      "faithfulness_score": 1.0,
      "summary": "Claim fully grounded in Knowledge Graph relations."
    }
    """

    evaluator = DocumentGroundingEvaluator(llm=mock_llm)

    chunks = [
        {"filename": "policy.pdf", "page_number": 1, "text": "Travel Policy details are summarized below."}
    ]
    graph_context = "• (Travel Policy [POLICY]) -[REQUIRES_APPROVAL_FROM]-> (VP Operations [ROLE]) [Approval required for > $5,000]"

    res = evaluator.evaluate_grounding(
        query="Who approves travel over $5,000?",
        answer="The VP of Operations approves travel expenses exceeding $5,000.",
        retrieved_chunks=chunks,
        graph_context=graph_context,
    )

    assert res["is_passed"] is True
    assert res["overall_grounding_score"] >= 0.85
    assert any("5,000" in n for n in res["supported_numbers"])
    assert res["unsupported_numbers"] == []

    # Verify that the judge LLM was called with the combined context including the graph relation
    called_prompt = mock_llm.generate.call_args[1].get("prompt", "") if mock_llm.generate.call_args[1] else mock_llm.generate.call_args[0][0]
    assert "[VERIFIED KNOWLEDGE GRAPH RELATIONS]:" in called_prompt
    assert "REQUIRES_APPROVAL_FROM" in called_prompt
