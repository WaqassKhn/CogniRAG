"""
tests/test_query_planner_graph.py - Unit tests for QueryPlannerAgent GraphRAG strategy routing.
"""

from unittest.mock import MagicMock
import pytest

from rag.agents.query_planner import QueryPlannerAgent


def test_heuristic_strategy_classification():
    # 1. Graph-only: governance, approval, hierarchy
    q1 = "Who approves travel requests exceeding $5,000?"
    assert QueryPlannerAgent.classify_strategy_heuristic(q1) == "graph_only"

    q2 = "Who does the VP of Engineering report to?"
    assert QueryPlannerAgent.classify_strategy_heuristic(q2) == "graph_only"

    q3 = "What is the escalation approval workflow for procurement?"
    assert QueryPlannerAgent.classify_strategy_heuristic(q3) == "graph_only"

    # 2. Vector-only: pure narrative / background
    q4 = "Summarize the history of the corporation"
    assert QueryPlannerAgent.classify_strategy_heuristic(q4) == "vector_only"

    q5 = "What is our corporate philosophy and mission statement?"
    assert QueryPlannerAgent.classify_strategy_heuristic(q5) == "vector_only"

    # 3. Hybrid: analytical, comparative, or multi-fact
    q6 = "Compare renewable capacity between Project Alpha and Beta in FY24"
    assert QueryPlannerAgent.classify_strategy_heuristic(q6) == "hybrid"


def test_heuristic_entity_extraction():
    q = "Does the VP of Engineering need approval from Human Resources under Section 4.2?"
    entities = QueryPlannerAgent.extract_heuristic_entities(q)

    assert "Vice President" in entities
    assert "Human Resources" in entities
    assert any("Section 4.2" in e for e in entities)


def test_planner_plan_with_mock_llm():
    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.generate.return_value = """
    {
      "complexity": "complex",
      "strategy": "hybrid",
      "sub_queries": [
        "What is the travel policy reimbursement threshold?",
        "Who is the approving authority for VP expenses?"
      ],
      "doc_scope": ["travel_policy.pdf"],
      "target_entities": ["Travel Policy", "Vice President"]
    }
    """

    planner = QueryPlannerAgent(llm=mock_llm, known_documents=["travel_policy.pdf"])
    plan = planner.plan("Compare travel thresholds and approving authorities for VPs")

    assert plan["complexity"] == "complex"
    assert plan["strategy"] == "hybrid"
    assert len(plan["sub_queries"]) == 2
    assert plan["doc_scope"] == ["travel_policy.pdf"]
    assert "Travel Policy" in plan["target_entities"]


def test_planner_fallback_when_llm_offline():
    planner = QueryPlannerAgent(llm=None, known_documents=["policy.pdf"])
    plan = planner.plan("Who approves reimbursement under Section 4.2?")

    assert plan["complexity"] == "simple"
    assert plan["strategy"] == "graph_only"
    assert plan["sub_queries"] == ["Who approves reimbursement under Section 4.2?"]
    assert any("Section 4.2" in e for e in plan["target_entities"])
