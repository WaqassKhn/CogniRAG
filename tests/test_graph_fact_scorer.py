"""
tests/test_graph_fact_scorer.py - Unit tests for GraphFactScorer and fact ranking.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from rag.graph.fact_scorer import GraphFactScorer, fact_to_text


def test_fact_to_text():
    fact = {
        "source": "Travel Policy",
        "source_type": "POLICY",
        "relation_type": "REQUIRES_APPROVAL_FROM",
        "target": "VP Operations",
        "target_type": "ROLE",
        "description": "Approval required for amounts > $5,000",
    }
    text = fact_to_text(fact)
    assert "Travel Policy (POLICY)" in text
    assert "requires approval from" in text
    assert "VP Operations (ROLE)" in text
    assert "Approval required for amounts > $5,000" in text


def test_fact_scorer_composite_and_ranking():
    mock_emb = MagicMock()
    # Return 384-d unit vector for query, fact 1, and fact 2
    mock_emb.embed_query.return_value = np.array([1.0] + [0.0] * 383, dtype=np.float32)
    mock_emb.embed_texts.return_value = np.array([
        [0.9] + [0.0] * 383,   # Fact 1: high cosine sim
        [0.1] + [0.0] * 383,   # Fact 2: low cosine sim
    ], dtype=np.float32)

    scorer = GraphFactScorer(embedding_manager=mock_emb)

    facts = [
        {
            "source": "Corporate Overview",
            "source_type": "DOCUMENT",
            "relation_type": "MENTIONS",
            "target": "Annual Report",
            "target_type": "DOCUMENT",
            "description": "General summary",
            "confidence": 0.8,
            "hop_distance": 2,
            "chunk_id": "c_irrelevant",
        },
        {
            "source": "Travel Policy",
            "source_type": "POLICY",
            "relation_type": "REQUIRES_APPROVAL_FROM",
            "target": "VP Operations",
            "target_type": "ROLE",
            "description": "Approval required for > $5,000",
            "confidence": 1.0,
            "hop_distance": 1,
            "chunk_id": "c_relevant",
        },
    ]

    query = "Who approves travel reimbursement requests exceeding $5,000?"
    ranked = scorer.score_and_rank_facts(
        query=query,
        facts=facts,
        query_entities=["Travel Policy", "VP Operations"],
        embedding_manager=mock_emb,
    )

    assert len(ranked) == 2
    # The relevant travel approval fact should rank #1
    assert ranked[0]["rank"] == 1
    assert ranked[0]["source"] == "Travel Policy"
    assert ranked[0]["relation_type"] == "REQUIRES_APPROVAL_FROM"
    assert ranked[0]["score"] > ranked[1]["score"]
    assert 0.0 <= ranked[0]["score"] <= 1.0
    assert 0.0 <= ranked[1]["score"] <= 1.0

    # Verify score breakdown exists
    breakdown = ranked[0]["score_breakdown"]
    assert "semantic" in breakdown
    assert "entity" in breakdown
    assert "relation" in breakdown
    assert "confidence" in breakdown
    assert "distance" in breakdown
    assert breakdown["distance"] == 1.0  # 1-hop = 1.0


def test_distance_penalty():
    scorer = GraphFactScorer(embedding_manager=None)

    fact_1hop = {
        "source": "A",
        "relation_type": "RELATES_TO",
        "target": "B",
        "hop_distance": 1,
    }
    fact_2hop = {
        "source": "A",
        "relation_type": "RELATES_TO",
        "target": "B",
        "hop_distance": 2,
    }
    fact_3hop = {
        "source": "A",
        "relation_type": "RELATES_TO",
        "target": "B",
        "hop_distance": 3,
    }

    assert scorer._compute_distance_score(fact_1hop, ["A"]) == 1.0
    assert scorer._compute_distance_score(fact_2hop, ["A"]) == 0.65
    assert scorer._compute_distance_score(fact_3hop, ["A"]) == 0.35


def test_relation_intent_match():
    scorer = GraphFactScorer(embedding_manager=None)

    approval_fact = {
        "source": "Policy",
        "relation_type": "REQUIRES_APPROVAL_FROM",
        "target": "Manager",
        "description": "Signoff needed",
    }
    unrelated_fact = {
        "source": "Company",
        "relation_type": "OWNS_SUBSIDIARY",
        "target": "Division",
        "description": "Branch office",
    }

    q_lower = "who approves the travel request"
    q_tokens = set(q_lower.split())

    score_app = scorer._compute_relation_match(approval_fact, q_lower, q_tokens)
    score_unrelated = scorer._compute_relation_match(unrelated_fact, q_lower, q_tokens)

    assert score_app > score_unrelated
    assert score_app == 1.0  # matches intent keyword "approv"


def test_lexical_fallback_without_embedding_manager():
    scorer = GraphFactScorer(embedding_manager=None)

    facts = [
        {
            "source": "Leave Policy",
            "relation_type": "APPLIES_TO",
            "target": "Engineering",
            "description": "Annual leave quota",
            "chunk_id": "c1",
        },
        {
            "source": "Security Standard",
            "relation_type": "GOVERNED_BY",
            "target": "IT Team",
            "description": "Access control",
            "chunk_id": "c2",
        },
    ]

    ranked = scorer.score_and_rank_facts(
        query="What is the leave policy for engineering?",
        facts=facts,
        query_entities=["Leave Policy", "Engineering"],
    )

    assert len(ranked) == 2
    assert ranked[0]["source"] == "Leave Policy"
    assert ranked[0]["score"] > ranked[1]["score"]
