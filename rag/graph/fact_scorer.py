"""
rag/graph/fact_scorer.py - Relevance scoring and ranking for Knowledge Graph facts.

Implements multi-feature composite scoring for Knowledge Graph paths:
  1. Semantic Similarity (40%): Cosine similarity between query and converted fact text.
  2. Entity Match Strength (25%): Exact, normalized, and partial match with query entities.
  3. Relationship Match Strength (15%): Query intent alignment with relation type & description.
  4. Extraction Confidence (10%): Ingestion/model confidence on the graph edge.
  5. Distance Penalty (10%): Proximity to the query seed entity (1-hop vs multi-hop).

Transforms unordered Neo4j path records into a ranked list of scored facts
with traceable evidence chunk IDs.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from rag.graph.extractor import EntityNormalizer

logger = logging.getLogger(__name__)

# Default scoring weights summing to 1.0
DEFAULT_SCORING_WEIGHTS: Dict[str, float] = {
    "semantic": 0.40,
    "entity": 0.25,
    "relation": 0.15,
    "confidence": 0.10,
    "distance": 0.10,
}

# Common enterprise intent keywords mapped to relation types
RELATION_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "REQUIRES_APPROVAL_FROM": ["approv", "signoff", "authorization", "permission", "who approves", "sanction"],
    "REPORTS_TO": ["report", "manager", "supervisor", "hierarchy", "organogram", "under who"],
    "APPLIES_TO": ["applies", "scope", "applicable", "who does this apply", "eligib"],
    "DEFINES_LIMIT": ["limit", "threshold", "maximum", "minimum", "ceiling", "cap", "upto", "exceed"],
    "RESPONSIBLE_FOR": ["responsible", "duties", "in charge", "handles", "owns", "tasks"],
    "ESCALATES_TO": ["escalat", "dispute", "exception", "higher authority"],
    "TRACKS_KPI": ["kpi", "metric", "target", "measure", "performance", "goal"],
    "HAS_KPI_METRIC": ["metric", "value", "target", "budget", "revenue", "ebitda", "figures"],
    "SUPERSEDES": ["supersede", "replaces", "prior", "previous", "former", "overrides"],
    "PART_OF_DEPT": ["department", "division", "unit", "team", "part of"],
    "OWNS_SUBSIDIARY": ["subsidiary", "owns", "parent company", "holding"],
}


def fact_to_text(fact: Dict[str, Any]) -> str:
    """
    Converts a raw Knowledge Graph path record into standardized natural language text.

    Format:
        "Source (Type) [relation humanized] Target (Type). Description"
    Example:
        "Travel Policy (POLICY) requires approval from VP Operations (ROLE). Approval required for > $5,000"
    """
    source = (fact.get("source") or "Unknown").strip()
    src_type = (fact.get("source_type") or "Entity").strip()
    rel_raw = (fact.get("relation_type") or "RELATES_TO").strip()
    target = (fact.get("target") or "Unknown").strip()
    tgt_type = (fact.get("target_type") or "Entity").strip()
    desc = (fact.get("description") or "").strip()

    rel_human = rel_raw.replace("_", " ").lower()

    text = f"{source} ({src_type}) {rel_human} {target} ({tgt_type})"
    if desc and desc.lower() not in ("relates to", rel_human):
        text += f". {desc}"

    return text


class GraphFactScorer:
    """
    Scores and ranks Knowledge Graph facts based on relevance to a given query.
    Combines semantic embeddings, lexical heuristics, confidence, and hop distance.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        embedding_manager: Optional[Any] = None,
    ):
        self.weights = weights or dict(DEFAULT_SCORING_WEIGHTS)
        self.embedding_manager = embedding_manager

    def score_and_rank_facts(
        self,
        query: str,
        facts: List[Dict[str, Any]],
        query_vec: Optional[np.ndarray] = None,
        query_entities: Optional[List[str]] = None,
        embedding_manager: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Calculates composite relevance scores for candidate graph facts and returns them
        sorted in descending order of score.
        """
        if not facts:
            return []

        emb_mgr = embedding_manager or self.embedding_manager

        # Step 1: Format searchable text for each fact
        fact_texts = [fact_to_text(f) for f in facts]

        # Step 2: Compute semantic similarities
        semantic_scores = self._compute_semantic_similarities(
            query=query,
            fact_texts=fact_texts,
            query_vec=query_vec,
            embedding_manager=emb_mgr,
        )

        # Step 3: Compute lexical and graph heuristic scores
        q_lower = query.lower()
        q_tokens = set(re.findall(r'\b\w+\b', q_lower))
        resolved_q_entities = [e.lower() for e in (query_entities or [])]

        scored_records: List[Dict[str, Any]] = []

        for idx, fact in enumerate(facts):
            sim_score = float(semantic_scores[idx]) if idx < len(semantic_scores) else 0.0
            entity_score = self._compute_entity_match(fact, q_lower, resolved_q_entities)
            rel_score = self._compute_relation_match(fact, q_lower, q_tokens)
            conf_score = self._compute_confidence(fact)
            dist_score = self._compute_distance_score(fact, resolved_q_entities)

            # Composite formula
            composite = (
                self.weights["semantic"] * sim_score
                + self.weights["entity"] * entity_score
                + self.weights["relation"] * rel_score
                + self.weights["confidence"] * conf_score
                + self.weights["distance"] * dist_score
            )
            composite = max(0.0, min(1.0, composite))

            src = fact.get("source", "Unknown")
            src_type = fact.get("source_type", "Entity")
            rel = fact.get("relation_type", "RELATES_TO")
            tgt = fact.get("target", "Unknown")
            tgt_type = fact.get("target_type", "Entity")
            desc = fact.get("description", "")
            chunk_id = fact.get("chunk_id")

            scored_records.append({
                "fact_text": fact_texts[idx],
                "score": round(composite, 4),
                "source": src,
                "source_type": src_type,
                "relation_type": rel,
                "target": tgt,
                "target_type": tgt_type,
                "description": desc,
                "chunk_id": chunk_id,
                "score_breakdown": {
                    "semantic": round(sim_score, 4),
                    "entity": round(entity_score, 4),
                    "relation": round(rel_score, 4),
                    "confidence": round(conf_score, 4),
                    "distance": round(dist_score, 4),
                },
            })

        # Step 4: Sort descending by final score
        scored_records.sort(key=lambda x: x["score"], reverse=True)

        # Assign 1-indexed rank
        for rank, record in enumerate(scored_records, 1):
            record["rank"] = rank

        return scored_records

    def _compute_semantic_similarities(
        self,
        query: str,
        fact_texts: List[str],
        query_vec: Optional[np.ndarray],
        embedding_manager: Optional[Any],
    ) -> List[float]:
        """
        Computes cosine similarity between query and each fact's text.
        Falls back to token overlap similarity if embeddings are unavailable.
        """
        if embedding_manager is not None:
            try:
                # Ensure query vector exists
                if query_vec is None:
                    query_vec = embedding_manager.embed_query(query)
                q_arr = np.asarray(query_vec, dtype=np.float32)
                q_norm = np.linalg.norm(q_arr)
                if q_norm > 1e-8:
                    q_arr = q_arr / q_norm

                # Batch embed fact texts
                if hasattr(embedding_manager, "embed_texts"):
                    fact_vecs = embedding_manager.embed_texts(fact_texts)
                elif hasattr(embedding_manager, "embed_chunks"):
                    fact_vecs = embedding_manager.embed_chunks([{"text": t} for t in fact_texts])
                else:
                    fact_vecs = np.array([embedding_manager.embed_query(t) for t in fact_texts])

                f_arr = np.asarray(fact_vecs, dtype=np.float32)
                f_norms = np.linalg.norm(f_arr, axis=1, keepdims=True)
                f_norms = np.where(f_norms > 1e-8, f_norms, 1.0)
                f_normed = f_arr / f_norms

                similarities = np.dot(f_normed, q_arr)
                # Clip cosine similarity to [0.0, 1.0] (MiniLM cosine range)
                return [float(np.clip(s, 0.0, 1.0)) for s in similarities]
            except Exception as exc:
                logger.debug(f"[GraphFactScorer] Embedding similarity failed, falling back to lexical: {exc}")

        # Lexical Jaccard/Overlap fallback if embedding manager offline
        q_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        scores = []
        for text in fact_texts:
            f_tokens = set(re.findall(r'\b\w+\b', text.lower()))
            if not f_tokens or not q_tokens:
                scores.append(0.1)
                continue
            intersection = q_tokens.intersection(f_tokens)
            union = q_tokens.union(f_tokens)
            jaccard = len(intersection) / len(union) if union else 0.0
            overlap = len(intersection) / len(q_tokens) if q_tokens else 0.0
            score = 0.5 * jaccard + 0.5 * overlap
            scores.append(float(min(1.0, max(0.0, score))))
        return scores

    def _compute_entity_match(
        self,
        fact: Dict[str, Any],
        q_lower: str,
        resolved_q_entities: List[str],
    ) -> float:
        """
        Scores how strongly the fact's source and target entities match the query.
        """
        src = (fact.get("source") or "").lower().strip()
        tgt = (fact.get("target") or "").lower().strip()

        def match_entity(ent: str) -> float:
            if not ent:
                return 0.0
            # 1. Exact substring match in query with word boundary
            pattern = rf'(?<!\w){re.escape(ent)}(?!\w)'
            if re.search(pattern, q_lower):
                return 1.0
            # 2. Match against resolved query entities list
            if ent in resolved_q_entities:
                return 1.0
            for rq in resolved_q_entities:
                if ent in rq or rq in ent:
                    return 0.85
            # 3. Token overlap (e.g. 'VP Operations' vs 'VP')
            ent_tokens = [t for t in re.findall(r'\b\w+\b', ent) if len(t) > 2]
            if ent_tokens:
                matched_tokens = sum(1 for t in ent_tokens if t in q_lower)
                if matched_tokens > 0:
                    return 0.6 + 0.3 * (matched_tokens / len(ent_tokens))
            return 0.0

        src_match = match_entity(src)
        tgt_match = match_entity(tgt)

        if src_match > 0.8 and tgt_match > 0.8:
            return 1.0
        if src_match > 0.0 and tgt_match > 0.0:
            return min(1.0, 0.5 * src_match + 0.5 * tgt_match + 0.2)
        if src_match > 0.0:
            return max(0.4, src_match * 0.85)
        if tgt_match > 0.0:
            return max(0.4, tgt_match * 0.85)

        # Neither entity matched directly in query (multi-hop expansion)
        return 0.2

    def _compute_relation_match(
        self,
        fact: Dict[str, Any],
        q_lower: str,
        q_tokens: set[str],
    ) -> float:
        """
        Scores how well the relationship type and description match user intent.
        """
        rel_type = (fact.get("relation_type") or "").upper()
        desc = (fact.get("description") or "").lower()

        # 1. Check known intent keywords
        intent_keywords = RELATION_INTENT_KEYWORDS.get(rel_type, [])
        for kw in intent_keywords:
            if kw in q_lower:
                return 1.0

        # 2. Relation type token match (e.g. "approval" in query, relation "REQUIRES_APPROVAL_FROM")
        rel_words = [w.lower() for w in rel_type.split("_") if len(w) > 2]
        matched_rel_words = [w for w in rel_words if w in q_lower or any(w in qt for qt in q_tokens)]
        if matched_rel_words:
            return min(1.0, 0.6 + 0.2 * len(matched_rel_words))

        # 3. Description token overlap
        if desc:
            desc_tokens = set(re.findall(r'\b\w+\b', desc))
            common = q_tokens.intersection(desc_tokens)
            if common:
                return min(0.9, 0.4 + 0.15 * len(common))

        return 0.2

    def _compute_confidence(self, fact: Dict[str, Any]) -> float:
        """Extracts and clamps extraction confidence score."""
        val = fact.get("confidence")
        if val is None:
            return 1.0
        try:
            return float(np.clip(float(val), 0.0, 1.0))
        except (ValueError, TypeError):
            return 1.0

    def _compute_distance_score(
        self,
        fact: Dict[str, Any],
        resolved_q_entities: List[str],
    ) -> float:
        """
        Computes proximity score (1.0 for 1-hop, with decay for higher hops).
        """
        hops = fact.get("hop_distance")
        if hops is not None:
            try:
                h = int(hops)
                if h <= 1:
                    return 1.0
                elif h == 2:
                    return 0.65
                else:
                    return 0.35
            except (ValueError, TypeError):
                pass

        # If hop distance not explicitly recorded, check if edge touches a seed query entity
        src = (fact.get("source") or "").lower().strip()
        tgt = (fact.get("target") or "").lower().strip()
        if src in resolved_q_entities or tgt in resolved_q_entities:
            return 1.0
        return 0.65
