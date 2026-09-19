"""
rag/reranker.py - Hybrid & Multi-Way Reciprocal Rank Fusion (RRF) Reranker.

Combines:
  1. Dense vector similarity search rankings
  2. BM25 sparse keyword search rankings
  3. Knowledge Graph relevance rankings (from GraphFactScorer)
into a unified, scored, sorted list of document chunks.
Also boosts chunks with exact numerical token matches from the user query.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from config import RRF_K, RERANKED_TOP_K

logger = logging.getLogger(__name__)


class HybridReranker:
    """
    Reranks candidate chunks using Dense, BM25, and Knowledge Graph rankings via Reciprocal Rank Fusion (RRF).
    Boosts chunks containing exact numerical tokens present in the user query.
    """

    def __init__(self, rrf_k: int = RRF_K):
        self.rrf_k = rrf_k

    def _tokenize(self, text: str) -> List[str]:
        """Simple alphanumeric tokenizer preserving numbers and symbols."""
        return re.findall(r'\b\w+(?:\.\w+)?\b', text.lower())

    def rerank(
        self,
        query: str,
        dense_results: Optional[List[Tuple[Dict[str, Any], float]]] = None,
        top_k: int = RERANKED_TOP_K,
        all_chunks: Optional[List[Dict[str, Any]]] = None,
        graph_results: Optional[List[Tuple[str, float]]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Unified multi-way RRF fusion combining Dense, BM25, and Graph evidence.

        Args:
            query: User question string.
            dense_results: [(chunk_dict, cosine_score), ...] from vector search.
            top_k: Number of top reranked chunks to return.
            all_chunks: Global or candidate list of chunk dicts for BM25 and metadata lookup.
            graph_results: [(chunk_id, graph_score), ...] ranked by GraphFactScorer.
            weights: Optional weights dict, e.g. {"dense": 1.0, "bm25": 1.0, "graph": 1.0}.

        Returns:
            [(chunk_dict, unified_score), ...] sorted descending by final score.
        """
        dense_results = dense_results or []
        graph_results = graph_results or []
        w = weights or {"dense": 1.0, "bm25": 1.0, "graph": 1.0}

        if not dense_results and not graph_results:
            return []

        # Build chunk lookup dictionary across all available sources
        chunk_map: Dict[str, Dict[str, Any]] = {}
        if all_chunks:
            for c in all_chunks:
                cid = c.get("chunk_id")
                if cid:
                    chunk_map[cid] = c

        for doc, _ in dense_results:
            cid = doc.get("chunk_id")
            if cid and cid not in chunk_map:
                chunk_map[cid] = doc

        # 1. Dense rank map
        dense_rank_map = {
            doc["chunk_id"]: rank + 1
            for rank, (doc, _) in enumerate(dense_results)
            if doc.get("chunk_id")
        }

        # 2. BM25 rank map (if dense candidates or all_chunks available)
        bm25_rank_map: Dict[str, int] = {}
        corpus_chunks = all_chunks if all_chunks else [doc for doc, _ in dense_results]
        if corpus_chunks and w.get("bm25", 1.0) > 0:
            try:
                tokenized_corpus = [self._tokenize(doc.get("text", "")) for doc in corpus_chunks]
                bm25 = BM25Okapi(tokenized_corpus)
                query_tokens = self._tokenize(query)
                bm25_scores = bm25.get_scores(query_tokens)
                bm25_ranked_indices = list(np.argsort(bm25_scores)[::-1]) if len(bm25_scores) > 0 else []
                bm25_rank_map = {
                    corpus_chunks[idx]["chunk_id"]: rank + 1
                    for rank, idx in enumerate(bm25_ranked_indices)
                    if corpus_chunks[idx].get("chunk_id")
                }
            except Exception as exc:
                logger.debug(f"[HybridReranker] BM25 calculation notice: {exc}")

        # 3. Graph rank map
        graph_rank_map: Dict[str, int] = {
            cid: rank + 1
            for rank, (cid, _) in enumerate(graph_results)
            if cid
        }

        # Collect all unique candidate chunk IDs to score
        candidate_chunk_ids = set(dense_rank_map.keys()) | set(graph_rank_map.keys())

        # Extract numeric tokens from query for boost
        query_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', query))

        rrf_scored_chunks: List[Tuple[Dict[str, Any], float]] = []

        for chunk_id in candidate_chunk_ids:
            doc = chunk_map.get(chunk_id)
            if not doc:
                # If chunk body is not in memory, create minimal chunk stub
                doc = {"chunk_id": chunk_id, "filename": "knowledge_graph", "page_number": 1, "text": f"[Graph Evidence Chunk: {chunk_id}]"}

            score = 0.0

            # Dense component
            if chunk_id in dense_rank_map and w.get("dense", 0.0) > 0:
                r_dense = dense_rank_map[chunk_id]
                score += w["dense"] * (1.0 / (self.rrf_k + r_dense))

            # BM25 component
            if chunk_id in bm25_rank_map and w.get("bm25", 0.0) > 0:
                r_sparse = bm25_rank_map[chunk_id]
                score += w["bm25"] * (1.0 / (self.rrf_k + r_sparse))

            # Graph component
            if chunk_id in graph_rank_map and w.get("graph", 0.0) > 0:
                r_graph = graph_rank_map[chunk_id]
                score += w["graph"] * (1.0 / (self.rrf_k + r_graph))

            # Numerical token bonus
            if query_numbers and "text" in doc:
                chunk_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', doc["text"]))
                matched_nums = query_numbers.intersection(chunk_numbers)
                if matched_nums:
                    score += 0.05 * len(matched_nums)

            rrf_scored_chunks.append((doc, round(score, 6)))

        # Sort descending by final fused score
        rrf_scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return rrf_scored_chunks[:top_k]
