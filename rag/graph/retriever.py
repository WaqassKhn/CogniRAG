"""
rag/graph/retriever.py - Knowledge Graph Retriever & Multi-Hop Traversal Engine.

Extracts query entities, traverses multi-hop neighborhoods in Neo4j,
and formats connected subgraphs into dense relational facts for LLM prompting.
Guarantees 100% free operation, non-blocking execution, and seamless fallback
when Neo4j is offline.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from config import GRAPHRAG_MAX_HOPS
from rag.graph.extractor import EntityNormalizer, GraphExtractor
from rag.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphRetriever:
    """
    Retriever for Knowledge Graph entity-relation subgraphs.
    Performs entity resolution on queries, traverses Neo4j subgraphs,
    and returns formatted relational facts with chunk provenance.
    """

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        max_hops: int = GRAPHRAG_MAX_HOPS,
        max_facts: int = 20,
    ):
        self.neo4j_client = neo4j_client
        self.max_hops = max_hops
        self.max_facts = max_facts

    def is_available(self) -> bool:
        """Returns True if the underlying Neo4j client is active and reachable."""
        return self.neo4j_client is not None and self.neo4j_client.is_available()

    def extract_query_entities(self, query: str) -> List[str]:
        """
        Extracts candidate entity names from a user question.
        Uses fast deterministic heuristics:
        1. Explicit enterprise departments & roles
        2. Policy sections, clauses, and SOP references
        3. Capitalized multi-word noun phrases
        4. Currency/metric figures
        """
        if not query or not query.strip():
            return []

        resolved_entities: List[str] = []
        seen: Set[str] = set()

        def add_entity(raw: str, etype: str = "ENTITY"):
            norm = EntityNormalizer.normalize(raw, etype)
            norm_key = norm.lower().strip()
            if norm and len(norm_key) >= 2 and norm_key not in seen:
                # Filter out generic stop words
                if norm_key not in {"what", "where", "which", "when", "who", "how", "does", "explain", "summarize", "compare", "list", "show", "tell"}:
                    seen.add(norm_key)
                    resolved_entities.append(norm)

        # 1. Department keywords
        for dept_key, canonical in EntityNormalizer.DEPARTMENT_MAP.items():
            pattern = rf'\b{re.escape(dept_key)}\b'
            if re.search(pattern, query, re.IGNORECASE):
                add_entity(canonical, "DEPARTMENT")

        # 2. Role keywords
        for role_key, canonical in EntityNormalizer.ROLE_MAP.items():
            pattern = rf'\b{re.escape(role_key)}\b'
            if re.search(pattern, query, re.IGNORECASE):
                add_entity(canonical, "ROLE")

        # 3. Policy & Clause patterns (e.g. "Section 4.2", "Clause 5", "Travel Policy")
        for m in re.finditer(
            r'\b(?:Section|Clause|Article|Policy|SOP)\s+(?:[A-Z0-9]+(?:\.[0-9]+)*|\"[^\"]+\"|[A-Z][a-zA-Z]+)',
            query,
            re.IGNORECASE,
        ):
            add_entity(m.group(0), "POLICY")

        # 4. Capitalized entity names (e.g. "Acme Corp", "Project Titan")
        cap_matches = re.finditer(r'\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b', query)
        for m in cap_matches:
            val = m.group(0).strip()
            add_entity(val)

        # 5. Financial metrics and amounts (e.g. "$5,000", "FY24")
        metric_matches = re.finditer(
            r'(?:(?:INR|USD|EUR|GBP|\$|₹)\s*[\d,]+(?:\.\d+)?(?:\s*(?:Crore|Lakh|Million|Billion|k|M|B))?|\bFY\s*[-_]?\s*(?:20)?\d{2}\b)',
            query,
            re.IGNORECASE,
        )
        for m in metric_matches:
            add_entity(m.group(0).strip())

        return resolved_entities

    def retrieve_subgraph(
        self,
        query: str,
        max_hops: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Resolves query entities and queries Neo4j for connected multi-hop paths.
        """
        if not self.is_available():
            logger.debug("[GraphRetriever] Neo4j is offline or unavailable. Skipping graph retrieval.")
            return []

        entity_names = self.extract_query_entities(query)
        if not entity_names:
            logger.debug(f"[GraphRetriever] No candidate entities identified in query: '{query}'")
            return []

        hops = max_hops or self.max_hops
        lim = limit or self.max_facts

        try:
            records = self.neo4j_client.get_entity_neighborhood(
                entity_names=entity_names,
                max_hops=hops,
                limit=lim,
            )
            logger.info(
                f"[GraphRetriever] Retrieved {len(records)} graph relationships for entities: {entity_names}"
            )
            return records
        except Exception as exc:
            logger.warning(f"[GraphRetriever] Error querying entity neighborhood: {exc}")
            return []

    def format_relational_facts(
        self,
        paths: List[Dict[str, Any]],
    ) -> Tuple[str, List[str]]:
        """
        Converts raw Neo4j path records into high-density relational facts for prompt injection.
        Also returns unique chunk citations associated with the relationships.

        Output format:
            • (Source [Type]) -[RELATION]-> (Target [Type]) [Description]
        """
        if not paths:
            return "", []

        fact_lines: List[str] = []
        evidence_chunks: List[str] = []
        seen_facts: Set[str] = set()

        for p in paths:
            src = p.get("source", "Unknown")
            src_type = p.get("source_type") or "Entity"
            rel = p.get("relation_type", "RELATES_TO")
            tgt = p.get("target", "Unknown")
            tgt_type = p.get("target_type") or "Entity"
            desc = p.get("description", "").strip()
            chunk_id = p.get("chunk_id")

            # Deduplicate facts
            fact_key = f"{src}_{rel}_{tgt}".lower()
            if fact_key in seen_facts:
                continue
            seen_facts.add(fact_key)

            desc_clause = f" [Note: {desc}]" if desc and desc.lower() != "relates to" else ""
            fact_line = f"• ({src} [{src_type}]) -[{rel}]-> ({tgt} [{tgt_type}]){desc_clause}"
            fact_lines.append(fact_line)

            if chunk_id and chunk_id not in evidence_chunks:
                evidence_chunks.append(chunk_id)

        formatted_block = "\n".join(fact_lines)
        return formatted_block, evidence_chunks

    def retrieve_facts(
        self,
        query: str,
        max_hops: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Tuple[str, List[str]]:
        """
        Convenience method: resolves entities, executes multi-hop traversal,
        and returns formatted knowledge graph facts + evidence chunk citations.
        """
        paths = self.retrieve_subgraph(query, max_hops=max_hops, limit=limit)
        if not paths:
            return "", []
        return self.format_relational_facts(paths)
