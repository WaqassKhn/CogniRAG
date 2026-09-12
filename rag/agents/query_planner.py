"""
rag/agents/query_planner.py
───────────────────────────
QueryPlannerAgent — classifies query complexity and produces a retrieval plan.

Uses a small, fast LLM (llama-3.1-8b or equivalent) to decide:
  - "simple"  → single-pass retrieval is sufficient
  - "complex" → the query needs 2–4 targeted sub-queries

Also identifies which specific documents to scope retrieval to (doc_scope)
when the query clearly references a specific file or topic.

Output schema:
    {
        "complexity": "simple" | "complex",
        "sub_queries": ["..."],        # 1 item if simple, 2-4 if complex
        "doc_scope": ["filename.pdf"] | null
    }

Usage:
    planner = QueryPlannerAgent(llm=openrouter_llm, known_documents=["report.pdf"])
    plan = planner.plan("Compare Q1 and Q4 revenue figures across all reports")
    # plan = {"complexity": "complex", "sub_queries": ["Q1 revenue", "Q4 revenue"], "doc_scope": null}
"""

import json
import logging
import re
from typing import Optional, TYPE_CHECKING

from rag.graph.extractor import EntityNormalizer

if TYPE_CHECKING:
    from rag.openrouter_llm import OpenRouterLLM

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = """You are an agentic query planning assistant for an enterprise Document & Knowledge Graph RAG system.
Given a user query and a list of available indexed documents, output a retrieval plan as JSON.

Rules:
- "complexity":
  - "simple": single factual question answerable in one retrieval pass.
  - "complex": requires comparison, aggregation across time periods, or multiple distinct facts.
- "strategy":
  - "graph_only": direct structural, relational, hierarchy, or approval workflow questions.
    Examples: "Who does the VP report to?", "Who approves travel requests over $5,000?", "List departments and their heads"
  - "vector_only": pure narrative overviews, background summaries, or broad conceptual descriptions.
    Examples: "Summarize company history", "What is our corporate philosophy?", "Provide an overview of the introduction"
  - "hybrid": multi-hop analytical questions combining document text with relational entity governance.
    Examples: "Compare reimbursement policies between HR and Engineering in FY24", "Summarize Q3 operational highlights and approving authorities"
- "sub_queries":
  - For simple, return [original_query].
  - For complex, return 2-4 focused sub-queries targeting specific aspects.
- "doc_scope": ONLY set if the query explicitly names a document or clear topic matching ONE of the listed documents. Otherwise null.
- "target_entities": list of key named entities (roles, policies, departments, metrics) referenced in the query, or null.

Output ONLY valid JSON. No explanation. No markdown fences.
Schema:
{
  "complexity": "simple" | "complex",
  "strategy": "hybrid" | "vector_only" | "graph_only",
  "sub_queries": ["..."],
  "doc_scope": ["filename.pdf"] | null,
  "target_entities": ["..."] | null
}"""

_PLANNER_USER_TEMPLATE = """Available documents: {doc_list}

User query: {query}

JSON plan:"""


class QueryPlannerAgent:
    """
    Classifies query complexity, strategy routing (hybrid / vector_only / graph_only),
    and decomposes complex queries into targeted sub-queries.
    Falls back to deterministic heuristics if the LLM is unavailable or returns malformed JSON.
    """

    def __init__(
        self,
        llm: Optional["OpenRouterLLM"] = None,
        known_documents: Optional[list[str]] = None,
    ):
        self.llm = llm
        self.known_documents = known_documents or []

    def plan(self, query: str, procedural_hint: Optional[str] = None) -> dict:
        """
        Produces a retrieval plan for the given query.

        Args:
            query: User question to plan.
            procedural_hint: Optional procedural recipe guidelines to steer decomposition.

        Returns:
            {
                "complexity": "simple" | "complex",
                "strategy": "hybrid" | "vector_only" | "graph_only",
                "sub_queries": list[str],
                "doc_scope": list[str] | None,
                "target_entities": list[str] | None,
            }
        """
        fallback_strategy = self.classify_strategy_heuristic(query)
        fallback = {
            "complexity": "simple",
            "strategy": fallback_strategy,
            "sub_queries": [query],
            "doc_scope": None,
            "target_entities": self.extract_heuristic_entities(query),
        }

        if not self.llm or not self.llm.is_available():
            logger.debug("[QueryPlanner] LLM unavailable — using heuristic plan fallback.")
            return fallback

        doc_list = ", ".join(self.known_documents) if self.known_documents else "No documents listed"
        prompt = _PLANNER_USER_TEMPLATE.format(doc_list=doc_list, query=query)
        if procedural_hint:
            prompt += f"\n\nWorkflow Guidance:\n{procedural_hint}"

        try:
            raw = self.llm.generate(
                prompt=prompt,
                task="decompose",
                system_instruction=_PLANNER_SYSTEM_PROMPT,
                temperature=0.0,
            )
            plan = self._parse_plan(raw, query)
            logger.info(
                f"[QueryPlanner] complexity={plan['complexity']}, "
                f"strategy={plan['strategy']}, "
                f"sub_queries={len(plan['sub_queries'])}, "
                f"doc_scope={plan['doc_scope']}"
            )
            return plan

        except Exception as exc:
            logger.warning(f"[QueryPlanner] Failed to plan query: {exc}. Using heuristic fallback.")
            return fallback

    def _parse_plan(self, raw: str, original_query: str) -> dict:
        """
        Parses LLM output into a validated plan dict.
        Falls back to simple heuristic plan on any parse error.
        """
        # Extract JSON block (handles markdown fences, leading text, etc.)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in planner output.")

        data = json.loads(json_match.group(0))

        complexity = data.get("complexity", "simple")
        if complexity not in ("simple", "complex"):
            complexity = "simple"

        strategy = data.get("strategy")
        if strategy not in ("hybrid", "vector_only", "graph_only"):
            strategy = self.classify_strategy_heuristic(original_query)

        sub_queries = data.get("sub_queries", [original_query])
        if not isinstance(sub_queries, list) or not sub_queries:
            sub_queries = [original_query]
        # Enforce 1 item for simple, 2-4 for complex
        if complexity == "simple":
            sub_queries = sub_queries[:1] or [original_query]
        else:
            sub_queries = sub_queries[:4] or [original_query]

        doc_scope = data.get("doc_scope")
        if doc_scope is not None:
            if not isinstance(doc_scope, list):
                doc_scope = None
            else:
                # Filter to only known documents
                doc_scope = [d for d in doc_scope if d in self.known_documents] or None

        target_entities = data.get("target_entities")
        if target_entities is not None and not isinstance(target_entities, list):
            target_entities = None
        if not target_entities:
            target_entities = self.extract_heuristic_entities(original_query) or None

        return {
            "complexity": complexity,
            "strategy": strategy,
            "sub_queries": sub_queries,
            "doc_scope": doc_scope,
            "target_entities": target_entities,
        }

    @staticmethod
    def classify_strategy_heuristic(query: str) -> str:
        """
        Deterministic fast classification into 'graph_only', 'vector_only', or 'hybrid'.
        """
        q_lower = query.lower()

        # 1. Structural / Relational / Governance intent
        graph_patterns = [
            r'\breports?\s+to\b',
            r'\bwho\s+approves\b',
            r'\bapprov(?:al|ed\s+by|ing\s+authority)\b',
            r'\bescalat(?:e|ion)\s+to\b',
            r'\borganogram\b',
            r'\bhierarchy\b',
            r'\bwho\s+is\s+the\s+(?:head|lead|director|manager|officer|vp|ceo|cfo|md)\b',
            r'\bdepartments?\s+and\s+their\s+heads?\b',
            r'\bwhich\s+department\s+owns\b',
        ]
        if any(re.search(p, q_lower) for p in graph_patterns):
            return "graph_only"

        # 2. Pure narrative / high-level summary intent
        narrative_patterns = [
            r'\bsummarize\s+(?:the\s+)?(?:history|overview|introduction|background)\b',
            r'\b(?:what\s+is|explain|describe)\s+.*?\b(?:mission|vision|philosophy|history|background)\b',
            r'\b(?:corporate\s+)?(?:philosophy|mission\s+statement|vision\s+statement)\b',
            r'\bgive\s+(?:a\s+)?general\s+overview\b',
        ]
        if any(re.search(p, q_lower) for p in narrative_patterns):
            return "vector_only"

        # 3. Default to hybrid dual retrieval
        return "hybrid"

    @staticmethod
    def extract_heuristic_entities(query: str) -> list[str]:
        """Extracts candidate target entities from query for graph targeting."""
        entities = []
        for dept_k, canonical in EntityNormalizer.DEPARTMENT_MAP.items():
            if re.search(rf'\b{re.escape(dept_k)}\b', query, re.IGNORECASE):
                if canonical not in entities:
                    entities.append(canonical)
        for role_k, canonical in EntityNormalizer.ROLE_MAP.items():
            if re.search(rf'\b{re.escape(role_k)}\b', query, re.IGNORECASE):
                if canonical not in entities:
                    entities.append(canonical)
        for m in re.finditer(r'\b(?:Section|Clause|Policy|SOP)\s+[A-Za-z0-9\.]+\b', query, re.IGNORECASE):
            val = m.group(0).strip()
            if val not in entities:
                entities.append(val)
        return entities
