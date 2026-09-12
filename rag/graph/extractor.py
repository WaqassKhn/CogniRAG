"""
rag/graph/extractor.py - Zero-Cost Hybrid Entity & Relationship Extractor.

Purpose-built for Universal Enterprise Documents (Policies, SOPs, Management Reports,
Strategy Roadmaps, and Staff Approval Workflows). Combines fast deterministic rules
with OpenRouter free-tier LLM semantic extraction and alias normalization.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from config import GRAPHRAG_EXTRACTION_MODE

logger = logging.getLogger(__name__)


class EntityNormalizer:
    """
    Normalizes common enterprise entity names to eliminate duplicate/fragmented nodes.
    Maps acronyms (e.g. 'HR' -> 'Human Resources', 'VP' -> 'Vice President')
    and cleans formatting.
    """

    DEPARTMENT_MAP = {
        "hr": "Human Resources",
        "human resources": "Human Resources",
        "human resource": "Human Resources",
        "finance": "Finance Department",
        "fin": "Finance Department",
        "procurement": "Procurement Department",
        "it": "Information Technology",
        "information technology": "Information Technology",
        "legal": "Legal Department",
        "ops": "Operations",
        "operations": "Operations",
        "eng": "Engineering",
        "engineering": "Engineering",
    }

    ROLE_MAP = {
        "vp": "Vice President",
        "vice president": "Vice President",
        "ceo": "Chief Executive Officer",
        "cfo": "Chief Financial Officer",
        "cto": "Chief Technology Officer",
        "coo": "Chief Operating Officer",
        "md": "Managing Director",
        "hod": "Head of Department",
        "head of department": "Head of Department",
        "pm": "Project Manager",
        "project manager": "Project Manager",
        "tl": "Team Lead",
        "team lead": "Team Lead",
    }

    @classmethod
    def normalize(cls, name: str, entity_type: str = "ENTITY") -> str:
        """Cleans and unifies entity names while preserving currency prefixes."""
        clean = name.strip()
        # Remove surrounding punctuation except leading currency symbols ($ € £ ₹)
        clean = re.sub(r'^[^\w\s\$€£₹]+|[^\w\s]+$', '', clean)
        clean_lower = clean.lower()

        if entity_type == "DEPARTMENT" and clean_lower in cls.DEPARTMENT_MAP:
            return cls.DEPARTMENT_MAP[clean_lower]

        if entity_type == "ROLE" and clean_lower in cls.ROLE_MAP:
            return cls.ROLE_MAP[clean_lower]

        if entity_type == "ROLE" and clean_lower.startswith("vp "):
            rest = clean[3:].strip()
            return f"Vice President of {rest}" if not rest.lower().startswith("of ") else f"Vice President {rest}"

        # Standardize Fiscal Years (e.g. FY 2024, FY-24 -> FY24)
        fy_match = re.match(r'^(?:fy|fiscal\s+year)\s*[-_]?\s*(?:20)?(\d{2})$', clean_lower)
        if fy_match:
            return f"FY{fy_match.group(1)}"

        return clean


class GraphExtractor:
    """
    Hybrid Entity and Knowledge Graph Triplet Extractor for corporate documents.
    """

    # Common corporate role and department keywords
    ROLE_KEYWORDS = {
        "director", "manager", "officer", "executive", "head", "lead",
        "supervisor", "auditor", "chairperson", "approver", "approving authority",
        "vice president", "ceo", "cfo", "cto", "coo", "md", "vp"
    }

    DEPT_KEYWORDS = {
        "human resources", "hr", "finance", "accounting", "engineering",
        "procurement", "supply chain", "legal", "compliance", "it",
        "operations", "marketing", "sales", "audit", "security"
    }

    POLICY_KEYWORDS = {
        "policy", "guideline", "standard", "procedure", "sop", "code of conduct",
        "charter", "framework", "regulation", "rule", "protocol"
    }

    def __init__(
        self,
        llm: Optional[Any] = None,
        gemini_llm: Optional[Any] = None,
        mode: Optional[str] = None,
    ):
        self.llm = llm
        self.gemini_llm = gemini_llm
        self.mode = mode or GRAPHRAG_EXTRACTION_MODE  # hybrid_fast | local_rules | hybrid | gemini_batch | llm_free | llm_all

    def extract_subgraph_from_document(
        self,
        document_text: str,
        filename: str,
        chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        High-performance 1-call document batch extraction:
        - Runs Tier 1 rule extraction across all chunks in milliseconds (<20ms).
        - Uses Gemini 1.5 Flash in exactly 1 batch API call for the entire document text
          (using Gemini's 1,500 daily requests and 1M token context window).
        - If Gemini is unavailable, falls back to selective OpenRouter extraction or Tier 1 rules.
        - Deduplicates and normalizes all entities and relationships across the entire document.
        """
        all_entities: List[Dict[str, Any]] = []
        all_relations: List[Dict[str, Any]] = []
        seen_entity_names: Set[str] = set()
        seen_relation_keys: Set[str] = set()

        # 1. Fast Tier 1 deterministic extraction across all chunks
        if chunks:
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id", f"{filename}_c0")
                page_num = chunk.get("page_number", 1)
                rule_res = self._extract_via_rules(chunk.get("text", ""), chunk_id, page_num)
                for ent in rule_res.get("entities", []):
                    ent_key = ent["name"].lower()
                    if ent_key not in seen_entity_names:
                        seen_entity_names.add(ent_key)
                        all_entities.append(ent)
                for rel in rule_res.get("relations", []):
                    rel_key = f"{rel['source'].lower()}_{rel['type']}_{rel['target'].lower()}"
                    if rel_key not in seen_relation_keys:
                        seen_relation_keys.add(rel_key)
                        all_relations.append(rel)
        else:
            rule_res = self._extract_via_rules(document_text, f"{filename}_c0", 1)
            for ent in rule_res.get("entities", []):
                ent_key = ent["name"].lower()
                if ent_key not in seen_entity_names:
                    seen_entity_names.add(ent_key)
                    all_entities.append(ent)
            for rel in rule_res.get("relations", []):
                rel_key = f"{rel['source'].lower()}_{rel['type']}_{rel['target'].lower()}"
                if rel_key not in seen_relation_keys:
                    seen_relation_keys.add(rel_key)
                    all_relations.append(rel)

        # 2. Tier 2: 1-Call Document Batch Extraction via Gemini (or OpenRouter fallback)
        extractor_client = self.gemini_llm if (self.gemini_llm and getattr(self.gemini_llm, "is_available", lambda: True)()) else self.llm
        if extractor_client is not None and self.mode in ("hybrid_fast", "hybrid", "gemini_batch", "llm_free") and document_text:
            try:
                first_chunk_id = chunks[0]["chunk_id"] if chunks else f"{filename}_c0"
                llm_res = self._extract_document_batch_via_llm(
                    document_text=document_text,
                    filename=filename,
                    default_chunk_id=first_chunk_id,
                    client=extractor_client,
                )
                for ent in llm_res.get("entities", []):
                    ent_key = ent["name"].lower()
                    if ent_key not in seen_entity_names:
                        seen_entity_names.add(ent_key)
                        all_entities.append(ent)
                for rel in llm_res.get("relations", []):
                    rel_key = f"{rel['source'].lower()}_{rel['type']}_{rel['target'].lower()}"
                    if rel_key not in seen_relation_keys:
                        seen_relation_keys.add(rel_key)
                        all_relations.append(rel)
                logger.info(
                    f"[GraphExtractor] 1-Call Batch Extraction for '{filename}' complete: "
                    f"{len(all_entities)} entities, {len(all_relations)} relations."
                )
            except Exception as exc:
                logger.warning(f"[GraphExtractor] Document batch extraction fallback to rules: {exc}")

        return all_entities, all_relations

    def extract_subgraph_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        filename: str,
        page_number: int = 1,
    ) -> Dict[str, Any]:
        """
        Extracts entities and relationship triples from a single text chunk.
        Uses local rule extraction, enriched with LLM extraction if enabled.
        """
        if not chunk_text or len(chunk_text.strip()) < 15:
            return {"entities": [], "relations": []}

        # 1. Tier 1: Fast deterministic rule extraction (0ms)
        rule_result = self._extract_via_rules(chunk_text, chunk_id, page_number)

        # 2. If in full LLM mode or hybrid mode on a single chunk
        if self.mode in ("hybrid", "llm_all", "llm_free") and self.llm is not None:
            try:
                llm_result = self._extract_via_llm(chunk_text, chunk_id)
                merged = self._merge_results(rule_result, llm_result)
                return merged
            except Exception as e:
                logger.warning(f"[GraphExtractor] LLM extraction fallback to rules: {e}")
                return rule_result

        return rule_result

    def extract_subgraph_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        filename: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        High-performance document-level extraction combining all chunks:
        Delegates to extract_subgraph_from_document for 1-call batching.
        """
        full_doc_text = "\n\n".join([f"[Page {c.get('page_number', 1)}]\n{c['text']}" for c in chunks])
        return self.extract_subgraph_from_document(
            document_text=full_doc_text,
            filename=filename,
            chunks=chunks,
        )

    def _extract_via_rules(
        self,
        text: str,
        chunk_id: str,
        page_number: int,
    ) -> Dict[str, Any]:
        """
        High-speed deterministic extractor for enterprise entities and relational patterns.
        Zero API calls, zero latency overhead.
        """
        entities: List[Dict[str, Any]] = []
        relations: List[Dict[str, Any]] = []
        seen_entities: Set[str] = set()

        def add_entity(name: str, etype: str, desc: str = ""):
            norm_name = EntityNormalizer.normalize(name, etype)
            if norm_name and len(norm_name) >= 2 and norm_name.lower() not in seen_entities:
                seen_entities.add(norm_name.lower())
                entities.append({
                    "name": norm_name,
                    "type": etype,
                    "description": desc or f"Extracted {etype.lower()} from {chunk_id}",
                    "chunk_id": chunk_id,
                })
            return norm_name

        # A. Policy Clauses & Sections (e.g. "Section 4.2", "Clause 12", "Travel Policy")
        policy_matches = re.finditer(
            r'\b(?:Section|Clause|Article|Policy|SOP)\s+(?:[A-Z0-9]+(?:\.[0-9]+)*|\"[^\"]+\"|[A-Z][a-zA-Z]+)',
            text
        )
        found_policies = []
        for m in policy_matches:
            p_name = add_entity(m.group(0), "POLICY")
            found_policies.append(p_name)

        # B. Roles & Approving Authorities
        for r_keyword in self.ROLE_KEYWORDS:
            pattern = rf'\b(?:[A-Z][a-zA-Z]+\s+)?{re.escape(r_keyword)}(?:\s+[A-Z][a-zA-Z]+)?\b'
            for m in re.finditer(pattern, text, re.IGNORECASE):
                # Only add if capitalized or explicit role
                val = m.group(0).strip()
                if any(c.isupper() for c in val):
                    add_entity(val.title(), "ROLE")

        # C. Departments & Organizations
        for d_keyword in self.DEPT_KEYWORDS:
            pattern = rf'\b(?:Department of\s+)?[A-Z][a-zA-Z\s]*{re.escape(d_keyword)}(?:\s+Department)?\b'
            for m in re.finditer(pattern, text, re.IGNORECASE):
                val = m.group(0).strip()
                add_entity(val.title(), "DEPARTMENT")

        # D. Financial & Numerical Metrics (e.g. "$5,000", "INR 50 Lakhs", "99.9% SLA", "EBITDA")
        metric_matches = re.finditer(
            r'(?:(?:INR|USD|EUR|GBP|\$|₹)\s*[\d,]+(?:\.\d+)?(?:\s*(?:Crore|Lakh|Million|Billion|k|M|B))?|\b\d+(?:\.\d+)?%\s*(?:SLA|Uptime|Margin|Growth)?)',
            text,
            re.IGNORECASE
        )
        found_metrics = []
        for m in metric_matches:
            val = m.group(0).strip()
            m_name = add_entity(val, "METRIC", desc=f"Value mentioned on page {page_number}")
            found_metrics.append(m_name)

        # E. Fiscal Periods & Dates (e.g. FY24, FY2025, Q3 2026)
        fiscal_matches = re.finditer(r'\b(?:FY\s*[-_]?\s*(?:20)?\d{2}|Q[1-4]\s*(?:20)?\d{2})\b', text, re.IGNORECASE)
        for m in fiscal_matches:
            add_entity(m.group(0), "FISCAL_PERIOD")

        # F. Heuristic Relationships
        # 1. Approval Workflows: "approved by <Role>" / "requires approval from <Role>"
        approval_matches = re.finditer(
            r'(?:approved\s+by|requires\s+approval\s+from|escalated\s+to)\s+([A-Z][a-zA-Z\s]{2,30})',
            text,
            re.IGNORECASE
        )
        for m in approval_matches:
            approver = add_entity(m.group(1).strip().title(), "ROLE")
            for pol in found_policies:
                relations.append({
                    "source": pol,
                    "target": approver,
                    "type": "REQUIRES_APPROVAL_FROM",
                    "description": f"Approval rule defined in {chunk_id}",
                    "confidence": 0.85,
                    "chunk_id": chunk_id,
                })

        # 2. Limit definitions: "<Policy> defines <Metric>"
        if found_policies and found_metrics:
            for pol in found_policies[:2]:
                for met in found_metrics[:2]:
                    relations.append({
                        "source": pol,
                        "target": met,
                        "type": "DEFINES_LIMIT",
                        "description": f"Metric specified under {pol}",
                        "confidence": 0.75,
                        "chunk_id": chunk_id,
                    })

        return {"entities": entities, "relations": relations}

    def _extract_document_batch_via_llm(
        self,
        document_text: str,
        filename: str,
        default_chunk_id: str,
        client: Any,
    ) -> Dict[str, Any]:
        """
        Deep semantic whole-document extraction in a single batched LLM call.
        Utilizes Gemini Flash's 1M-token context window to extract all global
        entities and relational triples across the document at once.
        """
        # Truncate to reasonable upper bound (e.g. 35,000 chars ~ 8,500 tokens)
        truncated_doc = document_text[:35000]

        system_instruction = """You are a Knowledge Graph Information Extraction Engine specialized in corporate and enterprise documents (internal policies, governance roadmaps, operational reviews, approval workflows, financial filings).

Analyze the entire document text and extract all significant entities and relationship triples.

Extraction Guidelines:
1. Entity Types:
   - ORGANIZATION: Companies, subsidiaries, external partners
   - DEPARTMENT: Functional units (Human Resources, Finance, IT, Engineering, Legal, Operations, Procurement)
   - POLICY: Policies, SOPs, Standards, Codes, Charters, Specific Sections/Clauses (e.g., Section 4.2 Reimbursement Policy)
   - ROLE: Corporate designations & approval authorities (VP, CEO, CFO, Approving Manager, Department Head)
   - PROJECT: Key initiatives, roadmaps, systems, ERP/Cloud projects
   - METRIC: Financial amounts, currency limits ($5,000, INR 50L), SLA percentages, KPIs (EBITDA, Capacity)
   - FISCAL_PERIOD: Quarters, fiscal years (FY24, Q3 FY25), deadlines, effective dates

2. Relationship Types:
   - Governance & Hierarchy: APPLIES_TO, REPORTS_TO, PART_OF_DEPT, OWNS_SUBSIDIARY, GOVERNED_BY
   - Workflow & Approvals: REQUIRES_APPROVAL_FROM, RESPONSIBLE_FOR, ESCALATES_TO, DEFINES_LIMIT
   - Operational & Strategic: TRACKS_KPI, DELIVERS_PROJECT, PARTNERS_WITH, SUPERSEDES

Output ONLY valid JSON matching this schema with no markdown commentary:
{
  "entities": [
    {"name": "Entity Name", "type": "ORGANIZATION|DEPARTMENT|POLICY|ROLE|PROJECT|METRIC|FISCAL_PERIOD", "description": "brief context from document"}
  ],
  "relations": [
    {"source": "Source Entity Name", "target": "Target Entity Name", "type": "APPLIES_TO|REQUIRES_APPROVAL_FROM|REPORTS_TO|DEFINES_LIMIT|DELIVERS_PROJECT|TRACKS_KPI", "description": "details of relationship"}
  ]
}"""

        prompt = f"""DOCUMENT FILENAME: {filename}

DOCUMENT FULL TEXT:
\"\"\"{truncated_doc}\"\"\"

JSON KNOWLEDGE GRAPH TRIPLES:"""

        try:
            # Check if client accepts 'task' (OpenRouter) or standard generate (Gemini)
            if hasattr(client, "generate"):
                import inspect
                sig = inspect.signature(client.generate)
                if "task" in sig.parameters:
                    raw_response = client.generate(
                        prompt=prompt,
                        task="triage",
                        system_instruction=system_instruction,
                        temperature=0.0,
                    )
                else:
                    raw_response = client.generate(
                        prompt=prompt,
                        system_instruction=system_instruction,
                        temperature=0.0,
                    )
            else:
                return {"entities": [], "relations": []}

            # Parse JSON from response
            cleaned_json = re.sub(r'^```(?:json)?\s*', '', raw_response.strip(), flags=re.MULTILINE)
            cleaned_json = re.sub(r'\s*```$', '', cleaned_json.strip(), flags=re.MULTILINE)
            
            # Extract JSON object
            json_match = re.search(r'\{.*\}', cleaned_json, re.DOTALL)
            if not json_match:
                return {"entities": [], "relations": []}

            data = json.loads(json_match.group(0))
            extracted_entities = data.get("entities", [])
            extracted_relations = data.get("relations", [])

            entities = []
            for ent in extracted_entities:
                if isinstance(ent, dict) and "name" in ent:
                    norm = EntityNormalizer.normalize(ent["name"], ent.get("type", "ENTITY"))
                    entities.append({
                        "name": norm,
                        "type": ent.get("type", "ENTITY").upper(),
                        "description": ent.get("description", f"Extracted from {filename}"),
                        "chunk_id": default_chunk_id,
                    })

            relations = []
            for rel in extracted_relations:
                if isinstance(rel, dict) and "source" in rel and "target" in rel:
                    relations.append({
                        "source": EntityNormalizer.normalize(rel["source"]),
                        "target": EntityNormalizer.normalize(rel["target"]),
                        "type": rel.get("type", "RELATES_TO").upper().replace(" ", "_"),
                        "description": rel.get("description", ""),
                        "confidence": 0.95,
                        "chunk_id": default_chunk_id,
                    })

            return {"entities": entities, "relations": relations}
        except Exception as e:
            logger.warning(f"[GraphExtractor] LLM batch parsing warning: {e}")
            return {"entities": [], "relations": []}

    def _extract_via_llm(
        self,
        text: str,
        chunk_id: str,
    ) -> Dict[str, Any]:
        """
        Deep semantic extraction using OpenRouter free-tier LLM.
        Extracts structured entities and relationship triples.
        """
        if not self.llm:
            return {"entities": [], "relations": []}

        prompt = f"""You are a Knowledge Graph Information Extraction Engine for enterprise corporate documents (policies, reports, strategy roadmaps, approval workflows).

Analyze the text chunk and extract key entities and relationship triples.

Return ONLY a valid JSON object matching this schema, with no markdown formatting or commentary:
{{
  "entities": [
    {{"name": "Entity Name", "type": "ORGANIZATION|DEPARTMENT|POLICY|ROLE|PROJECT|METRIC|FISCAL_PERIOD", "description": "brief context"}}
  ],
  "relations": [
    {{"source": "Entity Name", "target": "Target Entity Name", "type": "APPLIES_TO|REQUIRES_APPROVAL_FROM|REPORTS_TO|DEFINES_LIMIT|DELIVERS_PROJECT|TRACKS_KPI|OWNS_SUBSIDIARY", "description": "relationship detail"}}
  ]
}}

TEXT CHUNK:
\"\"\"{text[:1200]}\"\"\"
"""
        try:
            raw_response = self.llm.generate(prompt, task="triage")
            # Parse JSON from response
            cleaned_json = re.sub(r'^```(?:json)?\s*', '', raw_response.strip(), flags=re.MULTILINE)
            cleaned_json = re.sub(r'\s*```$', '', cleaned_json.strip(), flags=re.MULTILINE)
            
            data = json.loads(cleaned_json)
            extracted_entities = data.get("entities", [])
            extracted_relations = data.get("relations", [])

            entities = []
            for ent in extracted_entities:
                if isinstance(ent, dict) and "name" in ent:
                    norm = EntityNormalizer.normalize(ent["name"], ent.get("type", "ENTITY"))
                    entities.append({
                        "name": norm,
                        "type": ent.get("type", "ENTITY").upper(),
                        "description": ent.get("description", ""),
                        "chunk_id": chunk_id,
                    })

            relations = []
            for rel in extracted_relations:
                if isinstance(rel, dict) and "source" in rel and "target" in rel:
                    relations.append({
                        "source": EntityNormalizer.normalize(rel["source"]),
                        "target": EntityNormalizer.normalize(rel["target"]),
                        "type": rel.get("type", "RELATES_TO").upper().replace(" ", "_"),
                        "description": rel.get("description", ""),
                        "confidence": 0.90,
                        "chunk_id": chunk_id,
                    })

            return {"entities": entities, "relations": relations}
        except Exception as e:
            logger.debug(f"[GraphExtractor] LLM JSON parsing error: {e}")
            return {"entities": [], "relations": []}

    def _merge_results(
        self,
        rules: Dict[str, Any],
        llm: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merges rule-based and LLM-based extractions with deduplication."""
        entities_by_name = {e["name"].lower(): e for e in rules.get("entities", [])}
        for ent in llm.get("entities", []):
            k = ent["name"].lower()
            if k not in entities_by_name:
                entities_by_name[k] = ent
            elif ent.get("description"):
                entities_by_name[k]["description"] = ent["description"]

        relations_by_key = {
            f"{r['source'].lower()}_{r['type']}_{r['target'].lower()}": r
            for r in rules.get("relations", [])
        }
        for rel in llm.get("relations", []):
            k = f"{rel['source'].lower()}_{rel['type']}_{rel['target'].lower()}"
            if k not in relations_by_key:
                relations_by_key[k] = rel

        return {
            "entities": list(entities_by_name.values()),
            "relations": list(relations_by_key.values()),
        }
