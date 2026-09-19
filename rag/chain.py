"""
rag/chain.py
────────────
End-to-end RAG chain: embed → retrieve → rerank → generate.

Supports 4 distinct retrieval modes:
  1. "dense"        — Pure dense vector similarity search
  2. "dense_bm25"   — Dense vector + BM25 sparse RRF fusion (vector_only)
  3. "graph_only"   — Ranked Knowledge Graph facts & evidence chunks only
  4. "hybrid"       — Unified 3-way RRF fusion combining Dense, BM25, and Graph

Supports both Pinecone (production) and FAISS VectorDatabase (eval harness)
via duck-typing — both expose .search() and .chunks_metadata.

Two execution interfaces:
  run()        — returns a complete dict (used by MergeAgent, eval harness, cache)
  run_stream() — yields (type, data) tuples for token-by-token streaming in UI
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from config import INITIAL_TOP_K, RERANKED_TOP_K
from vectorstore.embeddings import EmbeddingManager
from rag.reranker import HybridReranker

logger = logging.getLogger(__name__)

# Accept either VectorDatabase (eval) or PineconeDB (production) via duck-typing
try:
    from vectorstore.pinecone_db import PineconeDB
except ImportError:
    PineconeDB = None

try:
    from vectorstore.vector_db import VectorDatabase
except ImportError:
    VectorDatabase = None

try:
    from rag.tracer import ExecutionTracer
except ImportError:
    ExecutionTracer = None


class RAGChain:
    """
    End-to-End Hybrid RAG Chain implementing:
    1. Query embedding
    2. Vector similarity retrieval (Pinecone or FAISS)
    3. Sparse/BM25 Hybrid Reranking via RRF
    4. Knowledge Graph Multi-Hop Subgraph Traversal & Fact Scoring (Neo4j)
    5. Unified Grounded Prompt Formulation (Vectors + Triples)
    6. LLM Answer Generation (OpenRouter or Gemini)
    """

    SYSTEM_PROMPT = """You are a strict, grounded AI assistant specialized in analyzing enterprise corporate documents, policies, operational reports, and multi-hop relationships.

Your core instruction: Answer the user's question using ONLY the provided document context chunks and knowledge graph relations below.

Rules:
1. Accuracy & Grounding: Every statement and numerical figure in your response MUST be directly supported by the context chunks or knowledge graph relations.
2. Numerical Precision: Quote exact figures, percentages, dates, and currency values as they appear in the source context. Do NOT round, estimate, or extrapolate figures unless explicitly requested.
3. Citations: Cite your sources inline using the format [Source: <Filename>, Page <PageNumber>] or [Knowledge Graph] for every key claim or figure.
4. Missing Information: If the provided context does not contain enough information to answer the question, explicitly state: "The provided document context does not contain sufficient information to answer this question." Do NOT use outside knowledge.
"""

    def __init__(
        self,
        vector_db,                          # PineconeDB or VectorDatabase (duck-typed)
        embedding_manager: EmbeddingManager,
        llm,                                # OpenRouterLLM or GeminiLLM (duck-typed)
        reranker: Optional[HybridReranker] = None,
        graph_retriever: Optional[Any] = None,
    ):
        self.vector_db = vector_db
        self.embedding_manager = embedding_manager
        self.llm = llm
        self.reranker = reranker or HybridReranker()
        self.graph_retriever = graph_retriever

    # ─── Retrieval & Ranking Subsystem ────────────────────────────────────────

    def retrieve_context(
        self,
        query: str,
        query_vec,
        initial_top_k: int = INITIAL_TOP_K,
        rerank_top_k: int = RERANKED_TOP_K,
        filter_filenames: Optional[List[str]] = None,
        mode: str = "hybrid",
        tracer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Executes mode-specific retrieval and ranking:
          - "dense": Pure dense vector search sorted by cosine score.
          - "dense_bm25" (or "vector_only"): Dense + BM25 sparse RRF fusion.
          - "graph_only": Scored and ranked Knowledge Graph facts and evidence chunks.
          - "hybrid": Unified 3-way RRF fusion combining Dense, BM25, and Graph.
        """
        norm_mode = mode.lower().strip() if mode else "hybrid"
        if norm_mode in ("vector_only", "dense_only_bm25"):
            norm_mode = "dense_bm25"

        all_chunks = getattr(self.vector_db, "chunks_metadata", None) or []

        dense_results: List[Tuple[Dict[str, Any], float]] = []
        ranked_facts: List[Dict[str, Any]] = []
        graph_facts_str: str = ""
        graph_chunk_tuples: List[Tuple[str, float]] = []

        # Step A: Vector Retrieval (skipped in graph_only mode)
        if norm_mode != "graph_only":
            retrieval_span = (
                tracer.start_span("vector_retrieval", component="retrieval", inputs={"top_k": initial_top_k, "filters": filter_filenames, "mode": norm_mode})
                if tracer else None
            )
            dense_results = self._retrieve(query_vec, initial_top_k, filter_filenames)
            if tracer and retrieval_span:
                tracer.finish_span(retrieval_span, outputs={"candidate_chunks_found": len(dense_results)})

        # Step B: Graph Retrieval & Fact Scoring (skipped in dense and dense_bm25 modes)
        if norm_mode in ("graph_only", "hybrid"):
            if self.graph_retriever and getattr(self.graph_retriever, "is_available", lambda: False)():
                graph_span = (
                    tracer.start_span("graph_retrieval", component="graph", inputs={"query": query, "mode": norm_mode})
                    if tracer else None
                )
                try:
                    # Attempt retrieve_ranked_facts first
                    if hasattr(self.graph_retriever, "retrieve_ranked_facts"):
                        rf = self.graph_retriever.retrieve_ranked_facts(
                            query=query,
                            query_vec=query_vec,
                            embedding_manager=self.embedding_manager,
                        )
                        if isinstance(rf, list):
                            ranked_facts = rf
                            fmt_res = self.graph_retriever.format_relational_facts(ranked_facts)
                            if isinstance(fmt_res, tuple) and len(fmt_res) == 2:
                                graph_facts_str, _ = fmt_res
                            elif isinstance(fmt_res, str):
                                graph_facts_str = fmt_res

                    # Fall back to retrieve_facts if graph_facts_str is still empty
                    if not graph_facts_str and hasattr(self.graph_retriever, "retrieve_facts"):
                        rf_res = self.graph_retriever.retrieve_facts(query)
                        if isinstance(rf_res, tuple) and len(rf_res) == 2:
                            graph_facts_str, evidence_ids = rf_res
                            if not graph_chunk_tuples and evidence_ids:
                                graph_chunk_tuples = [(cid, 1.0) for cid in evidence_ids]
                        elif isinstance(rf_res, str):
                            graph_facts_str = rf_res

                    if hasattr(self.graph_retriever, "get_ranked_evidence_chunks"):
                        rec = self.graph_retriever.get_ranked_evidence_chunks(
                            query=query,
                            query_vec=query_vec,
                            embedding_manager=self.embedding_manager,
                        )
                        if isinstance(rec, list):
                            graph_chunk_tuples = rec

                    if tracer and graph_span:
                        tracer.finish_span(graph_span, outputs={"facts_count": len(ranked_facts), "graph_evidence_chunks": len(graph_chunk_tuples)})
                except Exception as exc:
                    if tracer and graph_span:
                        tracer.finish_span(graph_span, outputs={"error": str(exc)})
                    logger.warning(f"[RAGChain] Graph retrieval notice: {exc}")

        # Step C: Ranking execution according to mode
        top_chunks: List[Dict[str, Any]] = []
        ranked_items: List[Dict[str, Any]] = []

        if norm_mode == "dense":
            # Pure dense ranking by cosine score
            sorted_dense = sorted(dense_results, key=lambda x: x[1], reverse=True)[:rerank_top_k]
            top_chunks = [c for c, s in sorted_dense]
            ranked_items = [
                {"chunk_id": c.get("chunk_id"), "score": round(float(s), 5), "rank": i + 1, "type": "dense"}
                for i, (c, s) in enumerate(sorted_dense)
            ]

        elif norm_mode == "dense_bm25":
            # 2-way RRF reranking (Dense + BM25)
            rerank_span = (
                tracer.start_span("hybrid_reranking", component="reranker", inputs={"initial_count": len(dense_results), "rerank_top_k": rerank_top_k})
                if tracer else None
            )
            reranked_results = self.reranker.rerank(
                query=query,
                dense_results=dense_results,
                top_k=rerank_top_k,
                all_chunks=all_chunks,
                weights={"dense": 1.0, "bm25": 1.0, "graph": 0.0},
            )
            top_chunks = [chunk for chunk, score in reranked_results]
            ranked_items = [
                {"chunk_id": c.get("chunk_id"), "score": round(float(s), 5), "rank": i + 1, "type": "dense_bm25"}
                for i, (c, s) in enumerate(reranked_results)
            ]
            if tracer and rerank_span:
                tracer.finish_span(rerank_span, outputs={"top_chunks_selected": len(top_chunks)})

        elif norm_mode == "graph_only":
            # Graph facts & evidence chunks only
            reranked_results = self.reranker.rerank(
                query=query,
                dense_results=[],
                top_k=rerank_top_k,
                all_chunks=all_chunks,
                graph_results=graph_chunk_tuples,
                weights={"dense": 0.0, "bm25": 0.0, "graph": 1.0},
            )
            top_chunks = [chunk for chunk, score in reranked_results]
            ranked_items = [
                {"chunk_id": c.get("chunk_id"), "score": round(float(s), 5), "rank": i + 1, "type": "graph_only"}
                for i, (c, s) in enumerate(reranked_results)
            ]

        else:  # "hybrid"
            # Unified 3-way RRF fusion
            rerank_span = (
                tracer.start_span("unified_fusion", component="reranker", inputs={"dense_count": len(dense_results), "graph_count": len(graph_chunk_tuples), "rerank_top_k": rerank_top_k})
                if tracer else None
            )
            reranked_results = self.reranker.rerank(
                query=query,
                dense_results=dense_results,
                top_k=rerank_top_k,
                all_chunks=all_chunks,
                graph_results=graph_chunk_tuples,
                weights={"dense": 1.0, "bm25": 1.0, "graph": 1.0},
            )
            top_chunks = [chunk for chunk, score in reranked_results]
            ranked_items = [
                {"chunk_id": c.get("chunk_id"), "score": round(float(s), 5), "rank": i + 1, "type": "hybrid"}
                for i, (c, s) in enumerate(reranked_results)
            ]
            if tracer and rerank_span:
                tracer.finish_span(rerank_span, outputs={"top_chunks_selected": len(top_chunks)})

        return {
            "mode": norm_mode,
            "top_chunks": top_chunks,
            "ranked_items": ranked_items,
            "dense_results": dense_results,
            "ranked_graph_facts": ranked_facts,
            "graph_context": graph_facts_str,
            "graph_evidence_chunks": [cid for cid, _ in graph_chunk_tuples],
        }

    # ─── Full (non-streaming) run ─────────────────────────────────────────────

    def run(
        self,
        query: str,
        initial_top_k: int = INITIAL_TOP_K,
        rerank_top_k: int = RERANKED_TOP_K,
        filter_filenames: Optional[List[str]] = None,
        memory_context: Optional[str] = None,
        tracer: Optional[Any] = None,
        mode: str = "hybrid",
    ) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline for a user query under the specified mode.
        """
        if not query or not query.strip():
            return self._empty_response(query, "Please provide a valid question.")

        # Step 1: Embed query
        embed_span = tracer.start_span("embed_query", component="embedding", inputs={"query": query}) if tracer else None
        query_vec = self.embedding_manager.embed_query(query)
        if tracer and embed_span:
            tracer.finish_span(embed_span, outputs={"vector_dim": len(query_vec)})

        # Step 2 & 3: Mode-specific retrieval and ranking
        retrieval_data = self.retrieve_context(
            query=query,
            query_vec=query_vec,
            initial_top_k=initial_top_k,
            rerank_top_k=rerank_top_k,
            filter_filenames=filter_filenames,
            mode=mode,
            tracer=tracer,
        )

        top_chunks = retrieval_data["top_chunks"]
        graph_facts = retrieval_data["graph_context"]
        dense_results = retrieval_data["dense_results"]
        active_mode = retrieval_data["mode"]

        if not top_chunks and not graph_facts:
            return self._empty_response(
                query,
                "No matching document context or knowledge graph relationships found for this question.",
            )

        # Step 4: Build grounded prompt
        user_prompt, formatted_context = self._build_prompt(
            query=query,
            top_chunks=top_chunks,
            memory_context=memory_context,
            graph_context=graph_facts,
            mode=active_mode,
        )

        # Step 5: Generate answer
        llm_span = tracer.start_span("llm_generation", component="llm", inputs={"task": "answer", "chunks_count": len(top_chunks), "mode": active_mode}) if tracer else None
        answer = self._generate(user_prompt)
        if tracer and llm_span:
            tracer.finish_span(llm_span, outputs={"answer_length": len(answer)}, metadata={"model": getattr(self.llm, "last_model_used", "openrouter")})

        citations = self._extract_citations(top_chunks)
        if graph_facts and "[Knowledge Graph]" not in citations:
            citations.append("[Knowledge Graph]")

        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": [c for c, s in dense_results],
            "reranked_chunks": top_chunks,
            "ranked_items": retrieval_data["ranked_items"],
            "ranked_graph_facts": retrieval_data["ranked_graph_facts"],
            "citations": citations,
            "formatted_context": formatted_context,
            "graph_context": graph_facts,
            "mode_used": active_mode,
        }

    # ─── Streaming run ────────────────────────────────────────────────────────

    def run_stream(
        self,
        query: str,
        initial_top_k: int = INITIAL_TOP_K,
        rerank_top_k: int = RERANKED_TOP_K,
        filter_filenames: Optional[List[str]] = None,
        memory_context: Optional[str] = None,
        tracer: Optional[Any] = None,
        mode: str = "hybrid",
    ) -> Generator[Tuple[str, Any], None, None]:
        """
        Streaming version of run(). Yields (event_type, data) tuples:
            ("context", reranked_chunks)
            ("token",   text_chunk)
            ("done",    metadata_dict)
        """
        if not query or not query.strip():
            yield ("token", "Please provide a valid question.")
            yield ("done", {"query": query, "citations": [], "reranked_chunks": [], "mode_used": mode})
            return

        # Step 1: Embed query
        embed_span = tracer.start_span("embed_query", component="embedding", inputs={"query": query}) if tracer else None
        query_vec = self.embedding_manager.embed_query(query)
        if tracer and embed_span:
            tracer.finish_span(embed_span, outputs={"vector_dim": len(query_vec)})

        # Step 2 & 3: Mode-specific retrieval and ranking
        retrieval_data = self.retrieve_context(
            query=query,
            query_vec=query_vec,
            initial_top_k=initial_top_k,
            rerank_top_k=rerank_top_k,
            filter_filenames=filter_filenames,
            mode=mode,
            tracer=tracer,
        )

        top_chunks = retrieval_data["top_chunks"]
        graph_facts = retrieval_data["graph_context"]
        dense_results = retrieval_data["dense_results"]
        active_mode = retrieval_data["mode"]

        if not top_chunks and not graph_facts:
            yield ("token", "No matching document context or knowledge graph relationships found for this question.")
            yield ("done", {"query": query, "citations": [], "reranked_chunks": [], "mode_used": active_mode})
            return

        # Emit context immediately so UI can show retrieved chunks while LLM generates
        yield ("context", top_chunks)

        user_prompt, formatted_context = self._build_prompt(
            query=query,
            top_chunks=top_chunks,
            memory_context=memory_context,
            graph_context=graph_facts,
            mode=active_mode,
        )
        citations = self._extract_citations(top_chunks)
        if graph_facts and "[Knowledge Graph]" not in citations:
            citations.append("[Knowledge Graph]")

        # Stream tokens from LLM
        llm_span = tracer.start_span("llm_stream", component="llm", inputs={"task": "answer", "chunks_count": len(top_chunks), "mode": active_mode}) if tracer else None

        token_count = 0
        if hasattr(self.llm, "generate_stream"):
            for token in self.llm.generate_stream(
                prompt=user_prompt,
                task="answer",
                system_instruction=self.SYSTEM_PROMPT,
                temperature=0.1,
            ):
                token_count += 1
                yield ("token", token)
        else:
            answer = self.llm.generate(
                prompt=user_prompt,
                system_instruction=self.SYSTEM_PROMPT,
                temperature=0.1,
            )
            token_count += len(answer.split())
            yield ("token", answer)

        if tracer and llm_span:
            tracer.finish_span(llm_span, outputs={"tokens_streamed": token_count}, metadata={"model": getattr(self.llm, "last_model_used", "openrouter")})

        yield (
            "done",
            {
                "query": query,
                "citations": citations,
                "reranked_chunks": top_chunks,
                "retrieved_chunks": [c for c, s in dense_results],
                "ranked_items": retrieval_data["ranked_items"],
                "ranked_graph_facts": retrieval_data["ranked_graph_facts"],
                "formatted_context": formatted_context,
                "graph_context": graph_facts,
                "mode_used": active_mode,
            },
        )

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _retrieve(
        self,
        query_vec,
        top_k: int,
        filter_filenames: Optional[List[str]],
    ) -> List[Tuple[Dict, float]]:
        """Routes retrieval to Pinecone or FAISS depending on the vector_db type."""
        search_kwargs: dict = {"top_k": top_k}

        if filter_filenames and hasattr(self.vector_db, "search"):
            import inspect
            sig = inspect.signature(self.vector_db.search)
            if "filter_filenames" in sig.parameters:
                search_kwargs["filter_filenames"] = filter_filenames

        return self.vector_db.search(query_vec, **search_kwargs)

    def _build_prompt(
        self,
        query: str,
        top_chunks: List[Dict],
        memory_context: Optional[str],
        graph_context: Optional[str] = None,
        mode: str = "hybrid",
    ) -> Tuple[str, str]:
        """Constructs the grounded user prompt with optional conversation memory and knowledge graph context."""
        context_blocks = []
        for idx, chunk in enumerate(top_chunks):
            citation_tag = f"Source: {chunk.get('filename', 'Doc')}, Page {chunk.get('page_number', 1)}"
            block = f"--- CONTEXT CHUNK #{idx+1} [{citation_tag}] ---\n{chunk.get('text', '')}\n"
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)

        memory_section = ""
        if memory_context and memory_context.strip():
            memory_section = f"CONVERSATION HISTORY & USER PREFERENCES:\n{memory_context.strip()}\n\n"

        graph_section = ""
        if graph_context and graph_context.strip() and mode in ("graph_only", "hybrid"):
            graph_section = f"KNOWLEDGE GRAPH CONTEXT (MULTI-HOP RELATIONS & GOVERNANCE):\n{graph_context.strip()}\n\n"

        doc_context_section = ""
        if formatted_context:
            doc_context_section = f"DOCUMENT CONTEXT:\n{formatted_context}\n\n"

        prompt = (
            f"{memory_section}"
            f"{graph_section}"
            f"{doc_context_section}"
            f"QUESTION:\n{query}\n\n"
            f"ANSWER (strictly grounded in the document and knowledge graph context above, quoting exact figures and citing sources):"
        )
        return prompt, formatted_context

    def _generate(self, prompt: str) -> str:
        """Calls the LLM (OpenRouter or Gemini) with the grounded prompt."""
        if hasattr(self.llm, "generate"):
            return self.llm.generate(
                prompt=prompt,
                task="answer",
                system_instruction=self.SYSTEM_PROMPT,
                temperature=0.1,
            )
        return "Error: No valid LLM backend configured."

    def _extract_citations(self, chunks: List[Dict]) -> List[str]:
        """Extracts unique [Source: <filename>, Page <page>] citation tags."""
        seen = set()
        citations = []
        for chunk in chunks:
            fn = chunk.get("filename")
            pg = chunk.get("page_number", 1)
            if fn and fn != "knowledge_graph":
                tag = f"{fn} (p. {pg})"
                if tag not in seen:
                    seen.add(tag)
                    citations.append(tag)
        return citations

    def _empty_response(self, query: str, message: str) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": message,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "ranked_items": [],
            "ranked_graph_facts": [],
            "citations": [],
            "formatted_context": "",
            "graph_context": "",
            "mode_used": "none",
        }
