"""
pipeline/ingestion.py - Dual Ingestion Pipeline for CogniRAG.

Coordinates the simultaneous ingestion of documents into:
1. Vector Database (Pinecone Serverless / SQLite WAL / FAISS) for dense semantic retrieval.
2. Knowledge Graph (Neo4j Community / AuraDB) for deterministic multi-hop entity and relationship traversal.

Guarantees 100% free operation, non-blocking execution, and seamless fallback
if Neo4j is offline or temporarily unreachable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, GRAPHRAG_EXTRACTION_MODE
from pipeline.chunker import DocumentChunker
from pipeline.parser import DocumentParser
from rag.graph.extractor import GraphExtractor, EntityNormalizer
from rag.graph.neo4j_client import Neo4jClient
from vectorstore.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class DualIngestionPipeline:
    """
    Unified ingestion pipeline that parses files, generates semantic chunks,
    calculates dense vector embeddings, extracts knowledge graph subgraphs,
    and synchronizes vector and graph stores.
    """

    def __init__(
        self,
        vector_db: Any,
        embedding_manager: EmbeddingManager,
        neo4j_client: Optional[Neo4jClient] = None,
        graph_extractor: Optional[GraphExtractor] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.vector_db = vector_db
        self.embedding_manager = embedding_manager
        self.neo4j_client = neo4j_client
        self.graph_extractor = graph_extractor or GraphExtractor()
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ingest_file(
        self,
        file_path: Union[str, Path],
        filename: Optional[str] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Parses, chunks, embeds, and indexes a single document into both
        the vector store and the knowledge graph.

        Args:
            file_path: Path to the target document (PDF, CSV, XLSX, DOCX, TXT).
            filename: Optional display name (defaults to file_path basename).
            progress_callback: Optional callback fn(stage: str, percent: float).

        Returns:
            Dict containing ingestion statistics:
                {
                    "filename": str,
                    "chunks": int,
                    "entities": int,
                    "relations": int,
                    "graph_synced": bool,
                }
        """
        path = Path(file_path)
        doc_filename = filename or path.name
        doc_type = path.suffix.lstrip(".").lower() or "txt"

        logger.info(f"[DualIngestion] Starting dual ingestion for {doc_filename}...")
        if progress_callback:
            progress_callback(f"Parsing {doc_filename}...", 0.15)

        # 1. Parse document structure
        parsed = DocumentParser.parse_file(str(path))
        chunks = self.chunker.chunk_parsed_document(parsed)

        if not chunks:
            logger.warning(f"[DualIngestion] No chunks generated for {doc_filename}")
            return {
                "filename": doc_filename,
                "chunks": 0,
                "entities": 0,
                "relations": 0,
                "graph_synced": False,
            }

        # Ensure filename is set on all chunks
        for c in chunks:
            c["filename"] = doc_filename

        # 2. Vector Embedding & Upsert
        if progress_callback:
            progress_callback(f"Vectorizing {len(chunks)} chunks into Pinecone...", 0.40)

        texts = [c["text"] for c in chunks]
        vecs = self.embedding_manager.embed_texts(texts)
        self.vector_db.upsert_chunks(chunks, vecs)
        logger.info(f"[DualIngestion] Vectorized {len(chunks)} chunks into Vector DB.")

        # 3. Knowledge Graph Subgraph Extraction & Upsert
        graph_synced = False
        all_entities: List[Dict[str, Any]] = []
        all_relations: List[Dict[str, Any]] = []

        if self.neo4j_client and self.neo4j_client.is_available():
            try:
                if progress_callback:
                    progress_callback(f"Extracting Knowledge Graph entities...", 0.70)

                # Ensure schema constraints are initialized
                self.neo4j_client.init_schema()

                # Fast 1-call document-level extraction across chunks with deduplication
                if hasattr(self.graph_extractor, "extract_subgraph_from_document"):
                    full_doc_text = "\n\n".join([f"[Page {c.get('page_number', 1)}]\n{c['text']}" for c in chunks])
                    res = self.graph_extractor.extract_subgraph_from_document(
                        document_text=full_doc_text,
                        filename=doc_filename,
                        chunks=chunks,
                    )
                    if isinstance(res, tuple) and len(res) == 2:
                        all_entities, all_relations = res
                    elif isinstance(res, dict):
                        all_entities = res.get("entities", [])
                        all_relations = res.get("relations", [])
                    else:
                        all_entities, all_relations = [], []
                elif hasattr(self.graph_extractor, "extract_subgraph_from_chunks"):
                    res = self.graph_extractor.extract_subgraph_from_chunks(
                        chunks=chunks,
                        filename=doc_filename,
                    )
                    if isinstance(res, tuple) and len(res) == 2:
                        all_entities, all_relations = res
                    elif isinstance(res, dict):
                        all_entities = res.get("entities", [])
                        all_relations = res.get("relations", [])
                    else:
                        all_entities, all_relations = [], []
                else:
                    seen_entity_names = set()
                    seen_relation_keys = set()
                    for chunk in chunks:
                        subgraph = self.graph_extractor.extract_subgraph_from_chunk(
                            chunk_text=chunk["text"],
                            chunk_id=chunk["chunk_id"],
                            filename=doc_filename,
                            page_number=chunk.get("page_number", 1),
                        )
                        for ent in subgraph.get("entities", []):
                            ent_k = ent["name"].lower()
                            if ent_k not in seen_entity_names:
                                seen_entity_names.add(ent_k)
                                all_entities.append(ent)
                        for rel in subgraph.get("relations", []):
                            rel_k = f"{rel['source'].lower()}_{rel['type']}_{rel['target'].lower()}"
                            if rel_k not in seen_relation_keys:
                                seen_relation_keys.add(rel_k)
                                all_relations.append(rel)

                if progress_callback:
                    progress_callback(f"Ingesting {len(all_entities)} entities to Neo4j...", 0.90)

                # Prepare document and chunk records for batch upsert
                doc_record = {
                    "filename": doc_filename,
                    "total_chunks": len(chunks),
                    "doc_type": doc_type,
                }
                chunk_records = [
                    {
                        "chunk_id": c["chunk_id"],
                        "filename": doc_filename,
                        "page_number": c.get("page_number", 1),
                        "has_table": c.get("has_table", False),
                        "text_preview": c["text"][:200],
                    }
                    for c in chunks
                ]

                graph_synced = self.neo4j_client.batch_upsert_subgraph(
                    document=doc_record,
                    chunks=chunk_records,
                    entities=all_entities,
                    relations=all_relations,
                )
                logger.info(
                    f"[DualIngestion] Graph sync for {doc_filename}: "
                    f"{len(all_entities)} entities, {len(all_relations)} relations, success={graph_synced}."
                )
            except Exception as exc:
                logger.warning(f"[DualIngestion] Neo4j graph ingestion encountered error: {exc}. Vector indexing preserved.")
                graph_synced = False
        else:
            logger.info("[DualIngestion] Neo4j is offline or disabled. Document indexed into vector store only.")

        if progress_callback:
            progress_callback(f"Completed {doc_filename} ({len(chunks)} chunks, {len(all_entities)} entities)", 1.0)

        return {
            "filename": doc_filename,
            "chunks": len(chunks),
            "entities": len(all_entities),
            "relations": len(all_relations),
            "graph_synced": graph_synced,
        }

    def delete_document(self, filename: str) -> Dict[str, Any]:
        """
        Removes a document from both vector storage and knowledge graph.

        Args:
            filename: Target document filename to delete.

        Returns:
            Dict containing deletion summary.
        """
        deleted_chunks = 0
        if hasattr(self.vector_db, "delete_by_filename"):
            try:
                deleted_chunks = self.vector_db.delete_by_filename(filename)
            except Exception as e:
                logger.warning(f"[DualIngestion] Error deleting from vector DB: {e}")

        graph_deleted = False
        if self.neo4j_client and self.neo4j_client.is_available():
            try:
                graph_deleted = self.neo4j_client.delete_document_subgraph(filename)
            except Exception as e:
                logger.warning(f"[DualIngestion] Error deleting from Neo4j: {e}")

        logger.info(f"[DualIngestion] Deleted {filename}: {deleted_chunks} vector chunks, graph_deleted={graph_deleted}")
        return {
            "filename": filename,
            "deleted_chunks": deleted_chunks,
            "graph_deleted": graph_deleted,
        }
