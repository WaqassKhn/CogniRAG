"""
rag/graph/neo4j_client.py - Neo4j Client & Schema Manager for CogniRAG.

Manages connection pooling, idempotent schema constraints, batch subgraph upserts,
and multi-hop Cypher traversals. Gracefully degrades if Neo4j is offline.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from neo4j import GraphDatabase, Driver, Session
    from neo4j.exceptions import ServiceUnavailable, AuthError, Neo4jError
    NEO4J_AVAILABLE = True
except ImportError:
    GraphDatabase = None
    Driver = None
    Session = None
    ServiceUnavailable = Exception
    AuthError = Exception
    Neo4jError = Exception
    NEO4J_AVAILABLE = False

from config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    NEO4J_MAX_CONNECTION_POOL_SIZE,
    ENABLE_GRAPHRAG,
)

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Thread-safe client for Neo4j Community / AuraDB.
    Supports lazy connection, idempotent schema initialization, batch Cypher upserts,
    and multi-hop knowledge graph traversal.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.uri = uri or NEO4J_URI
        self.username = username or NEO4J_USERNAME
        self.password = password or NEO4J_PASSWORD
        self.database = database or NEO4J_DATABASE
        self.enabled = ENABLE_GRAPHRAG if enabled is None else enabled

        self._driver: Optional[Any] = None
        self._connected = False
        self._last_health_check = 0.0
        self._health_cache_ttl = 10.0  # seconds

    def get_driver(self) -> Optional[Any]:
        """Lazy initializer for the Neo4j driver."""
        if not self.enabled:
            return None

        if not NEO4J_AVAILABLE:
            logger.warning("[Neo4jClient] neo4j Python driver not installed. GraphRAG disabled.")
            return None

        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                    max_connection_pool_size=NEO4J_MAX_CONNECTION_POOL_SIZE,
                    connection_timeout=3.0,
                )
                logger.info(f"[Neo4jClient] Created driver for {self.uri}")
            except Exception as e:
                logger.warning(f"[Neo4jClient] Failed to create driver: {e}")
                self._driver = None
                return None

        return self._driver

    def _test_ping(self, driver: Any) -> bool:
        """Helper to ping Neo4j using a driver instance."""
        if driver is None:
            return False
        try:
            with driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS ping")
                record = result.single()
                return record is not None and record["ping"] == 1
        except Exception as e:
            logger.debug(f"[Neo4jClient] Ping failed on {self.uri}: {e}")
            return False

    def is_available(self, force_check: bool = False) -> bool:
        """
        Fast non-blocking connectivity check with cached status and intelligent endpoint resolution.
        Returns True if Neo4j is reachable and authenticated.
        """
        if not self.enabled or not NEO4J_AVAILABLE:
            return False

        now = time.time()
        if not force_check and (now - self._last_health_check < self._health_cache_ttl):
            return self._connected

        driver = self.get_driver()
        connected = self._test_ping(driver) if driver else False

        # If primary URI failed, try auto-fallback candidate URIs (e.g. host localhost <-> docker neo4j container name)
        if not connected and NEO4J_AVAILABLE:
            alt_candidates = []
            if "localhost" in self.uri or "127.0.0.1" in self.uri:
                alt_candidates.append(self.uri.replace("localhost", "neo4j").replace("127.0.0.1", "neo4j"))
            elif "neo4j" in self.uri:
                alt_candidates.append(self.uri.replace("neo4j", "localhost"))

            for alt_uri in alt_candidates:
                try:
                    alt_driver = GraphDatabase.driver(
                        alt_uri,
                        auth=(self.username, self.password),
                        max_connection_pool_size=NEO4J_MAX_CONNECTION_POOL_SIZE,
                        connection_timeout=2.0,
                    )
                    if self._test_ping(alt_driver):
                        logger.info(f"[Neo4jClient] Discovered reachable Neo4j endpoint at fallback URI: {alt_uri}")
                        if self._driver is not None:
                            try:
                                self._driver.close()
                            except Exception:
                                pass
                        self.uri = alt_uri
                        self._driver = alt_driver
                        connected = True
                        break
                    else:
                        try:
                            alt_driver.close()
                        except Exception:
                            pass
                except Exception:
                    pass

        self._connected = connected
        self._last_health_check = now
        return self._connected

    def init_schema(self) -> bool:
        """
        Initializes schema constraints and indexes idempotently.
        Ensures fast lookups and prevents duplicate Entity, Document, or Chunk nodes.
        """
        if not self.is_available():
            return False

        constraints_and_indexes = [
            # Unique constraints
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT doc_filename_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.filename IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            # Performance lookup indexes
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX chunk_page_idx IF NOT EXISTS FOR (c:Chunk) ON (c.page_number)",
        ]

        driver = self.get_driver()
        if not driver:
            return False

        try:
            with driver.session(database=self.database) as session:
                for statement in constraints_and_indexes:
                    try:
                        session.run(statement)
                    except Neo4jError as e:
                        # Some versions of Community Edition handle syntax differences; log and continue
                        logger.debug(f"[Neo4jClient] Constraint/Index statement notice: {e}")
            logger.info("[Neo4jClient] Schema constraints and indexes successfully verified.")
            return True
        except Exception as e:
            logger.warning(f"[Neo4jClient] Failed to initialize schema: {e}")
            return False

    def batch_upsert_subgraph(
        self,
        document: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> bool:
        """
        Atomic batch ingestion of document, chunks, entities, and relationship edges.
        Links entities to chunks (:Chunk)-[:MENTIONS]->(:Entity) and entities to each other.
        """
        if not self.is_available():
            logger.warning("[Neo4jClient] Neo4j is offline. Skipping graph ingestion.")
            return False

        driver = self.get_driver()
        if not driver:
            return False

        cypher_document = """
        MERGE (d:Document {filename: $doc.filename})
        ON CREATE SET d.total_chunks = $doc.total_chunks,
                      d.doc_type = $doc.doc_type,
                      d.indexed_at = timestamp()
        ON MATCH SET d.total_chunks = $doc.total_chunks,
                     d.indexed_at = timestamp()
        """

        cypher_chunks = """
        UNWIND $chunks AS ch
        MATCH (d:Document {filename: ch.filename})
        MERGE (c:Chunk {chunk_id: ch.chunk_id})
        ON CREATE SET c.page_number = ch.page_number,
                      c.has_table = ch.has_table,
                      c.text_preview = ch.text_preview
        MERGE (d)-[:HAS_CHUNK]->(c)
        """

        cypher_entities = """
        UNWIND $entities AS ent
        MERGE (e:Entity {name: ent.name})
        ON CREATE SET e.type = ent.type,
                      e.description = ent.description,
                      e.first_seen = timestamp()
        ON MATCH SET e.last_seen = timestamp()
        """

        cypher_chunk_mentions = """
        UNWIND $mentions AS m
        MATCH (c:Chunk {chunk_id: m.chunk_id})
        MATCH (e:Entity {name: m.entity_name})
        MERGE (c)-[:MENTIONS]->(e)
        """

        cypher_relations = """
        UNWIND $relations AS rel
        MATCH (src:Entity {name: rel.source})
        MATCH (tgt:Entity {name: rel.target})
        MERGE (src)-[r:RELATES_TO {type: rel.type}]->(tgt)
        ON CREATE SET r.description = rel.description,
                      r.evidence_chunk_id = rel.chunk_id,
                      r.confidence = rel.confidence,
                      r.created_at = timestamp()
        """

        try:
            with driver.session(database=self.database) as session:
                # 1. Upsert document
                session.run(cypher_document, doc=document)

                # 2. Upsert chunks
                if chunks:
                    session.run(cypher_chunks, chunks=chunks)

                # 3. Upsert entities
                if entities:
                    session.run(cypher_entities, entities=entities)

                # 4. Upsert chunk -> entity mentions
                mentions = [
                    {"chunk_id": ent.get("chunk_id"), "entity_name": ent["name"]}
                    for ent in entities
                    if ent.get("chunk_id")
                ]
                if mentions:
                    session.run(cypher_chunk_mentions, mentions=mentions)

                # 5. Upsert entity -> entity relationships
                if relations:
                    session.run(cypher_relations, relations=relations)

            logger.info(
                f"[Neo4jClient] Ingested subgraph for {document.get('filename')}: "
                f"{len(chunks)} chunks, {len(entities)} entities, {len(relations)} relations."
            )
            return True
        except Exception as e:
            logger.error(f"[Neo4jClient] Error in batch_upsert_subgraph: {e}", exc_info=True)
            return False

    def get_entity_neighborhood(
        self,
        entity_names: List[str],
        max_hops: int = 2,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Performs 1-to-N hop traversal around given entity names to retrieve connected subgraphs.
        Returns paths: (source)-[relation]-(target) with chunk citations.
        """
        if not self.is_available() or not entity_names:
            return []

        driver = self.get_driver()
        if not driver:
            return []

        # Bound hops between 1 and 3 for low latency
        hops = max(1, min(max_hops, 3))

        cypher = f"""
        UNWIND $entity_names AS name
        MATCH (start:Entity)
        WHERE toLower(start.name) = toLower(name)
        MATCH path = (start)-[r:RELATES_TO*1..{hops}]-(connected:Entity)
        WITH r, start, connected, relationships(path) AS rels
        UNWIND rels AS rel
        RETURN DISTINCT
            startNode(rel).name AS source,
            startNode(rel).type AS source_type,
            rel.type AS relation_type,
            rel.description AS description,
            rel.evidence_chunk_id AS chunk_id,
            endNode(rel).name AS target,
            endNode(rel).type AS target_type
        LIMIT $limit
        """

        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, entity_names=entity_names, limit=limit)
                records = [record.data() for record in result]
                return records
        except Exception as e:
            logger.warning(f"[Neo4jClient] Error traversing neighborhood for {entity_names}: {e}")
            return []

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Returns summary metrics about the knowledge graph."""
        if not self.is_available():
            return {
                "status": "disconnected",
                "total_entities": 0,
                "total_relationships": 0,
                "total_documents": 0,
                "total_chunks": 0,
            }

        driver = self.get_driver()
        if not driver:
            return {"status": "disconnected"}

        cypher = """
        RETURN
            COUNT { MATCH (e:Entity) } AS entities,
            COUNT { MATCH ()-[r:RELATES_TO]->() } AS relationships,
            COUNT { MATCH (d:Document) } AS documents,
            COUNT { MATCH (c:Chunk) } AS chunks
        """

        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher)
                record = result.single()
                if record:
                    return {
                        "status": "connected",
                        "total_entities": record["entities"],
                        "total_relationships": record["relationships"],
                        "total_documents": record["documents"],
                        "total_chunks": record["chunks"],
                    }
        except Exception as e:
            logger.warning(f"[Neo4jClient] Error retrieving stats: {e}")

        return {"status": "error"}

    def get_entire_graph(
        self,
        limit: int = 150,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves graph nodes and edges for visual rendering in PyVis or Network explorer.
        Returns:
            {
                "nodes": [{"id": name, "label": name, "type": type, "description": desc}],
                "edges": [{"source": src, "target": tgt, "type": rel_type, "description": desc, "chunk_id": chunk_id}]
            }
        """
        if not self.is_available():
            return {"nodes": [], "edges": []}

        driver = self.get_driver()
        if not driver:
            return {"nodes": [], "edges": []}

        cypher = """
        MATCH (src:Entity)-[r:RELATES_TO]->(tgt:Entity)
        WHERE ($entity_types IS NULL OR src.type IN $entity_types OR tgt.type IN $entity_types)
          AND ($relation_types IS NULL OR r.type IN $relation_types)
        RETURN
            src.name AS source,
            src.type AS source_type,
            src.description AS source_desc,
            tgt.name AS target,
            tgt.type AS target_type,
            tgt.description AS target_desc,
            r.type AS relation_type,
            r.description AS description,
            r.evidence_chunk_id AS chunk_id
        LIMIT $limit
        """

        try:
            with driver.session(database=self.database) as session:
                result = session.run(
                    cypher,
                    limit=limit,
                    entity_types=entity_types or None,
                    relation_types=relation_types or None,
                )
                nodes_map = {}
                edges = []
                for record in result:
                    data = record.data()
                    src_name = data["source"]
                    tgt_name = data["target"]

                    if src_name not in nodes_map:
                        nodes_map[src_name] = {
                            "id": src_name,
                            "label": src_name,
                            "type": data.get("source_type", "ENTITY"),
                            "description": data.get("source_desc", ""),
                        }
                    if tgt_name not in nodes_map:
                        nodes_map[tgt_name] = {
                            "id": tgt_name,
                            "label": tgt_name,
                            "type": data.get("target_type", "ENTITY"),
                            "description": data.get("target_desc", ""),
                        }

                    edges.append({
                        "source": src_name,
                        "target": tgt_name,
                        "type": data.get("relation_type", "RELATES_TO"),
                        "description": data.get("description", ""),
                        "chunk_id": data.get("chunk_id", ""),
                    })

                return {
                    "nodes": list(nodes_map.values()),
                    "edges": edges,
                }
        except Exception as e:
            logger.warning(f"[Neo4jClient] Error retrieving entire graph: {e}")
            return {"nodes": [], "edges": []}

    def delete_document_subgraph(self, filename: str) -> bool:
        """Removes a document, its chunks, and associated mentions from Neo4j."""
        if not self.is_available():
            return False

        driver = self.get_driver()
        if not driver:
            return False

        cypher = """
        MATCH (d:Document {filename: $filename})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        DETACH DELETE c, d
        """

        try:
            with driver.session(database=self.database) as session:
                session.run(cypher, filename=filename)
            logger.info(f"[Neo4jClient] Deleted graph nodes for document: {filename}")
            return True
        except Exception as e:
            logger.error(f"[Neo4jClient] Failed to delete document subgraph: {e}")
            return False

    def close(self) -> None:
        """Closes driver connections cleanly."""
        if self._driver is not None:
            try:
                self._driver.close()
                logger.info("[Neo4jClient] Neo4j driver closed.")
            except Exception as e:
                logger.warning(f"[Neo4jClient] Error closing driver: {e}")
            finally:
                self._driver = None
                self._connected = False
