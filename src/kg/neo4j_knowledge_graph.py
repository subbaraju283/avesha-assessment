"""
Neo4j-based Knowledge Graph for handling relational reasoning and entity connections.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from neo4j import GraphDatabase
from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

from ..models.llm_manager import LLMManager

logger = logging.getLogger(__name__)


@dataclass
class Neo4jKGQueryResult:
    """Result of Neo4j Knowledge Graph query."""
    answer: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    paths: List[List[str]]
    confidence: float
    reasoning: str
    query_type: str
    metadata: Dict[str, Any]


class Neo4jKnowledgeGraph:
    """Neo4j-based Knowledge Graph for relational reasoning with NASA data."""
    
    def __init__(self, config: Dict[str, Any], llm_manager: LLMManager):
        self.config = config
        self.llm_manager = llm_manager
        self.embedder = OpenAIEmbeddings()
        
        # Neo4j connection
        self.neo4j_uri = config.get("neo4j", {}).get("database_url", "neo4j+s://72bbcca1.databases.neo4j.io")
        self.neo4j_username = config.get("neo4j", {}).get("username", "neo4j")
        self.neo4j_password = config.get("neo4j", {}).get("password", "TqEewyu6IUiWVzoaUXfHTjXAC2JHUg72C_p4AEIwJEY")
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # Initialize LLM for GraphRAG
        self.llm = OpenAILLM(
            model_name="gpt-4o",
            model_params={"temperature": 0.0}
        )
        
        # Define RAG prompt template
        self.rag_template = RagTemplate(
            template='''
                Answer the Question using the following Context. 
                Only respond with information mentioned in the Context. 
                Do not inject any speculative information not mentioned.

                # Question:
                {query_text}

                # Context:
                {context}

                # Answer:
            ''',
            expected_inputs=['query_text', 'context']
        )
        
        # Initialize GraphRAG
        self.graph_rag = None
        self._initialize_graph_rag()
        
        logger.info("Neo4j Knowledge Graph initialized")
    
    def _initialize_graph_rag(self):
        """Initialize the GraphRAG system."""
        try:
            # Create vector index if it doesn't exist
            self._create_vector_index()
            
            # Create vector retriever with proper parameters
            vector_retriever = VectorCypherRetriever(
                driver=self.driver,
                index_name="text_embeddings",
                embedder=self.embedder,
                retrieval_query="""
                    // 1) Go out 2-3 hops in the entity graph and get relationships
                    WITH node AS chunk
                    MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,2}()
                    UNWIND relList AS rel
                    
                    // 2) Collect relationships and text chunks
                    WITH collect(DISTINCT chunk) AS chunks, 
                      collect(DISTINCT rel) AS rels
                    
                    // 3) Format and return context
                    RETURN '=== text ===\n' + apoc.text.join([c in chunks | c.text], '\n---\n') + '\n\n=== kg_rels ===\n' +
                      apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r) + '(' + coalesce(r.details, '') + ')' +  ' -> ' + endNode(r).name ], '\n---\n') AS info
                """
            )
            
            # Create GraphRAG
            self.graph_rag = GraphRAG(
                llm=self.llm,
                retriever=vector_retriever,
                prompt_template=self.rag_template
            )
            
            logger.info("GraphRAG initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG: {e}")
            self.graph_rag = None
    
    def _create_vector_index(self):
        """Create vector index for embeddings if it doesn't exist."""
        try:
            from neo4j_graphrag.indexes import create_vector_index
            
            # Create vector index
            create_vector_index(
                self.driver, 
                name="text_embeddings", 
                label="Chunk",
                embedding_property="embedding", 
                dimensions=1536, 
                similarity_fn="cosine"
            )
            
            logger.info("Vector index 'text_embeddings' created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create vector index: {e}")
            # Continue without index - it might already exist
    
    async def query(self, query: str, debug: bool = False) -> Neo4jKGQueryResult:
        """
        Query the Neo4j knowledge graph for relational information.
        
        Args:
            query: The user's query
            debug: Whether to enable debug logging
            
        Returns:
            Neo4jKGQueryResult with answer and metadata
        """
        logger.info(f"Neo4j KG Query: {query}")
        
        if not self.graph_rag:
            return Neo4jKGQueryResult(
                answer="Neo4j Knowledge Graph is not available",
                entities=[],
                relationships=[],
                paths=[],
                confidence=0.0,
                reasoning="GraphRAG not initialized",
                query_type="error",
                metadata={"error": "GraphRAG not available"}
            )
        
        try:
            # Use GraphRAG to search and generate answer
            result = self.graph_rag.search(query, retriever_config={'top_k': 5})
            
            # Extract entities and relationships from the result
            entities = self._extract_entities_from_result(result)
            relationships = self._extract_relationships_from_result(result)
            paths = self._extract_paths_from_result(result)
            
            if debug:
                logger.info(f"GraphRAG result: {result}")
                logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
            
            return Neo4jKGQueryResult(
                answer=result.answer,
                entities=entities,
                relationships=relationships,
                paths=paths,
                confidence=0.8,  # GraphRAG confidence
                reasoning=f"Retrieved information using GraphRAG with {len(entities)} entities",
                query_type="graph_rag",
                metadata={
                    "retriever_config": {'top_k': 5},
                    "graph_rag_used": True
                }
            )
            
        except Exception as e:
            logger.error(f"Neo4j KG query failed: {e}")
            return Neo4jKGQueryResult(
                answer=f"Error querying Neo4j Knowledge Graph: {e}",
                entities=[],
                relationships=[],
                paths=[],
                confidence=0.0,
                reasoning=f"Query failed: {e}",
                query_type="error",
                metadata={"error": str(e)}
            )
    
    def _extract_entities_from_result(self, result) -> List[Dict[str, Any]]:
        """Extract entities from GraphRAG result."""
        entities = []
        try:
            # Extract entities from the context or metadata
            if hasattr(result, 'context') and result.context:
                # Parse entities from context
                # This is a simplified extraction - you might want to enhance this
                entities = self._parse_entities_from_context(result.context)
        except Exception as e:
            logger.warning(f"Failed to extract entities: {e}")
        
        return entities
    
    def _extract_relationships_from_result(self, result) -> List[Dict[str, Any]]:
        """Extract relationships from GraphRAG result."""
        relationships = []
        try:
            # Extract relationships from the context or metadata
            if hasattr(result, 'context') and result.context:
                # Parse relationships from context
                # This is a simplified extraction - you might want to enhance this
                relationships = self._parse_relationships_from_context(result.context)
        except Exception as e:
            logger.warning(f"Failed to extract relationships: {e}")
        
        return relationships
    
    def _extract_paths_from_result(self, result) -> List[List[str]]:
        """Extract paths from GraphRAG result."""
        paths = []
        try:
            # Extract paths from the context or metadata
            if hasattr(result, 'context') and result.context:
                # Parse paths from context
                # This is a simplified extraction - you might want to enhance this
                paths = self._parse_paths_from_context(result.context)
        except Exception as e:
            logger.warning(f"Failed to extract paths: {e}")
        
        return paths
    
    def _parse_entities_from_context(self, context: str) -> List[Dict[str, Any]]:
        """Parse entities from context string."""
        entities = []
        # This is a simplified parser - you might want to enhance this
        # For now, we'll return an empty list
        return entities
    
    def _parse_relationships_from_context(self, context: str) -> List[Dict[str, Any]]:
        """Parse relationships from context string."""
        relationships = []
        # This is a simplified parser - you might want to enhance this
        # For now, we'll return an empty list
        return relationships
    
    def _parse_paths_from_context(self, context: str) -> List[List[str]]:
        """Parse paths from context string."""
        paths = []
        # This is a simplified parser - you might want to enhance this
        # For now, we'll return an empty list
        return paths
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Neo4j Knowledge Graph statistics."""
        try:
            with self.driver.session() as session:
                # Get node count
                node_result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = node_result.single()["node_count"]
                
                # Get relationship count
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = rel_result.single()["rel_count"]
                
                # Get node labels
                label_result = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
                labels = label_result.single()["labels"]
                
                # Get relationship types
                type_result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")
                types = type_result.single()["types"]
                
                return {
                    "total_entities": node_count,
                    "total_relationships": rel_count,
                    "entity_types": labels,
                    "relationship_types": types,
                    "neo4j_uri": self.neo4j_uri,
                    "graph_rag_available": self.graph_rag is not None
                }
                
        except Exception as e:
            logger.error(f"Failed to get Neo4j stats: {e}")
            return {
                "total_entities": 0,
                "total_relationships": 0,
                "entity_types": [],
                "relationship_types": [],
                "neo4j_uri": self.neo4j_uri,
                "graph_rag_available": False,
                "error": str(e)
            }
    
    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close() 