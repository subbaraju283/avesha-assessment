"""
Knowledge Graph for handling relational reasoning and entity connections.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import networkx as nx
import json

from ..models.llm_manager import LLMManager
from .graph_store import GraphStore
from .models import Entity, Relationship, KGQueryResult

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Knowledge Graph for relational reasoning with NASA data."""
    
    def __init__(self, config: Dict[str, Any], llm_manager: LLMManager):
        self.config = config
        self.llm_manager = llm_manager
        self.max_depth = config.get("max_depth", 3)
        self.max_paths = config.get("max_paths", 5)
        self.relationship_types = config.get("relationship_types", [])
        
        # Initialize graph store
        self.graph_store = GraphStore(Path("data/kg"))
        
        # Load initial NASA graph data
        self._load_nasa_graph()
    
    def _load_nasa_graph(self):
        """Load initial NASA knowledge graph data."""
        # Create entities
        entities = [
            Entity("mission_voyager1", "Voyager 1", "mission", 
                  {"launch_date": "1977-09-05", "status": "active", "type": "interstellar"}, "NASA"),
            Entity("mission_voyager2", "Voyager 2", "mission", 
                  {"launch_date": "1977-08-20", "status": "active", "type": "interstellar"}, "NASA"),
            Entity("mission_curiosity", "Mars Curiosity", "mission", 
                  {"launch_date": "2011-11-26", "status": "active", "type": "rover"}, "NASA"),
            Entity("mission_perseverance", "Perseverance", "mission", 
                  {"launch_date": "2020-07-30", "status": "active", "type": "rover"}, "NASA"),
            Entity("mission_hubble", "Hubble Space Telescope", "mission", 
                  {"launch_date": "1990-04-24", "status": "active", "type": "telescope"}, "NASA"),
            Entity("mission_webb", "James Webb Space Telescope", "mission", 
                  {"launch_date": "2021-12-25", "status": "active", "type": "telescope"}, "NASA"),
            
            Entity("planet_mars", "Mars", "planet", 
                  {"type": "terrestrial", "distance_from_sun": "227.9M km"}, "NASA"),
            Entity("planet_jupiter", "Jupiter", "planet", 
                  {"type": "gas_giant", "distance_from_sun": "778.5M km"}, "NASA"),
            Entity("planet_saturn", "Saturn", "planet", 
                  {"type": "gas_giant", "distance_from_sun": "1.4B km"}, "NASA"),
            
            Entity("tech_ion_propulsion", "Ion Propulsion", "technology", 
                  {"type": "propulsion", "efficiency": "high"}, "NASA"),
            Entity("tech_nuclear_power", "Nuclear Power", "technology", 
                  {"type": "power", "reliability": "high"}, "NASA"),
            Entity("tech_solar_panels", "Solar Panels", "technology", 
                  {"type": "power", "renewable": True}, "NASA"),
            
            Entity("location_kennedy", "Kennedy Space Center", "location", 
                  {"type": "launch_facility", "country": "USA"}, "NASA"),
            Entity("location_jpl", "Jet Propulsion Laboratory", "location", 
                  {"type": "research_facility", "country": "USA"}, "NASA")
        ]
        
        # Create relationships
        relationships = [
            # Mission -> Planet relationships
            Relationship("rel_001", "mission_voyager1", "planet_jupiter", "STUDIED", 
                       {"study_type": "flyby", "year": 1979}, "NASA"),
            Relationship("rel_002", "mission_voyager1", "planet_saturn", "STUDIED", 
                       {"study_type": "flyby", "year": 1980}, "NASA"),
            Relationship("rel_003", "mission_voyager2", "planet_jupiter", "STUDIED", 
                       {"study_type": "flyby", "year": 1979}, "NASA"),
            Relationship("rel_004", "mission_voyager2", "planet_saturn", "STUDIED", 
                       {"study_type": "flyby", "year": 1981}, "NASA"),
            Relationship("rel_005", "mission_curiosity", "planet_mars", "STUDIED", 
                       {"study_type": "surface_exploration", "year": 2012}, "NASA"),
            Relationship("rel_006", "mission_perseverance", "planet_mars", "STUDIED", 
                       {"study_type": "surface_exploration", "year": 2021}, "NASA"),
            
            # Mission -> Technology relationships
            Relationship("rel_007", "mission_voyager1", "tech_ion_propulsion", "USED_TECHNOLOGY", 
                       {"usage_type": "propulsion"}, "NASA"),
            Relationship("rel_008", "mission_curiosity", "tech_nuclear_power", "USED_TECHNOLOGY", 
                       {"usage_type": "power"}, "NASA"),
            Relationship("rel_009", "mission_perseverance", "tech_nuclear_power", "USED_TECHNOLOGY", 
                       {"usage_type": "power"}, "NASA"),
            Relationship("rel_010", "mission_hubble", "tech_solar_panels", "USED_TECHNOLOGY", 
                       {"usage_type": "power"}, "NASA"),
            Relationship("rel_011", "mission_webb", "tech_solar_panels", "USED_TECHNOLOGY", 
                       {"usage_type": "power"}, "NASA"),
            
            # Mission -> Location relationships
            Relationship("rel_012", "mission_voyager1", "location_kennedy", "LAUNCHED_FROM", 
                       {"launch_date": "1977-09-05"}, "NASA"),
            Relationship("rel_013", "mission_voyager2", "location_kennedy", "LAUNCHED_FROM", 
                       {"launch_date": "1977-08-20"}, "NASA"),
            Relationship("rel_014", "mission_curiosity", "location_kennedy", "LAUNCHED_FROM", 
                       {"launch_date": "2011-11-26"}, "NASA"),
            Relationship("rel_015", "mission_perseverance", "location_kennedy", "LAUNCHED_FROM", 
                       {"launch_date": "2020-07-30"}, "NASA"),
            
            # Similar mission types
            Relationship("rel_016", "mission_voyager1", "mission_voyager2", "SIMILAR_TO", 
                       {"similarity_type": "interstellar_probe"}, "NASA"),
            Relationship("rel_017", "mission_curiosity", "mission_perseverance", "SIMILAR_TO", 
                       {"similarity_type": "mars_rover"}, "NASA"),
            Relationship("rel_018", "mission_hubble", "mission_webb", "SIMILAR_TO", 
                       {"similarity_type": "space_telescope"}, "NASA")
        ]
        
        # Add entities and relationships to graph store
        for entity in entities:
            self.graph_store.add_entity(entity)
        
        for relationship in relationships:
            self.graph_store.add_relationship(relationship)
        
        logger.info(f"Loaded {len(entities)} entities and {len(relationships)} relationships into knowledge graph")
    
    async def query(self, query: str, debug: bool = False) -> KGQueryResult:
        """
        Query the knowledge graph for relational information.
        
        Args:
            query: The user's query
            debug: Whether to enable debug logging
            
        Returns:
            KGQueryResult with entities, relationships, and paths
        """
        logger.info(f"KG Query: {query}")
        
        # Step 1: Analyze query to extract entities and relationship types
        query_analysis = await self._analyze_query(query)
        
        # Step 2: Find relevant entities and relationships
        entities, relationships = self._find_relevant_data(query, query_analysis)
        
        # Step 3: Find paths between entities
        paths = self._find_paths(entities, relationships, query_analysis)
        
        # Step 4: Generate response
        response = await self._generate_response(query, entities, relationships, paths, query_analysis)
        
        if debug:
            logger.info(f"Query analysis: {query_analysis}")
            logger.info(f"Found {len(entities)} entities and {len(relationships)} relationships")
            logger.info(f"Found {len(paths)} paths")
        
        return response
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to extract entities and relationship types."""
        prompt = f"""
        Analyze this NASA-related query and extract:
        1. Main entities (missions, planets, technologies, locations)
        2. Relationship types being asked about
        3. Query intent (comparison, connection, similarity, etc.)
        
        Query: {query}
        
        Respond in JSON format with keys: entities, relationship_types, query_intent
        """
        
        try:
            response = await self.llm_manager.generate(prompt)
            # Parse JSON response (simplified)
            analysis = {
                "entities": self._extract_entities_simple(query),
                "relationship_types": self._extract_relationship_types_simple(query),
                "query_intent": self._classify_query_intent(query)
            }
        except Exception as e:
            logger.warning(f"LLM query analysis failed: {e}")
            analysis = {
                "entities": self._extract_entities_simple(query),
                "relationship_types": self._extract_relationship_types_simple(query),
                "query_intent": self._classify_query_intent(query)
            }
        
        return analysis
    
    def _extract_entities_simple(self, query: str) -> List[str]:
        """Simple entity extraction using keyword matching."""
        entities = []
        query_lower = query.lower()
        
        # Mission names
        mission_keywords = ["voyager", "curiosity", "perseverance", "hubble", "webb", "apollo"]
        for keyword in mission_keywords:
            if keyword in query_lower:
                entities.append(keyword.title())
        
        # Planets
        planets = ["mars", "jupiter", "saturn", "venus", "mercury", "neptune", "uranus"]
        for planet in planets:
            if planet in query_lower:
                entities.append(planet.title())
        
        # Technologies
        tech_keywords = ["ion propulsion", "nuclear power", "solar panels", "telescope"]
        for tech in tech_keywords:
            if tech in query_lower:
                entities.append(tech)
        
        # Locations
        location_keywords = ["kennedy", "jpl", "cape canaveral"]
        for location in location_keywords:
            if location in query_lower:
                entities.append(location.title())
        
        return entities
    
    def _extract_relationship_types_simple(self, query: str) -> List[str]:
        """Simple relationship type extraction."""
        relationship_types = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["studied", "explored", "visited"]):
            relationship_types.append("STUDIED")
        if any(word in query_lower for word in ["used", "technology", "equipment"]):
            relationship_types.append("USED_TECHNOLOGY")
        if any(word in query_lower for word in ["launched", "launch"]):
            relationship_types.append("LAUNCHED_FROM")
        if any(word in query_lower for word in ["similar", "same", "both"]):
            relationship_types.append("SIMILAR_TO")
        
        return relationship_types
    
    def _classify_query_intent(self, query: str) -> str:
        """Classify the intent of the relational query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["which", "what missions"]):
            return "entity_discovery"
        elif any(word in query_lower for word in ["how are", "related", "connection"]):
            return "relationship_exploration"
        elif any(word in query_lower for word in ["compare", "similar", "different"]):
            return "comparison"
        elif any(word in query_lower for word in ["both", "either", "neither"]):
            return "set_operations"
        else:
            return "general"
    
    def _find_relevant_data(self, query: str, analysis: Dict[str, Any]) -> Tuple[List[Entity], List[Relationship]]:
        """Find relevant entities and relationships based on query analysis."""
        entities = []
        relationships = []
        
        # Get all entities and relationships from graph store
        all_entities = self.graph_store.get_all_entities()
        all_relationships = self.graph_store.get_all_relationships()
        
        # Find entities matching query criteria
        query_entities = analysis.get("entities", [])
        for entity in all_entities:
            if any(query_entity.lower() in entity.name.lower() for query_entity in query_entities):
                entities.append(entity)
        
        # If no specific entities found, get all entities
        if not entities:
            entities = all_entities
        
        # Find relationships matching query criteria
        query_relationship_types = analysis.get("relationship_types", [])
        for relationship in all_relationships:
            # Check if relationship involves any of the found entities
            entity_ids = [entity.id for entity in entities]
            if (relationship.source_id in entity_ids or relationship.target_id in entity_ids):
                if not query_relationship_types or relationship.type in query_relationship_types:
                    relationships.append(relationship)
        
        return entities, relationships
    
    def _find_paths(self, entities: List[Entity], relationships: List[Relationship], analysis: Dict[str, Any]) -> List[List[str]]:
        """Find paths between entities in the graph."""
        if len(entities) < 2:
            return []
        
        # Create a NetworkX graph for path finding
        G = nx.DiGraph()
        
        # Add relationships to graph
        for rel in relationships:
            G.add_edge(rel.source_id, rel.target_id, type=rel.type, properties=rel.properties)
        
        paths = []
        entity_ids = [entity.id for entity in entities]
        
        # Find paths between different entities
        for i, source_id in enumerate(entity_ids):
            for target_id in entity_ids[i+1:]:
                try:
                    # Find all simple paths between source and target
                    simple_paths = list(nx.all_simple_paths(G, source_id, target_id, cutoff=self.max_depth))
                    
                    # Limit number of paths
                    paths.extend(simple_paths[:self.max_paths])
                    
                except nx.NetworkXNoPath:
                    # No path exists between these entities
                    continue
        
        return paths[:self.max_paths]
    
    async def _generate_response(
        self, 
        query: str, 
        entities: List[Entity], 
        relationships: List[Relationship], 
        paths: List[List[str]], 
        analysis: Dict[str, Any]
    ) -> KGQueryResult:
        """Generate a response based on found entities, relationships, and paths."""
        if not entities and not relationships:
            return KGQueryResult(
                entities=[],
                relationships=[],
                paths=[],
                confidence=0.0,
                reasoning="No relevant entities or relationships found in knowledge graph",
                query_type=analysis.get("query_intent", "unknown"),
                metadata={"analysis": analysis}
            )
        
        # Create summary for LLM
        entity_summary = "\n".join([
            f"- {entity.name} ({entity.type}): {entity.properties}"
            for entity in entities[:10]  # Limit to first 10 entities
        ])
        
        relationship_summary = "\n".join([
            f"- {rel.source_id} --[{rel.type}]--> {rel.target_id}: {rel.properties}"
            for rel in relationships[:10]  # Limit to first 10 relationships
        ])
        
        path_summary = "\n".join([
            f"- Path: {' -> '.join(path)}"
            for path in paths[:5]  # Limit to first 5 paths
        ])
        
        prompt = f"""
        Based on this NASA knowledge graph data, answer the user's question:
        
        User Question: {query}
        
        Relevant Entities:
        {entity_summary}
        
        Relevant Relationships:
        {relationship_summary}
        
        Relevant Paths:
        {path_summary}
        
        Provide a clear answer that explains the relationships and connections between the entities.
        """
        
        try:
            response_text = await self.llm_manager.generate(prompt)
            confidence = 0.8 if entities and relationships else 0.3
            
            reasoning = f"Found {len(entities)} entities, {len(relationships)} relationships, and {len(paths)} paths"
            
        except Exception as e:
            logger.error(f"Failed to generate KG response: {e}")
            response_text = "Unable to generate response due to technical issues."
            confidence = 0.0
            reasoning = f"Error generating response: {e}"
        
        return KGQueryResult(
            entities=entities,
            relationships=relationships,
            paths=paths,
            confidence=confidence,
            reasoning=reasoning,
            query_type=analysis.get("query_intent", "unknown"),
            metadata={
                "analysis": analysis,
                "response_text": response_text,
                "entity_count": len(entities),
                "relationship_count": len(relationships),
                "path_count": len(paths)
            }
        )
    
    def add_entity(self, entity: Entity):
        """Add a new entity to the knowledge graph."""
        self.graph_store.add_entity(entity)
        logger.info(f"Added entity: {entity.name} ({entity.type})")
    
    def add_relationship(self, relationship: Relationship):
        """Add a new relationship to the knowledge graph."""
        self.graph_store.add_relationship(relationship)
        logger.info(f"Added relationship: {relationship.source_id} --[{relationship.type}]--> {relationship.target_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        all_entities = self.graph_store.get_all_entities()
        all_relationships = self.graph_store.get_all_relationships()
        
        entity_types = list(set(entity.type for entity in all_entities))
        relationship_types = list(set(rel.type for rel in all_relationships))
        
        return {
            "total_entities": len(all_entities),
            "total_relationships": len(all_relationships),
            "entity_types": entity_types,
            "relationship_types": relationship_types
        } 