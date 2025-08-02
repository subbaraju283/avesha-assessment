"""
Knowledge Base for handling factual, static Q&A queries.
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from ..models.llm_manager import LLMManager
from .fact_store import FactStore
from .models import Fact, KBQueryResult

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Knowledge Base for factual Q&A with NASA data."""
    
    def __init__(self, config: Dict[str, Any], llm_manager: LLMManager):
        self.config = config
        self.llm_manager = llm_manager
        self.storage_path = Path(config.get("storage_path", "data/kb"))
        self.similarity_threshold = config.get("similarity_threshold", 0.8)
        self.max_results = config.get("max_results", 10)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        # Initialize fact store
        self.fact_store = FactStore(self.storage_path)
        
        # Load initial NASA facts
        self._load_nasa_facts()
    
    def _load_nasa_facts(self):
        """Load initial NASA facts into the knowledge base."""
        nasa_facts = [
            {
                "id": "mission_001",
                "subject": "Voyager 1",
                "predicate": "launch_date",
                "object": "1977-09-05",
                "source": "NASA Mission Database",
                "confidence": 0.95,
                "metadata": {"mission_type": "interstellar", "status": "active"}
            },
            {
                "id": "mission_002",
                "subject": "Voyager 2",
                "predicate": "launch_date",
                "object": "1977-08-20",
                "source": "NASA Mission Database",
                "confidence": 0.95,
                "metadata": {"mission_type": "interstellar", "status": "active"}
            },
            {
                "id": "mission_003",
                "subject": "Mars Curiosity",
                "predicate": "launch_date",
                "object": "2011-11-26",
                "source": "NASA Mission Database",
                "confidence": 0.95,
                "metadata": {"mission_type": "rover", "status": "active", "planet": "Mars"}
            },
            {
                "id": "mission_004",
                "subject": "Perseverance",
                "predicate": "launch_date",
                "object": "2020-07-30",
                "source": "NASA Mission Database",
                "confidence": 0.95,
                "metadata": {"mission_type": "rover", "status": "active", "planet": "Mars"}
            },
            {
                "id": "mission_005",
                "subject": "Hubble Space Telescope",
                "predicate": "launch_date",
                "object": "1990-04-24",
                "source": "NASA Mission Database",
                "confidence": 0.95,
                "metadata": {"mission_type": "telescope", "status": "active"}
            },
            {
                "id": "mission_006",
                "subject": "James Webb Space Telescope",
                "predicate": "launch_date",
                "object": "2021-12-25",
                "source": "NASA Mission Database",
                "confidence": 0.95,
                "metadata": {"mission_type": "telescope", "status": "active"}
            },
            {
                "id": "tech_001",
                "subject": "Voyager 1",
                "predicate": "uses_technology",
                "object": "ion propulsion",
                "source": "NASA Mission Database",
                "confidence": 0.9,
                "metadata": {"technology_type": "propulsion"}
            },
            {
                "id": "tech_002",
                "subject": "Mars Curiosity",
                "predicate": "uses_technology",
                "object": "nuclear power",
                "source": "NASA Mission Database",
                "confidence": 0.9,
                "metadata": {"technology_type": "power"}
            },
            {
                "id": "planet_001",
                "subject": "Voyager 1",
                "predicate": "studied_planet",
                "object": "Jupiter",
                "source": "NASA Mission Database",
                "confidence": 0.95,
                "metadata": {"study_type": "flyby"}
            },
            {
                "id": "planet_002",
                "subject": "Voyager 1",
                "predicate": "studied_planet",
                "object": "Saturn",
                "source": "NASA Mission Database",
                "confidence": 0.95,
                "metadata": {"study_type": "flyby"}
            },
            {
                "id": "planet_003",
                "subject": "Mars Curiosity",
                "predicate": "studied_planet",
                "object": "Mars",
                "source": "NASA Mission Database",
                "confidence": 0.95,
                "metadata": {"study_type": "surface exploration"}
            },
            {
                "id": "planet_004",
                "subject": "Perseverance",
                "predicate": "studied_planet",
                "object": "Mars",
                "source": "NASA Mission Database",
                "confidence": 0.95,
                "metadata": {"study_type": "surface exploration"}
            }
        ]
        
        for fact_data in nasa_facts:
            fact = Fact(
                id=fact_data["id"],
                subject=fact_data["subject"],
                predicate=fact_data["predicate"],
                object=fact_data["object"],
                source=fact_data["source"],
                confidence=fact_data["confidence"],
                metadata=fact_data["metadata"]
            )
            self.fact_store.add_fact(fact)
        
        logger.info(f"Loaded {len(nasa_facts)} NASA facts into knowledge base")
    
    async def query(self, query: str, debug: bool = False) -> KBQueryResult:
        """
        Query the knowledge base for factual information.
        
        Args:
            query: The user's query
            debug: Whether to enable debug logging
            
        Returns:
            KBQueryResult with matching facts and reasoning
        """
        logger.info(f"KB Query: {query}")
        
        # Step 1: Parse query to extract entities and predicates
        query_analysis = await self._analyze_query(query)
        
        # Step 2: Search for relevant facts
        matching_facts = self._search_facts(query, query_analysis)
        
        # Step 3: Generate response
        response = await self._generate_response(query, matching_facts, query_analysis)
        
        if debug:
            logger.info(f"Query analysis: {query_analysis}")
            logger.info(f"Found {len(matching_facts)} matching facts")
        
        return response
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to extract entities and intent."""
        # Use LLM to extract entities and intent
        prompt = f"""
        Analyze this NASA-related query and extract:
        1. Main entities (missions, spacecraft, planets, technologies)
        2. Predicates (what information is being asked for)
        3. Query type (factual lookup, comparison, etc.)
        
        Query: {query}
        
        Respond in JSON format with keys: entities, predicates, query_type
        """
        
        try:
            response = await self.llm_manager.generate(prompt)
            # Parse JSON response (simplified)
            analysis = {
                "entities": self._extract_entities_simple(query),
                "predicates": self._extract_predicates_simple(query),
                "query_type": self._classify_query_type(query)
            }
        except Exception as e:
            logger.warning(f"LLM query analysis failed: {e}")
            analysis = {
                "entities": self._extract_entities_simple(query),
                "predicates": self._extract_predicates_simple(query),
                "query_type": self._classify_query_type(query)
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
        
        return entities
    
    def _extract_predicates_simple(self, query: str) -> List[str]:
        """Simple predicate extraction."""
        predicates = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["when", "date", "launch"]):
            predicates.append("launch_date")
        if any(word in query_lower for word in ["what", "name"]):
            predicates.append("name")
        if any(word in query_lower for word in ["technology", "uses", "equipment"]):
            predicates.append("uses_technology")
        if any(word in query_lower for word in ["planet", "studied", "explored"]):
            predicates.append("studied_planet")
        
        return predicates
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of factual query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["when", "date", "time"]):
            return "temporal"
        elif any(word in query_lower for word in ["what", "name", "is"]):
            return "descriptive"
        elif any(word in query_lower for word in ["which", "studied", "used"]):
            return "relational"
        else:
            return "general"
    
    def _search_facts(self, query: str, analysis: Dict[str, Any]) -> List[Fact]:
        """Search for relevant facts based on query analysis."""
        all_facts = self.fact_store.get_all_facts()
        matching_facts = []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        for fact in all_facts:
            # Create fact text for similarity comparison
            fact_text = f"{fact.subject} {fact.predicate} {fact.object}"
            fact_embedding = self.embedding_model.encode([fact_text])[0]
            
            # Calculate similarity using numpy (no FAISS required)
            similarity = np.dot(query_embedding, fact_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(fact_embedding)
            )
            
            # Check if fact matches query criteria
            matches_criteria = self._fact_matches_criteria(fact, analysis)
            
            if similarity >= self.similarity_threshold or matches_criteria:
                matching_facts.append((fact, similarity))
        
        # Sort by similarity and return top results
        matching_facts.sort(key=lambda x: x[1], reverse=True)
        return [fact for fact, _ in matching_facts[:self.max_results]]
    
    def _fact_matches_criteria(self, fact: Fact, analysis: Dict[str, Any]) -> bool:
        """Check if fact matches query criteria."""
        entities = analysis.get("entities", [])
        predicates = analysis.get("predicates", [])
        
        # Check if fact subject matches any entity
        subject_match = any(entity.lower() in fact.subject.lower() for entity in entities)
        
        # Check if fact predicate matches any predicate
        predicate_match = any(pred in fact.predicate for pred in predicates)
        
        # Check if fact object matches any entity
        object_match = any(entity.lower() in fact.object.lower() for entity in entities)
        
        return subject_match or predicate_match or object_match
    
    async def _generate_response(
        self, 
        query: str, 
        facts: List[Fact], 
        analysis: Dict[str, Any]
    ) -> KBQueryResult:
        """Generate a response based on found facts."""
        if not facts:
            return KBQueryResult(
                facts=[],
                confidence=0.0,
                reasoning="No matching facts found in knowledge base",
                query_type=analysis.get("query_type", "unknown"),
                metadata={"analysis": analysis}
            )
        
        # Create fact summary for LLM
        fact_summary = "\n".join([
            f"- {fact.subject} {fact.predicate} {fact.object} (confidence: {fact.confidence})"
            for fact in facts
        ])
        
        prompt = f"""
        Based on these NASA facts, answer the user's question:
        
        User Question: {query}
        
        Relevant Facts:
        {fact_summary}
        
        Provide a clear, factual answer based only on the given facts.
        """
        
        try:
            response_text = await self.llm_manager.generate(prompt)
            confidence = np.mean([fact.confidence for fact in facts])
            
            reasoning = f"Found {len(facts)} relevant facts with average confidence {confidence:.2f}"
            
        except Exception as e:
            logger.error(f"Failed to generate KB response: {e}")
            response_text = "Unable to generate response due to technical issues."
            confidence = 0.0
            reasoning = f"Error generating response: {e}"
        
        return KBQueryResult(
            facts=facts,
            confidence=confidence,
            reasoning=reasoning,
            query_type=analysis.get("query_type", "unknown"),
            metadata={
                "analysis": analysis,
                "response_text": response_text,
                "fact_count": len(facts)
            }
        )
    
    def add_fact(self, fact: Fact):
        """Add a new fact to the knowledge base."""
        self.fact_store.add_fact(fact)
        logger.info(f"Added fact: {fact.subject} {fact.predicate} {fact.object}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        all_facts = self.fact_store.get_all_facts()
        return {
            "total_facts": len(all_facts),
            "subjects": list(set(fact.subject for fact in all_facts)),
            "predicates": list(set(fact.predicate for fact in all_facts)),
            "sources": list(set(fact.source for fact in all_facts))
        } 