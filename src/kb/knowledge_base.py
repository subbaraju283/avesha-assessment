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
        # Initial facts are now loaded through document ingestion
        # The nasa_initial_knowledge.txt file will be processed during ingestion
        logger.info("Initial facts will be loaded through document ingestion")
        pass
    
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
        # Use LLM to extract entities and intent with structured prompt
        prompt = f"""Analyze this NASA-related query and extract specific information.

Query: "{query}"

Respond ONLY with a JSON object in this exact format:
{{
  "entities": ["entity1", "entity2"],
  "predicates": ["predicate1", "predicate2"],
  "query_type": "temporal|descriptive|relational|general"
}}

Rules:
- entities: List of NASA entities (missions, spacecraft, planets, technologies)
- predicates: List of information types being requested (launch_date, uses_technology, studied_planet, etc.)
- query_type: One of "temporal", "descriptive", "relational", or "general"

Do not include any other text or explanation. Only the JSON object."""
        
        try:
            response = await self.llm_manager.generate(prompt)
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis = json.loads(json_str)
            else:
                # Fallback: try to parse the entire response
                analysis = json.loads(response)
            
            # Validate and ensure required fields
            if "entities" not in analysis:
                analysis["entities"] = []
            if "predicates" not in analysis:
                analysis["predicates"] = []
            if "query_type" not in analysis:
                analysis["query_type"] = "general"
            
            # Ensure entities and predicates are lists
            if not isinstance(analysis["entities"], list):
                analysis["entities"] = [analysis["entities"]] if analysis["entities"] else []
            if not isinstance(analysis["predicates"], list):
                analysis["predicates"] = [analysis["predicates"]] if analysis["predicates"] else []
            
            logger.debug(f"LLM analysis successful: {analysis}")
            
        except Exception as e:
            logger.warning(f"LLM query analysis failed: {e}, using fallback")
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
        """Add a fact to the knowledge base."""
        self.fact_store.add_fact(fact)
    
    async def extract_facts_from_content(self, content: str, source: str) -> List[Fact]:
        """
        Extract facts from document content using LLM.
        Only extracts facts with 100% confidence that are simple and factual.
        
        Args:
            content: Document content to analyze
            source: Source document name
            
        Returns:
            List of extracted facts
        """
        try:
            logger.info(f"Starting fact extraction from {source}")
            
            # Create prompt for fact extraction
            prompt = f"""Analyze the following NASA document content and extract simple, factual statements.
            
            Content: {content[:2000]}  # Limit content length
            
            Extract ONLY facts that are:
            1. 100% certain and verifiable
            2. Simple subject-predicate-object format
            3. Specific dates, names, numbers, or clear relationships
            4. Not opinions, explanations, or complex descriptions
            
            Return ONLY a JSON array of facts in this exact format:
            [
                {{
                    "subject": "entity name",
                    "predicate": "relationship type",
                    "object": "value or target entity",
                    "confidence": 1.0
                }}
            ]
            
            Examples of good facts:
            - {{"subject": "Apollo 11", "predicate": "launch_date", "object": "1969-07-16", "confidence": 1.0}}
            - {{"subject": "Neil Armstrong", "predicate": "mission", "object": "Apollo 11", "confidence": 1.0}}
            - {{"subject": "Voyager 1", "predicate": "uses_technology", "object": "ion propulsion", "confidence": 1.0}}
            
            If no clear facts are found, return an empty array [].
            Do not include any explanations or additional text."""
            
            logger.debug(f"Sending prompt to LLM for fact extraction")
            
            # Get LLM response
            response = await self.llm_manager.generate(prompt)
            
            logger.debug(f"Received LLM response: {response[:200]}...")
            
            # Parse JSON response
            try:
                import json
                facts_data = json.loads(response)
                
                if not isinstance(facts_data, list):
                    logger.warning(f"Invalid facts format from LLM: {response}")
                    return []
                
                logger.info(f"Parsed {len(facts_data)} potential facts from LLM response")
                
                extracted_facts = []
                for i, fact_data in enumerate(facts_data):
                    if not isinstance(fact_data, dict):
                        logger.debug(f"Skipping non-dict fact data: {fact_data}")
                        continue
                    
                    # Validate required fields
                    required_fields = ["subject", "predicate", "object", "confidence"]
                    if not all(field in fact_data for field in required_fields):
                        logger.debug(f"Skipping fact with missing fields: {fact_data}")
                        continue
                    
                    # Only include facts with 100% confidence
                    if fact_data.get("confidence", 0) != 1.0:
                        logger.debug(f"Skipping fact with confidence != 1.0: {fact_data}")
                        continue
                    
                    # Create Fact object
                    fact = Fact(
                        id=f"extracted_{source}_{i}",
                        subject=fact_data["subject"],
                        predicate=fact_data["predicate"],
                        object=fact_data["object"],
                        source=source,
                        confidence=fact_data["confidence"],
                        metadata={"extraction_method": "llm", "source_document": source}
                    )
                    
                    # Check if fact already exists (simple deduplication)
                    existing_facts = self.fact_store.get_all_facts()
                    is_duplicate = False
                    for existing_fact in existing_facts:
                        if (existing_fact.subject == fact.subject and 
                            existing_fact.predicate == fact.predicate and 
                            existing_fact.object == fact.object):
                            is_duplicate = True
                            logger.debug(f"Skipping duplicate fact: {fact.subject} {fact.predicate} {fact.object}")
                            break
                    
                    if not is_duplicate:
                        extracted_facts.append(fact)
                        logger.info(f"Extracted fact: {fact.subject} {fact.predicate} {fact.object}")
                
                logger.info(f"Successfully extracted {len(extracted_facts)} unique facts from {source}")
                return extracted_facts
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                logger.warning(f"Raw response: {response}")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting facts from content: {e}")
            return []
    
    def update_knowledge_base_with_facts(self, facts: List[Fact]):
        """
        Update the knowledge base with extracted facts.
        
        Args:
            facts: List of facts to add to the knowledge base
        """
        for fact in facts:
            self.add_fact(fact)
            logger.info(f"Added fact to KB: {fact.subject} {fact.predicate} {fact.object}")
        
        logger.info(f"Updated Knowledge Base with {len(facts)} new facts")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        all_facts = self.fact_store.get_all_facts()
        return {
            "total_facts": len(all_facts),
            "subjects": list(set(fact.subject for fact in all_facts)),
            "predicates": list(set(fact.predicate for fact in all_facts)),
            "sources": list(set(fact.source for fact in all_facts))
        } 