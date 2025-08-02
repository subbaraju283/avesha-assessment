"""
Query Router for intelligently routing queries to appropriate subsystems.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries that can be handled."""
    FACTUAL = "factual"  # KB
    RELATIONAL = "relational"  # KG
    GENERATIVE = "generative"  # RAG


@dataclass
class RoutingDecision:
    """Result of query routing decision."""
    query_type: QueryType
    confidence: float
    reasoning: str
    subsystem: str
    metadata: Dict[str, Any]


class QueryRouter:
    """Intelligent query router for NASA documentation system."""
    
    def __init__(self, config: Dict[str, Any], llm_manager: LLMManager):
        self.config = config
        self.llm_manager = llm_manager
        self.kb_threshold = config.get("kb_threshold", 0.7)
        self.kg_threshold = config.get("kg_threshold", 0.6)
        self.rag_threshold = config.get("rag_threshold", 0.4)
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        
        # Query patterns for classification
        self.kb_patterns = [
            "what is", "when was", "who is", "where is", "how many",
            "mission date", "launch date", "spacecraft name", "mission name",
            "exact", "specific", "precise", "number", "date", "time"
        ]
        
        self.kg_patterns = [
            "which missions", "what missions", "how are", "relationship between",
            "connected to", "related to", "studied", "used", "involved",
            "compare", "similar", "different", "both", "either", "neither"
        ]
        
        self.rag_patterns = [
            "explain", "describe", "how does", "what is the process",
            "tell me about", "elaborate", "detailed", "comprehensive",
            "analysis", "synthesis", "overview", "summary"
        ]
    
    async def route_query(self, query: str, debug: bool = False) -> RoutingDecision:
        """
        Route a query to the appropriate subsystem.
        
        Args:
            query: The user's natural language query
            debug: Whether to enable debug logging
            
        Returns:
            RoutingDecision with subsystem choice and reasoning
        """
        logger.info(f"Routing query: {query}")
        
        # Step 1: Analyze query characteristics
        characteristics = self._analyze_query_characteristics(query)
        
        # Step 2: Classify query intent
        intent_scores = await self._classify_intent(query)
        
        # Step 3: Make routing decision
        decision = self._make_routing_decision(query, characteristics, intent_scores)
        
        if debug:
            logger.info(f"Query characteristics: {characteristics}")
            logger.info(f"Intent scores: {intent_scores}")
            logger.info(f"Routing decision: {decision}")
        
        return decision
    
    def _analyze_query_characteristics(self, query: str) -> Dict[str, Any]:
        """Analyze query characteristics for routing."""
        query_lower = query.lower()
        
        # Check for pattern matches
        kb_matches = sum(1 for pattern in self.kb_patterns if pattern in query_lower)
        kg_matches = sum(1 for pattern in self.kg_patterns if pattern in query_lower)
        rag_matches = sum(1 for pattern in self.rag_patterns if pattern in query_lower)
        
        # Analyze query complexity
        word_count = len(query.split())
        has_question_words = any(word in query_lower for word in ["what", "when", "where", "who", "how", "why", "which"])
        has_comparison_words = any(word in query_lower for word in ["and", "or", "but", "while", "however"])
        
        return {
            "word_count": word_count,
            "has_question_words": has_question_words,
            "has_comparison_words": has_comparison_words,
            "kb_pattern_matches": kb_matches,
            "kg_pattern_matches": kg_matches,
            "rag_pattern_matches": rag_matches,
            "complexity_score": self._calculate_complexity_score(query)
        }
    
    def _calculate_complexity_score(self, query: str) -> float:
        """Calculate query complexity score."""
        words = query.split()
        word_count = len(words)
        
        # Simple complexity based on word count and special words
        complexity = min(word_count / 10.0, 1.0)  # Normalize to 0-1
        
        # Boost complexity for certain patterns
        if any(word in query.lower() for word in ["compare", "analyze", "explain", "describe"]):
            complexity += 0.3
        
        if any(word in query.lower() for word in ["and", "or", "but", "while", "however"]):
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    async def _classify_intent(self, query: str) -> Dict[str, float]:
        """Classify query intent using LLM."""
        categories = ["factual", "relational", "generative"]
        
        try:
            scores = await self.llm_manager.classify(query, categories)
            return scores
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, using fallback")
            return self._fallback_intent_classification(query)
    
    def _fallback_intent_classification(self, query: str) -> Dict[str, float]:
        """Fallback intent classification using rule-based approach."""
        query_lower = query.lower()
        
        # Rule-based scoring
        factual_score = 0.0
        relational_score = 0.0
        generative_score = 0.0
        
        # Factual indicators
        if any(word in query_lower for word in ["what is", "when", "who", "where", "how many"]):
            factual_score += 0.4
        if any(word in query_lower for word in ["mission", "spacecraft", "date", "name"]):
            factual_score += 0.3
        
        # Relational indicators
        if any(word in query_lower for word in ["which", "studied", "used", "involved", "relationship"]):
            relational_score += 0.4
        if any(word in query_lower for word in ["and", "or", "both", "either"]):
            relational_score += 0.3
        
        # Generative indicators
        if any(word in query_lower for word in ["explain", "describe", "how does", "tell me about"]):
            generative_score += 0.4
        if any(word in query_lower for word in ["process", "analysis", "overview"]):
            generative_score += 0.3
        
        # Normalize scores
        total = factual_score + relational_score + generative_score
        if total > 0:
            factual_score /= total
            relational_score /= total
            generative_score /= total
        else:
            # Default to factual if no clear indicators
            factual_score = 0.5
            relational_score = 0.3
            generative_score = 0.2
        
        return {
            "factual": factual_score,
            "relational": relational_score,
            "generative": generative_score
        }
    
    def _make_routing_decision(
        self, 
        query: str, 
        characteristics: Dict[str, Any], 
        intent_scores: Dict[str, float]
    ) -> RoutingDecision:
        """Make final routing decision based on analysis."""
        
        # Get the highest scoring intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        confidence = best_intent[1]
        
        # Determine query type and subsystem
        if best_intent[0] == "factual" and confidence >= self.kb_threshold:
            query_type = QueryType.FACTUAL
            subsystem = "KB"
            reasoning = f"Query classified as factual with {confidence:.2f} confidence"
        elif best_intent[0] == "relational" and confidence >= self.kg_threshold:
            query_type = QueryType.RELATIONAL
            subsystem = "KG"
            reasoning = f"Query classified as relational with {confidence:.2f} confidence"
        elif best_intent[0] == "generative" and confidence >= self.rag_threshold:
            query_type = QueryType.GENERATIVE
            subsystem = "RAG"
            reasoning = f"Query classified as generative with {confidence:.2f} confidence"
        else:
            # Fallback to RAG for complex queries
            query_type = QueryType.GENERATIVE
            subsystem = "RAG"
            reasoning = f"Fallback to RAG due to low confidence ({confidence:.2f})"
        
        # Additional reasoning based on characteristics
        if characteristics["complexity_score"] > 0.7:
            reasoning += f". High complexity ({characteristics['complexity_score']:.2f})"
        
        if characteristics["word_count"] > 15:
            reasoning += f". Long query ({characteristics['word_count']} words)"
        
        return RoutingDecision(
            query_type=query_type,
            confidence=confidence,
            reasoning=reasoning,
            subsystem=subsystem,
            metadata={
                "characteristics": characteristics,
                "intent_scores": intent_scores,
                "query_length": len(query)
            }
        )
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics and configuration."""
        return {
            "thresholds": {
                "kb": self.kb_threshold,
                "kg": self.kg_threshold,
                "rag": self.rag_threshold,
                "confidence": self.confidence_threshold
            },
            "patterns": {
                "kb_patterns": self.kb_patterns,
                "kg_patterns": self.kg_patterns,
                "rag_patterns": self.rag_patterns
            }
        } 