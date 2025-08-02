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
        
        # Adjusted thresholds to prioritize KG and RAG over KB
        self.kb_threshold = config.get("kb_threshold", 0.85)  # Higher threshold for KB
        self.kg_threshold = config.get("kg_threshold", 0.5)   # Lower threshold for KG
        self.rag_threshold = config.get("rag_threshold", 0.3) # Lower threshold for RAG
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
            "compare", "similar", "different", "both", "either", "neither",
            "missions that", "spacecraft that", "technologies used by"
        ]
        
        self.rag_patterns = [
            "explain", "describe", "how does", "what is the process",
            "tell me about", "elaborate", "detailed", "comprehensive",
            "analysis", "synthesis", "overview", "summary", "how",
            "why", "what causes", "what led to", "background"
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
        
        # Step 1: Classify query intent
        intent_scores = await self._classify_intent(query)
        
        # Step 2: Make routing decision
        decision = self._make_routing_decision(query, intent_scores)
        
        if debug:
            logger.info(f"Intent scores: {intent_scores}")
            logger.info(f"Routing decision: {decision}")
        
        return decision
    

    
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
        """Fallback intent classification using rule-based approach with KG/RAG prioritization."""
        query_lower = query.lower()
        
        # Rule-based scoring using defined patterns
        factual_score = 0.0
        relational_score = 0.0
        generative_score = 0.0
        
        # Factual indicators (KB patterns) - reduced weight
        for pattern in self.kb_patterns:
            if pattern in query_lower:
                factual_score += 0.2  # Reduced from 0.3
        
        # Relational indicators (KG patterns) - increased weight
        for pattern in self.kg_patterns:
            if pattern in query_lower:
                relational_score += 0.4  # Increased from 0.3
        
        # Generative indicators (RAG patterns) - increased weight
        for pattern in self.rag_patterns:
            if pattern in query_lower:
                generative_score += 0.4  # Increased from 0.3
        
        # Normalize scores
        total = factual_score + relational_score + generative_score
        if total > 0:
            factual_score /= total
            relational_score /= total
            generative_score /= total
        else:
            # Default to RAG for comprehensive answers, then KG, then KB
            factual_score = 0.15   # Reduced from 0.2
            relational_score = 0.35 # Increased from 0.3
            generative_score = 0.5  # Increased from 0.5
        
        return {
            "factual": factual_score,
            "relational": relational_score,
            "generative": generative_score
        }
    
    def _make_routing_decision(
        self, 
        query: str, 
        intent_scores: Dict[str, float]
    ) -> RoutingDecision:
        """Make final routing decision based on analysis with KG/RAG prioritization."""
        
        # Get scores for each intent
        factual_score = intent_scores.get("factual", 0.0)
        relational_score = intent_scores.get("relational", 0.0)
        generative_score = intent_scores.get("generative", 0.0)
        
        # Enhanced routing logic with KG/RAG prioritization
        
        # 1. Check for very high factual confidence (only route to KB for very specific facts)
        if factual_score >= self.kb_threshold and factual_score > (relational_score + 0.2) and factual_score > (generative_score + 0.2):
            query_type = QueryType.FACTUAL
            subsystem = "KB"
            reasoning = f"High factual confidence ({factual_score:.2f}) - specific fact query"
        
        # 2. Check for relational queries (prioritize KG)
        elif relational_score >= self.kg_threshold:
            query_type = QueryType.RELATIONAL
            subsystem = "KG"
            reasoning = f"Relational query detected ({relational_score:.2f}) - using Knowledge Graph"
        
        # 3. Check for generative queries (prioritize RAG)
        elif generative_score >= self.rag_threshold:
            query_type = QueryType.GENERATIVE
            subsystem = "RAG"
            reasoning = f"Generative query detected ({generative_score:.2f}) - using RAG system"
        
        # 4. Mixed confidence scenarios - prioritize KG and RAG
        elif relational_score > 0.3 or generative_score > 0.3:
            # If both KG and RAG have decent scores, prefer the higher one
            if relational_score >= generative_score:
                query_type = QueryType.RELATIONAL
                subsystem = "KG"
                reasoning = f"Mixed confidence - preferring KG ({relational_score:.2f} vs RAG {generative_score:.2f})"
            else:
                query_type = QueryType.GENERATIVE
                subsystem = "RAG"
                reasoning = f"Mixed confidence - preferring RAG ({generative_score:.2f} vs KG {relational_score:.2f})"
        
        # 5. Low confidence scenarios - default to RAG for comprehensive answers
        else:
            query_type = QueryType.GENERATIVE
            subsystem = "RAG"
            reasoning = f"Low confidence across all systems - defaulting to RAG for comprehensive answer"
        
        # Use the highest score for confidence
        confidence = max(factual_score, relational_score, generative_score)
        
        return RoutingDecision(
            query_type=query_type,
            confidence=confidence,
            reasoning=reasoning,
            subsystem=subsystem,
            metadata={
                "intent_scores": intent_scores,
                "query_length": len(query),
                "routing_priority": "KG/RAG prioritized"
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