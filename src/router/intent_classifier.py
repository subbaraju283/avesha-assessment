"""
Intent Classifier for analyzing query intent and complexity.
"""

import logging
from typing import Dict, Any, List, Tuple
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IntentAnalysis:
    """Result of intent analysis."""
    primary_intent: str
    confidence: float
    secondary_intents: List[Tuple[str, float]]
    complexity_level: str
    reasoning: str
    entities: List[str]
    relationships: List[str]


class IntentClassifier:
    """Advanced intent classifier for query analysis."""
    
    def __init__(self):
        # Intent patterns and keywords
        self.intent_patterns = {
            "factual": {
                "patterns": [
                    r"what is (.+)",
                    r"when was (.+)",
                    r"who is (.+)",
                    r"where is (.+)",
                    r"how many (.+)",
                    r"what is the (.+) of (.+)",
                    r"when did (.+) launch",
                    r"what is the name of (.+)"
                ],
                "keywords": [
                    "mission", "spacecraft", "date", "time", "name", "number",
                    "launch", "landing", "duration", "distance", "mass", "size"
                ]
            },
            "relational": {
                "patterns": [
                    r"which (.+) (.+)",
                    r"what (.+) (.+)",
                    r"how are (.+) and (.+) related",
                    r"compare (.+) and (.+)",
                    r"what missions (.+) (.+)",
                    r"which missions (.+) (.+)"
                ],
                "keywords": [
                    "studied", "used", "involved", "connected", "related",
                    "compare", "similar", "different", "both", "either",
                    "relationship", "connection", "association"
                ]
            },
            "generative": {
                "patterns": [
                    r"explain (.+)",
                    r"describe (.+)",
                    r"how does (.+) work",
                    r"tell me about (.+)",
                    r"what is the process of (.+)",
                    r"elaborate on (.+)"
                ],
                "keywords": [
                    "explain", "describe", "elaborate", "process", "how",
                    "analysis", "overview", "summary", "detailed", "comprehensive"
                ]
            }
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            "low": ["what", "when", "who", "where", "how many"],
            "medium": ["which", "compare", "similar", "different"],
            "high": ["explain", "describe", "analyze", "elaborate", "comprehensive"]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            "missions": r"\b[A-Z][A-Z0-9\s]+(?:mission|probe|rover|satellite)\b",
            "planets": r"\b(Mars|Venus|Jupiter|Saturn|Neptune|Uranus|Mercury|Pluto)\b",
            "technologies": r"\b(ion propulsion|solar panels|radioisotope|spectrometer|telescope)\b",
            "dates": r"\b\d{4}-\d{2}-\d{2}\b|\b\d{4}\b",
            "locations": r"\b(Kennedy|Cape Canaveral|Jet Propulsion Laboratory|Johnson Space Center)\b"
        }
    
    def analyze_intent(self, query: str) -> IntentAnalysis:
        """
        Analyze query intent and complexity.
        
        Args:
            query: The user's natural language query
            
        Returns:
            IntentAnalysis with detailed intent information
        """
        query_lower = query.lower()
        
        # Analyze primary intent
        intent_scores = self._calculate_intent_scores(query_lower)
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        # Get secondary intents
        secondary_intents = sorted(
            [(intent, score) for intent, score in intent_scores.items() if intent != primary_intent[0]],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Analyze complexity
        complexity_level = self._analyze_complexity(query_lower)
        
        # Extract entities and relationships
        entities = self._extract_entities(query)
        relationships = self._extract_relationships(query_lower)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            primary_intent, intent_scores, complexity_level, entities, relationships
        )
        
        return IntentAnalysis(
            primary_intent=primary_intent[0],
            confidence=primary_intent[1],
            secondary_intents=secondary_intents,
            complexity_level=complexity_level,
            reasoning=reasoning,
            entities=entities,
            relationships=relationships
        )
    
    def _calculate_intent_scores(self, query: str) -> Dict[str, float]:
        """Calculate intent scores based on patterns and keywords."""
        scores = {"factual": 0.0, "relational": 0.0, "generative": 0.0}
        
        for intent, config in self.intent_patterns.items():
            # Pattern matching
            pattern_matches = 0
            for pattern in config["patterns"]:
                if re.search(pattern, query, re.IGNORECASE):
                    pattern_matches += 1
            
            # Keyword matching
            keyword_matches = 0
            for keyword in config["keywords"]:
                if keyword in query:
                    keyword_matches += 1
            
            # Calculate score
            pattern_score = min(pattern_matches / len(config["patterns"]), 1.0)
            keyword_score = min(keyword_matches / len(config["keywords"]), 1.0)
            
            # Weighted combination
            scores[intent] = (pattern_score * 0.7) + (keyword_score * 0.3)
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {intent: score / total for intent, score in scores.items()}
        else:
            # Default to factual if no clear indicators
            scores = {"factual": 0.5, "relational": 0.3, "generative": 0.2}
        
        return scores
    
    def _analyze_complexity(self, query: str) -> str:
        """Analyze query complexity level."""
        words = query.split()
        word_count = len(words)
        
        # Count complexity indicators
        low_complexity = sum(1 for word in self.complexity_indicators["low"] if word in query)
        medium_complexity = sum(1 for word in self.complexity_indicators["medium"] if word in query)
        high_complexity = sum(1 for word in self.complexity_indicators["high"] if word in query)
        
        # Determine complexity level
        if high_complexity > 0 or word_count > 15:
            return "high"
        elif medium_complexity > 0 or word_count > 10:
            return "medium"
        else:
            return "low"
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query."""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_relationships(self, query: str) -> List[str]:
        """Extract relationship indicators from query."""
        relationship_keywords = [
            "studied", "used", "involved", "connected", "related",
            "compare", "similar", "different", "both", "either",
            "relationship", "connection", "association", "between"
        ]
        
        relationships = []
        for keyword in relationship_keywords:
            if keyword in query:
                relationships.append(keyword)
        
        return relationships
    
    def _generate_reasoning(
        self,
        primary_intent: Tuple[str, float],
        intent_scores: Dict[str, float],
        complexity_level: str,
        entities: List[str],
        relationships: List[str]
    ) -> str:
        """Generate reasoning for the intent analysis."""
        reasoning_parts = []
        
        # Primary intent reasoning
        intent_name, confidence = primary_intent
        reasoning_parts.append(f"Primary intent: {intent_name} (confidence: {confidence:.2f})")
        
        # Complexity reasoning
        reasoning_parts.append(f"Complexity level: {complexity_level}")
        
        # Entity reasoning
        if entities:
            reasoning_parts.append(f"Detected entities: {', '.join(entities)}")
        
        # Relationship reasoning
        if relationships:
            reasoning_parts.append(f"Relationship indicators: {', '.join(relationships)}")
        
        # Secondary intents
        if intent_scores:
            secondary_intents = [
                f"{intent}: {score:.2f}" 
                for intent, score in intent_scores.items() 
                if intent != intent_name
            ]
            if secondary_intents:
                reasoning_parts.append(f"Secondary intents: {', '.join(secondary_intents)}")
        
        return ". ".join(reasoning_parts)
    
    def get_intent_patterns(self) -> Dict[str, Any]:
        """Get all intent patterns for debugging."""
        return self.intent_patterns 