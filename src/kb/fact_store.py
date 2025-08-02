"""
Fact Store for managing structured facts in the Knowledge Base.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from .models import Fact

logger = logging.getLogger(__name__)


class FactStore:
    """Persistent storage for structured facts."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.facts_file = storage_path / "facts.json"
        self.facts: List[Fact] = []
        self.fact_index: Dict[str, int] = {}  # fact_id -> index in facts list
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing facts
        self._load_facts()
    
    def _load_facts(self):
        """Load facts from persistent storage."""
        try:
            if self.facts_file.exists():
                with open(self.facts_file, 'r') as f:
                    facts_data = json.load(f)
                
                self.facts = []
                self.fact_index = {}
                
                for fact_data in facts_data:
                    fact = Fact(
                        id=fact_data["id"],
                        subject=fact_data["subject"],
                        predicate=fact_data["predicate"],
                        object=fact_data["object"],
                        source=fact_data["source"],
                        confidence=fact_data["confidence"],
                        metadata=fact_data.get("metadata", {})
                    )
                    self.facts.append(fact)
                    self.fact_index[fact.id] = len(self.facts) - 1
                
                logger.info(f"Loaded {len(self.facts)} facts from {self.facts_file}")
            else:
                logger.info(f"No existing facts file found at {self.facts_file}")
                
        except Exception as e:
            logger.error(f"Failed to load facts: {e}")
            self.facts = []
            self.fact_index = {}
    
    def add_fact(self, fact: Fact):
        """Add a fact to the store."""
        logger.info(f"Adding fact to store: {fact.subject} {fact.predicate} {fact.object}")
        
        if fact.id in self.fact_index:
            # Update existing fact
            index = self.fact_index[fact.id]
            self.facts[index] = fact
            logger.debug(f"Updated existing fact {fact.id}")
        else:
            # Add new fact
            self.facts.append(fact)
            self.fact_index[fact.id] = len(self.facts) - 1
            logger.debug(f"Added new fact {fact.id}")
        
        self._save_facts()
        logger.info(f"Saved fact {fact.id} to {self.facts_file}")
    
    def _save_facts(self):
        """Save facts to persistent storage."""
        try:
            facts_data = [asdict(fact) for fact in self.facts]
            with open(self.facts_file, 'w') as f:
                json.dump(facts_data, f, indent=2)
            logger.info(f"Saved {len(self.facts)} facts to {self.facts_file}")
        except Exception as e:
            logger.error(f"Failed to save facts: {e}")
    
    def get_all_facts(self) -> List[Fact]:
        """Get all facts in the store."""
        return self.facts.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the fact store."""
        if not self.facts:
            return {
                "total_facts": 0,
                "sources": [],
                "avg_confidence": 0.0,
                "predicates": []
            }
        
        sources = list(set(fact.source for fact in self.facts))
        predicates = list(set(fact.predicate for fact in self.facts))
        total_facts = len(self.facts)
        avg_confidence = sum(fact.confidence for fact in self.facts) / total_facts
        
        return {
            "total_facts": total_facts,
            "sources": sources,
            "avg_confidence": avg_confidence,
            "predicates": predicates
        } 