"""
Fact Store for managing and persisting facts in the Knowledge Base.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle

from .models import Fact

logger = logging.getLogger(__name__)


class FactStore:
    """Persistent storage for facts in the Knowledge Base."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.facts_file = self.storage_path / "facts.json"
        self.index_file = self.storage_path / "index.pkl"
        
        self.facts: List[Fact] = []
        self.fact_index: Dict[str, int] = {}  # id -> index mapping
        
        self._load_facts()
    
    def _load_facts(self):
        """Load facts from persistent storage."""
        try:
            if self.facts_file.exists():
                with open(self.facts_file, 'r') as f:
                    facts_data = json.load(f)
                
                self.facts = []
                for fact_data in facts_data:
                    fact = Fact(
                        id=fact_data["id"],
                        subject=fact_data["subject"],
                        predicate=fact_data["predicate"],
                        object=fact_data["object"],
                        source=fact_data["source"],
                        confidence=fact_data["confidence"],
                        metadata=fact_data["metadata"]
                    )
                    self.facts.append(fact)
                    self.fact_index[fact.id] = len(self.facts) - 1
                
                logger.info(f"Loaded {len(self.facts)} facts from storage")
            else:
                logger.info("No existing facts found, starting with empty store")
                
        except Exception as e:
            logger.error(f"Failed to load facts: {e}")
            self.facts = []
            self.fact_index = {}
    
    def _save_facts(self):
        """Save facts to persistent storage."""
        try:
            facts_data = []
            for fact in self.facts:
                fact_dict = asdict(fact)
                facts_data.append(fact_dict)
            
            with open(self.facts_file, 'w') as f:
                json.dump(facts_data, f, indent=2)
            
            logger.debug(f"Saved {len(self.facts)} facts to storage")
            
        except Exception as e:
            logger.error(f"Failed to save facts: {e}")
    
    def add_fact(self, fact: Fact):
        """Add a new fact to the store."""
        if fact.id in self.fact_index:
            logger.warning(f"Fact with id {fact.id} already exists, updating")
            # Update existing fact
            index = self.fact_index[fact.id]
            self.facts[index] = fact
        else:
            # Add new fact
            self.facts.append(fact)
            self.fact_index[fact.id] = len(self.facts) - 1
        
        self._save_facts()
    
    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get a fact by ID."""
        if fact_id in self.fact_index:
            return self.facts[self.fact_index[fact_id]]
        return None
    
    def get_all_facts(self) -> List[Fact]:
        """Get all facts in the store."""
        return self.facts.copy()
    
    def search_facts(
        self, 
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[Fact]:
        """Search facts by criteria."""
        matching_facts = []
        
        for fact in self.facts:
            matches = True
            
            if subject and subject.lower() not in fact.subject.lower():
                matches = False
            if predicate and predicate.lower() not in fact.predicate.lower():
                matches = False
            if object and object.lower() not in fact.object.lower():
                matches = False
            if source and source.lower() not in fact.source.lower():
                matches = False
            
            if matches:
                matching_facts.append(fact)
        
        return matching_facts
    
    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact by ID."""
        if fact_id in self.fact_index:
            index = self.fact_index[fact_id]
            del self.facts[index]
            
            # Update index for remaining facts
            self.fact_index.clear()
            for i, fact in enumerate(self.facts):
                self.fact_index[fact.id] = i
            
            self._save_facts()
            return True
        return False
    
    def update_fact(self, fact_id: str, **kwargs) -> bool:
        """Update a fact by ID with new values."""
        if fact_id not in self.fact_index:
            return False
        
        index = self.fact_index[fact_id]
        fact = self.facts[index]
        
        # Update fact attributes
        for key, value in kwargs.items():
            if hasattr(fact, key):
                setattr(fact, key, value)
        
        self._save_facts()
        return True
    
    def get_facts_by_subject(self, subject: str) -> List[Fact]:
        """Get all facts for a specific subject."""
        return [fact for fact in self.facts if subject.lower() in fact.subject.lower()]
    
    def get_facts_by_predicate(self, predicate: str) -> List[Fact]:
        """Get all facts for a specific predicate."""
        return [fact for fact in self.facts if predicate.lower() in fact.predicate.lower()]
    
    def get_facts_by_source(self, source: str) -> List[Fact]:
        """Get all facts from a specific source."""
        return [fact for fact in self.facts if source.lower() in fact.source.lower()]
    
    def get_high_confidence_facts(self, threshold: float = 0.8) -> List[Fact]:
        """Get facts with confidence above threshold."""
        return [fact for fact in self.facts if fact.confidence >= threshold]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the fact store."""
        if not self.facts:
            return {
                "total_facts": 0,
                "subjects": [],
                "predicates": [],
                "sources": [],
                "avg_confidence": 0.0
            }
        
        subjects = list(set(fact.subject for fact in self.facts))
        predicates = list(set(fact.predicate for fact in self.facts))
        sources = list(set(fact.source for fact in self.facts))
        avg_confidence = sum(fact.confidence for fact in self.facts) / len(self.facts)
        
        return {
            "total_facts": len(self.facts),
            "subjects": subjects,
            "predicates": predicates,
            "sources": sources,
            "avg_confidence": avg_confidence
        }
    
    def clear(self):
        """Clear all facts from the store."""
        self.facts = []
        self.fact_index = {}
        self._save_facts()
        logger.info("Cleared all facts from store")
    
    def export_facts(self, filepath: str, format: str = "json"):
        """Export facts to a file."""
        try:
            if format.lower() == "json":
                facts_data = [asdict(fact) for fact in self.facts]
                with open(filepath, 'w') as f:
                    json.dump(facts_data, f, indent=2)
            elif format.lower() == "csv":
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["id", "subject", "predicate", "object", "source", "confidence"])
                    for fact in self.facts:
                        writer.writerow([
                            fact.id, fact.subject, fact.predicate, 
                            fact.object, fact.source, fact.confidence
                        ])
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported {len(self.facts)} facts to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export facts: {e}")
            raise
    
    def import_facts(self, filepath: str, format: str = "json"):
        """Import facts from a file."""
        try:
            if format.lower() == "json":
                with open(filepath, 'r') as f:
                    facts_data = json.load(f)
                
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
                    self.add_fact(fact)
                    
            elif format.lower() == "csv":
                import csv
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fact = Fact(
                            id=row["id"],
                            subject=row["subject"],
                            predicate=row["predicate"],
                            object=row["object"],
                            source=row["source"],
                            confidence=float(row["confidence"]),
                            metadata={}
                        )
                        self.add_fact(fact)
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            logger.info(f"Imported facts from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to import facts: {e}")
            raise 