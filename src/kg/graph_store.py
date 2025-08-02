"""
Graph Store for managing and persisting graph data.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from .models import Entity, Relationship

logger = logging.getLogger(__name__)


class GraphStore:
    """Persistent storage for graph entities and relationships."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.entities_file = self.storage_path / "entities.json"
        self.relationships_file = self.storage_path / "relationships.json"
        
        self.entities: List[Entity] = []
        self.relationships: List[Relationship] = []
        self.entity_index: Dict[str, int] = {}  # id -> index mapping
        self.relationship_index: Dict[str, int] = {}  # id -> index mapping
        
        self._load_data()
    
    def _load_data(self):
        """Load entities and relationships from persistent storage."""
        try:
            # Load entities
            if self.entities_file.exists():
                with open(self.entities_file, 'r') as f:
                    entities_data = json.load(f)
                
                self.entities = []
                for entity_data in entities_data:
                    entity = Entity(
                        id=entity_data["id"],
                        name=entity_data["name"],
                        type=entity_data["type"],
                        properties=entity_data["properties"],
                        source=entity_data["source"]
                    )
                    self.entities.append(entity)
                    self.entity_index[entity.id] = len(self.entities) - 1
                
                logger.info(f"Loaded {len(self.entities)} entities from storage")
            else:
                logger.info("No existing entities found")
            
            # Load relationships
            if self.relationships_file.exists():
                with open(self.relationships_file, 'r') as f:
                    relationships_data = json.load(f)
                
                self.relationships = []
                for rel_data in relationships_data:
                    relationship = Relationship(
                        id=rel_data["id"],
                        source_id=rel_data["source_id"],
                        target_id=rel_data["target_id"],
                        type=rel_data["type"],
                        properties=rel_data["properties"],
                        source=rel_data["source"]
                    )
                    self.relationships.append(relationship)
                    self.relationship_index[relationship.id] = len(self.relationships) - 1
                
                logger.info(f"Loaded {len(self.relationships)} relationships from storage")
            else:
                logger.info("No existing relationships found")
                
        except Exception as e:
            logger.error(f"Failed to load graph data: {e}")
            self.entities = []
            self.relationships = []
            self.entity_index = {}
            self.relationship_index = {}
    
    def _save_entities(self):
        """Save entities to persistent storage."""
        try:
            entities_data = []
            for entity in self.entities:
                entity_dict = asdict(entity)
                entities_data.append(entity_dict)
            
            with open(self.entities_file, 'w') as f:
                json.dump(entities_data, f, indent=2)
            
            logger.debug(f"Saved {len(self.entities)} entities to storage")
            
        except Exception as e:
            logger.error(f"Failed to save entities: {e}")
    
    def _save_relationships(self):
        """Save relationships to persistent storage."""
        try:
            relationships_data = []
            for relationship in self.relationships:
                rel_dict = asdict(relationship)
                relationships_data.append(rel_dict)
            
            with open(self.relationships_file, 'w') as f:
                json.dump(relationships_data, f, indent=2)
            
            logger.debug(f"Saved {len(self.relationships)} relationships to storage")
            
        except Exception as e:
            logger.error(f"Failed to save relationships: {e}")
    
    def add_entity(self, entity: Entity):
        """Add a new entity to the store."""
        if entity.id in self.entity_index:
            logger.warning(f"Entity with id {entity.id} already exists, updating")
            # Update existing entity
            index = self.entity_index[entity.id]
            self.entities[index] = entity
        else:
            # Add new entity
            self.entities.append(entity)
            self.entity_index[entity.id] = len(self.entities) - 1
        
        self._save_entities()
    
    def add_relationship(self, relationship: Relationship):
        """Add a new relationship to the store."""
        if relationship.id in self.relationship_index:
            logger.warning(f"Relationship with id {relationship.id} already exists, updating")
            # Update existing relationship
            index = self.relationship_index[relationship.id]
            self.relationships[index] = relationship
        else:
            # Add new relationship
            self.relationships.append(relationship)
            self.relationship_index[relationship.id] = len(self.relationships) - 1
        
        self._save_relationships()
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        if entity_id in self.entity_index:
            return self.entities[self.entity_index[entity_id]]
        return None
    
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Get a relationship by ID."""
        if relationship_id in self.relationship_index:
            return self.relationships[self.relationship_index[relationship_id]]
        return None
    
    def get_all_entities(self) -> List[Entity]:
        """Get all entities in the store."""
        return self.entities.copy()
    
    def get_all_relationships(self) -> List[Relationship]:
        """Get all relationships in the store."""
        return self.relationships.copy()
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        return [entity for entity in self.entities if entity.type == entity_type]
    
    def get_relationships_by_type(self, relationship_type: str) -> List[Relationship]:
        """Get all relationships of a specific type."""
        return [rel for rel in self.relationships if rel.type == relationship_type]
    
    def get_relationships_by_entity(self, entity_id: str) -> List[Relationship]:
        """Get all relationships involving a specific entity."""
        return [
            rel for rel in self.relationships 
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]
    
    def get_connected_entities(self, entity_id: str) -> List[Entity]:
        """Get all entities connected to a specific entity."""
        connected_ids = set()
        
        for rel in self.relationships:
            if rel.source_id == entity_id:
                connected_ids.add(rel.target_id)
            elif rel.target_id == entity_id:
                connected_ids.add(rel.source_id)
        
        return [entity for entity in self.entities if entity.id in connected_ids]
    
    def search_entities(
        self, 
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[Entity]:
        """Search entities by criteria."""
        matching_entities = []
        
        for entity in self.entities:
            matches = True
            
            if name and name.lower() not in entity.name.lower():
                matches = False
            if entity_type and entity.type != entity_type:
                matches = False
            if source and source.lower() not in entity.source.lower():
                matches = False
            
            if matches:
                matching_entities.append(entity)
        
        return matching_entities
    
    def search_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[Relationship]:
        """Search relationships by criteria."""
        matching_relationships = []
        
        for relationship in self.relationships:
            matches = True
            
            if source_id and relationship.source_id != source_id:
                matches = False
            if target_id and relationship.target_id != target_id:
                matches = False
            if relationship_type and relationship.type != relationship_type:
                matches = False
            if source and source.lower() not in relationship.source.lower():
                matches = False
            
            if matches:
                matching_relationships.append(relationship)
        
        return matching_relationships
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity by ID and all its relationships."""
        if entity_id not in self.entity_index:
            return False
        
        # Delete all relationships involving this entity
        relationships_to_delete = []
        for rel in self.relationships:
            if rel.source_id == entity_id or rel.target_id == entity_id:
                relationships_to_delete.append(rel.id)
        
        for rel_id in relationships_to_delete:
            self.delete_relationship(rel_id)
        
        # Delete the entity
        index = self.entity_index[entity_id]
        del self.entities[index]
        
        # Update index for remaining entities
        self.entity_index.clear()
        for i, entity in enumerate(self.entities):
            self.entity_index[entity.id] = i
        
        self._save_entities()
        return True
    
    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship by ID."""
        if relationship_id in self.relationship_index:
            index = self.relationship_index[relationship_id]
            del self.relationships[index]
            
            # Update index for remaining relationships
            self.relationship_index.clear()
            for i, rel in enumerate(self.relationships):
                self.relationship_index[rel.id] = i
            
            self._save_relationships()
            return True
        return False
    
    def update_entity(self, entity_id: str, **kwargs) -> bool:
        """Update an entity by ID with new values."""
        if entity_id not in self.entity_index:
            return False
        
        index = self.entity_index[entity_id]
        entity = self.entities[index]
        
        # Update entity attributes
        for key, value in kwargs.items():
            if hasattr(entity, key):
                setattr(entity, key, value)
        
        self._save_entities()
        return True
    
    def update_relationship(self, relationship_id: str, **kwargs) -> bool:
        """Update a relationship by ID with new values."""
        if relationship_id not in self.relationship_index:
            return False
        
        index = self.relationship_index[relationship_id]
        relationship = self.relationships[index]
        
        # Update relationship attributes
        for key, value in kwargs.items():
            if hasattr(relationship, key):
                setattr(relationship, key, value)
        
        self._save_relationships()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph store."""
        entity_types = list(set(entity.type for entity in self.entities))
        relationship_types = list(set(rel.type for rel in self.relationships))
        sources = list(set(entity.source for entity in self.entities))
        
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "sources": sources
        }
    
    def clear(self):
        """Clear all entities and relationships from the store."""
        self.entities = []
        self.relationships = []
        self.entity_index = {}
        self.relationship_index = {}
        self._save_entities()
        self._save_relationships()
        logger.info("Cleared all entities and relationships from store")
    
    def export_graph(self, filepath: str, format: str = "json"):
        """Export graph data to a file."""
        try:
            if format.lower() == "json":
                graph_data = {
                    "entities": [asdict(entity) for entity in self.entities],
                    "relationships": [asdict(rel) for rel in self.relationships]
                }
                with open(filepath, 'w') as f:
                    json.dump(graph_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported graph with {len(self.entities)} entities and {len(self.relationships)} relationships to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            raise
    
    def import_graph(self, filepath: str, format: str = "json"):
        """Import graph data from a file."""
        try:
            if format.lower() == "json":
                with open(filepath, 'r') as f:
                    graph_data = json.load(f)
                
                # Import entities
                for entity_data in graph_data.get("entities", []):
                    entity = Entity(
                        id=entity_data["id"],
                        name=entity_data["name"],
                        type=entity_data["type"],
                        properties=entity_data["properties"],
                        source=entity_data["source"]
                    )
                    self.add_entity(entity)
                
                # Import relationships
                for rel_data in graph_data.get("relationships", []):
                    relationship = Relationship(
                        id=rel_data["id"],
                        source_id=rel_data["source_id"],
                        target_id=rel_data["target_id"],
                        type=rel_data["type"],
                        properties=rel_data["properties"],
                        source=rel_data["source"]
                    )
                    self.add_relationship(relationship)
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            logger.info(f"Imported graph from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to import graph: {e}")
            raise 