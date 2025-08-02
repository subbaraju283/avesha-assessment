"""
Data models for the Knowledge Graph module.
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]
    source: str


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]
    source: str


@dataclass
class KGQueryResult:
    """Result of a Knowledge Graph query."""
    entities: List[Entity]
    relationships: List[Relationship]
    paths: List[List[str]]
    confidence: float
    reasoning: str
    query_type: str
    metadata: Dict[str, Any] 