"""
Data models for the Knowledge Base module.
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class Fact:
    """Represents a factual piece of information."""
    id: str
    subject: str
    predicate: str
    object: str
    source: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class KBQueryResult:
    """Result of a Knowledge Base query."""
    facts: List[Fact]
    confidence: float
    reasoning: str
    query_type: str
    metadata: Dict[str, Any] 