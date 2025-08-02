"""
Data models for the RAG module.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class DocumentChunk:
    """Represents a chunk of document text."""
    id: str
    content: str
    source: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class RAGQueryResult:
    """Result of a RAG query."""
    answer: str
    retrieved_chunks: List[DocumentChunk]
    confidence: float
    reasoning: str
    query_type: str
    metadata: Dict[str, Any] 