"""
Document ingestion pipeline for processing new documents into the system.
"""

from .ingestion_pipeline import IngestionPipeline
from .document_processor import DocumentProcessor

__all__ = ["IngestionPipeline", "DocumentProcessor"] 