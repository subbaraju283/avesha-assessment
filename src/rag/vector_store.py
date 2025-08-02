"""
Vector Store for managing document chunks and embeddings.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle
import numpy as np

from .models import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent storage for document chunks and their embeddings."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.chunks_file = self.storage_path / "chunks.json"
        self.embeddings_file = self.storage_path / "embeddings.pkl"
        
        self.chunks: List[DocumentChunk] = []
        self.chunk_index: Dict[str, int] = {}  # id -> index mapping
        
        self._load_chunks()
    
    def _load_chunks(self):
        """Load chunks from persistent storage."""
        try:
            if self.chunks_file.exists():
                with open(self.chunks_file, 'r') as f:
                    chunks_data = json.load(f)
                
                self.chunks = []
                for chunk_data in chunks_data:
                    chunk = DocumentChunk(
                        id=chunk_data["id"],
                        content=chunk_data["content"],
                        source=chunk_data["source"],
                        metadata=chunk_data["metadata"],
                        embedding=None  # Will be loaded separately
                    )
                    self.chunks.append(chunk)
                    self.chunk_index[chunk.id] = len(self.chunks) - 1
                
                # Load embeddings if they exist
                if self.embeddings_file.exists():
                    with open(self.embeddings_file, 'rb') as f:
                        embeddings_data = pickle.load(f)
                    
                    for chunk_id, embedding in embeddings_data.items():
                        if chunk_id in self.chunk_index:
                            chunk_index = self.chunk_index[chunk_id]
                            self.chunks[chunk_index].embedding = embedding
                
                logger.info(f"Loaded {len(self.chunks)} chunks from storage")
            else:
                logger.info("No existing chunks found")
                
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")
            self.chunks = []
            self.chunk_index = {}
    
    def _save_chunks(self):
        """Save chunks to persistent storage."""
        try:
            chunks_data = []
            for chunk in self.chunks:
                chunk_dict = asdict(chunk)
                # Don't save embeddings in JSON
                chunk_dict["embedding"] = None
                chunks_data.append(chunk_dict)
            
            with open(self.chunks_file, 'w') as f:
                json.dump(chunks_data, f, indent=2)
            
            logger.debug(f"Saved {len(self.chunks)} chunks to storage")
            
        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")
    
    def _save_embeddings(self):
        """Save embeddings to persistent storage."""
        try:
            embeddings_data = {}
            for chunk in self.chunks:
                if chunk.embedding is not None:
                    embeddings_data[chunk.id] = chunk.embedding
            
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embeddings_data, f)
            
            logger.debug(f"Saved {len(embeddings_data)} embeddings to storage")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
    
    def add_chunk(self, chunk: DocumentChunk):
        """Add a new chunk to the store."""
        if chunk.id in self.chunk_index:
            logger.warning(f"Chunk with id {chunk.id} already exists, updating")
            # Update existing chunk
            index = self.chunk_index[chunk.id]
            self.chunks[index] = chunk
        else:
            # Add new chunk
            self.chunks.append(chunk)
            self.chunk_index[chunk.id] = len(self.chunks) - 1
        
        self._save_chunks()
        if chunk.embedding is not None:
            self._save_embeddings()
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a chunk by ID."""
        if chunk_id in self.chunk_index:
            return self.chunks[self.chunk_index[chunk_id]]
        return None
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        """Get all chunks in the store."""
        return self.chunks.copy()
    
    def get_chunks_by_source(self, source: str) -> List[DocumentChunk]:
        """Get all chunks from a specific source."""
        return [chunk for chunk in self.chunks if chunk.source == source]
    
    def search_chunks(
        self, 
        content: Optional[str] = None,
        source: Optional[str] = None,
        metadata_key: Optional[str] = None,
        metadata_value: Optional[Any] = None
    ) -> List[DocumentChunk]:
        """Search chunks by criteria."""
        matching_chunks = []
        
        for chunk in self.chunks:
            matches = True
            
            if content and content.lower() not in chunk.content.lower():
                matches = False
            if source and chunk.source != source:
                matches = False
            if metadata_key and metadata_value:
                if metadata_key not in chunk.metadata or chunk.metadata[metadata_key] != metadata_value:
                    matches = False
            
            if matches:
                matching_chunks.append(chunk)
        
        return matching_chunks
    
    def update_chunk_embedding(self, chunk_id: str, embedding: List[float]):
        """Update the embedding for a specific chunk."""
        if chunk_id in self.chunk_index:
            index = self.chunk_index[chunk_id]
            self.chunks[index].embedding = embedding
            self._save_embeddings()
        else:
            logger.warning(f"Chunk {chunk_id} not found for embedding update")
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk by ID."""
        if chunk_id in self.chunk_index:
            index = self.chunk_index[chunk_id]
            del self.chunks[index]
            
            # Update index for remaining chunks
            self.chunk_index.clear()
            for i, chunk in enumerate(self.chunks):
                self.chunk_index[chunk.id] = i
            
            self._save_chunks()
            self._save_embeddings()
            return True
        return False
    
    def update_chunk(self, chunk_id: str, **kwargs) -> bool:
        """Update a chunk by ID with new values."""
        if chunk_id not in self.chunk_index:
            return False
        
        index = self.chunk_index[chunk_id]
        chunk = self.chunks[index]
        
        # Update chunk attributes
        for key, value in kwargs.items():
            if hasattr(chunk, key):
                setattr(chunk, key, value)
        
        self._save_chunks()
        return True
    
    def get_chunks_with_embeddings(self) -> List[DocumentChunk]:
        """Get all chunks that have embeddings."""
        return [chunk for chunk in self.chunks if chunk.embedding is not None]
    
    def get_chunks_without_embeddings(self) -> List[DocumentChunk]:
        """Get all chunks that don't have embeddings."""
        return [chunk for chunk in self.chunks if chunk.embedding is None]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if not self.chunks:
            return {
                "total_chunks": 0,
                "sources": [],
                "avg_chunk_length": 0,
                "chunks_with_embeddings": 0
            }
        
        sources = list(set(chunk.source for chunk in self.chunks))
        total_chunks = len(self.chunks)
        chunks_with_embeddings = len([c for c in self.chunks if c.embedding is not None])
        avg_chunk_length = sum(len(chunk.content.split()) for chunk in self.chunks) / total_chunks
        
        return {
            "total_chunks": total_chunks,
            "sources": sources,
            "avg_chunk_length": avg_chunk_length,
            "chunks_with_embeddings": chunks_with_embeddings
        }
    
    def clear(self):
        """Clear all chunks from the store."""
        self.chunks = []
        self.chunk_index = {}
        self._save_chunks()
        self._save_embeddings()
        logger.info("Cleared all chunks from store")
    
    def export_chunks(self, filepath: str, format: str = "json"):
        """Export chunks to a file."""
        try:
            if format.lower() == "json":
                chunks_data = []
                for chunk in self.chunks:
                    chunk_dict = asdict(chunk)
                    # Convert embedding to list for JSON serialization
                    if chunk.embedding is not None:
                        chunk_dict["embedding"] = chunk.embedding.tolist() if isinstance(chunk.embedding, np.ndarray) else chunk.embedding
                    chunks_data.append(chunk_dict)
                
                with open(filepath, 'w') as f:
                    json.dump(chunks_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported {len(self.chunks)} chunks to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export chunks: {e}")
            raise
    
    def import_chunks(self, filepath: str, format: str = "json"):
        """Import chunks from a file."""
        try:
            if format.lower() == "json":
                with open(filepath, 'r') as f:
                    chunks_data = json.load(f)
                
                for chunk_data in chunks_data:
                    # Convert embedding back to numpy array if present
                    embedding = chunk_data.get("embedding")
                    if embedding is not None:
                        embedding = np.array(embedding) if isinstance(embedding, list) else embedding
                    
                    chunk = DocumentChunk(
                        id=chunk_data["id"],
                        content=chunk_data["content"],
                        source=chunk_data["source"],
                        metadata=chunk_data["metadata"],
                        embedding=embedding
                    )
                    self.add_chunk(chunk)
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            logger.info(f"Imported chunks from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to import chunks: {e}")
            raise 