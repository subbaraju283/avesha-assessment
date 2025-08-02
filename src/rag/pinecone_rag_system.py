"""
Pinecone-based RAG System for generative answers and synthesis from document context.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader

from ..models.llm_manager import LLMManager
from .models import DocumentChunk, RAGQueryResult

logger = logging.getLogger(__name__)


class PineconeRAGSystem:
    """Pinecone-based RAG System for generative answers with NASA documentation."""
    
    def __init__(self, config: Dict[str, Any], llm_manager: LLMManager):
        self.config = config
        self.llm_manager = llm_manager
        self.chunk_size = config.get("chunk_size", 500)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.retrieval_k = config.get("retrieval_k", 5)
        self.rerank_top_k = config.get("rerank_top_k", 3)
        
        # Pinecone configuration
        self.index_name = config.get("pinecone_index_name", "nasa-docs")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        print("pinecone_api_key: ", self.pinecone_api_key)
        
        # Initialize Pinecone
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key must be provided")
        
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Initialize embedding model
        self.embedder = OpenAIEmbeddings()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Get or create Pinecone index
        self._setup_index()
        
        logger.info("Pinecone RAG System initialized")
    
    def _setup_index(self):
        """Setup Pinecone index."""
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                # Create index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Created Pinecone index: {self.index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
            
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            logger.error(f"Failed to setup Pinecone index: {e}")
            raise
    
    def _load_documents(self, file_path: str) -> List:
        """Load documents from file."""
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            elif file_path.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            return loader.load()
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            raise
    
    def _split_documents(self, documents: List) -> List:
        """Split documents into chunks."""
        try:
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Failed to split documents: {e}")
            raise
    
    def _embed_chunks(self, chunks: List) -> List[List[float]]:
        """Embed document chunks."""
        try:
            texts = [chunk.page_content for chunk in chunks]
            return self.embedder.embed_documents(texts)
        except Exception as e:
            logger.error(f"Failed to embed chunks: {e}")
            raise
    
    def _upsert_to_pinecone(self, vectors: List[List[float]], chunks: List, doc_id: str):
        """Upsert vectors to Pinecone, ensuring no duplicates."""
        try:
            # Step 1: Delete any existing vectors for this document to prevent duplicates
            self._delete_existing_vectors(doc_id)
            
            # Step 2: Remove any content-based duplicates from new vectors
            vectors, chunks = self._remove_content_duplicates(vectors, chunks, doc_id)
            
            # Step 3: Construct Pinecone vectors
            pinecone_vectors = [
                {
                    "id": f"{doc_id}-chunk-{i}",
                    "values": vector,
                    "metadata": {
                        "text": chunks[i].page_content,
                        "source": doc_id,
                        "chunk_index": i
                    }
                }
                for i, vector in enumerate(vectors)
            ]
            
            # Step 4: Upsert in batches
            batch_size = 100
            for i in range(0, len(pinecone_vectors), batch_size):
                batch = pinecone_vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Upserted {len(pinecone_vectors)} vectors to Pinecone for {doc_id} (duplicates removed)")
            
        except Exception as e:
            logger.error(f"Failed to upsert to Pinecone: {e}")
            raise
    
    def _remove_content_duplicates(self, vectors: List[List[float]], chunks: List, doc_id: str) -> Tuple[List[List[float]], List]:
        """Remove duplicate vectors based on content similarity."""
        try:
            if len(vectors) <= 1:
                return vectors, chunks
            
            # Calculate cosine similarity between all pairs
            unique_vectors = []
            unique_chunks = []
            duplicate_count = 0
            
            for i, (vector, chunk) in enumerate(zip(vectors, chunks)):
                is_duplicate = False
                
                # Check against already selected vectors
                for existing_vector, existing_chunk in zip(unique_vectors, unique_chunks):
                    # Calculate cosine similarity
                    similarity = np.dot(vector, existing_vector) / (
                        np.linalg.norm(vector) * np.linalg.norm(existing_vector)
                    )
                    
                    # If similarity is very high (> 0.95), consider it a duplicate
                    if similarity > 0.95:
                        is_duplicate = True
                        duplicate_count += 1
                        logger.debug(f"Removed duplicate chunk {i} for document {doc_id} (similarity: {similarity:.3f})")
                        break
                
                if not is_duplicate:
                    unique_vectors.append(vector)
                    unique_chunks.append(chunk)
            
            if duplicate_count > 0:
                logger.info(f"Removed {duplicate_count} content-based duplicates from document {doc_id}")
            
            return unique_vectors, unique_chunks
            
        except Exception as e:
            logger.warning(f"Failed to remove content duplicates: {e}")
            return vectors, chunks
    
    def _delete_existing_vectors(self, doc_id: str):
        """Delete existing vectors for a document to prevent duplicates."""
        try:
            # Query to find all vectors for this document
            query_response = self.index.query(
                vector=[0] * 1536,  # Dummy vector for filtering
                top_k=10000,
                include_metadata=True,
                filter={"source": doc_id}
            )
            
            # Delete all vectors for this document
            vector_ids = [match.id for match in query_response.matches]
            if vector_ids:
                self.index.delete(ids=vector_ids)
                logger.info(f"Deleted {len(vector_ids)} existing vectors for document {doc_id} to prevent duplicates")
            else:
                logger.debug(f"No existing vectors found for document {doc_id}")
                
        except Exception as e:
            logger.warning(f"Failed to delete existing vectors for {doc_id}: {e}")
            # Continue with upsert even if deletion fails
    
    def add_document(self, content: str, doc_id: str, metadata: Dict[str, Any] = None):
        """Add a document to the Pinecone index."""
        try:
            # Create a temporary file to use with LangChain loaders
            import tempfile
            import os
            
            # Determine file extension from metadata or default to txt
            file_extension = metadata.get("file_extension", ".txt") if metadata else ".txt"
            temp_file_path = f"/tmp/{doc_id}{file_extension}"
            
            # Write content to temporary file
            with open(temp_file_path, 'w') as f:
                f.write(content)
            
            # Load documents
            documents = self._load_documents(temp_file_path)
            
            # Split into chunks
            chunks = self._split_documents(documents)
            
            # Embed chunks
            vectors = self._embed_chunks(chunks)
            
            # Upsert to Pinecone
            self._upsert_to_pinecone(vectors, chunks, doc_id)
            
            # Clean up temporary file
            os.remove(temp_file_path)
            
            logger.info(f"Successfully added document {doc_id} to Pinecone")
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            raise
    
    async def query(self, query: str, debug: bool = False) -> RAGQueryResult:
        """
        Query the Pinecone RAG system for generative answers.
        
        Args:
            query: The user's query
            debug: Whether to enable debug logging
            
        Returns:
            RAGQueryResult with answer and retrieved chunks
        """
        logger.info(f"Pinecone RAG Query: {query}")
        
        try:
            # Step 1: Generate query embedding
            query_embedding = self.embedder.embed_query(query)
            
            # Step 2: Query Pinecone
            query_response = self.index.query(
                vector=query_embedding,
                top_k=self.retrieval_k,
                include_metadata=True
            )
            
            # Step 3: Extract retrieved chunks
            retrieved_chunks = []
            for match in query_response.matches:
                chunk = DocumentChunk(
                    id=match.id,
                    content=match.metadata.get("text", ""),
                    source=match.metadata.get("source", ""),
                    metadata=match.metadata,
                    embedding=None  # Not needed for query result
                )
                retrieved_chunks.append(chunk)
            
            # Step 4: Rerank chunks (optional)
            if len(retrieved_chunks) > self.rerank_top_k:
                retrieved_chunks = self._rerank_chunks(query, retrieved_chunks)
            
            # Step 5: Generate answer
            answer = await self._generate_answer(query, retrieved_chunks)
            
            if debug:
                logger.info(f"Retrieved {len(retrieved_chunks)} chunks from Pinecone")
                logger.info(f"Generated answer: {answer[:100]}...")
            
            return RAGQueryResult(
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                confidence=0.8 if retrieved_chunks else 0.3,
                reasoning=f"Retrieved {len(retrieved_chunks)} relevant document chunks from Pinecone",
                query_type="generative",
                metadata={
                    "query_embedding_length": len(query_embedding),
                    "chunk_count": len(retrieved_chunks),
                    "pinecone_index": self.index_name
                }
            )
            
        except Exception as e:
            logger.error(f"Pinecone RAG query failed: {e}")
            return RAGQueryResult(
                answer=f"Error querying Pinecone RAG system: {e}",
                retrieved_chunks=[],
                confidence=0.0,
                reasoning=f"Query failed: {e}",
                query_type="error",
                metadata={"error": str(e)}
            )
    
    def _rerank_chunks(self, query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Rerank chunks based on relevance to query."""
        try:
            # Simple reranking based on cosine similarity
            def chunk_score(chunk):
                chunk_embedding = self.embedder.embed_query(chunk.content)
                query_embedding = self.embedder.embed_query(query)
                return np.dot(chunk_embedding, query_embedding) / (
                    np.linalg.norm(chunk_embedding) * np.linalg.norm(query_embedding)
                )
            
            # Sort by score and return top k
            scored_chunks = [(chunk, chunk_score(chunk)) for chunk in chunks]
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            return [chunk for chunk, score in scored_chunks[:self.rerank_top_k]]
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return chunks[:self.rerank_top_k]
    
    async def _generate_answer(self, query: str, chunks: List[DocumentChunk]) -> str:
        """Generate answer using LLM."""
        try:
            if not chunks:
                return "I don't have enough information to answer this question."
            
            # Prepare context from chunks
            context = "\n\n".join([chunk.content for chunk in chunks])
            
            # Generate prompt
            prompt = f"""
            Answer the following question based on the provided context. 
            Only use information from the context. If the context doesn't contain 
            enough information to answer the question, say so.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:
            """
            
            # Generate answer using LLM
            response = await self.llm_manager.generate(prompt)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"Error generating answer: {e}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        try:
            index_stats = self.index.describe_index_stats()
            
            # Get duplicate statistics
            duplicate_stats = self._get_duplicate_stats()
            
            return {
                "index_name": self.index_name,
                "total_vector_count": index_stats.total_vector_count,
                "dimension": index_stats.dimension,
                "metric": index_stats.metric,
                "namespaces": index_stats.namespaces,
                "duplicate_stats": duplicate_stats
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {e}")
            return {"error": str(e)}
    
    def _get_duplicate_stats(self) -> Dict[str, Any]:
        """Get statistics about potential duplicates in the index."""
        try:
            # Query all vectors to analyze
            query_response = self.index.query(
                vector=[0] * 1536,  # Dummy vector
                top_k=10000,
                include_metadata=True
            )
            
            if not query_response.matches:
                return {"total_vectors": 0, "potential_duplicates": 0, "duplicate_groups": 0}
            
            # Group by source document
            doc_groups = {}
            for match in query_response.matches:
                source = match.metadata.get("source", "unknown")
                if source not in doc_groups:
                    doc_groups[source] = []
                doc_groups[source].append(match)
            
            # Count potential duplicates within each document
            total_duplicates = 0
            duplicate_groups = 0
            
            for doc_id, matches in doc_groups.items():
                if len(matches) > 1:
                    # Check for content-based duplicates
                    content_groups = {}
                    for match in matches:
                        content = match.metadata.get("text", "")
                        content_hash = hash(content.strip().lower())
                        if content_hash not in content_groups:
                            content_groups[content_hash] = []
                        content_groups[content_hash].append(match)
                    
                    # Count duplicates
                    for content_hash, group in content_groups.items():
                        if len(group) > 1:
                            total_duplicates += len(group) - 1
                            duplicate_groups += 1
            
            return {
                "total_vectors": len(query_response.matches),
                "potential_duplicates": total_duplicates,
                "duplicate_groups": duplicate_groups,
                "documents_analyzed": len(doc_groups)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get duplicate stats: {e}")
            return {"error": str(e)}
    
    def cleanup_duplicates(self, similarity_threshold: float = 0.95):
        """Clean up duplicate vectors across the entire index."""
        try:
            logger.info("Starting duplicate cleanup across Pinecone index...")
            
            # Query all vectors
            query_response = self.index.query(
                vector=[0] * 1536,  # Dummy vector
                top_k=10000,
                include_metadata=True
            )
            
            if not query_response.matches:
                logger.info("No vectors found in index")
                return
            
            # Group by source document
            doc_groups = {}
            for match in query_response.matches:
                source = match.metadata.get("source", "unknown")
                if source not in doc_groups:
                    doc_groups[source] = []
                doc_groups[source].append(match)
            
            total_removed = 0
            
            for doc_id, matches in doc_groups.items():
                if len(matches) > 1:
                    # Find and remove duplicates within this document
                    removed_count = self._remove_document_duplicates(matches, similarity_threshold)
                    total_removed += removed_count
            
            logger.info(f"Duplicate cleanup completed. Removed {total_removed} duplicate vectors.")
            
        except Exception as e:
            logger.error(f"Failed to cleanup duplicates: {e}")
    
    def _remove_document_duplicates(self, matches: List, similarity_threshold: float) -> int:
        """Remove duplicate vectors within a document."""
        try:
            if len(matches) <= 1:
                return 0
            
            # Group by content hash first
            content_groups = {}
            for match in matches:
                content = match.metadata.get("text", "")
                content_hash = hash(content.strip().lower())
                if content_hash not in content_groups:
                    content_groups[content_hash] = []
                content_groups[content_hash].append(match)
            
            # Remove exact content duplicates
            duplicates_to_remove = []
            for content_hash, group in content_groups.items():
                if len(group) > 1:
                    # Keep the first one, mark others for deletion
                    duplicates_to_remove.extend(group[1:])
            
            # Remove the duplicates
            if duplicates_to_remove:
                duplicate_ids = [match.id for match in duplicates_to_remove]
                self.index.delete(ids=duplicate_ids)
                logger.info(f"Removed {len(duplicate_ids)} exact content duplicates")
                return len(duplicate_ids)
            
            return 0
            
        except Exception as e:
            logger.warning(f"Failed to remove document duplicates: {e}")
            return 0
    
    def delete_document(self, doc_id: str):
        """Delete all chunks for a document from Pinecone."""
        try:
            self._delete_existing_vectors(doc_id)
            logger.info(f"Successfully deleted document {doc_id} from Pinecone")
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
    
    def clear_index(self):
        """Clear all vectors from the Pinecone index."""
        try:
            self.index.delete(delete_all=True)
            logger.info("Cleared all vectors from Pinecone index")
        except Exception as e:
            logger.error(f"Failed to clear index: {e}") 