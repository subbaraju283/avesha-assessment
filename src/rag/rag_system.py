"""
RAG System for generative answers and synthesis from document context.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from ..models.llm_manager import LLMManager
from .vector_store import VectorStore
from .models import DocumentChunk, RAGQueryResult

logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG System for generative answers with NASA documentation."""
    
    def __init__(self, config: Dict[str, Any], llm_manager: LLMManager):
        self.config = config
        self.llm_manager = llm_manager
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.retrieval_k = config.get("retrieval_k", 5)
        self.rerank_top_k = config.get("rerank_top_k", 3)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        # Initialize vector store
        self.vector_store = VectorStore(Path(config.get("vector_store_path", "data/rag")))
        
        # Load initial NASA documents
        self._load_nasa_documents()
    
    def _load_nasa_documents(self):
        """Load initial NASA documents into the RAG system."""
        nasa_documents = [
            {
                "id": "doc_001",
                "title": "NASA Mars Exploration Program",
                "content": """
                NASA's Mars Exploration Program is a long-term effort to explore the planet Mars, 
                funded and led by NASA. Formed in 1993, the program has been exploring Mars with 
                orbiters, landers, and rovers. The program's scientific goals are to determine 
                whether Mars ever supported life, to characterize the climate and geology of Mars, 
                and to prepare for human exploration of Mars.
                
                The program includes several missions such as the Mars Science Laboratory (Curiosity rover), 
                the Mars 2020 mission (Perseverance rover), and the Mars Reconnaissance Orbiter. 
                These missions have provided unprecedented insights into the Red Planet's geology, 
                climate, and potential for past or present life.
                
                Key discoveries include evidence of ancient water flows, organic molecules, and 
                seasonal methane releases. The program continues to expand our understanding of 
                Mars and pave the way for future human exploration.
                """,
                "source": "NASA Official Documentation",
                "metadata": {"topic": "mars_exploration", "type": "program_overview"}
            },
            {
                "id": "doc_002",
                "title": "Voyager Program Overview",
                "content": """
                The Voyager program consists of two robotic probes, Voyager 1 and Voyager 2, 
                launched in 1977 to study the outer Solar System and interstellar space. 
                Both spacecraft are still operational and continue to send data back to Earth.
                
                Voyager 1 and Voyager 2 were designed to take advantage of a rare planetary 
                alignment that occurs only once every 176 years. This alignment allowed the 
                spacecraft to visit Jupiter, Saturn, Uranus, and Neptune using gravity assists.
                
                Key achievements include the first detailed images of Jupiter's moons, the discovery 
                of active volcanoes on Io, the complex ring system of Saturn, and the Great Dark 
                Spot on Neptune. Both spacecraft carry the Golden Record, a phonograph record 
                containing sounds and images selected to portray the diversity of life and culture on Earth.
                
                Voyager 1 entered interstellar space in 2012, becoming the first human-made object 
                to do so. Voyager 2 followed in 2018. Both spacecraft continue to operate and 
                send data about the interstellar medium.
                """,
                "source": "NASA Official Documentation",
                "metadata": {"topic": "interstellar_exploration", "type": "mission_overview"}
            },
            {
                "id": "doc_003",
                "title": "Hubble Space Telescope Science",
                "content": """
                The Hubble Space Telescope is one of NASA's most successful and long-lasting 
                science missions. Launched in 1990, Hubble has revolutionized our understanding 
                of the universe through its observations of distant galaxies, nebulae, and other 
                celestial objects.
                
                Hubble's key contributions include determining the rate of expansion of the universe, 
                discovering that most galaxies contain supermassive black holes, and providing 
                evidence for the existence of dark energy. The telescope has also captured 
                stunning images that have become iconic representations of space exploration.
                
                Hubble operates in low Earth orbit and has been serviced by astronauts on five 
                occasions. The telescope continues to make groundbreaking discoveries and remains 
                one of the most important tools in astronomy.
                
                The telescope's successor, the James Webb Space Telescope, was launched in 2021 
                and is designed to observe the universe in infrared wavelengths, complementing 
                Hubble's visible and ultraviolet observations.
                """,
                "source": "NASA Official Documentation",
                "metadata": {"topic": "space_telescopes", "type": "mission_overview"}
            },
            {
                "id": "doc_004",
                "title": "Ion Propulsion Technology",
                "content": """
                Ion propulsion is a form of electric propulsion used for spacecraft. It creates 
                thrust by accelerating ions using electricity. The most common type is the 
                electrostatic ion thruster, which uses Coulomb force to accelerate ions.
                
                Ion propulsion is much more efficient than traditional chemical rockets, with 
                specific impulses typically 5-10 times higher. This makes it ideal for long-duration 
                missions where fuel efficiency is critical. However, ion thrusters produce very 
                low thrust, so they are not suitable for launching spacecraft from Earth.
                
                NASA has used ion propulsion on several missions, including the Deep Space 1 
                mission, which demonstrated the technology, and the Dawn mission, which used 
                ion propulsion to visit the asteroids Vesta and Ceres. The technology is also 
                being considered for future missions to Mars and beyond.
                
                Key advantages include high efficiency, long operational life, and the ability 
                to provide continuous low-thrust propulsion. Challenges include the need for 
                large solar arrays to provide electrical power and the complexity of the 
                propulsion system.
                """,
                "source": "NASA Technical Documentation",
                "metadata": {"topic": "propulsion_technology", "type": "technical_overview"}
            },
            {
                "id": "doc_005",
                "title": "Exoplanet Detection Methods",
                "content": """
                NASA has developed and utilized several methods to detect and study exoplanets, 
                planets orbiting stars other than our Sun. The most successful methods include 
                the transit method, radial velocity method, and direct imaging.
                
                The transit method, used by the Kepler Space Telescope, detects planets by 
                measuring the slight dimming of a star when a planet passes in front of it. 
                This method has discovered thousands of exoplanets and provided valuable data 
                about their sizes and orbital periods.
                
                The radial velocity method measures the wobble of a star caused by the 
                gravitational pull of orbiting planets. This method provides information about 
                planet masses and orbital characteristics.
                
                Direct imaging, while challenging, allows astronomers to see exoplanets directly 
                and study their atmospheres. The James Webb Space Telescope is expected to 
                revolutionize this field with its infrared capabilities.
                
                These methods have revealed an incredible diversity of exoplanets, from 
                hot Jupiters to Earth-sized planets in habitable zones. The search for 
                potentially habitable worlds continues to be a major focus of NASA's 
                exoplanet research.
                """,
                "source": "NASA Scientific Documentation",
                "metadata": {"topic": "exoplanet_research", "type": "scientific_overview"}
            }
        ]
        
        # Process and add documents to vector store
        for doc_data in nasa_documents:
            chunks = self._chunk_document(doc_data["content"], doc_data["id"])
            for chunk in chunks:
                self.vector_store.add_chunk(chunk)
        
        logger.info(f"Loaded {len(nasa_documents)} NASA documents into RAG system")
    
    def _chunk_document(self, content: str, doc_id: str) -> List[DocumentChunk]:
        """Split document content into chunks."""
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_content = " ".join(chunk_words)
            
            chunk = DocumentChunk(
                id=f"{doc_id}_chunk_{i//self.chunk_size}",
                content=chunk_content,
                source=doc_id,
                metadata={"chunk_index": i//self.chunk_size}
            )
            chunks.append(chunk)
        
        return chunks
    
    async def query(self, query: str, debug: bool = False) -> RAGQueryResult:
        """
        Query the RAG system for generative answers.
        
        Args:
            query: The user's query
            debug: Whether to enable debug logging
            
        Returns:
            RAGQueryResult with answer and retrieved chunks
        """
        logger.info(f"RAG Query: {query}")
        
        # Step 1: Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Step 2: Retrieve relevant chunks
        retrieved_chunks = self._retrieve_chunks(query_embedding, query)
        
        # Step 3: Rerank chunks (optional)
        if len(retrieved_chunks) > self.rerank_top_k:
            retrieved_chunks = self._rerank_chunks(query, retrieved_chunks)
        
        # Step 4: Generate answer
        answer = await self._generate_answer(query, retrieved_chunks)
        
        if debug:
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            logger.info(f"Generated answer: {answer[:100]}...")
        
        return RAGQueryResult(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            confidence=0.8 if retrieved_chunks else 0.3,
            reasoning=f"Retrieved {len(retrieved_chunks)} relevant document chunks",
            query_type="generative",
            metadata={
                "query_embedding_length": len(query_embedding),
                "chunk_count": len(retrieved_chunks)
            }
        )
    
    def _retrieve_chunks(self, query_embedding: List[float], query: str) -> List[DocumentChunk]:
        """Retrieve relevant document chunks using vector similarity."""
        all_chunks = self.vector_store.get_all_chunks()
        
        if not all_chunks:
            return []
        
        # Calculate similarities
        chunk_similarities = []
        for chunk in all_chunks:
            if chunk.embedding is None:
                # Generate embedding for chunk
                chunk.embedding = self.embedding_model.encode([chunk.content])[0]
                self.vector_store.update_chunk_embedding(chunk.id, chunk.embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, chunk.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
            )
            chunk_similarities.append((chunk, similarity))
        
        # Sort by similarity and return top k
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in chunk_similarities[:self.retrieval_k]]
    
    def _rerank_chunks(self, query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Rerank chunks using more sophisticated methods (placeholder)."""
        # Simple reranking based on keyword overlap
        query_words = set(query.lower().split())
        
        def chunk_score(chunk):
            chunk_words = set(chunk.content.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            return overlap / len(query_words) if query_words else 0
        
        scored_chunks = [(chunk, chunk_score(chunk)) for chunk in chunks]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, _ in scored_chunks[:self.rerank_top_k]]
    
    async def _generate_answer(self, query: str, chunks: List[DocumentChunk]) -> str:
        """Generate answer using retrieved chunks and LLM."""
        if not chunks:
            return "I don't have enough information to answer that question about NASA."
        
        # Create context from chunks
        context = "\n\n".join([
            f"Document {i+1}:\n{chunk.content}"
            for i, chunk in enumerate(chunks)
        ])
        
        prompt = f"""
        Based on the following NASA documentation, answer the user's question.
        Provide a comprehensive and accurate answer using only the information given.
        
        User Question: {query}
        
        Relevant Documentation:
        {context}
        
        Answer:
        """
        
        try:
            answer = await self.llm_manager.generate(prompt)
            return answer
        except Exception as e:
            logger.error(f"Failed to generate RAG answer: {e}")
            return "I encountered an error while generating the answer. Please try again."
    
    def add_document(self, content: str, doc_id: str, metadata: Dict[str, Any] = None):
        """Add a new document to the RAG system."""
        chunks = self._chunk_document(content, doc_id)
        
        for chunk in chunks:
            if metadata:
                chunk.metadata.update(metadata)
            self.vector_store.add_chunk(chunk)
        
        logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        all_chunks = self.vector_store.get_all_chunks()
        
        sources = list(set(chunk.source for chunk in all_chunks))
        total_chunks = len(all_chunks)
        
        return {
            "total_chunks": total_chunks,
            "sources": sources,
            "avg_chunk_length": sum(len(chunk.content.split()) for chunk in all_chunks) / total_chunks if total_chunks > 0 else 0
        } 