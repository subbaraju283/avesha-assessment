"""
Tests for the NASA Documentation Query System.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import shutil

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.llm_manager import LLMManager
from src.router.query_router import QueryRouter, QueryType, RoutingDecision
from src.kb.knowledge_base import KnowledgeBase, Fact
from src.kg.knowledge_graph import KnowledgeGraph, Entity, Relationship
from src.rag.rag_system import RAGSystem, DocumentChunk
from src.ingestion.ingestion_pipeline import IngestionPipeline


class TestLLMManager:
    """Test LLM Manager functionality."""
    
    @pytest.fixture
    def config(self):
        return {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "api_key": "test_key",
                    "model": "gpt-4",
                    "temperature": 0.1,
                    "max_tokens": 2000
                }
            }
        }
    
    @pytest.fixture
    def llm_manager(self, config):
        with patch('src.models.llm_manager.openai') as mock_openai:
            mock_openai.AsyncOpenAI.return_value = Mock()
            return LLMManager(config)
    
    def test_initialization(self, llm_manager):
        """Test LLM manager initialization."""
        assert llm_manager.default_provider == "openai"
        assert "openai" in llm_manager.providers
    
    @pytest.mark.asyncio
    async def test_generate(self, llm_manager):
        """Test text generation."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        
        llm_manager.providers["openai"].client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await llm_manager.generate("Test prompt")
        assert result == "Test response"
    
    def test_get_available_providers(self, llm_manager):
        """Test getting available providers."""
        providers = llm_manager.get_available_providers()
        assert "openai" in providers


class TestQueryRouter:
    """Test Query Router functionality."""
    
    @pytest.fixture
    def config(self):
        return {
            "kb_threshold": 0.7,
            "kg_threshold": 0.6,
            "rag_threshold": 0.4,
            "confidence_threshold": 0.8
        }
    
    @pytest.fixture
    def llm_manager(self):
        return Mock(spec=LLMManager)
    
    @pytest.fixture
    def router(self, config, llm_manager):
        return QueryRouter(config, llm_manager)
    
    @pytest.mark.asyncio
    async def test_route_factual_query(self, router):
        """Test routing of factual queries to KB."""
        query = "When was Voyager 1 launched?"
        
        # Mock LLM classification
        router.llm_manager.classify = AsyncMock(return_value={
            "factual": 0.8,
            "relational": 0.1,
            "generative": 0.1
        })
        
        decision = await router.route_query(query)
        
        assert decision.subsystem == "KB"
        assert decision.query_type == QueryType.FACTUAL
        assert decision.confidence >= 0.7
    
    @pytest.mark.asyncio
    async def test_route_relational_query(self, router):
        """Test routing of relational queries to KG."""
        query = "Which missions studied Mars and used ion propulsion?"
        
        # Mock LLM classification
        router.llm_manager.classify = AsyncMock(return_value={
            "factual": 0.2,
            "relational": 0.8,
            "generative": 0.0
        })
        
        decision = await router.route_query(query)
        
        assert decision.subsystem == "KG"
        assert decision.query_type == QueryType.RELATIONAL
        assert decision.confidence >= 0.6
    
    @pytest.mark.asyncio
    async def test_route_generative_query(self, router):
        """Test routing of generative queries to RAG."""
        query = "Explain how NASA studies exoplanets"
        
        # Mock LLM classification
        router.llm_manager.classify = AsyncMock(return_value={
            "factual": 0.1,
            "relational": 0.2,
            "generative": 0.7
        })
        
        decision = await router.route_query(query)
        
        assert decision.subsystem == "RAG"
        assert decision.query_type == QueryType.GENERATIVE
        assert decision.confidence >= 0.4
    
    def test_analyze_query_characteristics(self, router):
        """Test query characteristics analysis."""
        query = "What missions studied Mars and used ion propulsion?"
        characteristics = router._analyze_query_characteristics(query)
        
        assert "word_count" in characteristics
        assert "complexity_score" in characteristics
        assert characteristics["word_count"] > 0
    
    def test_fallback_intent_classification(self, router):
        """Test fallback intent classification."""
        query = "When was Voyager 1 launched?"
        scores = router._fallback_intent_classification(query)
        
        assert "factual" in scores
        assert "relational" in scores
        assert "generative" in scores
        assert sum(scores.values()) > 0


class TestKnowledgeBase:
    """Test Knowledge Base functionality."""
    
    @pytest.fixture
    def config(self):
        return {
            "storage_path": "data/kb",
            "similarity_threshold": 0.8,
            "max_results": 10,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    
    @pytest.fixture
    def llm_manager(self):
        return Mock(spec=LLMManager)
    
    @pytest.fixture
    def kb(self, config, llm_manager):
        with patch('src.kb.knowledge_base.SentenceTransformer'):
            return KnowledgeBase(config, llm_manager)
    
    def test_initialization(self, kb):
        """Test KB initialization."""
        assert kb.similarity_threshold == 0.8
        assert kb.max_results == 10
        assert kb.fact_store is not None
    
    def test_add_fact(self, kb):
        """Test adding facts to KB."""
        fact = Fact(
            id="test_001",
            subject="Voyager 1",
            predicate="launch_date",
            object="1977-09-05",
            source="test",
            confidence=0.9,
            metadata={}
        )
        
        kb.add_fact(fact)
        facts = kb.fact_store.get_all_facts()
        assert len(facts) > 0
    
    @pytest.mark.asyncio
    async def test_query(self, kb):
        """Test KB querying."""
        # Mock LLM response
        kb.llm_manager.generate = AsyncMock(return_value="Voyager 1 was launched on September 5, 1977.")
        
        result = await kb.query("When was Voyager 1 launched?")
        
        assert result.facts is not None
        assert result.confidence > 0
        assert result.query_type in ["temporal", "descriptive", "relational", "general"]


class TestKnowledgeGraph:
    """Test Knowledge Graph functionality."""
    
    @pytest.fixture
    def config(self):
        return {
            "max_depth": 3,
            "max_paths": 5,
            "relationship_types": ["STUDIED", "USED_TECHNOLOGY"]
        }
    
    @pytest.fixture
    def llm_manager(self):
        return Mock(spec=LLMManager)
    
    @pytest.fixture
    def kg(self, config, llm_manager):
        with patch('src.kg.knowledge_graph.Path'):
            return KnowledgeGraph(config, llm_manager)
    
    def test_initialization(self, kg):
        """Test KG initialization."""
        assert kg.max_depth == 3
        assert kg.max_paths == 5
        assert kg.graph_store is not None
    
    def test_add_entity(self, kg):
        """Test adding entities to KG."""
        entity = Entity(
            id="test_entity",
            name="Voyager 1",
            type="mission",
            properties={"launch_date": "1977-09-05"},
            source="test"
        )
        
        kg.add_entity(entity)
        entities = kg.graph_store.get_all_entities()
        assert len(entities) > 0
    
    def test_add_relationship(self, kg):
        """Test adding relationships to KG."""
        relationship = Relationship(
            id="test_rel",
            source_id="voyager_1",
            target_id="jupiter",
            type="STUDIED",
            properties={"year": 1979},
            source="test"
        )
        
        kg.add_relationship(relationship)
        relationships = kg.graph_store.get_all_relationships()
        assert len(relationships) > 0
    
    @pytest.mark.asyncio
    async def test_query(self, kg):
        """Test KG querying."""
        # Mock LLM response
        kg.llm_manager.generate = AsyncMock(return_value="Voyager 1 studied Jupiter and Saturn.")
        
        result = await kg.query("Which missions studied Jupiter?")
        
        assert result.entities is not None
        assert result.relationships is not None
        assert result.confidence > 0


class TestRAGSystem:
    """Test RAG System functionality."""
    
    @pytest.fixture
    def config(self):
        return {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_k": 5,
            "rerank_top_k": 3,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_store_path": "data/rag"
        }
    
    @pytest.fixture
    def llm_manager(self):
        return Mock(spec=LLMManager)
    
    @pytest.fixture
    def rag(self, config, llm_manager):
        with patch('src.rag.rag_system.SentenceTransformer'):
            with patch('src.rag.rag_system.Path'):
                return RAGSystem(config, llm_manager)
    
    def test_initialization(self, rag):
        """Test RAG system initialization."""
        assert rag.chunk_size == 1000
        assert rag.retrieval_k == 5
        assert rag.vector_store is not None
    
    def test_chunk_document(self, rag):
        """Test document chunking."""
        content = "This is a test document with multiple sentences. " * 50
        chunks = rag._chunk_document(content, "test_doc")
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content is not None
            assert chunk.source == "test_doc"
    
    @pytest.mark.asyncio
    async def test_query(self, rag):
        """Test RAG querying."""
        # Mock LLM response
        rag.llm_manager.generate = AsyncMock(return_value="NASA studies exoplanets using various methods.")
        
        result = await rag.query("How does NASA study exoplanets?")
        
        assert result.answer is not None
        assert result.retrieved_chunks is not None
        assert result.confidence > 0


class TestIngestionPipeline:
    """Test Ingestion Pipeline functionality."""
    
    @pytest.fixture
    def config(self):
        return {
            "supported_formats": ["pdf", "json", "md", "txt"],
            "batch_size": 10,
            "max_workers": 4,
            "processing_timeout": 300
        }
    
    @pytest.fixture
    def kb(self):
        return Mock(spec=KnowledgeBase)
    
    @pytest.fixture
    def kg(self):
        return Mock(spec=KnowledgeGraph)
    
    @pytest.fixture
    def rag(self):
        return Mock(spec=RAGSystem)
    
    @pytest.fixture
    def pipeline(self, config, kb, kg, rag):
        return IngestionPipeline(config, kb, kg, rag)
    
    def test_initialization(self, pipeline):
        """Test ingestion pipeline initialization."""
        assert pipeline.batch_size == 10
        assert pipeline.max_workers == 4
        assert pipeline.supported_formats == ["pdf", "json", "md", "txt"]
    
    @pytest.mark.asyncio
    async def test_process_documents(self, pipeline):
        """Test document processing."""
        # Create temporary test directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("Voyager 1 was launched in 1977.")
            
            # Mock document processor
            pipeline.document_processor.extract_content = Mock(return_value="Voyager 1 was launched in 1977.")
            pipeline.document_processor.parse_content = Mock(return_value={
                "content": "Voyager 1 was launched in 1977.",
                "facts": [],
                "entities": [],
                "relationships": []
            })
            
            results = await pipeline.process_documents(temp_dir)
            
            assert len(results) > 0
            assert all(hasattr(result, 'success') for result in results)


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def config(self):
        return {
            "llm": {
                "default_provider": "openai",
                "providers": {
                    "openai": {
                        "api_key": "test_key",
                        "model": "gpt-4"
                    }
                }
            },
            "router": {
                "kb_threshold": 0.7,
                "kg_threshold": 0.6,
                "rag_threshold": 0.4
            },
            "kb": {
                "storage_path": "data/kb",
                "similarity_threshold": 0.8
            },
            "kg": {
                "max_depth": 3,
                "max_paths": 5
            },
            "rag": {
                "chunk_size": 1000,
                "retrieval_k": 5
            },
            "ingestion": {
                "supported_formats": ["txt"],
                "batch_size": 5
            }
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_query(self, config):
        """Test end-to-end query processing."""
        with patch('src.models.llm_manager.openai'):
            with patch('src.kb.knowledge_base.SentenceTransformer'):
                with patch('src.rag.rag_system.SentenceTransformer'):
                    with patch('src.kg.knowledge_graph.Path'):
                        with patch('src.rag.rag_system.Path'):
                            # Initialize system components
                            llm_manager = LLMManager(config["llm"])
                            router = QueryRouter(config["router"], llm_manager)
                            kb = KnowledgeBase(config["kb"], llm_manager)
                            kg = KnowledgeGraph(config["kg"], llm_manager)
                            rag = RAGSystem(config["rag"], llm_manager)
                            
                            # Mock LLM responses
                            llm_manager.generate = AsyncMock(return_value="Test response")
                            llm_manager.classify = AsyncMock(return_value={
                                "factual": 0.8,
                                "relational": 0.1,
                                "generative": 0.1
                            })
                            
                            # Test query routing
                            decision = await router.route_query("When was Voyager 1 launched?")
                            assert decision.subsystem in ["KB", "KG", "RAG"]
                            
                            # Test KB query
                            kb_result = await kb.query("When was Voyager 1 launched?")
                            assert kb_result is not None
                            
                            # Test KG query
                            kg_result = await kg.query("Which missions studied Mars?")
                            assert kg_result is not None
                            
                            # Test RAG query
                            rag_result = await rag.query("Explain exoplanet detection")
                            assert rag_result is not None


if __name__ == "__main__":
    pytest.main([__file__]) 