"""
Ingestion Pipeline for processing new documents into the system.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor

from .document_processor import DocumentProcessor
from ..kb.knowledge_base import KnowledgeBase
from ..kg.knowledge_graph import KnowledgeGraph
from ..rag.rag_system import RAGSystem

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    document_id: str
    file_path: str
    success: bool
    kb_facts_added: int
    kg_entities_added: int
    kg_relationships_added: int
    rag_chunks_added: int
    errors: List[str]
    processing_time: float


class IngestionPipeline:
    """Pipeline for ingesting documents into the NASA query system."""
    
    def __init__(self, config: Dict[str, Any], kb: KnowledgeBase, kg: KnowledgeGraph, rag: RAGSystem):
        self.config = config
        self.kb = kb
        self.kg = kg
        self.rag = rag
        self.document_processor = DocumentProcessor(config)
        
        self.supported_formats = config.get("supported_formats", ["pdf", "json", "md", "txt", "docx"])
        self.batch_size = config.get("batch_size", 10)
        self.max_workers = config.get("max_workers", 4)
        self.processing_timeout = config.get("processing_timeout", 300)
        self.auto_index = config.get("auto_index", True)
        
        # Track processed files to avoid reprocessing
        self.processed_files: Set[str] = set()
        self.processing_queue: List[Path] = []
    
    async def process_documents(self, data_path: str = "data/documents") -> List[IngestionResult]:
        """
        Process all documents in the specified directory.
        
        Args:
            data_path: Path to directory containing documents
            
        Returns:
            List of ingestion results
        """
        data_dir = Path(data_path)
        if not data_dir.exists():
            logger.warning(f"Data directory {data_path} does not exist")
            return []
        
        # Find all supported files
        files_to_process = []
        for file_path in data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower()[1:] in self.supported_formats:
                if str(file_path) not in self.processed_files:
                    files_to_process.append(file_path)
        
        if not files_to_process:
            logger.info("No new files to process")
            return []
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process files in batches
        results = []
        for i in range(0, len(files_to_process), self.batch_size):
            batch = files_to_process[i:i + self.batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
        
        # Mark files as processed
        for result in results:
            if result.success:
                self.processed_files.add(result.file_path)
        
        return results
    
    async def _process_batch(self, files: List[Path]) -> List[IngestionResult]:
        """Process a batch of files."""
        results = []
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path
                for file_path in files
            }
            
            # Collect results
            for future in future_to_file:
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, future.result, self.processing_timeout
                    )
                    results.append(result)
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.error(f"Failed to process {file_path}: {e}")
                    results.append(IngestionResult(
                        document_id=str(file_path.stem),
                        file_path=str(file_path),
                        success=False,
                        kb_facts_added=0,
                        kg_entities_added=0,
                        kg_relationships_added=0,
                        rag_chunks_added=0,
                        errors=[str(e)],
                        processing_time=0.0
                    ))
        
        return results
    
    def _process_single_file(self, file_path: Path) -> IngestionResult:
        """Process a single file."""
        start_time = time.time()
        errors = []
        
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Extract content from file
            content = self.document_processor.extract_content(file_path)
            if not content:
                raise ValueError("Could not extract content from file")
            
            # Parse content into structured data
            parsed_data = self.document_processor.parse_content(content, file_path.suffix.lower())
            
            # Ingest into appropriate subsystems
            kb_facts = 0
            kg_entities = 0
            kg_relationships = 0
            rag_chunks = 0
            
            # Add to KB (factual data)
            if parsed_data.get("facts"):
                for fact_data in parsed_data["facts"]:
                    try:
                        from ..kb.knowledge_base import Fact
                        fact = Fact(
                            id=fact_data.get("id", f"{file_path.stem}_{kb_facts}"),
                            subject=fact_data["subject"],
                            predicate=fact_data["predicate"],
                            object=fact_data["object"],
                            source=str(file_path),
                            confidence=fact_data.get("confidence", 0.8),
                            metadata=fact_data.get("metadata", {})
                        )
                        self.kb.add_fact(fact)
                        kb_facts += 1
                    except Exception as e:
                        errors.append(f"KB fact error: {e}")
            
            # Add to KG (entities and relationships)
            if parsed_data.get("entities"):
                for entity_data in parsed_data["entities"]:
                    try:
                        from ..kg.knowledge_graph import Entity
                        entity = Entity(
                            id=entity_data.get("id", f"{file_path.stem}_entity_{kg_entities}"),
                            name=entity_data["name"],
                            type=entity_data["type"],
                            properties=entity_data.get("properties", {}),
                            source=str(file_path)
                        )
                        self.kg.add_entity(entity)
                        kg_entities += 1
                    except Exception as e:
                        errors.append(f"KG entity error: {e}")
            
            if parsed_data.get("relationships"):
                for rel_data in parsed_data["relationships"]:
                    try:
                        from ..kg.knowledge_graph import Relationship
                        relationship = Relationship(
                            id=rel_data.get("id", f"{file_path.stem}_rel_{kg_relationships}"),
                            source_id=rel_data["source_id"],
                            target_id=rel_data["target_id"],
                            type=rel_data["type"],
                            properties=rel_data.get("properties", {}),
                            source=str(file_path)
                        )
                        self.kg.add_relationship(relationship)
                        kg_relationships += 1
                    except Exception as e:
                        errors.append(f"KG relationship error: {e}")
            
            # Add to RAG (document chunks)
            if parsed_data.get("content"):
                try:
                    self.rag.add_document(
                        content=parsed_data["content"],
                        doc_id=str(file_path.stem),
                        metadata={"source": str(file_path)}
                    )
                    # Estimate chunks added (simplified)
                    rag_chunks = len(parsed_data["content"].split()) // 1000 + 1
                except Exception as e:
                    errors.append(f"RAG error: {e}")
            
            processing_time = time.time() - start_time
            
            return IngestionResult(
                document_id=str(file_path.stem),
                file_path=str(file_path),
                success=len(errors) == 0,
                kb_facts_added=kb_facts,
                kg_entities_added=kg_entities,
                kg_relationships_added=kg_relationships,
                rag_chunks_added=rag_chunks,
                errors=errors,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to process {file_path}: {e}")
            return IngestionResult(
                document_id=str(file_path.stem),
                file_path=str(file_path),
                success=False,
                kb_facts_added=0,
                kg_entities_added=0,
                kg_relationships_added=0,
                rag_chunks_added=0,
                errors=[str(e)],
                processing_time=processing_time
            )
    
    async def watch_directory(self, data_path: str = "data/documents"):
        """Watch directory for new files and process them automatically."""
        data_dir = Path(data_path)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Watching directory: {data_dir}")
        
        while True:
            try:
                # Check for new files
                new_files = []
                for file_path in data_dir.rglob("*"):
                    if (file_path.is_file() and 
                        file_path.suffix.lower()[1:] in self.supported_formats and
                        str(file_path) not in self.processed_files):
                        new_files.append(file_path)
                
                if new_files:
                    logger.info(f"Found {len(new_files)} new files to process")
                    results = await self.process_documents(data_path)
                    
                    # Log results
                    successful = sum(1 for r in results if r.success)
                    logger.info(f"Processed {successful}/{len(results)} files successfully")
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in directory watcher: {e}")
                await asyncio.sleep(60)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "processed_files": len(self.processed_files),
            "supported_formats": self.supported_formats,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "processing_timeout": self.processing_timeout,
            "auto_index": self.auto_index
        }
    
    def reset_processed_files(self):
        """Reset the list of processed files."""
        self.processed_files.clear()
        logger.info("Reset processed files list")
    
    def add_processed_file(self, file_path: str):
        """Manually mark a file as processed."""
        self.processed_files.add(file_path)
        logger.info(f"Marked {file_path} as processed") 