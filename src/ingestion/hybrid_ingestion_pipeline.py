"""
Neo4j-based Ingestion Pipeline using neo4j-graphrag.
"""

import logging
import asyncio
import json
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
import time
from datetime import datetime

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Metadata for tracking file changes."""
    file_path: str
    last_modified: float
    file_size: int
    file_hash: str
    last_processed: float
    processing_status: str  # "success", "failed", "partial"


@dataclass
class HybridIngestionResult:
    """Result of hybrid document ingestion (Neo4j + RAG)."""
    document_id: str
    file_path: str
    success: bool
    added_to_kg: bool  # Whether file was added to Neo4j Knowledge Graph
    rag_chunks_added: int
    errors: List[str]
    processing_time: float
    extraction_method: str = "hybrid_graphrag"
    file_changed: bool = True  # Whether file was actually changed


class HybridIngestionPipeline:
    """Hybrid pipeline for ingesting documents using Neo4j graph and RAG vector store."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Neo4j connection
        self.neo4j_uri = config.get("neo4j", {}).get("database_url", "neo4j+s://72bbcca1.databases.neo4j.io")
        self.neo4j_username = config.get("neo4j", {}).get("username", "neo4j")
        self.neo4j_password = config.get("neo4j", {}).get("password", "TqEewyu6IUiWVzoaUXfHTjXAC2JHUg72C_p4AEIwJEY")
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # Initialize LLM
        self.llm = OpenAILLM(
            model_name="gpt-4o",
            model_params={
                "response_format": {"type": "json_object"},
                "temperature": 0
            }
        )
        
        # Initialize embedder
        self.embedder = OpenAIEmbeddings()
        
        # Define NASA-specific node labels and relationship types
        self.node_labels = [
            "mission", "spacecraft", "rover", "orbiter", "lander", "probe",
            "planet", "moon", "asteroid", "comet",
            "facility", "laboratory", "launch_site", "research_center",
            "company", "organization", "agency",
            "technology", "instrument", "system", "equipment",
            "scientist", "engineer", "researcher",
            "discovery", "finding", "observation"
        ]
        
        self.rel_types = [
            "MANAGES", "OPERATES", "BUILT", "LAUNCHED_FROM", "STUDIES",
            "EXPLORES", "DISCOVERS", "USES_TECHNOLOGY", "COLLABORATES_WITH",
            "LOCATED_AT", "PART_OF", "DEPENDS_ON", "COMMUNICATES_WITH",
            "TRANSMITS_TO", "RECEIVES_FROM", "ANALYZES", "MEASURES",
            "OBSERVES", "MONITORS", "CONTROLS", "SUPPORTS"
        ]
        
        # Store components for dynamic KG builder initialization
        self.llm = self.llm
        self.driver = self.driver
        self.embedder = self.embedder
        self.text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=100)
        self.entities = self.node_labels
        self.relations = self.rel_types
        self.prompt_template = self._get_nasa_prompt_template()
        
        # Create vector index for embeddings
        self._create_vector_index()
        
        # Initialize RAG system for vector store updates
        try:
            from ..rag.pinecone_rag_system import PineconeRAGSystem
            from ..models.llm_manager import LLMManager
            
            llm_manager = LLMManager(config)
            self.rag = PineconeRAGSystem(config["rag"], llm_manager)
            logger.info("Pinecone RAG system initialized for hybrid ingestion")
        except Exception as e:
            logger.warning(f"Failed to initialize Pinecone RAG system: {e}")
            self.rag = None
        
        # Initialize document processor
        try:
            from .document_processor import DocumentProcessor
            self.document_processor = DocumentProcessor(config)
            logger.info("Document processor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize document processor: {e}")
            self.document_processor = None
        
        # File metadata tracking
        self.metadata_file = Path("data/ingestion/file_metadata.json")
        self.file_metadata: Dict[str, FileMetadata] = {}
        self._load_file_metadata()
        
        # Supported file formats
        self.supported_formats = config.get("supported_formats", ["pdf", "json", "md", "txt", "docx"])
    
    def _load_file_metadata(self):
        """Load file metadata from persistent storage."""
        try:
            # Ensure metadata directory exists
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                # Convert back to FileMetadata objects
                for file_path, metadata in metadata_dict.items():
                    self.file_metadata[file_path] = FileMetadata(**metadata)
                
                logger.info(f"Loaded metadata for {len(self.file_metadata)} files")
            else:
                logger.info("No existing file metadata found")
                
        except Exception as e:
            logger.error(f"Failed to load file metadata: {e}")
            self.file_metadata = {}
    
    def _save_file_metadata(self):
        """Save file metadata to persistent storage."""
        try:
            # Convert FileMetadata objects to dictionaries
            metadata_dict = {}
            for file_path, metadata in self.file_metadata.items():
                metadata_dict[file_path] = {
                    "file_path": metadata.file_path,
                    "last_modified": metadata.last_modified,
                    "file_size": metadata.file_size,
                    "file_hash": metadata.file_hash,
                    "last_processed": metadata.last_processed,
                    "processing_status": metadata.processing_status
                }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            logger.debug(f"Saved metadata for {len(self.file_metadata)} files")
            
        except Exception as e:
            logger.error(f"Failed to save file metadata: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate file hash for {file_path}: {e}")
            return ""
    
    def _get_file_metadata(self, file_path: Path) -> FileMetadata:
        """Get current file metadata."""
        stat = file_path.stat()
        return FileMetadata(
            file_path=str(file_path),
            last_modified=stat.st_mtime,
            file_size=stat.st_size,
            file_hash=self._get_file_hash(file_path),
            last_processed=time.time(),
            processing_status="pending"
        )
    
    def _has_file_changed(self, file_path: Path) -> bool:
        """Check if file has changed since last processing."""
        file_path_str = str(file_path)
        
        # If file not in metadata, it's new
        if file_path_str not in self.file_metadata:
            logger.info(f"New file detected: {file_path.name}")
            return True
        
        # Get current file metadata
        current_metadata = self._get_file_metadata(file_path)
        stored_metadata = self.file_metadata[file_path_str]
        
        # Check if file has changed
        if (current_metadata.last_modified != stored_metadata.last_modified or
            current_metadata.file_size != stored_metadata.file_size or
            current_metadata.file_hash != stored_metadata.file_hash):
            
            logger.info(f"File changed detected: {file_path.name}")
            logger.debug(f"  Last modified: {stored_metadata.last_modified} -> {current_metadata.last_modified}")
            logger.debug(f"  File size: {stored_metadata.file_size} -> {current_metadata.file_size}")
            logger.debug(f"  File hash: {stored_metadata.file_hash[:8]}... -> {current_metadata.file_hash[:8]}...")
            return True
        
        logger.info(f"File unchanged: {file_path.name}")
        return False
    
    def _update_file_metadata(self, file_path: Path, status: str = "success"):
        """Update file metadata after processing."""
        file_path_str = str(file_path)
        current_metadata = self._get_file_metadata(file_path)
        current_metadata.processing_status = status
        current_metadata.last_processed = time.time()
        
        self.file_metadata[file_path_str] = current_metadata
        self._save_file_metadata()
        
        logger.debug(f"Updated metadata for {file_path.name} with status: {status}")
    
    def _create_kg_builder(self, file_path: Path) -> SimpleKGPipeline:
        """Create KG Builder with appropriate from_pdf setting based on file type."""
        file_extension = file_path.suffix.lower()
        from_pdf = file_extension == ".pdf"
        
        logger.info(f"Creating KG Builder for {file_path.name} with from_pdf={from_pdf}")
        
        return SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            text_splitter=self.text_splitter,
            embedder=self.embedder,
            entities=self.entities,
            relations=self.relations,
            prompt_template=self.prompt_template,
            from_pdf=from_pdf
        )
    
    def _create_vector_index(self):
        """Create vector index for embeddings if it doesn't exist."""
        try:
            from neo4j_graphrag.indexes import create_vector_index
            
            # Create vector index
            create_vector_index(
                self.driver, 
                name="text_embeddings", 
                label="Chunk",
                embedding_property="embedding", 
                dimensions=1536, 
                similarity_fn="cosine"
            )
            
            logger.info("Vector index 'text_embeddings' created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create vector index: {e}")
            # Continue without index - it might already exist
    
    def _get_nasa_prompt_template(self) -> str:
        """Get NASA-specific prompt template for graph extraction."""
        return """
        You are a NASA space exploration researcher tasked with extracting information from NASA documentation 
        and structuring it in a property graph to inform further space exploration and research Q&A.

        Extract the entities (nodes) and specify their type from the following Input text.
        Also extract the relationships between these nodes. The relationship direction goes from the start node to the end node.

        Return result as JSON using the following format:
        {{"nodes": [ {{"id": "0", "label": "the type of entity", "properties": {{"name": "name of entity" }} }}],
          "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}

        - Use only the information from the Input text. Do not add any additional information.
        - If the input text is empty, return empty JSON.
        - Make sure to create as many nodes and relationships as needed to offer rich NASA space exploration context for further research.
        - An AI knowledge assistant must be able to read this graph and immediately understand the context to inform detailed space exploration questions.
        - Multiple documents will be ingested from different sources and we are using this property graph to connect information, so make sure entity types are fairly general.

        Use only the following nodes and relationships (if provided):
        {schema}

        Assign a unique ID (string) to each node, and reuse it to define relationships.
        Do respect the source and target node types for relationship and the relationship direction.

        Do not return any additional information other than the JSON in it.

        Examples:
        {examples}

        Input text:
        {text}
        """
    
    async def process_documents(self, data_path: str = "data/documents") -> List[HybridIngestionResult]:
        """
        Process all documents in the specified directory using Neo4j.
        Only processes files that have changed since last processing.
        
        Args:
            data_path: Path to directory containing documents
            
        Returns:
            List of ingestion results
        """
        data_dir = Path(data_path)
        if not data_dir.exists():
            logger.warning(f"Data directory {data_path} does not exist")
            return []
        
        # Find all supported files and check for changes
        files_to_process = []
        unchanged_files = []
        
        for file_path in data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower()[1:] in self.supported_formats:
                if self._has_file_changed(file_path):
                    files_to_process.append(file_path)
                else:
                    unchanged_files.append(file_path)
        
        if not files_to_process:
            logger.info(f"No files to process. {len(unchanged_files)} files unchanged.")
            return []
        
        logger.info(f"Found {len(files_to_process)} files to process (changed/new)")
        logger.info(f"Skipping {len(unchanged_files)} unchanged files")
        
        # Process files sequentially (original behavior)
        results = []
        for file_path in files_to_process:
            result = await self._process_single_file(file_path)
            results.append(result)
        
        return results
    
    async def process_documents_parallel(self, data_path: str = "data/documents", max_concurrent: int = 3) -> List[HybridIngestionResult]:
        """
        Process documents in parallel with concurrency control.
        
        Args:
            data_path: Path to directory containing documents
            max_concurrent: Maximum number of files to process simultaneously
            
        Returns:
            List of ingestion results
        """
        data_dir = Path(data_path)
        if not data_dir.exists():
            logger.warning(f"Data directory {data_path} does not exist")
            return []
        
        # Find all supported files and check for changes
        files_to_process = []
        unchanged_files = []
        
        for file_path in data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower()[1:] in self.supported_formats:
                if self._has_file_changed(file_path):
                    files_to_process.append(file_path)
                else:
                    unchanged_files.append(file_path)
        
        if not files_to_process:
            logger.info(f"No files to process. {len(unchanged_files)} files unchanged.")
            return []
        
        logger.info(f"Found {len(files_to_process)} files to process in parallel (max {max_concurrent} concurrent)")
        logger.info(f"Skipping {len(unchanged_files)} unchanged files")
        
        # Process files in parallel
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_file(file_path):
            async with semaphore:
                try:
                    return await self._process_single_file(file_path)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    return HybridIngestionResult(
                        document_id=file_path.stem,
                        file_path=str(file_path),
                        success=False,
                        added_to_kg=False,
                        rag_chunks_added=0,
                        errors=[str(e)],
                        processing_time=0.0,
                        extraction_method="parallel_hybrid_graphrag",
                        file_changed=True
                    )
        
        # Create tasks for all files
        tasks = [process_file(f) for f in files_to_process]
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to proper results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task failed for {files_to_process[i]}: {result}")
                processed_results.append(HybridIngestionResult(
                    document_id=files_to_process[i].stem,
                    file_path=str(files_to_process[i]),
                    success=False,
                    added_to_kg=False,
                    rag_chunks_added=0,
                    errors=[str(result)],
                    processing_time=0.0,
                    extraction_method="parallel_hybrid_graphrag",
                    file_changed=True
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def process_documents_batch(self, data_path: str = "data/documents", batch_size: int = 5) -> List[HybridIngestionResult]:
        """
        Process documents in batches for better resource management.
        
        Args:
            data_path: Path to directory containing documents
            batch_size: Number of files to process in each batch
            
        Returns:
            List of ingestion results
        """
        data_dir = Path(data_path)
        if not data_dir.exists():
            logger.warning(f"Data directory {data_path} does not exist")
            return []
        
        # Find all supported files and check for changes
        files_to_process = []
        unchanged_files = []
        
        for file_path in data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower()[1:] in self.supported_formats:
                if self._has_file_changed(file_path):
                    files_to_process.append(file_path)
                else:
                    unchanged_files.append(file_path)
        
        if not files_to_process:
            logger.info(f"No files to process. {len(unchanged_files)} files unchanged.")
            return []
        
        logger.info(f"Found {len(files_to_process)} files to process in batches of {batch_size}")
        logger.info(f"Skipping {len(unchanged_files)} unchanged files")
        
        # Process files in batches
        all_results = []
        total_batches = (len(files_to_process) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(files_to_process))
            batch_files = files_to_process[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_files)} files)")
            
            # Process batch
            batch_results = await self._process_file_batch(batch_files)
            all_results.extend(batch_results)
            
            logger.info(f"Completed batch {batch_num + 1}/{total_batches}")
        
        return all_results
    
    async def _process_file_batch(self, file_paths: List[Path]) -> List[HybridIngestionResult]:
        """
        Process a batch of files with optimized operations.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of ingestion results
        """
        results = []
        
        # Step 1: Extract content for all files in batch
        file_contents = {}
        for file_path in file_paths:
            try:
                content = None
                if self.document_processor:
                    content = self.document_processor.extract_content(file_path)
                else:
                    # Fallback: read file directly
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                file_contents[file_path] = content
                logger.debug(f"Extracted content for {file_path.name}: {len(content) if content else 0} chars")
                
            except Exception as e:
                logger.error(f"Failed to extract content for {file_path}: {e}")
                file_contents[file_path] = None
        
        # Step 2: Process KG operations in parallel
        kg_tasks = []
        for file_path in file_paths:
            if file_contents.get(file_path):
                task = self._process_kg_for_file(file_path, file_contents[file_path])
                kg_tasks.append(task)
            else:
                kg_tasks.append(None)
        
        # Execute KG tasks in parallel
        kg_results = await asyncio.gather(*[task for task in kg_tasks if task is not None], return_exceptions=True)
        
        # Step 3: Process RAG operations in batch
        rag_batch_data = []
        for i, file_path in enumerate(file_paths):
            if file_contents.get(file_path):
                rag_batch_data.append((file_path, file_contents[file_path]))
        
        if rag_batch_data and self.rag:
            await self._batch_upsert_to_rag(rag_batch_data)
        
        # Step 4: Compile results
        kg_result_idx = 0
        for i, file_path in enumerate(file_paths):
            start_time = time.time()
            errors = []
            file_changed = self._has_file_changed(file_path)
            
            # Get KG result
            added_to_kg = False
            if kg_tasks[i] is not None:
                try:
                    kg_result = kg_results[kg_result_idx]
                    if isinstance(kg_result, Exception):
                        errors.append(f"KG processing failed: {kg_result}")
                    else:
                        added_to_kg = kg_result
                    kg_result_idx += 1
                except IndexError:
                    errors.append("KG result not available")
            
            # Get RAG result
            rag_chunks = 0
            if file_contents.get(file_path):
                rag_chunks = len(file_contents[file_path].split()) // 1000 + 1
            
            processing_time = time.time() - start_time
            
            # Update file metadata
            status = "success" if not errors else "failed"
            self._update_file_metadata(file_path, status)
            
            results.append(HybridIngestionResult(
                document_id=file_path.stem,
                file_path=str(file_path),
                success=len(errors) == 0,
                added_to_kg=added_to_kg,
                rag_chunks_added=rag_chunks,
                errors=errors,
                processing_time=processing_time,
                extraction_method="batch_hybrid_graphrag",
                file_changed=file_changed
            ))
        
        return results
    
    async def _process_kg_for_file(self, file_path: Path, content: str) -> bool:
        """
        Process KG operations for a single file.
        
        Args:
            file_path: Path to the file
            content: Extracted content
            
        Returns:
            True if nodes were added to KG, False otherwise
        """
        try:
            kg_builder = self._create_kg_builder(file_path)
            
            file_extension = file_path.suffix.lower()
            if file_extension == ".pdf":
                graph_data = await kg_builder.run_async(file_path=str(file_path))
            else:
                graph_data = await kg_builder.run_async(text=content)
            
            if graph_data is not None and hasattr(graph_data, 'result'):
                try:
                    return graph_data.result['resolver']['number_of_created_nodes'] > 0
                except (KeyError, TypeError):
                    return False
            return False
            
        except Exception as e:
            logger.error(f"KG processing failed for {file_path}: {e}")
            return False
    
    async def _batch_upsert_to_rag(self, batch_data: List[Tuple[Path, str]]):
        """
        Batch upsert multiple files to RAG system.
        
        Args:
            batch_data: List of (file_path, content) tuples
        """
        try:
            all_vectors = []
            all_chunks = []
            
            for file_path, content in batch_data:
                doc_id = str(file_path.stem)
                
                # Create proper LangChain Document objects
                documents = [Document(
                    page_content=content,
                    metadata={"source": str(file_path)}
                )]
                
                # Split content into chunks
                chunks = self.rag._split_documents(documents)
                
                # Embed chunks
                vectors = self.rag._embed_chunks(chunks)
                
                # Remove duplicates
                vectors, chunks = self.rag._remove_content_duplicates(vectors, chunks, doc_id)
                
                # Prepare for batch upsert
                for i, vector in enumerate(vectors):
                    all_vectors.append({
                        "id": f"{doc_id}-chunk-{i}",
                        "values": vector,
                        "metadata": {
                            "text": chunks[i].page_content,
                            "source": doc_id,
                            "chunk_index": i
                        }
                    })
                    all_chunks.append(chunks[i])
            
            # Batch upsert to Pinecone
            if all_vectors:
                batch_size = 100
                for i in range(0, len(all_vectors), batch_size):
                    batch = all_vectors[i:i + batch_size]
                    self.rag.index.upsert(vectors=batch)
                
                logger.info(f"Batch upserted {len(all_vectors)} vectors for {len(batch_data)} files")
            
        except Exception as e:
            logger.error(f"Batch RAG upsert failed: {e}")
            raise
    
    async def _process_single_file(self, file_path: Path) -> HybridIngestionResult:
        """Process a single file using Neo4j graph builder and RAG."""
        start_time = time.time()
        errors = []
        file_changed = self._has_file_changed(file_path)
        
        try:
            logger.info(f"Processing file with Neo4j: {file_path}")
            if not file_changed:
                logger.info(f"Skipping unchanged file: {file_path.name}")
            
            # Extract content for RAG
            content = None
            if self.document_processor:
                try:
                    content = self.document_processor.extract_content(file_path)
                    logger.info(f"Extracted content length: {len(content) if content else 0} characters")
                except Exception as e:
                    logger.warning(f"Failed to extract content: {e}")
                    errors.append(f"Content extraction error: {e}")
            
            # Create KG Builder dynamically based on file type
            kg_builder = self._create_kg_builder(file_path)
            
            # Run the KG builder for Neo4j graph
            graph_data = None
            try:
                file_extension = file_path.suffix.lower()
                if file_extension == ".pdf":
                    # For PDF files, use file_path
                    graph_data = await kg_builder.run_async(file_path=str(file_path))
                else:
                    # For text files, extract content and use text parameter
                    if content:
                        graph_data = await kg_builder.run_async(text=content)
                    else:
                        # Fallback: try to read file content
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                            graph_data = await kg_builder.run_async(text=file_content)
                        except Exception as e:
                            logger.error(f"Failed to read file content for KG processing: {e}")
                            errors.append(f"File content reading error: {e}")
            except Exception as e:
                error_msg = f"KG Builder execution failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
            
            # Determine if file was added to KG
            added_to_kg = False
            if graph_data is not None and hasattr(graph_data, 'result'):
                try:
                    added_to_kg = graph_data.result['resolver']['number_of_created_nodes'] > 0
                except (KeyError, TypeError) as e:
                    logger.warning(f"Could not determine KG creation status: {e}")
                    added_to_kg = False
            else:
                logger.warning("No graph data returned from KG Builder")
            
            # Add to RAG vector store
            rag_chunks = 0
            if content and self.rag:
                try:
                    self.rag.add_document(
                        content=content,
                        doc_id=str(file_path.stem),
                        metadata={"source": str(file_path)}
                    )
                    # Estimate chunks added (simplified)
                    rag_chunks = len(content.split()) // 1000 + 1
                    logger.info(f"Added document to RAG vector store: {rag_chunks} estimated chunks")
                except Exception as e:
                    error_msg = f"RAG vector store error: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            elif not self.rag:
                logger.warning("RAG system not available, skipping vector store update")
            
            processing_time = time.time() - start_time
            
            # Determine processing status
            if errors:
                processing_status = "failed"
                kg_status = "failed"
            elif added_to_kg:
                processing_status = "success"
                kg_status = "added to KG"
            else:
                processing_status = "partial"
                kg_status = "not added to KG"
            
            logger.info(f"Neo4j processing completed: {kg_status}, {rag_chunks} RAG chunks")
            
            # Update file metadata
            self._update_file_metadata(file_path, processing_status)
            
            return HybridIngestionResult(
                document_id=file_path.stem,
                file_path=str(file_path),
                success=len(errors) == 0,
                added_to_kg=added_to_kg,
                rag_chunks_added=rag_chunks,
                errors=errors,
                processing_time=processing_time,
                extraction_method="hybrid_graphrag",
                file_changed=file_changed
            )
            
        except Exception as e:
            error_msg = f"Neo4j processing failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            # Update file metadata with failure status
            self._update_file_metadata(file_path, "failed")
            
            return HybridIngestionResult(
                document_id=file_path.stem,
                file_path=str(file_path),
                success=False,
                added_to_kg=False, # KG was not built
                rag_chunks_added=0,
                errors=errors,
                processing_time=time.time() - start_time,
                extraction_method="hybrid_graphrag",
                file_changed=file_changed
            )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_files = len(self.file_metadata)
        successful_files = len([m for m in self.file_metadata.values() if m.processing_status == "success"])
        failed_files = len([m for m in self.file_metadata.values() if m.processing_status == "failed"])
        partial_files = len([m for m in self.file_metadata.values() if m.processing_status == "partial"])
        
        return {
            "total_files_tracked": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "partial_files": partial_files,
            "neo4j_uri": self.neo4j_uri,
            "node_labels": self.node_labels,
            "relationship_types": self.rel_types,
            "rag_available": self.rag is not None,
            "document_processor_available": self.document_processor is not None,
            "metadata_file": str(self.metadata_file)
        }
    
    def reset_processed_files(self):
        """Reset the file metadata tracking."""
        self.file_metadata.clear()
        self._save_file_metadata()
        logger.info("Reset file metadata tracking")
    
    def add_processed_file(self, file_path: str):
        """Add a file to the processed metadata."""
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            self._update_file_metadata(file_path_obj, "success")
            logger.info(f"Added {file_path} to processed files")
        else:
            logger.warning(f"File {file_path} does not exist")
    
    def get_file_status(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get the processing status of a specific file."""
        if file_path in self.file_metadata:
            metadata = self.file_metadata[file_path]
            return {
                "file_path": metadata.file_path,
                "last_modified": datetime.fromtimestamp(metadata.last_modified).isoformat(),
                "file_size": metadata.file_size,
                "file_hash": metadata.file_hash[:16] + "...",
                "last_processed": datetime.fromtimestamp(metadata.last_processed).isoformat(),
                "processing_status": metadata.processing_status,
                "has_changed": self._has_file_changed(Path(file_path))
            }
        return None
    
    def force_reprocess_file(self, file_path: str):
        """Force reprocessing of a specific file by removing its metadata."""
        if file_path in self.file_metadata:
            del self.file_metadata[file_path]
            self._save_file_metadata()
            logger.info(f"Removed metadata for {file_path}, will be reprocessed")
        else:
            logger.info(f"File {file_path} not in metadata, will be processed")
    
    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close() 