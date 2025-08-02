# NASA Query System - Hybrid KB/KG/RAG Architecture

A production-grade system for querying NASA documentation using a hybrid architecture combining Knowledge Base (KB), Knowledge Graph (KG), and Retrieval-Augmented Generation (RAG) with intelligent query routing.

## 🏗️ Architecture Overview

### **Hybrid Query Processing System**

The system uses a three-tier architecture to handle different types of queries:

1. **Knowledge Base (KB)** - For factual, static Q&A
2. **Knowledge Graph (KG)** - For relational reasoning and pathfinding  
3. **Retrieval-Augmented Generation (RAG)** - For generative answers and synthesis

### **Intelligent Query Router**

The system automatically routes queries to the most appropriate subsystem based on:
- Query complexity and structure
- Semantic analysis using LLMs
- Rule-based pattern matching
- Confidence scoring

## 🎯 Key Features

### **Multi-Layered Query Handling**
- **KB**: Exact factual Q&A (mission dates, spacecraft specs)
- **KG**: Relational reasoning ("Which missions studied Mars and used ion propulsion?")
- **RAG**: Generative answers and explanations ("Explain how NASA studies exoplanets")

### **Runtime Extensibility**
- Automatically index new documents (PDF, JSON, Markdown, TXT, DOCX)
- Drop files into `data/documents/` for automatic processing
- Hybrid ingestion pipeline updates all subsystems

### **Cloud-Native Vector Storage**
- **Pinecone Integration**: Scalable vector database for RAG
- **Neo4j Knowledge Graph**: Graph database for relational data
- **Local Knowledge Base**: JSON-based fact storage

### **CLI Interface with Visual Debugging**
- Rich terminal interface with progress indicators
- Debug mode for path visualization
- Real-time query routing display

## 🏛️ System Architecture

```
                    ┌─────────────────┐
                    │   Query Router  │
                    │                 │
                    │ • Intent        │
                    │ • Classification│
                    │ • Routing       │
                    └─────────────────┘
                              │
                    ┌────────────────────────────────────────────┐
                    │                   │                        │
         ┌─────────────────┐ │ ┌─────────────────┐ │ ┌─────────────────┐
         │  Knowledge Base │ │ │ Knowledge Graph │ │ │   RAG System    │
         │                 │ │ │                 │ │ │                 │
         │ • Fact Storage  │ │ │ • Neo4j Graph   │ │ │ • Pinecone      │
         │ • Embedding     │ │ │ • GraphRAG      │ │ │ • LangChain     │
         │ • Similarity    │ │ │ • Vector Index  │ │ │ • OpenAI LLM    │
         └─────────────────┘ │ └─────────────────┘ │ └─────────────────┘
                    
```

## 📁 Project Structure

```
avesha-v2/
├── src/
│   ├── kb/                    # Knowledge Base (Local Storage)
│   │   ├── knowledge_base.py  # Main KB system
│   │   ├── fact_store.py      # Local JSON fact storage
│   │   └── models.py          # KB data models
│   ├── kg/                    # Knowledge Graph (Cloud Storage)
│   │   ├── neo4j_knowledge_graph.py  # Neo4j AuraDB integration
│   │   └── models.py          # KG data models
│   ├── rag/                   # RAG System (Cloud Storage)
│   │   ├── pinecone_rag_system.py    # Pinecone vector database
│   │   └── models.py          # RAG data models
│   ├── router/                # Query Router
│   │   ├── query_router.py    # Main router with KG/RAG prioritization
│   │   └── intent_classifier.py      # Intent classification
│   ├── ingestion/             # Document Processing
│   │   ├── hybrid_ingestion_pipeline.py  # Batch/parallel processing
│   │   └── document_processor.py    # Multi-format document processing
│   └── models/                # LLM Management
│       ├── llm_manager.py     # LLM abstraction layer
│       └── providers.py       # OpenAI/Anthropic providers
├── config/
│   └── config.yaml           # System configuration
├── data/
│   ├── documents/            # Input documents
│   ├── kb/                   # Local KB storage (JSON files)
│   └── ingestion/            # File metadata tracking
├── logs/                     # System logs
├── main.py                   # CLI interface
├── requirements.txt          # Dependencies
├── requirements-frozen.txt   # Pinned dependencies
├── Dockerfile               # Docker configuration
└── .dockerignore           # Docker ignore rules
```

## 💾 Storage Architecture

### **Local Storage**
- **Knowledge Base Facts**: Stored locally in `data/kb/` as JSON files
- **File Metadata**: Stored locally in `data/ingestion/file_metadata.json`
- **System Logs**: Stored locally in `logs/` directory

### **Cloud Storage**
- **Knowledge Graph**: Neo4j AuraDB cloud database
  - Database URL: `neo4j+s://72bbcca1.databases.neo4j.io`
  - Entities and relationships stored in cloud
  - Requires `NEO4J_PASSWORD` in `.env` file
- **Vector Database**: Pinecone cloud vector database
  - Document chunks and embeddings stored in cloud
  - Requires `PINECONE_API_KEY` in `.env` file
  - Automatic scaling and management

## 🚀 Quick Start

### **1. Setup Environment**
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
python setup_env.py
```

### **2. Setup External Services**
- **Neo4j AuraDB**: Create a cloud database and get connection details
- **Pinecone**: Create an index and get API key
- **OpenAI/Anthropic**: Get API keys for LLM services

### **3. Configure Environment**
Create `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
PINECONE_API_KEY=your_pinecone_key
NEO4J_PASSWORD=your_neo4j_password
```

### **4. Initial Data Ingestion**
```bash
# Ingest initial NASA knowledge data
python main.py ingest-initial

# Or ingest all documents in data/documents/
python main.py ingest data/documents/
```

### **5. Start Querying**
```bash
# Interactive mode
python main.py interactive

# Single query
python main.py query "When was Voyager 1 launched?"
```

## 🔧 Configuration

### **System Configuration (`config/config.yaml`)**

```yaml
# LLM Configuration
llm:
  default_provider: "openai"
  openai:
    model: "gpt-4"
    temperature: 0.1
    api_key: "${OPENAI_API_KEY}"

# Query Router
router:
  kb_threshold: 0.7
  kg_threshold: 0.6
  rag_threshold: 0.4

# Knowledge Base
kb:
  storage_path: "data/kb"
  similarity_threshold: 0.8
  max_results: 10

# Knowledge Graph
kg:
  database_url: "bolt://localhost:7687"
  username: "neo4j"
  password: "password"

# RAG System
rag:
  pinecone_index_name: "nasa-docs"
  pinecone_api_key: "${PINECONE_API_KEY}"
  chunk_size: 500
  chunk_overlap: 50
  retrieval_k: 5
```

## 📊 Query Examples

### **Knowledge Base Queries** (Factual Questions)
```bash
# Mission launch dates and basic facts
python main.py query "When was Apollo 11 launched?"
python main.py query "What is the launch date of Voyager 1?"
python main.py query "When was the Hubble Space Telescope launched?"
python main.py query "What is the launch date of Mars Curiosity?"
python main.py query "When was the James Webb Space Telescope launched?"
```

### **Knowledge Graph Queries** (Relational Reasoning)
```bash
# Mission relationships and connections
python main.py query "Which missions used ion propulsion technology?"
python main.py query "Which missions use nuclear power technology?"
python main.py query "What missions explore Mars and what do they study?"
python main.py query "How do Voyager missions connect to planetary exploration?"
python main.py query "Which missions are related to interstellar exploration?"
```

### **RAG Queries** (Generative Explanations)
```bash
# Detailed mission explanations
python main.py query "Explain the Apollo 11 mission and its significance"
python main.py query "Explain the Hubble Space Telescope and its contributions to astronomy"
python main.py query "How do spacecraft navigate in deep space?"
python main.py query "What is the future of space telescope technology?"
```

### **Duplicate Management**
```bash
# Check for duplicates in Pinecone index
python main.py stats

# Clean up duplicates with default threshold (0.95)
python main.py cleanup-duplicates

# Clean up duplicates with custom threshold
python main.py cleanup-duplicates --similarity-threshold 0.90
```

## 📈 System Statistics

Check system status and statistics:

```bash
python main.py stats
```

**Shows:**
- Total documents processed
- KB facts count
- KG entities and relationships
- Pinecone index statistics
- Processing times

## 🔄 Document Ingestion

### **Supported Formats**
- **PDF**: Research papers, reports
- **TXT**: Plain text documents
- **MD**: Markdown files
- **JSON**: Structured data
- **DOCX**: Word documents

### **Automatic Processing**
1. **Content Extraction**: Extract text from documents
2. **Chunking**: Split into manageable chunks
3. **Embedding**: Generate vector embeddings
4. **Graph Extraction**: Extract entities and relationships
5. **Storage**: Update all subsystems

### **Ingestion Commands**
```bash
# Ingest all documents (sequential processing)
python main.py ingest

# Ingest documents in parallel (faster for multiple files)
python main.py ingest-parallel --max-concurrent 5

# Ingest documents in batches (optimized resource usage)
python main.py ingest-batch --batch-size 10

# Ingest specific file
python main.py ingest-file document.pdf

# Watch directory for changes
python main.py watch
```

## 📦 **Batch Processing & File Management**

### **Advanced File Processing Options**

The system provides three sophisticated processing strategies optimized for different scenarios:

#### **1. Sequential Processing (Default)**
```bash
python main.py ingest
```
- **Best for**: Small document sets, debugging, reliable processing
- **Pros**: Simple, predictable, easy error tracking, memory efficient
- **Cons**: Slower for multiple files
- **Use case**: Development, testing, small document sets

#### **2. Parallel Processing**
```bash
python main.py ingest-parallel --max-concurrent 5
```
- **Best for**: Multiple files, faster processing
- **Pros**: Significantly faster (2-5x), concurrent operations, individual error isolation
- **Cons**: Higher resource usage, potential API rate limits
- **Use case**: Medium document sets, production environments

#### **3. Batch Processing**
```bash
python main.py ingest-batch --batch-size 10
```
- **Best for**: Large document sets, optimized resource usage
- **Pros**: Efficient memory usage, optimized API calls, parallel KG processing
- **Cons**: More complex error handling
- **Use case**: Large document sets, resource-constrained environments

### **File Change Detection System**

The system intelligently tracks file changes to avoid reprocessing unchanged files:

#### **Metadata Tracking**
```json
{
  "data/documents/voyager_program.pdf": {
    "file_path": "data/documents/voyager_program.pdf",
    "last_modified": 1703123456.789,
    "file_size": 15360,
    "file_hash": "a1b2c3d4e5f6...",
    "last_processed": 1703123500.123,
    "processing_status": "success"
  }
}
```

#### **Change Detection Methods**
- **Timestamp Comparison**: File modification time
- **Size Comparison**: File size changes
- **Content Hash**: SHA-256 hash for content changes
- **Multi-level Validation**: All three checks must pass

#### **File Management Commands**
```bash
# Check file processing status
python main.py file-status data/documents/voyager_program.pdf

# Force reprocess specific file
python main.py force-reprocess data/documents/voyager_program.pdf

# Reset all file metadata (reprocess everything)
python main.py reset-metadata

# Check duplicate vectors in Pinecone
python main.py cleanup-duplicates --similarity-threshold 0.95
```

### **Processing Flow Comparison**

#### **Sequential Processing**
```
File 1 → KG → RAG → Complete
File 2 → KG → RAG → Complete  
File 3 → KG → RAG → Complete
```

#### **Parallel Processing**
```
File 1 → KG → RAG ↗
File 2 → KG → RAG → All Complete
File 3 → KG → RAG ↗
```

#### **Batch Processing**
```
Batch 1: [File1, File2, File3] → Parallel KG → Batch RAG → Complete
Batch 2: [File4, File5] → Parallel KG → Batch RAG → Complete
```

### **Performance Optimization Features**

#### **Memory Management**
- **Sequential**: Low memory usage, one file at a time
- **Parallel**: Moderate memory usage, controlled concurrency
- **Batch**: Optimized memory usage, batch operations

#### **API Efficiency**
- **Sequential**: Individual API calls
- **Parallel**: Concurrent API calls with rate limiting
- **Batch**: Batched API calls to Pinecone

#### **Error Handling**
- **Sequential**: Stop on first error
- **Parallel**: Continue processing other files
- **Batch**: Batch-level error isolation

### **Resource Usage Guidelines**

#### **Small Document Sets (< 10 files)**
```bash
python main.py ingest  # Sequential processing
```

#### **Medium Document Sets (10-50 files)**
```bash
python main.py ingest-parallel --max-concurrent 3-5
```