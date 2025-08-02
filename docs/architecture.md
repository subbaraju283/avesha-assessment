# NASA Documentation Query System - Architecture

## Overview

The NASA Documentation Query System is a production-grade hybrid architecture that combines Knowledge Base (KB), Knowledge Graph (KG), and Retrieval-Augmented Generation (RAG) to provide intelligent querying capabilities over NASA documentation.

## System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query Router  │    │  Knowledge Base │    │ Knowledge Graph │
│                 │    │                 │    │                 │
│ • Intent        │    │ • Factual Data  │    │ • Entity        │
│ • Classification│    │ • Structured    │    │ • Relationships │
│ • Routing       │    │ • Exact Match   │    │ • Path Finding  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   RAG System    │
                    │                 │
                    │ • Document      │
                    │ • Vector Search │
                    │ • Generation    │
                    └─────────────────┘
```

### Data Flow

1. **Query Input**: User submits natural language query
2. **Query Router**: Analyzes intent and routes to appropriate subsystem
3. **Subsystem Processing**: KB, KG, or RAG processes the query
4. **Response Generation**: LLM generates final answer
5. **Result Display**: Formatted response with debug information

## Component Details

### 1. Query Router

**Purpose**: Intelligently routes queries to the most appropriate subsystem based on intent and complexity.

**Key Features**:
- Intent classification using LLM and rule-based fallback
- Query complexity analysis
- Confidence-based routing decisions
- Debug mode for routing trace

**Routing Logic**:
- **KB**: Factual queries (dates, names, specifications)
- **KG**: Relational queries (connections, comparisons)
- **RAG**: Generative queries (explanations, synthesis)

### 2. Knowledge Base (KB)

**Purpose**: Handles factual, static Q&A with structured data.

**Components**:
- **Fact Store**: Persistent storage for structured facts
- **Embedding Model**: Semantic similarity for fact retrieval
- **LLM Integration**: Answer generation from retrieved facts

**Data Structure**:
```python
@dataclass
class Fact:
    id: str
    subject: str
    predicate: str
    object: str
    source: str
    confidence: float
    metadata: Dict[str, Any]
```

**Example Facts**:
- Voyager 1 launch_date 1977-09-05
- Mars Curiosity uses_technology nuclear power
- Hubble Space Telescope studied_planet Jupiter

### 3. Knowledge Graph (KG)

**Purpose**: Manages relational reasoning and entity connections.

**Components**:
- **Graph Store**: Persistent storage for entities and relationships
- **NetworkX Integration**: Path finding and graph algorithms
- **LLM Integration**: Relationship analysis and answer generation

**Data Structure**:
```python
@dataclass
class Entity:
    id: str
    name: str
    type: str
    properties: Dict[str, Any]
    source: str

@dataclass
class Relationship:
    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]
    source: str
```

**Example Relationships**:
- Voyager 1 STUDIED Jupiter
- Mars Curiosity USED_TECHNOLOGY nuclear power
- Voyager 1 SIMILAR_TO Voyager 2

### 4. RAG System

**Purpose**: Provides generative answers and synthesis from document context.

**Components**:
- **Vector Store**: Document chunk storage with embeddings
- **Embedding Model**: Semantic similarity for retrieval
- **LLM Integration**: Answer generation from retrieved chunks

**Data Structure**:
```python
@dataclass
class DocumentChunk:
    id: str
    content: str
    source: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]]
```

### 5. Ingestion Pipeline

**Purpose**: Automatically processes new documents into appropriate subsystems.

**Features**:
- Multi-format support (PDF, JSON, Markdown, TXT, DOCX)
- Batch processing with configurable workers
- Automatic routing to KB, KG, or RAG based on content
- Directory watching for real-time ingestion

**Processing Flow**:
1. **Content Extraction**: Extract text from various file formats
2. **Content Parsing**: Parse into structured data (facts, entities, relationships)
3. **Subsystem Routing**: Route data to appropriate subsystems
4. **Indexing**: Update search indices and embeddings

## LLM Integration

### Model Abstraction

The system supports multiple LLM providers through a unified interface:

- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude-3-sonnet, Claude-3-haiku
- **Local**: LLaMA, other local models

### Provider Configuration

```yaml
llm:
  default_provider: "openai"
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4"
      temperature: 0.1
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      model: "claude-3-sonnet-20240229"
```

## Data Storage

### File Structure

```
data/
├── kb/                    # Knowledge Base storage
│   ├── facts.json        # Structured facts
│   └── index.pkl         # Search index
├── kg/                    # Knowledge Graph storage
│   ├── entities.json     # Graph entities
│   ├── relationships.json # Graph relationships
│   └── index.pkl         # Graph index
├── rag/                   # RAG system storage
│   ├── chunks.json       # Document chunks
│   └── embeddings.pkl    # Vector embeddings
└── documents/             # User document storage
    ├── *.pdf
    ├── *.json
    ├── *.md
    └── *.txt
```

### Persistence Strategy

- **JSON**: Human-readable storage for facts, entities, relationships
- **Pickle**: Binary storage for embeddings and indices
- **Incremental Updates**: Support for adding new data without full rebuild

## Configuration Management

### Configuration Structure

```yaml
# LLM Configuration
llm:
  default_provider: "openai"
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4"

# Router Configuration
router:
  kb_threshold: 0.7
  kg_threshold: 0.6
  rag_threshold: 0.4

# Subsystem Configurations
kb:
  storage_path: "data/kb"
  similarity_threshold: 0.8

kg:
  max_depth: 3
  max_paths: 5

rag:
  chunk_size: 1000
  retrieval_k: 5
```

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `DEBUG`: Enable debug mode
- `LOG_LEVEL`: Logging level

## Performance Considerations

### Scalability

- **Async Processing**: Non-blocking query processing
- **Batch Operations**: Efficient document ingestion
- **Caching**: Embedding and result caching
- **Indexing**: Fast retrieval from large datasets

### Optimization

- **Vector Similarity**: Efficient cosine similarity calculations
- **Graph Algorithms**: Optimized path finding
- **Memory Management**: Lazy loading of large datasets
- **Parallel Processing**: Multi-worker document processing

## Security Considerations

### API Key Management

- Environment variable storage
- Secure configuration loading
- No hardcoded credentials

### Data Privacy

- Local storage of sensitive data
- No external data transmission beyond LLM APIs
- Configurable data retention policies

## Monitoring and Debugging

### Debug Mode

- Query routing decisions
- Subsystem selection reasoning
- Retrieval trace
- Performance metrics

### Logging

- Structured logging with JSON format
- Configurable log levels
- File and console output
- Performance monitoring

### Statistics

- System usage metrics
- Subsystem performance
- Data volume statistics
- Query success rates

## Deployment Considerations

### Dependencies

- Python 3.8+
- Required packages in requirements.txt
- Optional: Neo4j for graph database
- Optional: ChromaDB for vector database

### Environment Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables
3. Configure config/config.yaml
4. Initialize data directories
5. Run system: `python main.py interactive`

### Production Deployment

- **Docker**: Containerized deployment
- **API Server**: FastAPI integration
- **Load Balancing**: Multiple instances
- **Monitoring**: Prometheus metrics
- **Backup**: Automated data backups

## Future Enhancements

### Planned Features

1. **Multi-modal Support**: Image and video processing
2. **Real-time Updates**: Live data ingestion
3. **Advanced Analytics**: Query pattern analysis
4. **Custom Models**: Fine-tuned domain models
5. **API Integration**: RESTful API endpoints

### Scalability Improvements

1. **Distributed Processing**: Multi-node deployment
2. **Database Integration**: PostgreSQL, MongoDB
3. **Caching Layer**: Redis integration
4. **CDN Integration**: Static asset delivery
5. **Microservices**: Component separation 