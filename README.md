# NASA Documentation Query System

A production-grade system for querying NASA documentation using a hybrid architecture combining Knowledge Base (KB), Knowledge Graph (KG), and Retrieval-Augmented Generation (RAG) with intelligent query routing.

## Architecture Overview

### Core Components

1. **Query Router**: Intelligently routes queries to KB, KG, or RAG based on query intent and complexity
2. **Knowledge Base (KB)**: Handles factual, static Q&A with structured data
3. **Knowledge Graph (KG)**: Manages relational reasoning and entity connections
4. **RAG System**: Provides generative answers and synthesis from document context
5. **Data Ingestion Pipeline**: Automatically processes new documents into appropriate subsystems

### System Flow

```
User Query → Query Router → [KB/KG/RAG] → Response
                ↓
        Debug Mode: Visual trace
```

## Features

- **Multi-layered Query Handling**: Route queries to appropriate subsystem based on intent
- **Runtime Extensibility**: Drop new documents into `data/` folder for automatic indexing
- **CLI Interface**: Interactive command-line interface with visual debugging
- **Model Abstraction**: Configurable LLM backend (GPT-4, Claude, LLaMA)
- **Production Ingestion**: Async pipeline for document processing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Query

```bash
python main.py --query "What missions studied Mars?"
```

### Debug Mode

```bash
python main.py --query "Which missions used ion propulsion?" --debug
```

### Add New Documents

```bash
# Drop PDF, JSON, or Markdown files into data/ folder
python main.py --ingest
```

### Configure LLM

```bash
python main.py --query "Explain exoplanet detection" --model claude
```

## Project Structure

```
nasa-query-system/
├── src/
│   ├── router/          # Query routing logic
│   ├── kb/             # Knowledge Base implementation
│   ├── kg/             # Knowledge Graph implementation
│   ├── rag/            # RAG system implementation
│   ├── ingestion/      # Document processing pipeline
│   └── models/         # LLM abstraction layer
├── data/               # Document storage
├── config/             # Configuration files
├── docs/               # Architectural documentation
└── tests/              # Test suite
```

## Configuration

Edit `config/config.yaml` to customize:
- LLM providers and models
- Vector database settings
- Graph database configuration
- Query routing thresholds

## Development

```bash
# Run tests
pytest tests/

# Run with development mode
python main.py --dev --query "test query"
``` 