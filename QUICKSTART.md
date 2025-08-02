# NASA Query System - Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd nasa-query-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional)
   ```bash
   export OPENAI_API_KEY="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"
   export ANTHROPIC_API_KEY="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"
   ```

## Quick Start

### 1. Basic Query

Ask a question about NASA missions:

```bash
python main.py query "When was Voyager 1 launched?"
```

### 2. Interactive Mode

Start an interactive session:

```bash
python main.py interactive
```

Then ask questions like:
- "What missions studied Mars?"
- "Which missions used ion propulsion?"
- "Explain how NASA studies exoplanets"

### 3. Ingest Documents

Add your own NASA documents:

```bash
# Place PDF, JSON, Markdown, or text files in data/documents/
python main.py ingest
```

### 4. View System Statistics

```bash
python main.py stats
```

## Example Queries

### Factual Queries (KB)
- "When was Voyager 1 launched?"
- "What is the launch date of Mars Curiosity?"
- "How many planets did Voyager 1 study?"

### Relational Queries (KG)
- "Which missions studied Mars and used ion propulsion?"
- "What missions are similar to Voyager 1?"
- "How are Curiosity and Perseverance related?"

### Generative Queries (RAG)
- "Explain how NASA studies exoplanets"
- "Describe the Mars exploration program"
- "Tell me about ion propulsion technology"

## Debug Mode

Enable debug mode to see routing decisions and processing details:

```bash
python main.py --debug query "When was Voyager 1 launched?"
```

## Configuration

Edit `config/config.yaml` to customize:

- LLM providers and models
- Routing thresholds
- Storage paths
- Processing parameters

## Sample Data

The system comes with pre-loaded NASA data including:

- **Missions**: Voyager 1 & 2, Mars Curiosity, Hubble, James Webb
- **Planets**: Mars, Jupiter, Saturn
- **Technologies**: Ion propulsion, nuclear power, solar panels
- **Documents**: Exoplanet research, mission overviews

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the project directory
2. **API key errors**: Set environment variables or update config
3. **Missing dependencies**: Run `pip install -r requirements.txt`

### Getting Help

- Check the logs in `logs/nasa_query.log`
- Run with debug mode: `python main.py --debug interactive`
- View system stats: `python main.py stats`

## Next Steps

1. **Add your own documents** to `data/documents/`
2. **Customize the configuration** in `config/config.yaml`
3. **Explore the architecture** in `docs/architecture.md`
4. **Run tests** with `pytest tests/`

## Advanced Usage

### Watch Directory for New Documents

```bash
python main.py watch
```

### Use Different LLM Models

```bash
python main.py query "Your question" --model claude
```

### Custom Configuration

```bash
python main.py --config my_config.yaml interactive
```

## System Architecture

The system uses a hybrid approach:

- **Knowledge Base (KB)**: For factual queries
- **Knowledge Graph (KG)**: For relational queries  
- **RAG System**: For generative queries
- **Query Router**: Intelligently routes to appropriate subsystem

Each query is automatically routed to the best subsystem based on intent analysis. 