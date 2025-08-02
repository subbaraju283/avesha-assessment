#!/usr/bin/env python3
"""
NASA Query System - Hybrid KB/KG/RAG System
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

import click
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Setup logger
logger = logging.getLogger(__name__)

from src.models.llm_manager import LLMManager
from src.router.query_router import QueryRouter
from src.kb.knowledge_base import KnowledgeBase
from src.kg.neo4j_knowledge_graph import Neo4jKnowledgeGraph
from src.ingestion.hybrid_ingestion_pipeline import HybridIngestionPipeline, HybridIngestionResult


def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing configuration file: {e}")
        sys.exit(1)


def setup_logging(config: dict):
    """Setup logging configuration."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get("file", "logs/nasa_query_system.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


class NASAQuerySystem:
    """Main NASA Query System class."""
    
    def __init__(self, config: dict):
        self.config = config
        self.console = Console()
        
        # Initialize components
        self.llm_manager = LLMManager(config)  # Pass entire config
        self.router = QueryRouter(config["router"], self.llm_manager)
        self.kb = KnowledgeBase(config["kb"], self.llm_manager)
        self.kg = Neo4jKnowledgeGraph(config["kg"], self.llm_manager)
        # Initialize RAG system
        try:
            from src.rag.pinecone_rag_system import PineconeRAGSystem
            self.rag = PineconeRAGSystem(config["rag"], self.llm_manager)
            logger.info("Pinecone RAG system initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Pinecone RAG system: {e}")
            self.rag = None
        self.ingestion = HybridIngestionPipeline(config)
        
        self.debug_mode = config.get("debug", {}).get("enabled", False)
    
    async def query(self, user_query: str, model: Optional[str] = None, debug: bool = False) -> dict:
        """
        Process a user query through the hybrid system.
        
        Args:
            user_query: The user's natural language query
            model: Optional LLM model to use
            debug: Whether to enable debug mode
            
        Returns:
            Dictionary with query results and metadata
        """
        debug = debug or self.debug_mode
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Step 1: Route query
            task = progress.add_task("Routing query...", total=None)
            routing_decision = await self.router.route_query(user_query, debug=debug)
            progress.update(task, description=f"Routed to {routing_decision.subsystem}")
            
            # Step 2: Process query based on routing decision
            if routing_decision.subsystem == "KB":
                task = progress.add_task("Querying Knowledge Base...", total=None)
                result = await self.kb.query(user_query, debug=debug)
                progress.update(task, description="KB query complete")
                
            elif routing_decision.subsystem == "KG":
                task = progress.add_task("Querying Knowledge Graph...", total=None)
                result = await self.kg.query(user_query, debug=debug)
                progress.update(task, description="KG query complete")
                
            else:  # RAG
                task = progress.add_task("Querying RAG system...", total=None)
                result = await self.rag.query(user_query, debug=debug)
                progress.update(task, description="RAG query complete")
            
            # Step 3: Format response
            response = {
                "query": user_query,
                "routing_decision": {
                    "subsystem": routing_decision.subsystem,
                    "confidence": routing_decision.confidence,
                    "reasoning": routing_decision.reasoning,
                    "query_type": routing_decision.query_type.value
                },
                "result": result,
                "debug_info": {
                    "routing_metadata": routing_decision.metadata,
                    "result_metadata": getattr(result, 'metadata', {})
                } if debug else {}
            }
            
            return response
    
    def display_result(self, response: dict, debug: bool = False):
        """Display query results in a formatted way."""
        query = response["query"]
        routing = response["routing_decision"]
        result = response["result"]
        
        # Create main result panel
        if hasattr(result, 'answer'):
            # RAG result
            answer = result.answer
        elif hasattr(result, 'facts'):
            # KB result
            answer = getattr(result, 'metadata', {}).get('response_text', 'No answer generated')
        elif hasattr(result, 'entities'):
            # KG result
            answer = getattr(result, 'metadata', {}).get('response_text', 'No answer generated')
        else:
            answer = "Unable to generate answer"
        
        # Display routing information
        routing_table = Table(title="Query Routing Information")
        routing_table.add_column("Property", style="cyan")
        routing_table.add_column("Value", style="white")
        
        routing_table.add_row("Subsystem", routing["subsystem"])
        routing_table.add_row("Confidence", f"{routing['confidence']:.2f}")
        routing_table.add_row("Query Type", routing["query_type"])
        routing_table.add_row("Reasoning", routing["reasoning"])
        
        self.console.print(routing_table)
        
        # Display answer
        answer_panel = Panel(
            answer,
            title="[bold blue]Answer[/bold blue]",
            border_style="blue"
        )
        self.console.print(answer_panel)
        
        # Display debug information
        if debug and response.get("debug_info"):
            debug_table = Table(title="Debug Information")
            debug_table.add_column("Component", style="cyan")
            debug_table.add_column("Details", style="white")
            
            debug_info = response["debug_info"]
            
            # Routing metadata
            if "routing_metadata" in debug_info:
                routing_meta = debug_info["routing_metadata"]
                debug_table.add_row("Query Characteristics", str(routing_meta.get("characteristics", {})))
                debug_table.add_row("Intent Scores", str(routing_meta.get("intent_scores", {})))
            
            # Result metadata
            if "result_metadata" in debug_info:
                result_meta = debug_info["result_metadata"]
                debug_table.add_row("Result Metadata", str(result_meta))
            
            self.console.print(debug_table)
    
    async def ingest_documents(self, data_path: str = "data/documents"):
        """Ingest documents from the specified directory."""
        self.console.print(f"[yellow]Processing documents from {data_path}...[/yellow]")
        
        results = await self.ingestion.process_documents(data_path)
        
        if not results:
            self.console.print("[green]No new documents to process.[/green]")
            return
        
        # Use the proper display method that includes KB Facts
        self.display_ingestion_results(results)
    
    async def ingest_single_file(self, file_path: str):
        """Ingest a single file."""
        try:
            results = await self.ingestion.process_documents(file_path)
            self.display_ingestion_results(results)
        except Exception as e:
            self.console.print(f"[red]Error ingesting file: {e}[/red]")
    
    def display_ingestion_results(self, results: List):
        """Display ingestion results in a formatted table."""
        if not results:
            return
        
        # Create results table
        table = Table(title="Ingestion Results")
        table.add_column("File", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("KG Added", style="blue")
        table.add_column("RAG Chunks", style="magenta")
        table.add_column("KB Facts", style="yellow")
        table.add_column("Time (s)", style="white")
        table.add_column("Method", style="dim")
        
        total_success = 0
        total_kg_added = 0
        total_rag_chunks = 0
        total_kb_facts = 0
        total_time = 0
        
        for result in results:
            status = "✅ Success" if result.success else "❌ Failed"
            kg_status = "✅ Yes" if result.added_to_kg else "❌ No"
            
            table.add_row(
                Path(result.file_path).name,
                status,
                kg_status,
                str(result.rag_chunks_added),
                str(result.kb_facts_added),
                f"{result.processing_time:.2f}",
                result.extraction_method
            )
            
            if result.success:
                total_success += 1
            if result.added_to_kg:
                total_kg_added += 1
            total_rag_chunks += result.rag_chunks_added
            total_kb_facts += result.kb_facts_added
            total_time += result.processing_time
        
        self.console.print(table)
        
        # Summary
        summary = Table(title="Summary")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="white")
        
        summary.add_row("Total Files", str(len(results)))
        summary.add_row("Successful", str(total_success))
        summary.add_row("Failed", str(len(results) - total_success))
        summary.add_row("Added to KG", str(total_kg_added))
        summary.add_row("Total RAG Chunks", str(total_rag_chunks))
        summary.add_row("Total KB Facts", str(total_kb_facts))
        summary.add_row("Total Time", f"{total_time:.2f}s")
        summary.add_row("Avg Time/File", f"{total_time/len(results):.2f}s")
        
        self.console.print(summary)
        
        # Show errors if any
        errors = []
        for result in results:
            if result.errors:
                errors.extend([f"{Path(result.file_path).name}: {error}" for error in result.errors])
        
        if errors:
            self.console.print("\n[red]Errors:[/red]")
            for error in errors:
                self.console.print(f"  • {error}")

    def show_stats(self):
        """Display system statistics."""
        kb_stats = self.kb.get_stats()
        kg_stats = self.kg.get_stats()
        rag_stats = self.rag.get_stats()
        router_stats = self.router.get_routing_stats()
        ingestion_stats = self.ingestion.get_processing_stats()
        
        # KB Stats
        kb_table = Table(title="Knowledge Base Statistics")
        kb_table.add_column("Metric", style="cyan")
        kb_table.add_column("Value", style="white")
        
        kb_table.add_row("Total Facts", str(kb_stats["total_facts"]))
        kb_table.add_row("Subjects", str(len(kb_stats["subjects"])))
        kb_table.add_row("Predicates", str(len(kb_stats["predicates"])))
        kb_table.add_row("Sources", str(len(kb_stats["sources"])))
        
        # KG Stats
        kg_table = Table(title="Knowledge Graph Statistics")
        kg_table.add_column("Metric", style="cyan")
        kg_table.add_column("Value", style="white")
        
        kg_table.add_row("Total Entities", str(kg_stats["total_entities"]))
        kg_table.add_row("Total Relationships", str(kg_stats["total_relationships"]))
        kg_table.add_row("Entity Types", str(len(kg_stats["entity_types"])))
        kg_table.add_row("Relationship Types", str(len(kg_stats["relationship_types"])))
        
        # RAG Stats
        rag_table = Table(title="RAG System Statistics")
        rag_table.add_column("Metric", style="cyan")
        rag_table.add_column("Value", style="white")
        
        if "error" not in rag_stats:
            rag_table.add_row("Index Name", rag_stats.get("index_name", "Unknown"))
            rag_table.add_row("Total Vectors", str(rag_stats.get("total_vector_count", 0)))
            rag_table.add_row("Dimension", str(rag_stats.get("dimension", 0)))
            rag_table.add_row("Metric", rag_stats.get("metric", "Unknown"))
            
            # Add duplicate stats if available
            if "duplicate_stats" in rag_stats:
                dup_stats = rag_stats["duplicate_stats"]
                if "error" not in dup_stats:
                    rag_table.add_row("Potential Duplicates", str(dup_stats.get("potential_duplicates", 0)))
                    rag_table.add_row("Duplicate Groups", str(dup_stats.get("duplicate_groups", 0)))
                    rag_table.add_row("Documents Analyzed", str(dup_stats.get("documents_analyzed", 0)))
        else:
            rag_table.add_row("Status", f"Error: {rag_stats['error']}")
        
        # Ingestion Stats
        ingestion_table = Table(title="File Processing Statistics")
        ingestion_table.add_column("Metric", style="cyan")
        ingestion_table.add_column("Value", style="white")
        
        ingestion_table.add_row("Total Files Tracked", str(ingestion_stats["total_files_tracked"]))
        ingestion_table.add_row("Successful Files", str(ingestion_stats["successful_files"]))
        ingestion_table.add_row("Failed Files", str(ingestion_stats["failed_files"]))
        ingestion_table.add_row("Partial Files", str(ingestion_stats["partial_files"]))
        ingestion_table.add_row("Metadata File", ingestion_stats["metadata_file"])
        
        self.console.print(kb_table)
        self.console.print(kg_table)
        self.console.print(rag_table)
        self.console.print(ingestion_table)
    
    async def interactive_mode(self):
        """Run the system in interactive mode."""
        self.console.print(Panel(
            "[bold blue]NASA Documentation Query System[/bold blue]\n"
            "Ask questions about NASA missions, spacecraft, and space exploration!\n"
            "Type 'quit' to exit, 'stats' for system statistics, 'help' for commands.",
            border_style="blue"
        ))
        
        while True:
            try:
                query = click.prompt("\n[bold cyan]Query[/bold cyan]")
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'stats':
                    self.show_stats()
                    continue
                elif query.lower() == 'help':
                    self.console.print("""
                    [bold]Available Commands:[/bold]
                    • Ask any question about NASA missions, spacecraft, planets, etc.
                    • 'stats' - Show system statistics
                    • 'help' - Show this help message
                    • 'quit' - Exit the system
                    """)
                    continue
                elif not query.strip():
                    continue
                
                # Process query
                response = await self.query(query, debug=self.debug_mode)
                self.display_result(response, debug=self.debug_mode)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Exiting...[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")


@click.group()
@click.option('--config', '-c', default='config/config.yaml', help='Configuration file path')
@click.option('--debug', '-d', is_flag=True, help='Enable debug mode')
@click.pass_context
def cli(ctx, config, debug):
    """NASA Documentation Query System CLI."""
    # Load environment variables first
    load_env_file()
    
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)
    ctx.obj['debug'] = debug
    
    # Setup logging
    setup_logging(ctx.obj['config'])
    
    # Enable debug mode in config
    if debug:
        ctx.obj['config']['debug']['enabled'] = True


@cli.command()
@click.argument('query')
@click.option('--model', '-m', help='LLM model to use')
@click.pass_context
def query(ctx, query, model):
    """Query the NASA documentation system."""
    system = NASAQuerySystem(ctx.obj['config'])
    
    async def run_query():
        response = await system.query(query, model=model, debug=ctx.obj['debug'])
        system.display_result(response, debug=ctx.obj['debug'])
    
    asyncio.run(run_query())


@cli.command()
@click.argument('data_path', required=False, default='data/documents')
@click.pass_context
def ingest(ctx, data_path):
    """Ingest documents using sequential processing (original method)."""
    config = ctx.obj['config']
    setup_logging(config)
    
    async def run_ingest():
        system = NASAQuerySystem(config)
        results = await system.ingest_documents(data_path)
        system.display_ingestion_results(results)
    
    asyncio.run(run_ingest())


@cli.command()
@click.argument('data_path', required=False, default='data/documents')
@click.option('--max-concurrent', '-c', default=3, help='Maximum concurrent files to process')
@click.pass_context
def ingest_parallel(ctx, data_path, max_concurrent):
    """Ingest documents using parallel processing for better performance."""
    config = ctx.obj['config']
    setup_logging(config)
    
    async def run_ingest():
        system = NASAQuerySystem(config)
        results = await system.ingestion.process_documents_parallel(data_path, max_concurrent)
        system.display_ingestion_results(results)
    
    asyncio.run(run_ingest())


@cli.command()
@click.argument('data_path', required=False, default='data/documents')
@click.option('--batch-size', '-b', default=5, help='Number of files to process in each batch')
@click.pass_context
def ingest_batch(ctx, data_path, batch_size):
    """Ingest documents using batch processing for optimized resource usage."""
    config = ctx.obj['config']
    setup_logging(config)
    
    async def run_ingest():
        system = NASAQuerySystem(config)
        results = await system.ingestion.process_documents_batch(data_path, batch_size)
        system.display_ingestion_results(results)
    
    asyncio.run(run_ingest())


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.pass_context
def ingest_file(ctx, file_path):
    """Ingest a specific file."""
    system = NASAQuerySystem(ctx.obj['config'])
    
    async def run_ingest():
        await system.ingest_single_file(file_path)
    
    asyncio.run(run_ingest())


@cli.command()
@click.pass_context
def ingest_initial(ctx):
    """Ingest initial NASA knowledge data."""
    system = NASAQuerySystem(ctx.obj['config'])
    
    console = Console()
    console.print("[yellow]Ingesting initial NASA knowledge data...[/yellow]")
    
    # Check if initial data file exists
    initial_data_file = Path("data/documents/nasa_initial_knowledge.txt")
    if not initial_data_file.exists():
        console.print("[red]Initial NASA data file not found: data/documents/nasa_initial_knowledge.txt[/red]")
        console.print("[yellow]Please ensure the file exists before running this command.[/yellow]")
        return
    
    # Ingest the initial data
    try:
        asyncio.run(system.ingest_documents("data/documents"))
        console.print("[green]✅ Successfully ingested initial NASA knowledge data![/green]")
        
        # Show stats
        stats = system.kg.get_stats()
        console.print(f"[blue]Knowledge Graph now contains:[/blue]")
        console.print(f"  • {stats['total_entities']} entities")
        console.print(f"  • {stats['total_relationships']} relationships")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to ingest initial data: {e}[/red]")


@cli.command()
@click.pass_context
def ingest_rag_documents(ctx):
    """Ingest RAG documents for generative answers."""
    system = NASAQuerySystem(ctx.obj['config'])
    
    console = Console()
    console.print("[yellow]Ingesting RAG documents...[/yellow]")
    
    # Check for RAG documents
    rag_files = [
        "data/documents/nasa_mars_exploration.txt",
        "data/documents/voyager_program.txt",
        "data/documents/hubble_space_telescope.txt", 
        "data/documents/ion_propulsion_technology.txt",
        "data/documents/exoplanet_detection_methods.txt"
    ]
    
    missing_files = []
    for file_path in rag_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        console.print("[red]Missing RAG document files:[/red]")
        for file_path in missing_files:
            console.print(f"  • {file_path}")
        console.print("[yellow]Please ensure all RAG documents exist before running this command.[/yellow]")
        return
    
    # Ingest the RAG documents
    try:
        asyncio.run(system.ingest_documents("data/documents"))
        console.print("[green]✅ Successfully ingested RAG documents![/green]")
        
        # Show RAG stats
        rag_stats = system.rag.get_stats()
        console.print(f"[blue]RAG System now contains:[/blue]")
        console.print(f"  • {rag_stats['total_chunks']} document chunks")
        console.print(f"  • {len(rag_stats['sources'])} document sources")
        console.print(f"  • {rag_stats['avg_chunk_length']:.1f} avg words per chunk")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to ingest RAG documents: {e}[/red]")


@cli.command()
@click.pass_context
def stats(ctx):
    """Show system statistics."""
    system = NASAQuerySystem(ctx.obj['config'])
    system.show_stats()


@cli.command()
@click.pass_context
def interactive(ctx):
    """Start interactive query mode."""
    config = load_config(ctx.obj['config'])
    setup_logging(config)
    
    system = NASAQuerySystem(config)
    asyncio.run(system.interactive_mode())


@cli.command()
@click.argument('file_path')
@click.pass_context
def file_status(ctx, file_path):
    """Check the processing status of a specific file."""
    config = load_config(ctx.obj['config'])
    setup_logging(config)
    
    system = NASAQuerySystem(config)
    status = system.ingestion.get_file_status(file_path)
    
    console = Console()
    if status:
        console.print(f"[blue]File Status: {file_path}[/blue]")
        console.print(f"  • Processing Status: {status['processing_status']}")
        console.print(f"  • Last Modified: {status['last_modified']}")
        console.print(f"  • Last Processed: {status['last_processed']}")
        console.print(f"  • File Size: {status['file_size']} bytes")
        console.print(f"  • File Hash: {status['file_hash']}")
        console.print(f"  • Has Changed: {'Yes' if status['has_changed'] else 'No'}")
    else:
        console.print(f"[yellow]File {file_path} not found in metadata[/yellow]")


@cli.command()
@click.argument('file_path')
@click.pass_context
def force_reprocess(ctx, file_path):
    """Force reprocessing of a specific file."""
    config = load_config(ctx.obj['config'])
    setup_logging(config)
    
    system = NASAQuerySystem(config)
    system.ingestion.force_reprocess_file(file_path)
    
    console = Console()
    console.print(f"[green]✅ File {file_path} will be reprocessed on next ingestion[/green]")


@cli.command()
@click.pass_context
def reset_metadata(ctx):
    """Reset all file metadata tracking."""
    config = load_config(ctx.obj['config'])
    setup_logging(config)
    
    system = NASAQuerySystem(config)
    system.ingestion.reset_processed_files()
    
    console = Console()
    console.print("[green]✅ File metadata tracking reset. All files will be reprocessed.[/green]")


@cli.command()
@click.option('--similarity-threshold', '-t', default=0.95, help='Similarity threshold for duplicate detection')
@click.pass_context
def cleanup_duplicates(ctx, similarity_threshold):
    """Clean up duplicate vectors in Pinecone index."""
    config = load_config(ctx.obj['config'])
    setup_logging(config)
    
    system = NASAQuerySystem(config)
    
    console = Console()
    console.print(f"[yellow]Cleaning up duplicates with similarity threshold: {similarity_threshold}[/yellow]")
    
    try:
        # Get stats before cleanup
        stats_before = system.rag.get_stats()
        console.print(f"[blue]Before cleanup:[/blue]")
        console.print(f"  • Total vectors: {stats_before.get('total_vector_count', 0)}")
        if 'duplicate_stats' in stats_before:
            dup_stats = stats_before['duplicate_stats']
            console.print(f"  • Potential duplicates: {dup_stats.get('potential_duplicates', 0)}")
            console.print(f"  • Duplicate groups: {dup_stats.get('duplicate_groups', 0)}")
        
        # Perform cleanup
        system.rag.cleanup_duplicates(similarity_threshold)
        
        # Get stats after cleanup
        stats_after = system.rag.get_stats()
        console.print(f"[blue]After cleanup:[/blue]")
        console.print(f"  • Total vectors: {stats_after.get('total_vector_count', 0)}")
        if 'duplicate_stats' in stats_after:
            dup_stats = stats_after['duplicate_stats']
            console.print(f"  • Potential duplicates: {dup_stats.get('potential_duplicates', 0)}")
            console.print(f"  • Duplicate groups: {dup_stats.get('duplicate_groups', 0)}")
        
        console.print("[green]✅ Duplicate cleanup completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to cleanup duplicates: {e}[/red]")


if __name__ == "__main__":
    cli() 