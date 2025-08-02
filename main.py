#!/usr/bin/env python3
"""
NASA Documentation Query System - Main CLI Interface

A production-grade system for querying NASA documentation using a hybrid architecture
combining Knowledge Base (KB), Knowledge Graph (KG), and Retrieval-Augmented Generation (RAG).
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional
import yaml
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich import print as rprint

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.llm_manager import LLMManager
from src.router.query_router import QueryRouter
from src.kb.knowledge_base import KnowledgeBase
from src.kg.knowledge_graph import KnowledgeGraph
from src.rag.rag_system import RAGSystem
from src.ingestion.ingestion_pipeline import IngestionPipeline

console = Console()


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)


def setup_logging(config: dict):
    """Setup logging configuration."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_config.get("output_file", "logs/nasa_query.log")),
            logging.StreamHandler()
        ]
    )


class NASAQuerySystem:
    """Main orchestrator for the NASA query system."""
    
    def __init__(self, config: dict):
        self.config = config
        self.console = Console()
        
        # Initialize components
        self.llm_manager = LLMManager(config["llm"])
        self.router = QueryRouter(config["router"], self.llm_manager)
        self.kb = KnowledgeBase(config["kb"], self.llm_manager)
        self.kg = KnowledgeGraph(config["kg"], self.llm_manager)
        self.rag = RAGSystem(config["rag"], self.llm_manager)
        self.ingestion = IngestionPipeline(config["ingestion"], self.kb, self.kg, self.rag)
        
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
        
        # Display results
        results_table = Table(title="Document Ingestion Results")
        results_table.add_column("Document", style="cyan")
        results_table.add_column("Status", style="white")
        results_table.add_column("KB Facts", style="green")
        results_table.add_column("KG Entities", style="blue")
        results_table.add_column("KG Relations", style="magenta")
        results_table.add_column("RAG Chunks", style="yellow")
        results_table.add_column("Time (s)", style="white")
        
        for result in results:
            status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
            results_table.add_row(
                result.document_id,
                status,
                str(result.kb_facts_added),
                str(result.kg_entities_added),
                str(result.kg_relationships_added),
                str(result.rag_chunks_added),
                f"{result.processing_time:.2f}"
            )
        
        self.console.print(results_table)
        
        # Summary
        successful = sum(1 for r in results if r.success)
        total_facts = sum(r.kb_facts_added for r in results)
        total_entities = sum(r.kg_entities_added for r in results)
        total_relationships = sum(r.kg_relationships_added for r in results)
        total_chunks = sum(r.rag_chunks_added for r in results)
        
        summary = f"""
        [bold]Ingestion Summary:[/bold]
        • Processed: {len(results)} documents
        • Successful: {successful} documents
        • Added: {total_facts} KB facts, {total_entities} KG entities, 
          {total_relationships} KG relationships, {total_chunks} RAG chunks
        """
        
        self.console.print(Panel(summary, title="[bold green]Ingestion Complete[/bold green]"))
    
    def show_stats(self):
        """Display system statistics."""
        kb_stats = self.kb.get_stats()
        kg_stats = self.kg.get_stats()
        rag_stats = self.rag.get_stats()
        router_stats = self.router.get_routing_stats()
        
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
        
        rag_table.add_row("Total Chunks", str(rag_stats["total_chunks"]))
        rag_table.add_row("Sources", str(len(rag_stats["sources"])))
        rag_table.add_row("Avg Chunk Length", f"{rag_stats['avg_chunk_length']:.1f} words")
        rag_table.add_row("Chunks with Embeddings", str(rag_stats["chunks_with_embeddings"]))
        
        self.console.print(kb_table)
        self.console.print(kg_table)
        self.console.print(rag_table)
    
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
                query = Prompt.ask("\n[bold cyan]Query[/bold cyan]")
                
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
@click.option('--data-path', '-p', default='data/documents', help='Path to documents directory')
@click.pass_context
def ingest(ctx, data_path):
    """Ingest documents from the specified directory."""
    system = NASAQuerySystem(ctx.obj['config'])
    asyncio.run(system.ingest_documents(data_path))


@cli.command()
@click.pass_context
def stats(ctx):
    """Show system statistics."""
    system = NASAQuerySystem(ctx.obj['config'])
    system.show_stats()


@cli.command()
@click.pass_context
def interactive(ctx):
    """Run the system in interactive mode."""
    system = NASAQuerySystem(ctx.obj['config'])
    asyncio.run(system.interactive_mode())


@cli.command()
@click.option('--data-path', '-p', default='data/documents', help='Path to documents directory')
@click.pass_context
def watch(ctx, data_path):
    """Watch directory for new documents and process them automatically."""
    system = NASAQuerySystem(ctx.obj['config'])
    
    console.print(f"[yellow]Watching {data_path} for new documents...[/yellow]")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]")
    
    try:
        asyncio.run(system.ingestion.watch_directory(data_path))
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped watching directory.[/yellow]")


if __name__ == "__main__":
    cli() 