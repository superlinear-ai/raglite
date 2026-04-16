"""RAGLite CLI."""

import json
import os
from typing import ClassVar

import typer
from pydantic_settings import BaseSettings, SettingsConfigDict

from raglite._config import RAGLiteConfig


class RAGLiteCLIConfig(BaseSettings):
    """RAGLite CLI config."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="RAGLITE_", env_file=".env", extra="allow"
    )

    mcp_server_name: str = "RAGLite"
    db_url: str = str(RAGLiteConfig().db_url)
    llm: str = RAGLiteConfig().llm
    embedder: str = RAGLiteConfig().embedder


cli = typer.Typer()
cli.add_typer(mcp_cli := typer.Typer(), name="mcp")


@cli.callback()
def main(
    ctx: typer.Context,
    db_url: str = typer.Option(RAGLiteCLIConfig().db_url, help="Database URL"),
    llm: str = typer.Option(RAGLiteCLIConfig().llm, help="LiteLLM LLM"),
    embedder: str = typer.Option(RAGLiteCLIConfig().embedder, help="LiteLLM embedder"),
) -> None:
    """RAGLite CLI."""
    ctx.obj = {"db_url": db_url, "llm": llm, "embedder": embedder}


@cli.command()
def chainlit(ctx: typer.Context) -> None:
    """Serve a Chainlit frontend."""
    # Set the environment variables for the Chainlit frontend.
    os.environ["RAGLITE_DB_URL"] = ctx.obj["db_url"]
    os.environ["RAGLITE_LLM"] = ctx.obj["llm"]
    os.environ["RAGLITE_EMBEDDER"] = ctx.obj["embedder"]
    # Import Chainlit here as it's an optional dependency.
    try:
        from chainlit.cli import run_chainlit
    except ModuleNotFoundError as error:
        error_message = "To serve a Chainlit frontend, please install the `chainlit` extra."
        raise ModuleNotFoundError(error_message) from error
    # Serve the frontend.
    run_chainlit(__file__.replace("_cli.py", "_chainlit.py"))


@mcp_cli.command("install")
def install_mcp_server(
    ctx: typer.Context,
    server_name: str = typer.Option(RAGLiteCLIConfig().mcp_server_name, help="MCP server name"),
) -> None:
    """Install MCP server in the Claude desktop app."""
    from fastmcp.cli.claude import get_claude_config_path

    # Get the Claude config path.
    claude_config_path = get_claude_config_path()
    if not claude_config_path:
        typer.echo(
            "Please download the Claude desktop app from https://claude.ai/download before installing an MCP server."
        )
        return
    claude_config_filepath = claude_config_path / "claude_desktop_config.json"
    # Parse the Claude config.
    claude_config = (
        json.loads(claude_config_filepath.read_text()) if claude_config_filepath.exists() else {}
    )
    # Update the Claude config with the MCP server.
    mcp_config = RAGLiteCLIConfig(
        mcp_server_name=server_name,
        db_url=ctx.obj["db_url"],
        llm=ctx.obj["llm"],
        embedder=ctx.obj["embedder"],
    )
    claude_config["mcpServers"][server_name] = {
        "command": "uvx",
        "args": [
            "--python",
            "3.11",
            "--with",
            "numpy<2.0.0",  # TODO: Remove this constraint when uv no longer needs it to solve the environment.
            "raglite",
            "mcp",
            "run",
        ],
        "env": {
            f"RAGLITE_{key.upper()}" if key in RAGLiteCLIConfig.model_fields else key.upper(): value
            for key, value in mcp_config.model_dump().items()
            if value
        },
    }
    # Write the updated Claude config to disk.
    claude_config_filepath.write_text(json.dumps(claude_config, indent=2))


@mcp_cli.command("run")
def run_mcp_server(
    ctx: typer.Context,
    server_name: str = typer.Option(RAGLiteCLIConfig().mcp_server_name, help="MCP server name"),
) -> None:
    """Run MCP server."""
    from raglite._mcp import create_mcp_server

    config = RAGLiteConfig(
        db_url=ctx.obj["db_url"], llm=ctx.obj["llm"], embedder=ctx.obj["embedder"]
    )
    mcp = create_mcp_server(server_name, config=config)
    mcp.run()


@cli.command()
def bench(
    ctx: typer.Context,
    dataset_name: str = typer.Option(
        "nano-beir/hotpotqa", "--dataset", "-d", help="Dataset to use from https://ir-datasets.com/"
    ),
    measure: str = typer.Option(
        "AP@10",
        "--measure",
        "-m",
        help="Evaluation measure from https://ir-measur.es/en/latest/measures.html",
    ),
) -> None:
    """Run benchmark."""
    import ir_datasets
    import ir_measures
    import pandas as pd

    from raglite._bench import (
        IREvaluator,
        LlamaIndexEvaluator,
        OpenAIVectorStoreEvaluator,
        RAGLiteEvaluator,
    )

    # Initialise the benchmark.
    evaluator: IREvaluator
    measures = [ir_measures.parse_measure(measure)]
    index, results = [], []
    # Evaluate RAGLite (single-vector) + DuckDB HNSW + text-embedding-3-large.
    chunk_max_size = 2048
    config = RAGLiteConfig(
        embedder="text-embedding-3-large",
        chunk_max_size=chunk_max_size,
        vector_search_multivector=False,
        vector_search_query_adapter=False,
    )
    dataset = ir_datasets.load(dataset_name)
    evaluator = RAGLiteEvaluator(
        dataset, insert_variant=f"single-vector-{chunk_max_size // 4}t", config=config
    )
    index.append("RAGLite (single-vector)")
    results.append(ir_measures.calc_aggregate(measures, dataset.qrels_iter(), evaluator.score()))
    # Evaluate RAGLite (multi-vector) + DuckDB HNSW + text-embedding-3-large.
    config = RAGLiteConfig(
        embedder="text-embedding-3-large",
        chunk_max_size=chunk_max_size,
        vector_search_multivector=True,
        vector_search_query_adapter=False,
    )
    dataset = ir_datasets.load(dataset_name)
    evaluator = RAGLiteEvaluator(
        dataset, insert_variant=f"multi-vector-{chunk_max_size // 4}t", config=config
    )
    index.append("RAGLite (multi-vector)")
    results.append(ir_measures.calc_aggregate(measures, dataset.qrels_iter(), evaluator.score()))
    # Evaluate RAGLite (query adapter) + DuckDB HNSW + text-embedding-3-large.
    config = RAGLiteConfig(
        llm=(llm := "gpt-4.1"),
        embedder="text-embedding-3-large",
        chunk_max_size=chunk_max_size,
        vector_search_multivector=True,
        vector_search_query_adapter=True,
    )
    dataset = ir_datasets.load(dataset_name)
    evaluator = RAGLiteEvaluator(
        dataset,
        insert_variant=f"multi-vector-{chunk_max_size // 4}t",
        search_variant=f"query-adapter-{llm}",
        config=config,
    )
    index.append("RAGLite (query adapter)")
    results.append(ir_measures.calc_aggregate(measures, dataset.qrels_iter(), evaluator.score()))
    # Evaluate LLamaIndex + FAISS HNSW + text-embedding-3-large.
    dataset = ir_datasets.load(dataset_name)
    evaluator = LlamaIndexEvaluator(dataset)
    index.append("LlamaIndex")
    results.append(ir_measures.calc_aggregate(measures, dataset.qrels_iter(), evaluator.score()))
    # Evaluate OpenAI Vector Store.
    dataset = ir_datasets.load(dataset_name)
    evaluator = OpenAIVectorStoreEvaluator(dataset)
    index.append("OpenAI Vector Store")
    results.append(ir_measures.calc_aggregate(measures, dataset.qrels_iter(), evaluator.score()))
    # Print the results.
    results_df = pd.DataFrame.from_records(results, index=index)
    typer.echo(results_df)


if __name__ == "__main__":
    cli()
