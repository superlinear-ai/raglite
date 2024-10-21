"""RAGLite CLI."""

import os

import typer

from raglite._config import RAGLiteConfig

cli = typer.Typer()


@cli.callback()
def main() -> None:
    """RAGLite CLI."""


@cli.command()
def chainlit(
    db_url: str = typer.Option(RAGLiteConfig().db_url, help="Database URL"),
    llm: str = typer.Option(RAGLiteConfig().llm, help="LiteLLM LLM"),
    embedder: str = typer.Option(RAGLiteConfig().embedder, help="LiteLLM embedder"),
) -> None:
    """Serve a Chainlit frontend."""
    # Set the environment variables for the Chainlit frontend.
    os.environ["RAGLITE_DB_URL"] = os.environ.get("RAGLITE_DB_URL", db_url)
    os.environ["RAGLITE_LLM"] = os.environ.get("RAGLITE_LLM", llm)
    os.environ["RAGLITE_EMBEDDER"] = os.environ.get("RAGLITE_EMBEDDER", embedder)
    # Import Chainlit here as it's an optional dependency.
    try:
        from chainlit.cli import run_chainlit
    except ImportError as error:
        error_message = "To serve a Chainlit frontend, please install the `chainlit` extra."
        raise ImportError(error_message) from error
    # Serve the frontend.
    run_chainlit(__file__.replace("_cli.py", "_chainlit.py"))


if __name__ == "__main__":
    cli()
