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
    except ImportError as error:
        error_message = "To serve a Chainlit frontend, please install the `chainlit` extra."
        raise ImportError(error_message) from error
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


if __name__ == "__main__":
    cli()
