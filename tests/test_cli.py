"""Test RAGLite's command line interface."""

import json
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from raglite._cli import cli


@pytest.mark.usefixtures("cp1252_default_text_encoding")
def test_mcp_install_reads_and_writes_config_as_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that installing the MCP server reads and writes the desktop app config as UTF-8."""
    # Create a desktop app config that is encoded as UTF-8 and contains non-ASCII characters.
    config_filepath = tmp_path / "claude_desktop_config.json"
    other_mcp_server = {"command": "uvx", "args": ["--from", "“café”"]}
    config_filepath.write_bytes(
        json.dumps({"mcpServers": {"other": other_mcp_server}}, ensure_ascii=False).encode("utf-8")
    )
    # Point the CLI to this config.
    config_module = types.ModuleType("fastmcp.cli.claude")
    config_module.get_claude_config_path = lambda: tmp_path  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastmcp.cli.claude", config_module)
    # Install the MCP server.
    args = ["--db-url", "duckdb:///:memory:", "--llm", "gpt-4o-mini", "mcp", "install"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.exception
    # Verify that the existing MCP server is preserved and that the new one is added.
    config = json.loads(config_filepath.read_bytes().decode("utf-8"))
    assert config["mcpServers"]["other"] == other_mcp_server
    assert "RAGLite" in config["mcpServers"]
