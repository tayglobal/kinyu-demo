"""Utilities for working with the Model Context Protocol (MCP).

This subpackage intentionally keeps its dependencies limited to Python's
standard library so it can be used in environments where additional MCP
client libraries are unavailable.
"""

from .git import MCPClient, MCP_CONFIG  # noqa: F401

__all__ = ["MCPClient", "MCP_CONFIG"]
