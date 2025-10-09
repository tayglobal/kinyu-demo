# MCP Client Demo

This package illustrates how to communicate with an MCP server using nothing
but Python's standard library.  The reusable `MCPClient` can launch any
server described in the configuration below and drive the JSON-RPC handshake
manually.  The included CLI demonstrates the
[`mcp-server-git`](https://pypi.org/project/mcp-server-git/) server.

```json
{
  "servers": {
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git"]
    }
  }
}
```

Run the git-focused demo with::

    PYTHONPATH=src python -m kinyu.mcp.git --repository /path/to/git/repo

After establishing the connection it lists the available tools and invokes
`git_status` by default. You can call a different tool by passing
`--tool <name>` plus optional key/value pairs that will be converted into JSON
arguments::

    PYTHONPATH=src python -m kinyu.mcp.git --repository . --tool git_log --args max_count=5

Because the implementation talks JSON-RPC directly over stdio it does not rely
on the `mcp` Python package or any other third-party dependencies. To
experiment with another server, add it to the configuration and pass its
identifier via `--server-name`. Additional CLI arguments for the server can be
provided with `--server-arg`.
