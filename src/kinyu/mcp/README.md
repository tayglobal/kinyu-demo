# MCP Git Client Demo

This package illustrates how to communicate with an MCP server using nothing
but Python's standard library.  The example client launches
[`mcp-server-git`](https://pypi.org/project/mcp-server-git/) through the
configuration below and then drives the JSON-RPC handshake manually.

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

Run the demo with::

    PYTHONPATH=src python -m kinyu.mcp.git --repository /path/to/git/repo

After establishing the connection it lists the available Git tools and invokes
`git_status` by default. You can call a different tool by passing
`--tool <name>` plus optional key/value pairs that will be converted into JSON
arguments::

    PYTHONPATH=src python -m kinyu.mcp.git --repository . --tool git_log --args max_count=5

Because the implementation talks JSON-RPC directly over stdio it does not rely
on the `mcp` Python package or any other third-party dependencies.
