"""Example MCP client that talks to an arbitrary MCP server.

The goal of this module is to demonstrate the minimum plumbing that a Python
program needs in order to communicate with an MCP server using only the
standard library.  It follows the JSON-RPC interface that MCP servers expose
over stdio.  The included CLI showcases the ``mcp-server-git`` configuration,
but the ``MCPClient`` class itself is agnostic to the server it launches.

Usage
-----
The module exposes a :func:`main` entrypoint so it can be executed directly::

    python -m kinyu.mcp.git --repository /path/to/repository

The script will:

* launch ``mcp-server-git`` using the configuration described in the
  ``MCP_CONFIG`` constant,
* perform the JSON-RPC handshake (`initialize` followed by
  ``notifications/initialized``),
* list the tools that the server exposes, and
* call ``git_status`` as a concrete example.

Because the transport uses newline-delimited JSON, the implementation below
just writes JSON payloads to the server's stdin and reads responses line by
line from stdout.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from asyncio.subprocess import Process
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

JsonDict = Dict[str, Any]

MCP_CONFIG: JsonDict = {
    "servers": {
        "git": {
            "command": "uvx",
            "args": ["mcp-server-git"],
        }
    }
}
"""Demonstration Model Context Protocol configuration."""


@dataclass
class PendingRequest:
    """Bookkeeping container for in-flight JSON-RPC requests."""

    future: asyncio.Future[JsonDict]


@dataclass
class MCPClient:
    """Minimal asynchronous MCP client that can launch any configured server."""

    server_name: str = "git"
    config: JsonDict = field(default_factory=lambda: json.loads(json.dumps(MCP_CONFIG)))
    server_args: list[str] = field(default_factory=list)
    _process: Process | None = field(default=None, init=False, repr=False)
    _stdout_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _stderr_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _pending: Dict[int, PendingRequest] = field(default_factory=dict, init=False, repr=False)
    _next_request_id: int = field(default=1, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> "MCPClient":
        await self._start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _start(self) -> None:
        """Launch the configured MCP server subprocess and set up background tasks."""

        server_cfg = self.config["servers"][self.server_name]
        command = server_cfg["command"]
        args = list(server_cfg.get("args", []))
        args.extend(self.server_args)

        self._loop = asyncio.get_running_loop()

        self._process = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert self._process.stdout is not None
        assert self._process.stderr is not None

        self._stdout_task = asyncio.create_task(self._read_stdout(self._process.stdout))
        self._stderr_task = asyncio.create_task(self._log_stderr(self._process.stderr))

        await self._initialize()

    async def close(self) -> None:
        """Tear down the client and stop the subprocess."""

        if self._process:
            if self._process.stdin:
                self._process.stdin.close()
                with contextlib.suppress(Exception):
                    await self._process.stdin.wait_closed()  # type: ignore[func-returns-value]

            if self._process.returncode is None:
                self._process.terminate()
                with contextlib.suppress(ProcessLookupError):
                    await self._process.wait()

        if self._stdout_task:
            self._stdout_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stdout_task

        if self._stderr_task:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task

    async def _initialize(self) -> None:
        """Perform the MCP initialization handshake."""

        request_id = self._next_id()
        initialize_request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "kinyu-demo", "version": "0.1.0"},
            },
        }

        await self._send_request(request_id, initialize_request)
        await self._send_notification(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
        )

    async def list_tools(self, cursor: str | None = None) -> JsonDict:
        """Return the list of tools exposed by the connected server."""

        request_id = self._next_id()
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/list",
            "params": {"cursor": cursor} if cursor is not None else {},
        }

        return await self._send_request(request_id, request)

    async def call_tool(self, name: str, arguments: Optional[JsonDict] = None) -> JsonDict:
        """Invoke a tool by name with the supplied arguments."""

        request_id = self._next_id()
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments or {},
            },
        }
        return await self._send_request(request_id, request)

    async def _send_request(self, request_id: int, payload: JsonDict) -> JsonDict:
        assert self._process and self._process.stdin
        assert self._loop is not None

        future: asyncio.Future[JsonDict] = self._loop.create_future()
        self._pending[request_id] = PendingRequest(future=future)

        await self._write_json(payload)

        response = await future

        if "error" in response:
            error = response["error"]
            message = error.get("message", "Unknown MCP error")
            raise RuntimeError(f"MCP request failed: {message}")

        return response["result"]

    async def _send_notification(self, payload: JsonDict) -> None:
        await self._write_json(payload)

    async def _write_json(self, payload: JsonDict) -> None:
        assert self._process and self._process.stdin
        message = json.dumps(payload, separators=(",", ":")) + "\n"
        self._process.stdin.write(message.encode("utf-8"))
        await self._process.stdin.drain()

    async def _read_stdout(self, stream: asyncio.StreamReader) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break
            message = json.loads(line.decode("utf-8"))
            await self._handle_server_message(message)

    async def _log_stderr(self, stream: asyncio.StreamReader) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break
            sys.stderr.write(line.decode("utf-8"))
            sys.stderr.flush()

    async def _handle_server_message(self, message: JsonDict) -> None:
        if "id" in message and "result" in message:
            await self._resolve_pending(message)
        elif "id" in message and "error" in message:
            await self._resolve_pending(message)
        elif "id" in message and "method" in message:
            await self._handle_server_request(message)
        elif "method" in message:
            # Notification from server – log it so the user can see what's happening.
            params = message.get("params", {})
            print(f"[notification] {message['method']}: {params}")
        else:
            print(f"[unknown message] {message}")

    async def _resolve_pending(self, message: JsonDict) -> None:
        request_id = message["id"]
        pending = self._pending.pop(request_id, None)
        if pending:
            pending.future.set_result(message)

    async def _handle_server_request(self, message: JsonDict) -> None:
        method = message["method"]
        request_id = message["id"]

        if method == "roots/list":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"roots": []},
            }
        elif method == "ping":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {},
            }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Client does not implement {method}",
                },
            }

        await self._write_json(response)

    def _next_id(self) -> int:
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal MCP client example that can launch servers defined in the"
            " provided configuration."
        )
    )
    parser.add_argument(
        "--server-name",
        default="git",
        help="Identifier of the server to launch from the MCP configuration",
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        dest="server_args",
        default=None,
        help="Additional CLI arguments for the server command (repeatable)",
    )
    parser.add_argument(
        "--repository",
        "-r",
        type=Path,
        default=None,
        help=(
            "Convenience flag for the git demo: append --repository <path> when"
            " launching the server"
        ),
    )
    parser.add_argument(
        "--tool",
        default="git_status",
        help="Tool name to call after listing the available tools",
    )
    parser.add_argument(
        "--args",
        nargs="*",
        default=None,
        help="Optional JSON arguments for the tool call (key=value pairs)",
    )
    return parser


def parse_tool_arguments(arguments: list[str] | None) -> JsonDict:
    if not arguments:
        return {}

    parsed: JsonDict = {}
    for item in arguments:
        if "=" not in item:
            raise ValueError(f"Invalid argument '{item}', expected key=value")
        key, value = item.split("=", 1)
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed


async def demo(
    tool_name: str,
    server_name: str,
    tool_arguments: JsonDict | None = None,
    *,
    server_args: list[str] | None = None,
    config: JsonDict | None = None,
) -> None:
    source_config = config if config is not None else MCP_CONFIG
    client_config = json.loads(json.dumps(source_config))
    async with MCPClient(
        server_name=server_name,
        server_args=list(server_args or []),
        config=client_config,
    ) as client:
        tools: list[JsonDict] = []
        cursor: str | None = None
        while True:
            result = await client.list_tools(cursor)
            tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                break

        print("Available tools:")
        for tool in tools:
            description = tool.get("description", "")
            print(f"  - {tool['name']}: {description}")

        result = await client.call_tool(tool_name, arguments=tool_arguments or {})
        print("\nTool response:")
        content_blocks = result.get("content", [])
        if content_blocks:
            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    print(block.get("text", ""))
                else:
                    print(json.dumps(block, indent=2))
        elif "structuredContent" in result:
            print(json.dumps(result["structuredContent"], indent=2))
        else:
            print(json.dumps(result, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    tool_arguments = parse_tool_arguments(args.args)

    server_args = list(args.server_args or [])

    repo_for_tool: Path | None = None

    if args.repository is not None:
        server_args.extend(["--repository", str(args.repository)])
        repo_for_tool = args.repository

    if args.server_name == "git":
        if repo_for_tool is None:
            repo_for_tool = Path.cwd()
            if "--repository" not in server_args and "-r" not in server_args:
                server_args.extend(["--repository", str(repo_for_tool)])

        if "repo_path" not in tool_arguments and repo_for_tool is not None:
            tool_arguments["repo_path"] = str(repo_for_tool)

    try:
        asyncio.run(
            demo(
                args.tool,
                args.server_name,
                tool_arguments,
                server_args=server_args,
            )
        )
    except KeyboardInterrupt:
        return 1
    except Exception as exc:  # noqa: BLE001 - display readable error to users
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
