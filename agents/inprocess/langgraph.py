"""LangGraphAgent -- in-process agent backend using LangGraph.

Stub implementation. When built out, this would construct a LangGraph
graph with nodes per RCA phase, wire MCP tools from the provided
mcp_client, and execute the graph.

``available()`` returns False until langgraph is installed, so this
backend is inert until the dependency is added.
"""

from __future__ import annotations

from typing import Any, Callable

from agents.types import AgentConfig, AgentResult


class LangGraphAgent:

    name = "langgraph"

    def available(self) -> bool:
        try:
            import langgraph  # noqa: F401

            return True
        except ImportError:
            return False

    async def run(
        self,
        prompt: str,
        config: AgentConfig,
        mcp_client: Any = None,
        on_progress: Callable | None = None,
    ) -> AgentResult:
        raise NotImplementedError("LangGraph agent not yet implemented")

    async def cancel(self) -> bool:
        return False
