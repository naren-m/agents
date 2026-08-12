"""In-process agent protocol for Python-native agent backends.

Unlike CLI agents, in-process agents run inside the Python process as
coroutines. They receive tools via an MCP client and can stream progress
events back to the caller.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from agents.types import AgentConfig, AgentResult


@runtime_checkable
class InProcessAgent(Protocol):
    """Interface for agents that run inside the Python process."""

    @property
    def name(self) -> str: ...

    def available(self) -> bool:
        """Are the required dependencies installed?"""
        ...

    async def run(
        self,
        prompt: str,
        config: AgentConfig,
        mcp_client: Any = None,
        on_progress: Callable | None = None,
    ) -> AgentResult:
        """Execute the agent to completion.

        Args:
            prompt: The task description / instructions for the agent.
            config: Common agent configuration.
            mcp_client: A connected MCP client (e.g. fastmcp.Client) for
                tool access. The agent uses this to list and call tools.
            on_progress: Optional async callback with signature
                ``async def on_progress(event_type: str, data: dict) -> None``.
                Event types: "tool_call", "step_complete", "error".
        """
        ...

    async def cancel(self) -> bool:
        """Request graceful cancellation of a running agent."""
        ...
