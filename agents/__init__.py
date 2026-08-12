"""agents -- pluggable agent backend abstraction.

Two plugin categories:
- CLI agents (subprocess-based): cursor-agent, copilot, codex
- In-process agents (Python-native): LangGraph, direct SDK loops
"""

from agents.manager import AgentManager
from agents.types import AgentConfig, AgentResult, AgentRun

__version__ = "0.1.0"

__all__ = [
    "AgentConfig",
    "AgentManager",
    "AgentResult",
    "AgentRun",
    "__version__",
]
