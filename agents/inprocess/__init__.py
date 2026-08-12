"""In-process agent backends (Python-native)."""

from agents.inprocess.base import InProcessAgent
from agents.inprocess.langgraph import LangGraphAgent

__all__ = [
    "InProcessAgent",
    "LangGraphAgent",
]
