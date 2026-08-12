"""CopilotAgent -- CLI agent backend for GitHub Copilot CLI.

Stub implementation. CLI flags are placeholders until the Copilot CLI
agent mode is tested. ``available()`` returns False until the binary
is installed, so this backend is inert until needed.
"""

from __future__ import annotations

from agents.cli.base import BaseCLIAgent
from agents.types import AgentConfig


class CopilotAgent(BaseCLIAgent):

    name = "copilot"
    _binary_names = ["copilot"]

    def build_command(self, prompt: str, config: AgentConfig) -> list[str]:
        binary = self.binary_path()
        return [
            binary, "agent",
            "--prompt", prompt,
            "--workspace", str(config.workspace),
        ]
