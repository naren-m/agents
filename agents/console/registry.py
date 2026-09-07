"""Backend registry and flag validation for the console."""

from __future__ import annotations

from agents.cli.agy import AgyAgent
from agents.inprocess.ollama import OllamaAgent
from agents.types import Capability


def build_registry() -> dict:
    """All backends the console can drive, by name."""
    return {"agy": AgyAgent(), "ollama": OllamaAgent()}


def validate_flags(backend, workspace, files) -> tuple[str, str] | None:
    """Check flags against the backend's declared capabilities.

    Returns ``(message, example)`` describing the problem and a corrected
    invocation, or ``None`` when the combination is valid.
    """
    caps = backend.capabilities
    name = backend.name

    if workspace and Capability.WORKSPACE not in caps:
        return (
            f"backend '{name}' has no tools and cannot explore a workspace.",
            f"Inline the files instead:\n"
            f"  agents run --backend {name} -f src/a.py -f src/b.py --task \"...\"",
        )

    if files and Capability.CONTEXT not in caps:
        return (
            f"backend '{name}' explores a workspace with its own tools; "
            f"-f/--file does not apply.",
            f"agents run --backend {name} --workspace . --task \"...\"",
        )

    return None
