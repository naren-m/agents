"""One execution path for every front end (CLI, MCP, background start).

Every finished run must land in the ledger, otherwise ``agents stats`` goes
stale and the routing policy built on it stops being evidence. Keeping the
run logic here means a new front end cannot forget that step.
"""

from __future__ import annotations

from pathlib import Path

from agents.console.registry import build_registry, validate_flags
from agents.console.store import RunStore
from agents.inprocess.ollama import embed_files
from agents.types import AgentConfig, AgentResult, Capability


class UsageError(ValueError):
    """Bad invocation: no run happened, nothing was recorded."""

    def __init__(self, message: str, example: str = ""):
        super().__init__(message)
        self.message = message
        self.example = example


class BackendUnavailable(RuntimeError):
    """Backend exists but cannot run on this machine (binary/dep missing)."""


def resolve_backend(name: str, registry: dict | None = None):
    registry = registry if registry is not None else build_registry()
    backend = registry.get(name)
    if backend is None:
        known = ", ".join(sorted(registry))
        raise UsageError(
            f"unknown backend '{name}'. Known backends: {known}.",
            f"agents run --backend {sorted(registry)[0]} --task \"...\"",
        )
    return backend


def build_prompt(task: str, files: list[str]) -> str:
    """Inline named files (with line numbers) ahead of the question."""
    if not files:
        return task
    context = embed_files([Path(p) for p in files])
    return f"{context}\n\n===== QUESTION =====\n{task}"


def check_files(files: list[str]) -> None:
    missing = [p for p in files if not Path(p).is_file()]
    if missing:
        raise UsageError(
            "no such file: " + ", ".join(missing),
            "agents run --backend ollama -f path/to/existing.py --task \"...\"",
        )


async def _run_cli_backend(backend, prompt: str, config: AgentConfig):
    run = await backend.spawn(prompt, config)
    return await backend.wait(run)


async def execute_run(
    backend_name: str,
    task: str,
    *,
    files: list[str] | None = None,
    workspace: str | None = None,
    model: str | None = None,
    timeout: int = 300,
    run_id: str | None = None,
    registry: dict | None = None,
    store: RunStore | None = None,
) -> AgentResult:
    """Validate, run, and record. Raises ``UsageError`` / ``BackendUnavailable``
    before anything runs; once a backend has run, the result (success or not)
    is always appended to the ledger."""
    files = list(files or [])
    backend = resolve_backend(backend_name, registry)

    problem = validate_flags(backend, workspace, files)
    if problem is not None:
        raise UsageError(*problem)
    check_files(files)

    if not backend.available():
        raise BackendUnavailable(
            f"backend '{backend_name}' is not available on this machine."
        )

    env = {}
    if model:
        env["AGY_MODEL" if backend_name == "agy" else "OLLAMA_MODEL"] = model

    config = AgentConfig(
        workspace=Path(workspace or "."),
        timeout_seconds=timeout,
        env=env,
    )
    prompt = build_prompt(task, files)

    if Capability.CONTEXT in backend.capabilities:
        result = await backend.run(prompt, config)
    else:
        result = await _run_cli_backend(backend, prompt, config)

    store = store if store is not None else RunStore()
    # When launched by `agents start`, adopt the run_id the parent recorded and
    # persist the terminal state, otherwise status/logs would never see the run
    # finish and --follow would hang forever.
    if run_id:
        result.run.run_id = run_id
        store.save(result.run)

    # Every finished run lands in the ledger; without it `stats` has nothing
    # to read and delegation savings stay unmeasurable.
    store.append_ledger(result.run)
    return result
