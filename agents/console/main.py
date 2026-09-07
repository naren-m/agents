"""``agents`` console entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from agents.console.envelope import (
    EXIT_AGENT_FAILED, EXIT_OK, EXIT_UNAVAILABLE,
    build_envelope, usage_error,
)
from agents.console.lifecycle import cancel_run, show_logs, start_run, status_run
from agents.console.registry import build_registry, validate_flags
from agents.console.store import RunStore
from agents.inprocess.ollama import embed_files
from agents.types import AgentConfig, Capability


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend", required=True,
        help="Backend name (see: agents backends list)")
    parser.add_argument("--task", required=True, help="The task description")
    parser.add_argument(
        "--workspace",
        help="Directory for backends that explore with their own tools")
    parser.add_argument(
        "-f", "--file", action="append", default=[],
        help="Inline a file (context-only backends)")
    parser.add_argument("--model", help="Model override")
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Timeout in seconds (default: 300)")
    parser.add_argument(
        "--run-id",
        help=argparse.SUPPRESS)  # internal: set by `agents start`


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agents",
        description="Hand a task to an agent backend and get the result back.",
        epilog=(
            "Examples:\n"
            "  agents run --backend ollama -f src/a.py --task \"list the public functions\"\n"
            "  agents run --backend agy --workspace . --task \"where is auth handled?\"\n"
            "  agents backends list\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run",
        help="Run a task to completion and print the result envelope",
        epilog=(
            "Examples:\n"
            "  agents run --backend ollama -f a.py -f b.py --task \"what changed?\"\n"
            "  agents run --backend agy --workspace . --task \"find the retry logic\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_flags(run_p)

    backends_p = sub.add_parser("backends", help="Inspect available backends")
    backends_sub = backends_p.add_subparsers(dest="backends_command", required=True)
    backends_sub.add_parser(
        "list",
        help="List backends and their capabilities",
        epilog="Examples:\n  agents backends list\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    start_p = sub.add_parser(
        "start",
        help="Start a task in the background and return a run_id",
        epilog=(
            "Examples:\n"
            "  agents start --backend agy --workspace . --task \"audit error handling\"\n"
            "  agents start --backend ollama -f a.py --task \"...\" --idempotency-key job-42\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_flags(start_p)
    start_p.add_argument(
        "--idempotency-key",
        help="Reuse the existing run_id if this key was already started")

    status_p = sub.add_parser(
        "status", help="Show the state of a run",
        epilog="Examples:\n  agents status r_8f3a\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    status_p.add_argument("run_id")

    cancel_p = sub.add_parser(
        "cancel", help="Cancel a running task",
        epilog="Examples:\n  agents cancel r_8f3a\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    cancel_p.add_argument("run_id")

    logs_p = sub.add_parser(
        "logs", help="Show captured output for a run",
        epilog="Examples:\n  agents logs r_8f3a\n  agents logs r_8f3a --follow\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    logs_p.add_argument("run_id")
    logs_p.add_argument(
        "--follow", action="store_true",
        help="Stream until the run finishes")

    return parser


def _cmd_backends_list(registry: dict) -> int:
    for name, backend in sorted(registry.items()):
        caps = ", ".join(sorted(c.value for c in backend.capabilities))
        state = "available" if backend.available() else "unavailable"
        print(f"{name:<8} {state:<12} {caps}")
    return EXIT_OK


async def _run_cli_backend(backend, prompt: str, config: AgentConfig):
    run = await backend.spawn(prompt, config)
    return await backend.wait(run)


def _cmd_run(args, registry: dict) -> int:
    backend = registry.get(args.backend)
    if backend is None:
        known = ", ".join(sorted(registry))
        return usage_error(
            f"unknown backend '{args.backend}'. Known backends: {known}.",
            f"agents run --backend {sorted(registry)[0]} --task \"...\"",
        )

    problem = validate_flags(backend, args.workspace, args.file)
    if problem is not None:
        return usage_error(*problem)

    if not backend.available():
        print(
            f"Error: backend '{args.backend}' is not available on this machine.",
            file=sys.stderr,
        )
        return EXIT_UNAVAILABLE

    env = {}
    if args.model:
        env["AGY_MODEL" if args.backend == "agy" else "OLLAMA_MODEL"] = args.model

    config = AgentConfig(
        workspace=Path(args.workspace or "."),
        timeout_seconds=args.timeout,
        env=env,
    )

    prompt = args.task
    if args.file:
        context = embed_files([Path(p) for p in args.file])
        prompt = f"{context}\n\n===== QUESTION =====\n{args.task}"

    if Capability.CONTEXT in backend.capabilities:
        result = asyncio.run(backend.run(prompt, config))
    else:
        result = asyncio.run(_run_cli_backend(backend, prompt, config))

    # When launched by `agents start`, adopt the run_id the parent recorded and
    # persist the terminal state, otherwise status/logs would never see the run
    # finish and --follow would hang forever.
    run_id = getattr(args, "run_id", None)
    if run_id:
        result.run.run_id = run_id
        RunStore().save(result.run)

    print(json.dumps(build_envelope(result), indent=2))
    return EXIT_OK if result.success else EXIT_AGENT_FAILED


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    registry = build_registry()

    if args.command == "backends":
        return _cmd_backends_list(registry)
    if args.command == "run":
        return _cmd_run(args, registry)
    if args.command == "start":
        return start_run(args, RunStore())
    if args.command == "status":
        return status_run(args.run_id, RunStore())
    if args.command == "cancel":
        return cancel_run(args.run_id, RunStore())
    if args.command == "logs":
        return show_logs(args.run_id, RunStore(), args.follow)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
