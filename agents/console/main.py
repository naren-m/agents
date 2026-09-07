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
from agents.console.registry import build_registry, validate_flags
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

    print(json.dumps(build_envelope(result), indent=2))
    return EXIT_OK if result.success else EXIT_AGENT_FAILED


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    registry = build_registry()

    if args.command == "backends":
        return _cmd_backends_list(registry)
    if args.command == "run":
        return _cmd_run(args, registry)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
