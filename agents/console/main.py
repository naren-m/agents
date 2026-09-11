"""``agents`` console entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from agents.console.envelope import (
    EXIT_AGENT_FAILED, EXIT_OK, EXIT_UNAVAILABLE,
    build_envelope, usage_error,
)
from agents.console.lifecycle import cancel_run, show_logs, start_run, status_run
from agents.console.registry import build_registry
from agents.console.runner import BackendUnavailable, UsageError, execute_run
from agents.console.stats import stats_command
from agents.console.store import RunStore


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

    stats_p = sub.add_parser(
        "stats",
        help="Summarise past runs from the ledger",
        epilog=(
            "Examples:\n"
            "  agents stats\n"
            "  agents stats --since 7d --backend ollama\n"
            "  agents stats --format table\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    stats_p.add_argument("--since", help="Window, e.g. 7d, 12h, 30m")
    stats_p.add_argument("--backend", help="Only this backend")
    stats_p.add_argument(
        "--format", dest="fmt", choices=["json", "table"], default="json")

    return parser


def _cmd_backends_list(registry: dict) -> int:
    for name, backend in sorted(registry.items()):
        caps = ", ".join(sorted(c.value for c in backend.capabilities))
        state = "available" if backend.available() else "unavailable"
        print(f"{name:<8} {state:<12} {caps}")
    return EXIT_OK


def _cmd_run(args, registry: dict) -> int:
    try:
        result = asyncio.run(execute_run(
            args.backend, args.task,
            files=args.file, workspace=args.workspace, model=args.model,
            timeout=args.timeout, run_id=getattr(args, "run_id", None),
            registry=registry,
        ))
    except UsageError as exc:
        return usage_error(exc.message, exc.example)
    except BackendUnavailable as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE

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
    if args.command == "stats":
        return stats_command(RunStore(), args.since, args.backend, args.fmt)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
