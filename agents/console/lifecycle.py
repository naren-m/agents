"""Background run lifecycle: start, status, cancel, logs.

``start`` re-invokes ``agents run`` as a detached process. One mechanism for
every backend, so the in-process ollama backend gets background execution
without a second code path.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from agents.console.envelope import EXIT_OK, usage_error
from agents.console.store import RunStore
from agents.types import AgentRun

_TERMINAL = {"completed", "failed", "cancelled", "timed_out"}


def _spawn_detached(cmd: list[str], log_path: Path) -> int:
    """Start the command detached, redirecting output to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
    )
    return process.pid


def _pid_alive(pid: int) -> bool:
    """Whether the detached child is still running."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


def _rebuild_run_command(args, run_id: str) -> list[str]:
    # --run-id is what lets the child persist its own terminal state. Without
    # it the store record stays 'running' forever and --follow never ends.
    cmd = [sys.executable, "-m", "agents.console.main", "run",
           "--backend", args.backend, "--task", args.task,
           "--run-id", run_id]
    if args.workspace:
        cmd += ["--workspace", args.workspace]
    for f in args.file:
        cmd += ["-f", f]
    if args.model:
        cmd += ["--model", args.model]
    cmd += ["--timeout", str(args.timeout)]
    return cmd


def start_run(args, store: RunStore) -> int:
    key = getattr(args, "idempotency_key", None)
    if key:
        existing = store.find_by_key(key)
        if existing:
            print(json.dumps({"run_id": existing, "reused": True}, indent=2))
            return EXIT_OK

    run = AgentRun.create(backend=args.backend, prompt=args.task)
    log_path = store.root / "logs" / f"{run.run_id}.log"
    run.pid = _spawn_detached(_rebuild_run_command(args, run.run_id), log_path)
    run.transcript_path = log_path
    store.save(run)

    if key:
        store.save_key(key, run.run_id)

    print(json.dumps({"run_id": run.run_id, "reused": False}, indent=2))
    return EXIT_OK


def status_run(run_id: str, store: RunStore) -> int:
    data = store.load(run_id)
    if data is None:
        return usage_error(
            f"no run with id '{run_id}'.",
            "agents start --backend ollama --task \"...\"   # returns a run_id",
        )
    print(json.dumps(data, indent=2))
    return EXIT_OK


def cancel_run(run_id: str, store: RunStore) -> int:
    data = store.load(run_id)
    if data is None:
        return usage_error(
            f"no run with id '{run_id}'.",
            "agents status <run_id>",
        )

    # Cancelling a finished run is a no-op that reports the terminal status,
    # because orchestrators retry and a hard error there is noise.
    if data.get("status") in _TERMINAL:
        print(json.dumps({"run_id": run_id, "status": data["status"],
                          "cancelled": False}, indent=2))
        return EXIT_OK

    pid = data.get("pid")
    cancelled = False
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            cancelled = True
        except (ProcessLookupError, PermissionError):
            cancelled = False

    data["status"] = "cancelled"
    (store.runs_dir / f"{run_id}.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "status": "cancelled",
                      "cancelled": cancelled}, indent=2))
    return EXIT_OK


def show_logs(run_id: str, store: RunStore, follow: bool) -> int:
    data = store.load(run_id)
    if data is None:
        return usage_error(
            f"no run with id '{run_id}'.",
            "agents status <run_id>",
        )

    log_path = store.root / "logs" / f"{run_id}.log"
    if not log_path.exists():
        print("")
        return EXIT_OK

    if not follow:
        print(log_path.read_text(encoding="utf-8"), end="")
        return EXIT_OK

    # agy streams step_update events, so following shows progress live.
    # ollama emits nothing until it finishes, so this simply blocks to the end.
    with log_path.open("r", encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if line:
                print(line, end="")
                continue
            current = store.load(run_id) or {}
            if current.get("status") in _TERMINAL:
                print(fh.read(), end="")
                return EXIT_OK
            # A killed child can never mark itself terminal, so liveness is
            # the backstop that keeps this from hanging on a stale record.
            pid = current.get("pid")
            if pid and not _pid_alive(pid):
                print(fh.read(), end="")
                return EXIT_OK
            time.sleep(0.2)
