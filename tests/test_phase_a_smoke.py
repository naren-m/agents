"""End-to-end smoke for Phase A of the agent subprocess streaming design.

Exercises the full agents-layer stack against a real subprocess:

- :class:`AgentManager` provisions a :class:`RunEventBus` via
  :class:`EventBusRegistry`.
- :class:`BaseCLIAgent` streams stdout line-by-line into the bus's
  ring buffer and an on-disk transcript via :class:`FileStreamSink`.
- Manager publishes ``run.started`` / ``run.finished`` and closes the
  bus on terminal.
- The reusable ``agents.http`` router surfaces everything to HTTP
  clients.

This test deliberately uses ``printf`` (present on all CI hosts) so
stdout output is deterministic and finishes fast.  If any layer in
Phase A regresses, this single test catches it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("fastapi")


@pytest.fixture
def config(tmp_path: Path):
    from agents.types import AgentConfig

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    return AgentConfig(
        workspace=workspace,
        transcript_dir=transcripts,
        timeout_seconds=5,
    )


class _PrintfCLIAgent:
    """Minimal real-subprocess backend.

    ``BaseCLIAgent`` is an abstract-ish class; this adapter just
    plugs in ``printf`` so the subprocess path, not the mock path,
    is what actually runs.
    """

    def __init__(self, lines: list[str]) -> None:
        from agents.cli.base import BaseCLIAgent

        self._inner = BaseCLIAgent()
        self._inner._binary_names = ["printf"]
        self._inner.name = "printf-agent"
        fmt = "".join(f"{line}\\n" for line in lines)
        self._inner.build_command = lambda p, c: ["printf", fmt]

    @property
    def name(self) -> str:
        return self._inner.name

    def available(self) -> bool:
        return self._inner.available()

    def binary_path(self) -> str | None:
        return self._inner.binary_path()

    def build_command(self, prompt, config):
        return self._inner.build_command(prompt, config)

    async def spawn(self, prompt, config, run_id_hint=None):
        return await self._inner.spawn(
            prompt, config, run_id_hint=run_id_hint,
        )

    async def wait(self, run):
        return await self._inner.wait(run)

    async def cancel(self, run):
        return await self._inner.cancel(run)


@pytest.mark.asyncio
async def test_phase_a_end_to_end(config):
    """Spawn a real subprocess and verify every Phase-A contract."""
    from agents.event_bus import EventBusRegistry
    from agents.http import build_router
    from agents.manager import AgentManager
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    registry = EventBusRegistry()
    mgr = AgentManager(config=config, event_bus_registry=registry)
    mgr.register_cli(_PrintfCLIAgent(["alpha", "beta", "gamma"]))

    app = FastAPI()
    app.include_router(build_router(registry), prefix="/api/agents")

    run = await mgr.spawn("smoke")
    # auto-monitor drives the run to terminal; give it a moment and
    # also wait explicitly to be deterministic.
    try:
        await asyncio.wait_for(mgr.wait(run.run_id), timeout=3.0)
    except RuntimeError:
        # ``_monitor_run`` may have already finalised it; that's fine.
        pass
    # Let the monitor task finish its bus-close + notify sequence.
    for _ in range(20):
        bus = registry.get(run.run_id)
        if bus is not None and bus.closed:
            break
        await asyncio.sleep(0.02)

    bus = registry.get(run.run_id)
    assert bus is not None, "bus should live on until retention expires"
    assert bus.closed, "bus should be closed after run terminates"

    events = bus.events_since()
    types = [e["type"] for e in events]
    assert "run.started" in types
    assert "run.finished" in types
    started = next(e for e in events if e["type"] == "run.started")
    assert started["payload"]["backend"] == "printf-agent"
    assert started["payload"]["transcript_path"]

    data, _, lost = bus.read_buffer(0)
    assert lost == 0
    for line in (b"alpha", b"beta", b"gamma"):
        assert line in data, f"Ring buffer missing {line!r}: got {data!r}"

    transcript = Path(started["payload"]["transcript_path"])
    assert transcript.exists()
    transcript_bytes = transcript.read_bytes()
    for line in (b"alpha", b"beta", b"gamma"):
        assert line in transcript_bytes

    with TestClient(app) as client:
        summary = client.get(
            f"/api/agents/{run.run_id}/summary",
        ).json()
        assert summary["closed"] is True
        assert summary["event_count"] >= 2
        assert summary["log_bytes_available"] >= len(b"alpha\nbeta\ngamma\n")

        http_events = client.get(
            f"/api/agents/{run.run_id}/events",
        ).json()["events"]
        assert [e["type"] for e in http_events] == types

        log_body = client.get(f"/api/agents/{run.run_id}/log").json()
        assert "alpha" in log_body["data"]
        assert "gamma" in log_body["data"]
