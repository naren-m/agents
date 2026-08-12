"""AgentManager -- registry, backend selection, and lifecycle orchestrator.

The single entry point consumers use. Handles backend registration,
auto-selection, spawn/cancel/wait, concurrency limits, and run tracking.
Consumers (rca-server, inspector) never touch backend classes directly.

Every spawn() also provisions a :class:`RunEventBus` in an
:class:`EventBusRegistry` so downstream consumers (the dashboard SSE
endpoint, CLI log tailers, tests) can subscribe to structured run
events and raw log bytes in real time.  Lifecycle events
``run.started``, ``run.finished``, ``run.cancelled`` and
``run.failed`` are published by the manager; content events (phase
progress, subagent output) are published by task-specific producers
that sit on top of this bus.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Any, Callable, Union

from agents.cli.base import BaseCLIAgent, CLIAgent
from agents.event_bus import EventBusRegistry, RunEventBus
from agents.inprocess.base import InProcessAgent
from agents.types import AgentConfig, AgentResult, AgentRun, _generate_run_id

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "timed_out"})


class AgentManager:
    """Registry and lifecycle manager for agent backends.

    Discovers available backends, selects the right one based on
    preference or auto-detection, spawns/manages runs, and tracks history.
    """

    MAX_CONCURRENT = 2

    def __init__(
        self,
        config: AgentConfig,
        preferred_backend: str = "auto",
        on_run_changed: Callable[[AgentRun], None] | None = None,
        event_bus_registry: EventBusRegistry | None = None,
    ):
        self._config = config
        self._preferred = preferred_backend
        self._on_run_changed = on_run_changed
        self._cli_backends: dict[str, Any] = {}
        self._inprocess_backends: dict[str, Any] = {}
        self._active_runs: dict[str, _TrackedRun] = {}
        self._history: list[AgentRun] = []
        self._event_bus_registry = (
            event_bus_registry
            if event_bus_registry is not None
            else EventBusRegistry()
        )
        self._janitor_task: asyncio.Task[None] | None = None

    @property
    def event_bus_registry(self) -> EventBusRegistry:
        """Expose the registry so web layers can mount SSE/REST endpoints."""
        return self._event_bus_registry

    def start_bus_janitor(self, interval_seconds: float = 60.0) -> None:
        """Start a background task that periodically evicts expired buses.

        The registry retains closed buses for its configured window so
        late consumers (a reloaded browser tab, a curl retry) can still
        pick up the tail of a finished run.  Once that window elapses the
        bus is garbage -- this task trims it so the registry does not
        grow unbounded over a long-running process.

        Idempotent: a second call while a janitor is already running is
        a no-op.  Start is eager (first tick is scheduled immediately)
        but the actual cleanup cadence is ``interval_seconds``.
        """
        if self._janitor_task is not None and not self._janitor_task.done():
            return
        self._janitor_task = asyncio.create_task(
            self._janitor_loop(interval_seconds),
        )

    async def stop_bus_janitor(self) -> None:
        """Cancel the janitor task and wait for it to finish.

        Safe to call without a prior ``start_bus_janitor()``.  Always
        leaves ``_janitor_task`` set back to ``None`` so a subsequent
        start works cleanly -- which matters for tests that spin
        managers up and down repeatedly.
        """
        task = self._janitor_task
        self._janitor_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            # We own the task: swallow whatever it raises on shutdown.
            pass

    async def _janitor_loop(self, interval_seconds: float) -> None:
        """Run ``cleanup_expired()`` forever at the configured cadence.

        Errors from the registry are logged and swallowed so a transient
        failure never kills the loop; the next tick retries.
        """
        while True:
            try:
                await self._event_bus_registry.cleanup_expired()
            except Exception:
                logger.exception(
                    "Event-bus janitor cleanup tick failed; will retry.",
                )
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                raise

    def _notify(self, run: AgentRun) -> None:
        """Fire the on_run_changed callback if registered."""
        if self._on_run_changed is not None:
            self._on_run_changed(run)

    async def _publish_event(
        self, run_id: str, event_type: str, payload: dict[str, Any],
    ) -> None:
        """Publish a run.* event to the bus if one exists. Never raises."""
        bus = self._event_bus_registry.get(run_id)
        if bus is None:
            return
        try:
            await bus.publish_event(event_type, payload)
        except Exception:
            logger.exception(
                "Failed to publish %s for run %s", event_type, run_id,
            )

    async def _close_bus(self, run_id: str) -> None:
        """Close the run's bus so subscribers see the terminal sentinel."""
        bus = self._event_bus_registry.get(run_id)
        if bus is None:
            return
        try:
            await bus.close()
        except Exception:
            logger.exception("Failed to close bus for run %s", run_id)

    def register_cli(self, agent: Any) -> None:
        """Register a CLI agent backend."""
        self._cli_backends[agent.name] = agent

    def register_inprocess(self, agent: Any) -> None:
        """Register an in-process agent backend."""
        self._inprocess_backends[agent.name] = agent

    def list_available(self) -> dict[str, list[str]]:
        """Return names of backends that are currently usable."""
        return {
            "cli": [
                name for name, a in self._cli_backends.items() if a.available()
            ],
            "inprocess": [
                name for name, a in self._inprocess_backends.items() if a.available()
            ],
        }

    def resolve_backend(self, preference: str = "") -> Any:
        """Pick a backend by name, or auto-select the first available.

        Resolution order for "auto":
        1. First available CLI backend (in registration order)
        2. First available in-process backend

        Raises RuntimeError if no suitable backend is found.
        """
        pref = preference or self._preferred

        if pref == "auto":
            for agent in self._cli_backends.values():
                if agent.available():
                    return agent
            for agent in self._inprocess_backends.values():
                if agent.available():
                    return agent
            raise RuntimeError(
                "No agent backend available. "
                "Install a CLI agent (cursor-agent, copilot, codex) "
                "or an in-process framework (langgraph)."
            )

        all_backends = {**self._cli_backends, **self._inprocess_backends}
        if pref not in all_backends:
            raise RuntimeError(
                f"Agent backend {pref!r} not found. "
                f"Registered: {sorted(all_backends.keys())}"
            )

        agent = all_backends[pref]
        if not agent.available():
            raise RuntimeError(
                f"Agent backend {pref!r} is registered but not available. "
                f"Check that its binary or dependencies are installed."
            )
        return agent

    async def spawn(
        self,
        prompt: str,
        backend: str = "",
        agent_name: str | None = None,
    ) -> AgentRun:
        """Spawn an agent. Main entry point for consumers.

        For CLI backends, launches a subprocess. For in-process backends,
        wraps ``run()`` in an asyncio.Task.

        ``agent_name`` is an optional human-friendly label. When set, it
        is validated against a strict allow-list (raises ``ValueError``
        on bad input) and forwarded into the backend so the resulting
        transcript file is named ``<agent_name>_<run_id>.log``.

        Every spawn provisions a per-run :class:`RunEventBus` in the
        registry so subscribers can consume structured events + raw log
        bytes. The bus is bound to a pre-generated ``run_id`` which is
        forwarded to CLI backends via ``run_id_hint`` so the bus and
        the returned ``AgentRun`` share the same identity.

        A background monitor task is created automatically so the run's
        completion is always tracked -- callers do not need to call
        ``wait()``.

        Raises RuntimeError if concurrency limit is reached, ValueError
        if ``agent_name`` is invalid.
        """
        from agents.cli.base import _validate_agent_name

        if agent_name is not None:
            _validate_agent_name(agent_name)

        active_count = len(self._active_runs)
        if active_count >= self.MAX_CONCURRENT:
            raise RuntimeError(
                f"Concurrency limit reached ({self.MAX_CONCURRENT} max). "
                f"Wait for a running agent to finish or cancel one."
            )

        agent = self.resolve_backend(backend)

        # Reserve the run id + bus up front so the backend sees a live
        # bus during spawn() and the registry entry exists before any
        # subscribers race us.
        run_id = _generate_run_id()
        bus = self._event_bus_registry.get_or_create(run_id)
        run_config = replace(self._config, event_bus=bus)
        start_monotonic = time.monotonic()

        if hasattr(agent, "build_command"):
            run = await agent.spawn(
                prompt, run_config,
                run_id_hint=run_id,
                agent_name=agent_name,
            )
            tracked = _TrackedRun(
                run=run,
                agent=agent,
                kind="cli",
                start_monotonic=start_monotonic,
            )
            self._active_runs[run.run_id] = tracked
        else:
            run = AgentRun.create(
                backend=agent.name, prompt=prompt, run_id=run_id,
            )
            run.agent_name = agent_name
            task = asyncio.create_task(
                agent.run(prompt, run_config)
            )
            tracked = _TrackedRun(
                run=run,
                agent=agent,
                kind="inprocess",
                task=task,
                start_monotonic=start_monotonic,
            )
            self._active_runs[run.run_id] = tracked
            # CLI backends publish run.started themselves from spawn();
            # in-process backends never call into BaseCLIAgent, so we
            # publish here to keep the event contract consistent.
            await self._publish_event(
                run_id,
                "run.started",
                {
                    "backend": agent.name,
                    "prompt_len": len(prompt),
                    "transcript_path": None,
                },
            )

        logger.info(
            "Agent spawned: backend=%s, run_id=%s, agent_name=%s",
            agent.name, run.run_id, agent_name or "<none>",
        )
        self._notify(run)

        asyncio.create_task(self._monitor_run(run.run_id))

        return run

    async def _monitor_run(self, run_id: str) -> None:
        """Background task: wait for a spawned run to finish and update state.

        If the backend's wait() raises, the run is marked failed so it never
        stays stuck as 'running' in the DB.  In all paths the per-run event
        bus is guaranteed to receive a terminal event and be closed.
        """
        try:
            await self.wait(run_id)
        except Exception:
            tracked = self._active_runs.pop(run_id, None)
            if tracked is not None and not tracked.run.is_terminal:
                tracked.run.mark_failed(exit_code=-1)
                self._history.append(tracked.run)
                logger.exception(
                    "Monitor: run %s failed unexpectedly", run_id,
                )
                self._notify(tracked.run)
                duration = time.monotonic() - tracked.start_monotonic
                await self._publish_event(
                    run_id,
                    "run.failed",
                    {
                        "exit_code": -1,
                        "duration_seconds": duration,
                    },
                )
                await self._close_bus(run_id)

    async def wait(self, run_id: str) -> AgentResult:
        """Wait for a run to complete. Returns the result.

        On normal termination, publishes ``run.finished`` with exit code,
        status, and duration, then closes the bus so subscribers see the
        end-of-stream sentinel.
        """
        tracked = self._active_runs.get(run_id)
        if tracked is None:
            raise RuntimeError(f"No active run with id {run_id!r}")

        if tracked.kind == "cli":
            result = await tracked.agent.wait(tracked.run)
        else:
            result = await tracked.task

        # cancel() may have already moved this run out of _active_runs
        # while we were awaiting.  Only finalize if still tracked.
        if self._active_runs.pop(run_id, None) is not None:
            self._history.append(result.run)
            self._notify(result.run)
            duration = time.monotonic() - tracked.start_monotonic
            await self._publish_event(
                run_id,
                "run.finished",
                {
                    "status": result.run.status,
                    "exit_code": result.run.exit_code,
                    "duration_seconds": duration,
                },
            )
            await self._close_bus(run_id)
        return result

    async def cancel(self, run_id: str) -> bool:
        """Cancel a running agent by run_id.

        When cancellation succeeds, publishes ``run.cancelled`` and closes
        the bus so subscribers see the terminal sentinel.
        """
        tracked = self._active_runs.get(run_id)
        if tracked is None:
            return False

        if tracked.kind == "cli":
            cancelled = await tracked.agent.cancel(tracked.run)
        else:
            if tracked.task and not tracked.task.done():
                tracked.task.cancel()
                cancelled = await tracked.agent.cancel()
            else:
                cancelled = False

        if cancelled:
            self._active_runs.pop(run_id, None)
            self._history.append(tracked.run)
            self._notify(tracked.run)
            duration = time.monotonic() - tracked.start_monotonic
            await self._publish_event(
                run_id,
                "run.cancelled",
                {"duration_seconds": duration},
            )
            await self._close_bus(run_id)

        return cancelled

    async def cancel_all(self) -> list[str]:
        """Cancel every still-active run and return the list of cancelled ids.

        Terminal runs (success/failed/cancelled) are skipped so that
        shutdown or ``POST /cancel-all`` handlers do not double-finalize
        a completed run.  Cancellation is issued sequentially against a
        snapshot of currently-active run ids; individual failures are
        swallowed so one unresponsive backend cannot block the others.
        """
        run_ids = [
            run_id
            for run_id, tracked in list(self._active_runs.items())
            if tracked.run.status not in _TERMINAL_STATUSES
        ]
        cancelled: list[str] = []
        for run_id in run_ids:
            try:
                if await self.cancel(run_id):
                    cancelled.append(run_id)
            except Exception:
                logger.exception("cancel_all: failed to cancel %s", run_id)
        return cancelled

    def get_run(self, run_id: str) -> AgentRun | None:
        """Look up a run by ID (active or historical)."""
        tracked = self._active_runs.get(run_id)
        if tracked:
            return tracked.run
        for run in self._history:
            if run.run_id == run_id:
                return run
        return None

    def list_active(self) -> list[AgentRun]:
        """Return all currently running agent runs."""
        return [t.run for t in self._active_runs.values()]

    def list_history(self) -> list[AgentRun]:
        """Return completed/cancelled/failed runs."""
        return list(self._history)


class _TrackedRun:
    """Internal bookkeeping for an active run.

    ``start_monotonic`` captures ``time.monotonic()`` at spawn so we can
    compute duration-since-start for lifecycle events regardless of
    wall-clock skew.
    """

    __slots__ = ("run", "agent", "kind", "task", "start_monotonic")

    def __init__(
        self,
        run: AgentRun,
        agent: Any,
        kind: str,
        task: asyncio.Task | None = None,
        start_monotonic: float = 0.0,
    ):
        self.run = run
        self.agent = agent
        self.kind = kind
        self.task = task
        self.start_monotonic = start_monotonic
