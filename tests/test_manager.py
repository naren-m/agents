"""Tests for AgentManager -- registry, backend selection, lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.types import AgentConfig, AgentResult, AgentRun


@pytest.fixture
def config(tmp_path):
    return AgentConfig(
        workspace=tmp_path / "workspace",
        transcript_dir=tmp_path / "transcripts",
        timeout_seconds=5,
    )


class _FakeCLIAgent:
    """Controllable fake for testing manager without real subprocesses."""

    def __init__(self, name: str = "fake-cli", is_available: bool = True):
        self.name = name
        self._available = is_available
        self._spawned_runs: list[AgentRun] = []

    def available(self) -> bool:
        return self._available

    def binary_path(self) -> str | None:
        return "/usr/bin/fake" if self._available else None

    def build_command(self, prompt, config):
        return ["fake", prompt]

    async def spawn(
        self,
        prompt: str,
        config: AgentConfig,
        run_id_hint: str | None = None,
        agent_name: str | None = None,
    ) -> AgentRun:
        run = AgentRun.create(
            backend=self.name, prompt=prompt, pid=12345, run_id=run_id_hint,
        )
        run.agent_name = agent_name
        self._spawned_runs.append(run)
        self._last_config = config
        self._last_agent_name = agent_name
        return run

    async def cancel(self, run: AgentRun) -> bool:
        run.mark_cancelled()
        return True

    async def wait(self, run: AgentRun) -> AgentResult:
        run.mark_completed()
        return AgentResult(success=True, output="done", run=run)


class _FakeInProcessAgent:
    """Controllable fake for in-process backend testing."""

    def __init__(self, name: str = "fake-inprocess", is_available: bool = True):
        self.name = name
        self._available = is_available

    def available(self) -> bool:
        return self._available

    async def run(self, prompt, config, mcp_client=None, on_progress=None):
        run = AgentRun.create(backend=self.name, prompt=prompt)
        run.mark_completed()
        return AgentResult(success=True, output="in-process done", run=run)

    async def cancel(self) -> bool:
        return True


class TestManagerRegistration:
    def test_register_cli_backend(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        fake = _FakeCLIAgent("test-cli")
        mgr.register_cli(fake)

        available = mgr.list_available()
        assert "test-cli" in available["cli"]

    def test_register_inprocess_backend(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        fake = _FakeInProcessAgent("test-ip")
        mgr.register_inprocess(fake)

        available = mgr.list_available()
        assert "test-ip" in available["inprocess"]

    def test_list_available_excludes_unavailable(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        mgr.register_cli(_FakeCLIAgent("yes-cli", is_available=True))
        mgr.register_cli(_FakeCLIAgent("no-cli", is_available=False))

        available = mgr.list_available()
        assert "yes-cli" in available["cli"]
        assert "no-cli" not in available["cli"]


class TestManagerResolveBackend:
    def test_auto_prefers_cli(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        cli = _FakeCLIAgent("cli-one")
        ip = _FakeInProcessAgent("ip-one")
        mgr.register_cli(cli)
        mgr.register_inprocess(ip)

        resolved = mgr.resolve_backend()
        assert resolved.name == "cli-one"

    def test_auto_falls_back_to_inprocess(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        ip = _FakeInProcessAgent("ip-one")
        mgr.register_inprocess(ip)

        resolved = mgr.resolve_backend()
        assert resolved.name == "ip-one"

    def test_explicit_backend_selection(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        mgr.register_cli(_FakeCLIAgent("a"))
        mgr.register_cli(_FakeCLIAgent("b"))

        resolved = mgr.resolve_backend("b")
        assert resolved.name == "b"

    def test_explicit_inprocess_selection(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        mgr.register_inprocess(_FakeInProcessAgent("my-ip"))

        resolved = mgr.resolve_backend("my-ip")
        assert resolved.name == "my-ip"

    def test_raises_when_nothing_available(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")

        with pytest.raises(RuntimeError, match="No agent backend available"):
            mgr.resolve_backend()

    def test_raises_when_explicit_not_found(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        mgr.register_cli(_FakeCLIAgent("exists"))

        with pytest.raises(RuntimeError, match="not found"):
            mgr.resolve_backend("nonexistent")

    def test_raises_when_explicit_not_available(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        mgr.register_cli(_FakeCLIAgent("unavail", is_available=False))

        with pytest.raises(RuntimeError, match="not available"):
            mgr.resolve_backend("unavail")


class TestManagerSpawn:
    @pytest.mark.asyncio
    async def test_spawn_cli(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        fake = _FakeCLIAgent("test-cli")
        mgr.register_cli(fake)

        run = await mgr.spawn("do stuff")
        assert run.backend == "test-cli"
        assert run.status == "running"
        assert run.run_id in [r.run_id for r in mgr.list_active()]

    @pytest.mark.asyncio
    async def test_spawn_with_explicit_backend(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        mgr.register_cli(_FakeCLIAgent("a"))
        mgr.register_cli(_FakeCLIAgent("b"))

        run = await mgr.spawn("task", backend="b")
        assert run.backend == "b"

    @pytest.mark.asyncio
    async def test_concurrency_limit(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        mgr.MAX_CONCURRENT = 1
        mgr.register_cli(_FakeCLIAgent("test"))

        await mgr.spawn("first")
        with pytest.raises(RuntimeError, match="[Cc]oncurren"):
            await mgr.spawn("second")


class TestManagerWaitAndCancel:
    @pytest.mark.asyncio
    async def test_wait_returns_result(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        fake = _FakeCLIAgent("test")
        mgr.register_cli(fake)

        run = await mgr.spawn("task")
        result = await mgr.wait(run.run_id)

        assert result.success is True
        assert result.run.is_terminal is True
        assert run.run_id not in [r.run_id for r in mgr.list_active()]

    @pytest.mark.asyncio
    async def test_wait_moves_to_history(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        mgr.register_cli(_FakeCLIAgent("test"))

        run = await mgr.spawn("task")
        await mgr.wait(run.run_id)

        assert run.run_id in [r.run_id for r in mgr.list_history()]

    @pytest.mark.asyncio
    async def test_cancel(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        mgr.register_cli(_FakeCLIAgent("test"))

        run = await mgr.spawn("task")
        cancelled = await mgr.cancel(run.run_id)

        assert cancelled is True
        assert run.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_unknown_run(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        cancelled = await mgr.cancel("nonexistent")
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_cancel_all_returns_cancelled_ids(self, config):
        """``cancel_all()`` is the bulk shutdown API used by hosts.

        Every run that was still alive at call time must end up
        cancelled, and the list of cancelled run_ids must be returned
        so callers (e.g. FastAPI /cancel-all or app shutdown hooks) can
        log/report what they did.
        """
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        mgr.register_cli(_FakeCLIAgent("test"))

        run_a = await mgr.spawn("task a")
        run_b = await mgr.spawn("task b")

        cancelled_ids = await mgr.cancel_all()

        assert set(cancelled_ids) == {run_a.run_id, run_b.run_id}
        assert run_a.status == "cancelled"
        assert run_b.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_all_skips_terminal_runs(self, config):
        """Already-terminal runs (success / failed / cancelled) are left
        alone and not included in the returned list; otherwise shutdown
        would double-finalize a completed run."""
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        mgr.register_cli(_FakeCLIAgent("test"))

        finished = await mgr.spawn("done")
        await mgr.wait(finished.run_id)
        live = await mgr.spawn("alive")

        cancelled_ids = await mgr.cancel_all()

        assert cancelled_ids == [live.run_id]
        assert finished.status != "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_all_empty_manager_returns_empty_list(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        assert await mgr.cancel_all() == []


class TestManagerTracking:
    @pytest.mark.asyncio
    async def test_get_run(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        mgr.register_cli(_FakeCLIAgent("test"))

        run = await mgr.spawn("task")
        found = mgr.get_run(run.run_id)
        assert found is not None
        assert found.run_id == run.run_id

    def test_get_run_not_found(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        assert mgr.get_run("nonexistent") is None

    @pytest.mark.asyncio
    async def test_list_active(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        mgr.MAX_CONCURRENT = 3
        mgr.register_cli(_FakeCLIAgent("test"))

        r1 = await mgr.spawn("one")
        r2 = await mgr.spawn("two")

        active = mgr.list_active()
        assert len(active) == 2

    @pytest.mark.asyncio
    async def test_uses_preferred_backend(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="b")
        mgr.register_cli(_FakeCLIAgent("a"))
        mgr.register_cli(_FakeCLIAgent("b"))

        run = await mgr.spawn("task")
        assert run.backend == "b"


class TestOnRunChangedCallback:
    """Verify the on_run_changed callback fires on state transitions."""

    @pytest.fixture
    def callback_log(self):
        return []

    @pytest.fixture
    def manager_with_callback(self, config, callback_log):
        from agents.manager import AgentManager
        def on_changed(run):
            callback_log.append({"run_id": run.run_id, "status": run.status})
        mgr = AgentManager(config=config, on_run_changed=on_changed)
        mgr.register_cli(_FakeCLIAgent("fake-cli"))
        return mgr

    @pytest.mark.asyncio
    async def test_callback_fires_on_spawn(self, manager_with_callback, callback_log):
        run = await manager_with_callback.spawn("test prompt")
        assert len(callback_log) == 1
        assert callback_log[0]["run_id"] == run.run_id
        assert callback_log[0]["status"] == "running"

    @pytest.mark.asyncio
    async def test_callback_fires_on_wait_completed(self, manager_with_callback, callback_log):
        run = await manager_with_callback.spawn("test prompt")
        await manager_with_callback.wait(run.run_id)
        assert len(callback_log) == 2
        assert callback_log[1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_callback_fires_on_cancel(self, manager_with_callback, callback_log):
        run = await manager_with_callback.spawn("test prompt")
        await manager_with_callback.cancel(run.run_id)
        assert len(callback_log) == 2
        assert callback_log[1]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_no_callback_when_none(self, config):
        from agents.manager import AgentManager
        mgr = AgentManager(config=config)
        mgr.register_cli(_FakeCLIAgent("fake-cli"))
        run = await mgr.spawn("test prompt")
        assert run.status == "running"


class _SlowFakeCLIAgent(_FakeCLIAgent):
    """CLI agent whose wait() completes after a short delay."""

    def __init__(self, name="slow-cli", delay: float = 0.05):
        super().__init__(name=name)
        self._delay = delay
        self.wait_called = False

    async def wait(self, run: AgentRun) -> AgentResult:
        self.wait_called = True
        await asyncio.sleep(self._delay)
        run.mark_completed()
        return AgentResult(success=True, output="done", run=run)


class _FailingFakeCLIAgent(_FakeCLIAgent):
    """CLI agent whose wait() raises an exception."""

    def __init__(self, name="failing-cli"):
        super().__init__(name=name)

    async def wait(self, run: AgentRun) -> AgentResult:
        raise RuntimeError("process exploded")


class TestAutoMonitor:
    """Verify spawn() creates a background task that waits for the run."""

    @pytest.mark.asyncio
    async def test_spawn_auto_monitors_to_completion(self, config):
        """After spawn, the run should reach terminal state without explicit wait()."""
        from agents.manager import AgentManager

        callback_log = []
        def on_changed(run):
            callback_log.append({"run_id": run.run_id, "status": run.status})

        mgr = AgentManager(config=config, on_run_changed=on_changed)
        agent = _SlowFakeCLIAgent(delay=0.05)
        mgr.register_cli(agent)

        run = await mgr.spawn("test prompt")
        assert run.status == "running"

        # Give the background monitor task time to complete
        await asyncio.sleep(0.2)

        assert agent.wait_called is True
        assert run.is_terminal is True
        assert run.status == "completed"
        # Callback fired for spawn (running) and completion
        statuses = [e["status"] for e in callback_log]
        assert "running" in statuses
        assert "completed" in statuses

    @pytest.mark.asyncio
    async def test_spawn_auto_monitor_marks_failed_on_exception(self, config):
        """If the backend wait() crashes, monitor marks run as failed."""
        from agents.manager import AgentManager

        callback_log = []
        def on_changed(run):
            callback_log.append({"run_id": run.run_id, "status": run.status})

        mgr = AgentManager(config=config, on_run_changed=on_changed)
        mgr.register_cli(_FailingFakeCLIAgent())

        run = await mgr.spawn("test prompt")
        await asyncio.sleep(0.1)

        assert run.is_terminal is True
        assert run.status == "failed"
        statuses = [e["status"] for e in callback_log]
        assert "failed" in statuses

    @pytest.mark.asyncio
    async def test_auto_monitor_removes_from_active(self, config):
        """After auto-monitor completes, the run should not be in active list."""
        from agents.manager import AgentManager

        mgr = AgentManager(config=config)
        mgr.register_cli(_SlowFakeCLIAgent(delay=0.05))

        run = await mgr.spawn("test prompt")
        assert len(mgr.list_active()) == 1

        await asyncio.sleep(0.2)

        assert len(mgr.list_active()) == 0
        assert run.run_id in [r.run_id for r in mgr.list_history()]

    @pytest.mark.asyncio
    async def test_cancel_before_monitor_completes(self, config):
        """Cancelling a run before the monitor wait finishes should work cleanly."""
        from agents.manager import AgentManager

        mgr = AgentManager(config=config)
        agent = _SlowFakeCLIAgent(delay=1.0)
        mgr.register_cli(agent)

        run = await mgr.spawn("test prompt")
        cancelled = await mgr.cancel(run.run_id)

        assert cancelled is True
        assert run.status == "cancelled"


class TestManagerEventBusIntegration:
    """Manager creates a RunEventBus per spawn and emits run.* events."""

    @pytest.mark.asyncio
    async def test_spawn_creates_bus_in_registry_for_run(self, config):
        from agents.event_bus import EventBusRegistry
        from agents.manager import AgentManager

        registry = EventBusRegistry()
        mgr = AgentManager(config=config, event_bus_registry=registry)
        mgr.register_cli(_FakeCLIAgent("fake-cli"))

        run = await mgr.spawn("hello")

        bus = registry.get(run.run_id)
        assert bus is not None, "Registry should have a bus for the run"
        assert bus.run_id == run.run_id

    @pytest.mark.asyncio
    async def test_spawn_passes_bus_to_backend_config(self, config):
        """Backend must see a config where event_bus points at the live bus."""
        from agents.event_bus import EventBusRegistry
        from agents.manager import AgentManager

        registry = EventBusRegistry()
        mgr = AgentManager(config=config, event_bus_registry=registry)
        agent = _FakeCLIAgent("fake-cli")
        mgr.register_cli(agent)

        run = await mgr.spawn("hello")

        assert agent._last_config.event_bus is not None
        assert agent._last_config.event_bus is registry.get(run.run_id)

    @pytest.mark.asyncio
    async def test_manager_creates_default_registry(self, config):
        """If no registry is passed, one should be created automatically."""
        from agents.event_bus import EventBusRegistry
        from agents.manager import AgentManager

        mgr = AgentManager(config=config)
        assert isinstance(mgr.event_bus_registry, EventBusRegistry)

    @pytest.mark.asyncio
    async def test_wait_publishes_run_finished(self, config):
        from agents.event_bus import EventBusRegistry
        from agents.manager import AgentManager

        registry = EventBusRegistry()
        mgr = AgentManager(config=config, event_bus_registry=registry)
        mgr.register_cli(_FakeCLIAgent("fake-cli"))

        run = await mgr.spawn("hello")
        bus = registry.get(run.run_id)
        await mgr.wait(run.run_id)

        events = bus.events_since(None)
        finished = [e for e in events if e["type"] == "run.finished"]
        assert len(finished) == 1
        payload = finished[0]["payload"]
        assert payload["status"] == "completed"
        assert "duration_seconds" in payload

    @pytest.mark.asyncio
    async def test_cancel_publishes_run_cancelled(self, config):
        from agents.event_bus import EventBusRegistry
        from agents.manager import AgentManager

        registry = EventBusRegistry()
        mgr = AgentManager(config=config, event_bus_registry=registry)
        mgr.register_cli(_FakeCLIAgent("fake-cli"))

        run = await mgr.spawn("hello")
        bus = registry.get(run.run_id)
        await mgr.cancel(run.run_id)

        events = bus.events_since(None)
        cancelled = [e for e in events if e["type"] == "run.cancelled"]
        assert len(cancelled) == 1

    @pytest.mark.asyncio
    async def test_bus_is_closed_after_terminal(self, config):
        """Subscribers should see a sentinel once the bus is closed."""
        from agents.event_bus import EventBusRegistry
        from agents.manager import AgentManager

        registry = EventBusRegistry()
        mgr = AgentManager(config=config, event_bus_registry=registry)
        mgr.register_cli(_SlowFakeCLIAgent(delay=0.02))

        run = await mgr.spawn("hello")
        bus = registry.get(run.run_id)
        await asyncio.sleep(0.15)

        assert bus.closed is True

    @pytest.mark.asyncio
    async def test_failed_monitor_publishes_run_failed(self, config):
        """A crashing wait() should still terminate the bus with run.failed."""
        from agents.event_bus import EventBusRegistry
        from agents.manager import AgentManager

        registry = EventBusRegistry()
        mgr = AgentManager(config=config, event_bus_registry=registry)
        mgr.register_cli(_FailingFakeCLIAgent())

        run = await mgr.spawn("hello")
        bus = registry.get(run.run_id)
        await asyncio.sleep(0.1)

        events = bus.events_since(None)
        failed = [e for e in events if e["type"] == "run.failed"]
        assert len(failed) == 1
        assert bus.closed is True


class TestBusJanitor:
    """The manager runs a periodic cleanup of expired buses.

    Retention behaviour lives on :class:`EventBusRegistry`; the manager
    just drives it on a timer so callers do not have to wire their own
    background task.  These tests use short retention windows so they
    stay fast.
    """

    @pytest.mark.asyncio
    async def test_start_janitor_removes_expired_bus(self, config):
        from agents.event_bus import EventBusRegistry
        from agents.manager import AgentManager

        registry = EventBusRegistry(retention_seconds=0.05)
        mgr = AgentManager(config=config, event_bus_registry=registry)

        bus = registry.get_or_create("run-to-evict")
        await bus.close()

        mgr.start_bus_janitor(interval_seconds=0.02)
        try:
            for _ in range(50):
                if registry.get("run-to-evict") is None:
                    break
                await asyncio.sleep(0.02)
        finally:
            await mgr.stop_bus_janitor()

        assert registry.get("run-to-evict") is None, (
            "Janitor should have evicted the closed bus after retention "
            "expired."
        )

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config=config)
        mgr.start_bus_janitor(interval_seconds=0.05)
        first_task = mgr._janitor_task  # intentional private check
        mgr.start_bus_janitor(interval_seconds=0.05)
        second_task = mgr._janitor_task
        try:
            assert first_task is second_task, (
                "Calling start_bus_janitor twice must not spawn a second task."
            )
        finally:
            await mgr.stop_bus_janitor()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config=config)
        await mgr.stop_bus_janitor()  # should not raise
        assert mgr._janitor_task is None

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config=config)
        mgr.start_bus_janitor(interval_seconds=0.01)
        task = mgr._janitor_task
        assert task is not None
        await mgr.stop_bus_janitor()
        assert task.done()
        assert mgr._janitor_task is None


class TestManagerAgentName:
    """`AgentManager.spawn()` accepts and forwards agent_name."""

    @pytest.mark.asyncio
    async def test_spawn_forwards_agent_name_to_backend(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        fake = _FakeCLIAgent("test-cli")
        mgr.register_cli(fake)

        run = await mgr.spawn("task", agent_name="rca-gather")

        assert fake._last_agent_name == "rca-gather"
        assert run.agent_name == "rca-gather"

    @pytest.mark.asyncio
    async def test_spawn_forwards_none_when_omitted(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        fake = _FakeCLIAgent("test-cli")
        mgr.register_cli(fake)

        run = await mgr.spawn("task")

        assert fake._last_agent_name is None
        assert run.agent_name is None

    @pytest.mark.asyncio
    async def test_spawn_rejects_invalid_agent_name_at_manager_layer(
        self, config
    ):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        fake = _FakeCLIAgent("test-cli")
        mgr.register_cli(fake)

        with pytest.raises(ValueError, match="Invalid agent_name"):
            await mgr.spawn("task", agent_name="../etc")

        # Backend must NOT have been called -- validation runs first.
        assert fake._spawned_runs == []

    @pytest.mark.asyncio
    async def test_inprocess_spawn_records_agent_name(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        ip = _FakeInProcessAgent("ip-name-test")
        mgr.register_inprocess(ip)

        run = await mgr.spawn("task", agent_name="phase_1")

        assert run.agent_name == "phase_1"

    @pytest.mark.asyncio
    async def test_history_preserves_agent_name(self, config):
        from agents.manager import AgentManager

        mgr = AgentManager(config, preferred_backend="auto")
        mgr.register_cli(_FakeCLIAgent("test-cli"))

        run = await mgr.spawn("task", agent_name="rca-gather")
        await mgr.wait(run.run_id)

        history = mgr.list_history()
        assert any(
            r.run_id == run.run_id and r.agent_name == "rca-gather"
            for r in history
        )
