"""Tests for CLI agent protocol and BaseCLIAgent subprocess lifecycle."""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.types import AgentConfig


class _StubCLIAgent:
    """Minimal CLI agent for testing BaseCLIAgent plumbing.

    Simulates a CLI that prints a line and exits.
    """

    name = "stub"
    _binary_names = ["stub-agent"]

    def build_command(self, prompt: str, config: AgentConfig) -> list[str]:
        return ["echo", prompt]


@pytest.fixture
def config(tmp_path):
    return AgentConfig(
        workspace=tmp_path / "workspace",
        transcript_dir=tmp_path / "transcripts",
        timeout_seconds=5,
    )


class TestBaseCLIAgentAvailable:
    def test_available_when_binary_exists(self):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["python3"]
        assert agent.available() is True

    def test_not_available_when_binary_missing(self):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["nonexistent-binary-xyz"]
        assert agent.available() is False

    def test_binary_path_returns_path_when_found(self):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["python3"]
        path = agent.binary_path()
        assert path is not None
        assert "python" in path

    def test_binary_path_returns_none_when_missing(self):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["nonexistent-binary-xyz"]
        assert agent.binary_path() is None

    def test_tries_multiple_names_in_order(self):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["nonexistent-xyz", "python3"]
        path = agent.binary_path()
        assert path is not None


class TestBaseCLIAgentSpawn:
    @pytest.mark.asyncio
    async def test_spawn_returns_agent_run(self, config):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["echo"]
        agent.name = "test"
        agent.build_command = lambda prompt, cfg: ["echo", prompt]

        config.transcript_dir.mkdir(parents=True, exist_ok=True)
        run = await agent.spawn("hello", config)

        assert run.backend == "test"
        assert run.prompt == "hello"
        assert run.status == "running"
        assert run.pid is not None
        assert run.transcript_path is not None

    @pytest.mark.asyncio
    async def test_spawn_creates_transcript_dir(self, config):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["echo"]
        agent.name = "test"
        agent.build_command = lambda prompt, cfg: ["echo", prompt]

        assert not config.transcript_dir.exists()
        run = await agent.spawn("hello", config)
        assert config.transcript_dir.exists()

    @pytest.mark.asyncio
    async def test_spawn_raises_when_not_available(self, config):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["nonexistent-xyz"]
        agent.name = "missing"

        with pytest.raises(RuntimeError, match="not available"):
            await agent.spawn("hello", config)


class TestBaseCLIAgentWait:
    @pytest.mark.asyncio
    async def test_wait_success(self, config):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["echo"]
        agent.name = "test"
        agent.build_command = lambda prompt, cfg: ["echo", prompt]

        config.transcript_dir.mkdir(parents=True, exist_ok=True)
        run = await agent.spawn("hello world", config)
        result = await agent.wait(run)

        assert result.success is True
        assert result.run.status == "completed"
        assert result.run.exit_code == 0
        assert result.run.is_terminal is True

    @pytest.mark.asyncio
    async def test_wait_captures_output(self, config):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["echo"]
        agent.name = "test"
        agent.build_command = lambda prompt, cfg: ["echo", prompt]

        config.transcript_dir.mkdir(parents=True, exist_ok=True)
        run = await agent.spawn("test output", config)
        result = await agent.wait(run)

        assert "test output" in result.output

    @pytest.mark.asyncio
    async def test_wait_failure(self, config):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["false"]
        agent.name = "test"
        agent.build_command = lambda prompt, cfg: ["false"]

        config.transcript_dir.mkdir(parents=True, exist_ok=True)
        run = await agent.spawn("fail", config)
        result = await agent.wait(run)

        assert result.success is False
        assert result.run.status == "failed"
        assert result.run.exit_code != 0

    @pytest.mark.asyncio
    async def test_wait_writes_transcript(self, config):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["echo"]
        agent.name = "test"
        agent.build_command = lambda prompt, cfg: ["echo", "transcript content"]

        config.transcript_dir.mkdir(parents=True, exist_ok=True)
        run = await agent.spawn("test", config)
        await agent.wait(run)

        assert run.transcript_path.exists()
        content = run.transcript_path.read_text()
        assert "transcript content" in content


class TestBaseCLIAgentCancel:
    @pytest.mark.asyncio
    async def test_cancel_running_process(self, config):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["sleep"]
        agent.name = "test"
        agent.build_command = lambda prompt, cfg: ["sleep", "60"]

        config.transcript_dir.mkdir(parents=True, exist_ok=True)
        run = await agent.spawn("long task", config)

        cancelled = await agent.cancel(run)
        assert cancelled is True
        assert run.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_already_finished(self, config):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["echo"]
        agent.name = "test"
        agent.build_command = lambda prompt, cfg: ["echo", "done"]

        config.transcript_dir.mkdir(parents=True, exist_ok=True)
        run = await agent.spawn("quick", config)
        await agent.wait(run)

        cancelled = await agent.cancel(run)
        assert cancelled is False


class TestBaseCLIAgentTimeout:
    @pytest.mark.asyncio
    async def test_timeout_kills_process(self, config):
        from agents.cli.base import BaseCLIAgent

        config.timeout_seconds = 1

        agent = BaseCLIAgent()
        agent._binary_names = ["sleep"]
        agent.name = "test"
        agent.build_command = lambda prompt, cfg: ["sleep", "60"]

        config.transcript_dir.mkdir(parents=True, exist_ok=True)
        run = await agent.spawn("slow task", config)
        result = await agent.wait(run)

        assert result.success is False
        assert result.run.status == "timed_out"


class TestBaseCLIAgentStreaming:
    """Streaming stdout pump + event bus + file sink integration."""

    @pytest.mark.asyncio
    async def test_events_published_when_bus_supplied(self, config):
        from agents.cli.base import BaseCLIAgent
        from agents.event_bus import RunEventBus

        bus = RunEventBus(run_id="forced")
        config.event_bus = bus

        agent = BaseCLIAgent()
        agent._binary_names = ["printf"]
        agent.name = "stream-test"
        agent.build_command = lambda p, c: ["printf", "line1\nline2\n"]

        config.transcript_dir.mkdir(parents=True, exist_ok=True)
        run = await agent.spawn("hello", config)
        result = await agent.wait(run)

        assert result.success
        types = [e["type"] for e in bus.events_since()]
        assert "run.started" in types
        started = next(
            e for e in bus.events_since() if e["type"] == "run.started"
        )
        assert started["payload"]["backend"] == "stream-test"
        assert started["payload"]["pid"] == run.pid
        assert started["payload"]["transcript_path"] == str(run.transcript_path)
        assert started["payload"]["prompt_len"] == len("hello")

    @pytest.mark.asyncio
    async def test_ring_buffer_captures_stdout(self, config):
        from agents.cli.base import BaseCLIAgent
        from agents.event_bus import RunEventBus

        bus = RunEventBus(run_id="rb")
        config.event_bus = bus

        agent = BaseCLIAgent()
        agent._binary_names = ["printf"]
        agent.name = "stream-test"
        agent.build_command = lambda p, c: ["printf", "alpha\nbeta\n"]

        config.transcript_dir.mkdir(parents=True, exist_ok=True)
        run = await agent.spawn("x", config)
        await agent.wait(run)

        data, _, lost = bus.read_buffer(0)
        assert lost == 0
        assert b"alpha" in data
        assert b"beta" in data

    @pytest.mark.asyncio
    async def test_transcript_written_incrementally(self, config):
        """Transcript must contain first line well before the process ends.

        Reads the transcript mid-``wait()`` by racing a peek task against
        the wait -- this mirrors how the dashboard will ``tail -f`` the
        file while the agent is still running.
        """
        from agents.cli.base import BaseCLIAgent
        from agents.event_bus import RunEventBus

        bus = RunEventBus(run_id="inc")
        config.event_bus = bus
        config.transcript_dir.mkdir(parents=True, exist_ok=True)

        agent = BaseCLIAgent()
        agent._binary_names = ["bash"]
        agent.name = "stream-test"
        agent.build_command = lambda p, c: [
            "bash", "-c", "echo first; sleep 0.4; echo second",
        ]

        run = await agent.spawn("x", config)

        mid_run_snapshot: dict[str, bytes] = {}

        async def _peek() -> None:
            await asyncio.sleep(0.15)
            mid_run_snapshot["data"] = (
                run.transcript_path.read_bytes()
                if run.transcript_path and run.transcript_path.exists()
                else b""
            )

        peek_task = asyncio.create_task(_peek())
        await agent.wait(run)
        await peek_task

        partial = mid_run_snapshot.get("data", b"")
        assert b"first" in partial, (
            f"Expected first line in transcript mid-run, got {partial!r}"
        )
        # "second" appears after the sleep, so it should NOT be in the
        # snapshot taken before wait() finished.
        assert b"second" not in partial

    @pytest.mark.asyncio
    async def test_parse_output_still_receives_full_stdout(self, config):
        """Ring buffer may wrap, but parse_output must see everything."""
        from agents.cli.base import BaseCLIAgent
        from agents.event_bus import RunEventBus

        bus = RunEventBus(run_id="po", ring_buffer_bytes=16)
        config.event_bus = bus
        config.transcript_dir.mkdir(parents=True, exist_ok=True)

        seen: list[str] = []

        class _PA(BaseCLIAgent):
            name = "po"
            _binary_names = ["printf"]

            def build_command(self, prompt, cfg):
                return ["printf", "A" * 200 + "\n"]

            def parse_output(self, output, run):
                seen.append(output)

        agent = _PA()
        run = await agent.spawn("x", config)
        await agent.wait(run)

        assert seen and len(seen[0]) >= 200

    @pytest.mark.asyncio
    async def test_streaming_preserves_lines_larger_than_stream_limit(self, config):
        from agents.cli.base import BaseCLIAgent
        from agents.event_bus import RunEventBus

        config.event_bus = RunEventBus(run_id="oversized-line")
        config.transcript_dir.mkdir(parents=True, exist_ok=True)

        agent = BaseCLIAgent()
        agent._binary_names = ["python3"]
        agent.name = "oversized-line"
        agent.build_command = lambda p, c: [
            "python3",
            "-c",
            "import sys; sys.stdout.write('A' * 70000 + '\\n')",
        ]

        run = await agent.spawn("x", config)
        result = await agent.wait(run)

        assert result.success
        assert run.transcript_path is not None
        transcript = run.transcript_path.read_text()
        assert transcript.endswith("\n")
        assert len(transcript) == 70001

    @pytest.mark.asyncio
    async def test_no_bus_preserves_legacy_behaviour(self, config):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["echo"]
        agent.name = "legacy"
        agent.build_command = lambda p, c: ["echo", "legacy"]

        config.transcript_dir.mkdir(parents=True, exist_ok=True)
        run = await agent.spawn("x", config)
        result = await agent.wait(run)
        assert result.success
        assert "legacy" in result.output
        assert run.transcript_path.read_text().strip() == "legacy"

    @pytest.mark.asyncio
    async def test_spawn_respects_run_id_hint(self, config):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["echo"]
        agent.name = "t"
        agent.build_command = lambda p, c: ["echo", p]

        config.transcript_dir.mkdir(parents=True, exist_ok=True)
        run = await agent.spawn("hi", config, run_id_hint="fixed0001")
        assert run.run_id == "fixed0001"
        await agent.wait(run)

    @pytest.mark.asyncio
    async def test_wait_returns_when_orphan_keeps_stdout_open(self, config):
        """``wait()`` must not block on stdout EOF when a forked daemon
        survives the direct child.

        Real-world trigger: cursor-agent forks a long-lived
        ``worker-server`` child that inherits the parent's stdout fd.
        When the cursor-agent process exits, that fd remains open via
        the daemon, so ``readline()`` never sees EOF.  Without an
        explicit drain bound, ``await reader_task`` after
        ``process.wait()`` blocks for the daemon's full lifetime and
        the AgentRun stays in ``running`` indefinitely.

        The bash command below echoes its line, then forks
        ``sleep 30`` into a detached subshell that inherits stdout.
        The direct bash exits in milliseconds; the orphan keeps the
        pipe open for 30s.  ``wait()`` must finalize the run within a
        bounded drain window (well under 30s).
        """
        from agents.cli.base import BaseCLIAgent
        from agents.event_bus import RunEventBus

        bus = RunEventBus(run_id="orphan")
        config.event_bus = bus
        config.transcript_dir.mkdir(parents=True, exist_ok=True)

        agent = BaseCLIAgent()
        agent._binary_names = ["bash"]
        agent.name = "orphan-test"
        agent.build_command = lambda p, c: [
            "bash", "-c", "(sleep 30 &) ; echo done",
        ]

        run = await agent.spawn("x", config)

        try:
            result = await asyncio.wait_for(agent.wait(run), timeout=4.0)
        except asyncio.TimeoutError:
            pytest.fail(
                "wait() blocked on orphan daemon's stdout fd; the "
                "drain after process.wait() must be bounded.",
            )

        assert result.success
        assert "done" in result.output


class TestValidateAgentName:
    """Strict allow-list validation for caller-supplied agent_name."""

    @pytest.mark.parametrize(
        "name",
        [
            "rca-gather",
            "phase_1",
            "a",
            "a.b.c",
            "X-Y-Z",
            "abc123",
            "A1B2-c3.d4_e5",
            "x" * 64,
        ],
    )
    def test_accepts_valid_name(self, name):
        from agents.cli.base import _validate_agent_name

        _validate_agent_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "",
            " ",
            "   ",
            "rca gather",
            "slash/inside",
            "back\\slash",
            "../etc",
            ".hidden",
            "-flag",
            "..",
            ".",
            "foo\x00bar",
            "foo\nbar",
            "foo\tbar",
            "\u202ertl",
            "x" * 65,
            "name with spaces and / inside",
            "a..b",
        ],
    )
    def test_rejects_invalid_name(self, name):
        from agents.cli.base import _validate_agent_name

        with pytest.raises(ValueError, match="Invalid agent_name"):
            _validate_agent_name(name)

    def test_rejects_non_string(self):
        from agents.cli.base import _validate_agent_name

        with pytest.raises(ValueError, match="Invalid agent_name"):
            _validate_agent_name(123)


class TestBuildTranscriptPath:
    """Path resolution: (transcript_dir, agent_name) -> Path."""

    def test_default_dir_when_transcript_dir_none(self, tmp_path, monkeypatch):
        from agents.cli import base as base_mod
        from agents.cli.base import _build_transcript_path
        from agents.types import AgentRun

        monkeypatch.setattr(base_mod, "_DEFAULT_TRANSCRIPT_DIR", tmp_path)

        cfg = AgentConfig(workspace=tmp_path / "ws", transcript_dir=None)
        run = AgentRun.create(backend="t", prompt="x")

        path = _build_transcript_path(cfg, run, agent_name=None)
        assert path == tmp_path / f"{run.run_id}.log"

    def test_default_dir_with_agent_name(self, tmp_path, monkeypatch):
        from agents.cli import base as base_mod
        from agents.cli.base import _build_transcript_path
        from agents.types import AgentRun

        monkeypatch.setattr(base_mod, "_DEFAULT_TRANSCRIPT_DIR", tmp_path)

        cfg = AgentConfig(workspace=tmp_path / "ws", transcript_dir=None)
        run = AgentRun.create(backend="t", prompt="x")

        path = _build_transcript_path(cfg, run, agent_name="rca-gather")
        assert path == tmp_path / f"rca-gather_{run.run_id}.log"

    def test_custom_dir_no_name(self, tmp_path):
        from agents.cli.base import _build_transcript_path
        from agents.types import AgentRun

        custom = tmp_path / "custom"
        cfg = AgentConfig(workspace=tmp_path / "ws", transcript_dir=custom)
        run = AgentRun.create(backend="t", prompt="x")

        path = _build_transcript_path(cfg, run, agent_name=None)
        assert path == custom / f"{run.run_id}.log"

    def test_custom_dir_with_name(self, tmp_path):
        from agents.cli.base import _build_transcript_path
        from agents.types import AgentRun

        custom = tmp_path / "custom"
        cfg = AgentConfig(workspace=tmp_path / "ws", transcript_dir=custom)
        run = AgentRun.create(backend="t", prompt="x")

        path = _build_transcript_path(cfg, run, agent_name="phase_1")
        assert path == custom / f"phase_1_{run.run_id}.log"

    def test_creates_parent_directory(self, tmp_path):
        from agents.cli.base import _build_transcript_path
        from agents.types import AgentRun

        custom = tmp_path / "newly" / "made"
        assert not custom.exists()

        cfg = AgentConfig(workspace=tmp_path / "ws", transcript_dir=custom)
        run = AgentRun.create(backend="t", prompt="x")

        _build_transcript_path(cfg, run, agent_name=None)
        assert custom.is_dir()

    def test_default_constant_value(self):
        from agents.cli.base import _DEFAULT_TRANSCRIPT_DIR

        assert _DEFAULT_TRANSCRIPT_DIR == Path("/tmp/agent_logs")


class TestBaseCLIAgentSpawnAgentName:
    """Spawn integration: agent_name flows from spawn into transcript_path."""

    @pytest.mark.asyncio
    async def test_spawn_uses_default_dir_when_transcript_dir_none(
        self, tmp_path, monkeypatch
    ):
        from agents.cli import base as base_mod
        from agents.cli.base import BaseCLIAgent

        monkeypatch.setattr(base_mod, "_DEFAULT_TRANSCRIPT_DIR", tmp_path)

        agent = BaseCLIAgent()
        agent._binary_names = ["echo"]
        agent.name = "spawn-default-dir"
        agent.build_command = lambda p, c: ["echo", p]

        cfg = AgentConfig(workspace=tmp_path / "ws", transcript_dir=None)
        run = await agent.spawn("hello", cfg)
        await agent.wait(run)

        assert run.transcript_path is not None
        assert run.transcript_path.parent == tmp_path
        assert run.transcript_path.name == f"{run.run_id}.log"

    @pytest.mark.asyncio
    async def test_spawn_includes_agent_name_in_filename(
        self, tmp_path, monkeypatch
    ):
        from agents.cli import base as base_mod
        from agents.cli.base import BaseCLIAgent

        monkeypatch.setattr(base_mod, "_DEFAULT_TRANSCRIPT_DIR", tmp_path)

        agent = BaseCLIAgent()
        agent._binary_names = ["echo"]
        agent.name = "spawn-named"
        agent.build_command = lambda p, c: ["echo", p]

        cfg = AgentConfig(workspace=tmp_path / "ws", transcript_dir=None)
        run = await agent.spawn("hello", cfg, agent_name="rca-gather")
        await agent.wait(run)

        assert run.transcript_path.name == f"rca-gather_{run.run_id}.log"
        assert run.agent_name == "rca-gather"

    @pytest.mark.asyncio
    async def test_spawn_omits_name_segment_when_absent(
        self, tmp_path, monkeypatch
    ):
        from agents.cli import base as base_mod
        from agents.cli.base import BaseCLIAgent

        monkeypatch.setattr(base_mod, "_DEFAULT_TRANSCRIPT_DIR", tmp_path)

        agent = BaseCLIAgent()
        agent._binary_names = ["echo"]
        agent.name = "spawn-unnamed"
        agent.build_command = lambda p, c: ["echo", p]

        cfg = AgentConfig(workspace=tmp_path / "ws", transcript_dir=None)
        run = await agent.spawn("hello", cfg)
        await agent.wait(run)

        assert run.transcript_path.name == f"{run.run_id}.log"
        assert run.agent_name is None

    @pytest.mark.asyncio
    async def test_spawn_uses_custom_transcript_dir(self, tmp_path):
        from agents.cli.base import BaseCLIAgent

        custom = tmp_path / "custom"
        agent = BaseCLIAgent()
        agent._binary_names = ["echo"]
        agent.name = "spawn-custom-dir"
        agent.build_command = lambda p, c: ["echo", p]

        cfg = AgentConfig(workspace=tmp_path / "ws", transcript_dir=custom)
        run = await agent.spawn("hello", cfg, agent_name="phase_1")
        await agent.wait(run)

        assert run.transcript_path.parent == custom
        assert run.transcript_path.name == f"phase_1_{run.run_id}.log"

    @pytest.mark.asyncio
    async def test_spawn_rejects_invalid_agent_name(self, tmp_path):
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["echo"]
        agent.name = "spawn-rejects"
        agent.build_command = lambda p, c: ["echo", p]

        cfg = AgentConfig(workspace=tmp_path / "ws", transcript_dir=tmp_path)

        with pytest.raises(ValueError, match="Invalid agent_name"):
            await agent.spawn("hello", cfg, agent_name="../etc")

        # Validation must happen before any subprocess is started.
        assert agent._processes == {}

    @pytest.mark.asyncio
    async def test_spawn_logs_transcript_path(self, tmp_path, caplog):
        import logging
        from agents.cli.base import BaseCLIAgent

        agent = BaseCLIAgent()
        agent._binary_names = ["echo"]
        agent.name = "spawn-logs"
        agent.build_command = lambda p, c: ["echo", p]

        cfg = AgentConfig(workspace=tmp_path / "ws", transcript_dir=tmp_path)

        with caplog.at_level(logging.INFO, logger="agents.cli.base"):
            run = await agent.spawn("hello", cfg, agent_name="phase_2")
            await agent.wait(run)

        records = [r for r in caplog.records if "Transcript:" in r.getMessage()]
        assert records, "Expected an INFO log line containing 'Transcript:'"
        msg = records[0].getMessage()
        assert str(run.transcript_path) in msg
        assert run.run_id in msg
        assert "phase_2" in msg

    @pytest.mark.asyncio
    async def test_two_spawns_same_name_get_distinct_files(
        self, tmp_path, monkeypatch
    ):
        from agents.cli import base as base_mod
        from agents.cli.base import BaseCLIAgent

        monkeypatch.setattr(base_mod, "_DEFAULT_TRANSCRIPT_DIR", tmp_path)

        def _make():
            a = BaseCLIAgent()
            a._binary_names = ["echo"]
            a.name = "twin"
            a.build_command = lambda p, c: ["echo", p]
            return a

        cfg = AgentConfig(workspace=tmp_path / "ws", transcript_dir=None)
        run1 = await _make().spawn("a", cfg, agent_name="dup")
        run2 = await _make().spawn("b", cfg, agent_name="dup")

        assert run1.transcript_path != run2.transcript_path
        assert run1.run_id != run2.run_id
        assert run1.transcript_path.name == f"dup_{run1.run_id}.log"
        assert run2.transcript_path.name == f"dup_{run2.run_id}.log"
