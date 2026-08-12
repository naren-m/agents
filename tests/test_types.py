"""Tests for agents.types -- shared data types."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agents.types import AgentConfig, AgentResult, AgentRun, _generate_run_id


class TestGenerateRunId:
    def test_length(self):
        rid = _generate_run_id()
        assert len(rid) == 12

    def test_hex_chars(self):
        rid = _generate_run_id()
        int(rid, 16)  # raises if not valid hex

    def test_unique(self):
        ids = {_generate_run_id() for _ in range(100)}
        assert len(ids) == 100


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig(workspace=Path("/tmp/work"))
        assert cfg.workspace == Path("/tmp/work")
        assert cfg.mcp_server_url == ""
        assert cfg.timeout_seconds == 900
        assert cfg.transcript_dir is None
        assert cfg.env == {}

    def test_custom_values(self):
        cfg = AgentConfig(
            workspace=Path("/w"),
            mcp_server_url="http://localhost:8400/mcp",
            timeout_seconds=300,
            transcript_dir=Path("/logs"),
            env={"FOO": "bar"},
        )
        assert cfg.mcp_server_url == "http://localhost:8400/mcp"
        assert cfg.timeout_seconds == 300
        assert cfg.transcript_dir == Path("/logs")
        assert cfg.env == {"FOO": "bar"}

    def test_event_bus_default_none(self):
        cfg = AgentConfig(workspace=Path("/tmp/work"))
        assert cfg.event_bus is None

    def test_event_bus_accepts_instance(self):
        from agents.event_bus import RunEventBus

        bus = RunEventBus(run_id="x")
        cfg = AgentConfig(workspace=Path("/tmp/work"), event_bus=bus)
        assert cfg.event_bus is bus


class TestAgentRun:
    def _make_run(self, **kwargs):
        defaults = {
            "run_id": "abc123def456",
            "backend": "cursor",
            "prompt": "do stuff",
            "started_at": datetime(2026, 4, 14, 10, 0, 0, tzinfo=timezone.utc),
        }
        defaults.update(kwargs)
        return AgentRun(**defaults)

    def test_defaults(self):
        run = self._make_run()
        assert run.status == "running"
        assert run.pid is None
        assert run.exit_code is None
        assert run.finished_at is None
        assert run.tool_call_count == 0
        assert run.llm_total_cost_usd == 0.0

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="Invalid status"):
            self._make_run(status="bogus")

    @pytest.mark.parametrize(
        "status",
        ["running", "completed", "failed", "cancelled", "timed_out"],
    )
    def test_valid_statuses(self, status):
        run = self._make_run(status=status)
        assert run.status == status

    def test_create_factory(self):
        run = AgentRun.create(backend="cursor", prompt="hello")
        assert len(run.run_id) == 12
        assert run.backend == "cursor"
        assert run.prompt == "hello"
        assert run.status == "running"
        assert run.started_at is not None

    def test_mark_completed(self):
        run = self._make_run()
        run.mark_completed(exit_code=0)
        assert run.status == "completed"
        assert run.exit_code == 0
        assert run.finished_at is not None
        assert run.is_terminal is True

    def test_mark_failed(self):
        run = self._make_run()
        run.mark_failed(exit_code=1)
        assert run.status == "failed"
        assert run.exit_code == 1
        assert run.is_terminal is True

    def test_mark_cancelled(self):
        run = self._make_run()
        run.mark_cancelled()
        assert run.status == "cancelled"
        assert run.is_terminal is True

    def test_mark_timed_out(self):
        run = self._make_run()
        run.mark_timed_out()
        assert run.status == "timed_out"
        assert run.is_terminal is True

    def test_is_terminal_false_when_running(self):
        run = self._make_run()
        assert run.is_terminal is False

    def test_duration_none_when_running(self):
        run = self._make_run()
        assert run.duration_seconds is None

    def test_duration_computed(self):
        run = self._make_run(
            started_at=datetime(2026, 4, 14, 10, 0, 0, tzinfo=timezone.utc),
        )
        run.finished_at = datetime(2026, 4, 14, 10, 5, 0, tzinfo=timezone.utc)
        assert run.duration_seconds == 300.0

    def test_telemetry_fields(self):
        run = self._make_run()
        run.tool_call_count = 15
        run.llm_request_count = 8
        run.llm_input_tokens = 10000
        run.llm_output_tokens = 5000
        run.llm_cache_read_tokens = 2000
        run.llm_total_cost_usd = 0.12
        assert run.tool_call_count == 15
        assert run.llm_total_cost_usd == 0.12

    def test_session_id_default_none(self):
        run = AgentRun.create(backend="test", prompt="hello")
        assert run.session_id is None

    def test_session_id_settable(self):
        run = AgentRun.create(backend="test", prompt="hello")
        run.session_id = "abc123"
        assert run.session_id == "abc123"

    def test_metadata_default_empty_dict(self):
        run = AgentRun.create(backend="test", prompt="hello")
        assert run.metadata == {}

    def test_metadata_settable(self):
        run = AgentRun.create(backend="test", prompt="hello")
        run.metadata = {"jenkins_url": "https://jenkins.example.com/job/foo/123"}
        assert run.metadata["jenkins_url"] == "https://jenkins.example.com/job/foo/123"

    def test_metadata_not_shared_between_instances(self):
        run1 = AgentRun.create(backend="test", prompt="a")
        run2 = AgentRun.create(backend="test", prompt="b")
        run1.metadata["key"] = "val"
        assert "key" not in run2.metadata

    def test_agent_name_default_none(self):
        run = AgentRun.create(backend="test", prompt="hello")
        assert run.agent_name is None

    def test_agent_name_assignable(self):
        run = AgentRun.create(backend="test", prompt="hello")
        run.agent_name = "rca-gather"
        assert run.agent_name == "rca-gather"


class TestAgentResult:
    def test_success(self):
        run = AgentRun.create(backend="cursor", prompt="test")
        run.mark_completed()
        result = AgentResult(success=True, output="done", run=run)
        assert result.success is True
        assert result.output == "done"
        assert result.run.status == "completed"

    def test_failure(self):
        run = AgentRun.create(backend="cursor", prompt="test")
        run.mark_failed()
        result = AgentResult(success=False, output="error text", run=run)
        assert result.success is False
        assert result.run.status == "failed"
