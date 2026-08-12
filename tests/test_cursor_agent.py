"""Tests for CursorAgent -- command building, availability, telemetry parsing."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.cli.cursor import CursorAgent, parse_cursor_telemetry
from agents.types import AgentConfig, AgentRun


@pytest.fixture
def config(tmp_path):
    return AgentConfig(
        workspace=tmp_path / "workspace",
        transcript_dir=tmp_path / "transcripts",
    )


class TestCursorAgentProperties:
    def test_name(self):
        agent = CursorAgent()
        assert agent.name == "cursor"

    def test_binary_names(self):
        agent = CursorAgent()
        assert "cursor-agent" in agent._binary_names
        assert "cursor" in agent._binary_names
        assert agent._binary_names.index("cursor-agent") < agent._binary_names.index("cursor")


class TestCursorAgentBuildCommand:
    @pytest.mark.parametrize(
        "binary_name,expected_prefix",
        [
            (
                "cursor-agent",
                ["cursor-agent", "-p", "--yolo", "--trust", "--approve-mcps"],
            ),
            (
                "cursor",
                ["cursor", "agent", "-p", "--yolo", "--trust", "--approve-mcps"],
            ),
        ],
    )
    def test_command_shape(self, binary_name, expected_prefix, config):
        agent = CursorAgent()
        with patch.object(agent, "binary_path", return_value=binary_name):
            cmd = agent.build_command("analyze this", config)

        assert cmd[:len(expected_prefix)] == expected_prefix
        assert "--workspace" in cmd
        assert str(config.workspace) in cmd
        assert "analyze this" in cmd

    def test_includes_output_format_json(self, config):
        agent = CursorAgent()
        with patch.object(agent, "binary_path", return_value="cursor-agent"):
            cmd = agent.build_command("test", config)
        assert "--output-format" in cmd
        fmt_idx = cmd.index("--output-format")
        assert cmd[fmt_idx + 1] == "json"

    def test_prompt_is_last_arg(self, config):
        agent = CursorAgent()
        with patch.object(agent, "binary_path", return_value="cursor-agent"):
            cmd = agent.build_command("my prompt", config)
        assert cmd[-1] == "my prompt"

    def test_workspace_path(self, config):
        agent = CursorAgent()
        with patch.object(agent, "binary_path", return_value="cursor-agent"):
            cmd = agent.build_command("test", config)
        ws_idx = cmd.index("--workspace")
        assert cmd[ws_idx + 1] == str(config.workspace)

    @pytest.mark.parametrize(
        "binary_name",
        ["cursor-agent", "cursor"],
    )
    def test_includes_trust_flag(self, binary_name, config):
        """--trust is required for headless mode to skip workspace trust prompts."""
        agent = CursorAgent()
        with patch.object(agent, "binary_path", return_value=binary_name):
            cmd = agent.build_command("test", config)
        assert "--trust" in cmd

    @pytest.mark.parametrize(
        "binary_name",
        ["cursor-agent", "cursor"],
    )
    def test_includes_approve_mcps_flag(self, binary_name, config):
        """--approve-mcps auto-approves MCP servers the RCA agent depends on."""
        agent = CursorAgent()
        with patch.object(agent, "binary_path", return_value=binary_name):
            cmd = agent.build_command("test", config)
        assert "--approve-mcps" in cmd

    @pytest.mark.parametrize(
        "binary_name",
        ["cursor-agent", "cursor"],
    )
    def test_includes_yolo_flag(self, binary_name, config):
        """--yolo (alias for --force) auto-approves shell/command execution."""
        agent = CursorAgent()
        with patch.object(agent, "binary_path", return_value=binary_name):
            cmd = agent.build_command("test", config)
        assert "--yolo" in cmd


class TestCursorAgentAvailability:
    def test_available_when_cursor_agent_exists(self):
        agent = CursorAgent()
        with patch("shutil.which", side_effect=lambda n: "/usr/bin/cursor-agent" if n == "cursor-agent" else None):
            assert agent.available() is True
            assert agent.binary_path() == "/usr/bin/cursor-agent"

    def test_available_when_cursor_exists(self):
        agent = CursorAgent()
        with patch("shutil.which", side_effect=lambda n: "/usr/bin/cursor" if n == "cursor" else None):
            assert agent.available() is True
            assert agent.binary_path() == "/usr/bin/cursor"

    def test_not_available(self):
        agent = CursorAgent()
        with patch("shutil.which", return_value=None):
            assert agent.available() is False
            assert agent.binary_path() is None

    def test_prefers_cursor_agent_over_cursor(self):
        agent = CursorAgent()
        def mock_which(name):
            return f"/usr/bin/{name}" if name in ("cursor-agent", "cursor") else None
        with patch("shutil.which", side_effect=mock_which):
            assert agent.binary_path() == "/usr/bin/cursor-agent"


# -- Telemetry JSON parsing --------------------------------------------------

# Realistic sample JSON output from cursor --output-format json.
# Each line is a separate JSON object dumped during execution.
SAMPLE_CURSOR_JSON_LINES = [
    {"type": "assistant", "message": "Starting analysis..."},
    {"type": "tool_use", "tool": "read_file", "args": {"path": "foo.py"}},
    {"type": "tool_result", "tool": "read_file", "result": "contents..."},
    {"type": "tool_use", "tool": "grep", "args": {"pattern": "error"}},
    {"type": "tool_result", "tool": "grep", "result": "match found"},
    {"type": "assistant", "message": "Found the issue."},
    {
        "type": "usage",
        "total_cost_usd": 0.0532,
        "total_input_tokens": 12500,
        "total_output_tokens": 3200,
        "total_cache_read_tokens": 8000,
        "num_requests": 4,
        "num_tool_calls": 2,
    },
]

SAMPLE_OUTPUT = "\n".join(json.dumps(line) for line in SAMPLE_CURSOR_JSON_LINES)


class TestParseCursorTelemetry:
    """Tests for parsing telemetry from cursor JSON output."""

    def test_extracts_usage_block(self):
        run = AgentRun.create(backend="cursor", prompt="test")
        parse_cursor_telemetry(SAMPLE_OUTPUT, run)

        assert run.llm_input_tokens == 12500
        assert run.llm_output_tokens == 3200
        assert run.llm_cache_read_tokens == 8000
        assert run.llm_request_count == 4
        assert run.tool_call_count == 2
        assert run.llm_total_cost_usd == pytest.approx(0.0532)

    def test_counts_tool_use_lines_when_no_usage_block(self):
        """If there's no summary 'usage' block, count tool_use lines."""
        lines = [
            {"type": "tool_use", "tool": "read_file"},
            {"type": "tool_result", "tool": "read_file"},
            {"type": "tool_use", "tool": "grep"},
            {"type": "tool_result", "tool": "grep"},
            {"type": "assistant", "message": "done"},
        ]
        output = "\n".join(json.dumps(l) for l in lines)
        run = AgentRun.create(backend="cursor", prompt="test")
        parse_cursor_telemetry(output, run)

        assert run.tool_call_count == 2
        assert run.llm_request_count == 0
        assert run.llm_total_cost_usd == 0.0

    def test_handles_mixed_json_and_text(self):
        """Real output may have non-JSON lines (ANSI, progress bars, etc)."""
        output = (
            "Loading workspace...\n"
            + json.dumps({"type": "assistant", "message": "hi"}) + "\n"
            + "some random ANSI garbage \x1b[32m\n"
            + json.dumps(SAMPLE_CURSOR_JSON_LINES[-1]) + "\n"
            + "Done.\n"
        )
        run = AgentRun.create(backend="cursor", prompt="test")
        parse_cursor_telemetry(output, run)

        assert run.llm_input_tokens == 12500
        assert run.llm_total_cost_usd == pytest.approx(0.0532)

    def test_empty_output(self):
        run = AgentRun.create(backend="cursor", prompt="test")
        parse_cursor_telemetry("", run)

        assert run.tool_call_count == 0
        assert run.llm_request_count == 0
        assert run.llm_total_cost_usd == 0.0

    def test_no_json_at_all(self):
        run = AgentRun.create(backend="cursor", prompt="test")
        parse_cursor_telemetry("just plain text\nnothing here\n", run)

        assert run.tool_call_count == 0

    def test_multiple_usage_blocks_takes_last(self):
        """If cursor emits multiple usage summaries, take the last one."""
        usage1 = {"type": "usage", "total_cost_usd": 0.01, "total_input_tokens": 100,
                  "total_output_tokens": 50, "num_requests": 1}
        usage2 = {"type": "usage", "total_cost_usd": 0.05, "total_input_tokens": 500,
                  "total_output_tokens": 200, "num_requests": 3}
        output = json.dumps(usage1) + "\n" + json.dumps(usage2) + "\n"
        run = AgentRun.create(backend="cursor", prompt="test")
        parse_cursor_telemetry(output, run)

        assert run.llm_total_cost_usd == pytest.approx(0.05)
        assert run.llm_input_tokens == 500
        assert run.llm_request_count == 3

    def test_partial_usage_block(self):
        """Usage block with only some fields present."""
        usage = {"type": "usage", "total_cost_usd": 0.02}
        output = json.dumps(usage) + "\n"
        run = AgentRun.create(backend="cursor", prompt="test")
        parse_cursor_telemetry(output, run)

        assert run.llm_total_cost_usd == pytest.approx(0.02)
        assert run.llm_input_tokens == 0
        assert run.llm_output_tokens == 0


class TestCursorAgentParseOutputHook:
    """Tests that CursorAgent.parse_output wires into BaseCLIAgent.wait."""

    def test_parse_output_called_on_run(self):
        agent = CursorAgent()
        run = AgentRun.create(backend="cursor", prompt="test")
        agent.parse_output(SAMPLE_OUTPUT, run)

        assert run.llm_input_tokens == 12500
        assert run.tool_call_count == 2


class TestAgentRunNewFields:
    def test_new_telemetry_fields_exist(self):
        run = AgentRun.create(backend="cursor", prompt="test")
        assert run.llm_cache_write_tokens == 0
        assert run.duration_ms == 0
        assert run.model == ""

    def test_new_fields_assignable(self):
        run = AgentRun.create(backend="cursor", prompt="test")
        run.llm_cache_write_tokens = 53007
        run.duration_ms = 7551
        run.model = "Opus 4.6 1M Max Thinking"
        assert run.llm_cache_write_tokens == 53007
        assert run.duration_ms == 7551
        assert run.model == "Opus 4.6 1M Max Thinking"


SAMPLE_CURSOR_RESULT_FORMAT = "\n".join([
    json.dumps({
        "type": "system", "subtype": "init",
        "model": "Opus 4.6 1M Max Thinking",
        "session_id": "c6f20c1c-0882-4968-a651-deb9456a4433",
    }),
    json.dumps({
        "type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        "session_id": "c6f20c1c-0882-4968-a651-deb9456a4433",
    }),
    json.dumps({
        "type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
        "session_id": "c6f20c1c-0882-4968-a651-deb9456a4433",
    }),
    json.dumps({
        "type": "result", "subtype": "success",
        "duration_ms": 7551, "duration_api_ms": 7551,
        "is_error": False, "result": "Hello!",
        "session_id": "c6f20c1c-0882-4968-a651-deb9456a4433",
        "usage": {
            "inputTokens": 43106,
            "outputTokens": 512,
            "cacheReadTokens": 8000,
            "cacheWriteTokens": 53007,
        },
    }),
])


class TestParseCursorResultFormat:
    """Tests for parsing the real Cursor CLI type=result JSON format."""

    def test_extracts_result_usage(self):
        run = AgentRun.create(backend="cursor", prompt="test")
        parse_cursor_telemetry(SAMPLE_CURSOR_RESULT_FORMAT, run)

        assert run.llm_input_tokens == 43106
        assert run.llm_output_tokens == 512
        assert run.llm_cache_read_tokens == 8000
        assert run.llm_cache_write_tokens == 53007
        assert run.duration_ms == 7551

    def test_extracts_model_from_init(self):
        run = AgentRun.create(backend="cursor", prompt="test")
        parse_cursor_telemetry(SAMPLE_CURSOR_RESULT_FORMAT, run)

        assert run.model == "Opus 4.6 1M Max Thinking"

    def test_result_format_with_no_init_line(self):
        output = json.dumps({
            "type": "result", "duration_ms": 5000,
            "usage": {"inputTokens": 100, "outputTokens": 50,
                      "cacheReadTokens": 0, "cacheWriteTokens": 0},
        })
        run = AgentRun.create(backend="cursor", prompt="test")
        parse_cursor_telemetry(output, run)

        assert run.llm_input_tokens == 100
        assert run.llm_output_tokens == 50
        assert run.duration_ms == 5000
        assert run.model == ""

    def test_result_format_partial_usage(self):
        output = json.dumps({
            "type": "result", "duration_ms": 3000,
            "usage": {"inputTokens": 200, "outputTokens": 10},
        })
        run = AgentRun.create(backend="cursor", prompt="test")
        parse_cursor_telemetry(output, run)

        assert run.llm_input_tokens == 200
        assert run.llm_output_tokens == 10
        assert run.llm_cache_read_tokens == 0
        assert run.llm_cache_write_tokens == 0

    def test_legacy_usage_format_still_works(self):
        """The old type=usage format must continue to work."""
        run = AgentRun.create(backend="cursor", prompt="test")
        parse_cursor_telemetry(SAMPLE_OUTPUT, run)

        assert run.llm_input_tokens == 12500
        assert run.llm_output_tokens == 3200
        assert run.llm_total_cost_usd == pytest.approx(0.0532)

    def test_result_takes_precedence_over_usage(self):
        """When both type=usage and type=result exist, result wins."""
        output = "\n".join([
            json.dumps({
                "type": "usage", "total_input_tokens": 100,
                "total_output_tokens": 50, "total_cost_usd": 0.01,
            }),
            json.dumps({
                "type": "result", "duration_ms": 5000,
                "usage": {"inputTokens": 43106, "outputTokens": 512,
                          "cacheReadTokens": 8000, "cacheWriteTokens": 53007},
            }),
        ])
        run = AgentRun.create(backend="cursor", prompt="test")
        parse_cursor_telemetry(output, run)

        assert run.llm_input_tokens == 43106
        assert run.llm_output_tokens == 512
