"""Tests for CodexAgent -- command building, availability, telemetry parsing.

Mirrors the structure of ``test_cursor_agent.py``. The telemetry samples are
real ``codex exec --json`` output captured from
``codex exec --json --skip-git-repo-check --cd <dir>
  --dangerously-bypass-approvals-and-sandbox <prompt>``.
"""

import json
from unittest.mock import patch

import pytest

from agents.cli.codex import CodexAgent, parse_codex_telemetry
from agents.types import AgentConfig, AgentRun


@pytest.fixture
def config(tmp_path):
    return AgentConfig(
        workspace=tmp_path / "workspace",
        transcript_dir=tmp_path / "transcripts",
    )


class TestCodexAgentProperties:
    def test_name(self):
        agent = CodexAgent()
        assert agent.name == "codex"

    def test_binary_names(self):
        agent = CodexAgent()
        assert agent._binary_names == ["codex"]


class TestCodexAgentBuildCommand:
    def test_command_starts_with_exec_json(self, config):
        agent = CodexAgent()
        with patch.object(agent, "binary_path", return_value="codex"):
            cmd = agent.build_command("analyze this", config)
        assert cmd[0] == "codex"
        assert cmd[1] == "exec"
        assert "--json" in cmd

    def test_includes_workspace_via_cd(self, config):
        agent = CodexAgent()
        with patch.object(agent, "binary_path", return_value="codex"):
            cmd = agent.build_command("test", config)
        assert "--cd" in cmd
        cd_idx = cmd.index("--cd")
        assert cmd[cd_idx + 1] == str(config.workspace)

    def test_prompt_is_last_arg(self, config):
        agent = CodexAgent()
        with patch.object(agent, "binary_path", return_value="codex"):
            cmd = agent.build_command("my prompt", config)
        assert cmd[-1] == "my prompt"

    def test_includes_skip_git_repo_check(self, config):
        """The workspace is rarely a git repo; without this codex refuses to start."""
        agent = CodexAgent()
        with patch.object(agent, "binary_path", return_value="codex"):
            cmd = agent.build_command("test", config)
        assert "--skip-git-repo-check" in cmd

    def test_includes_unattended_flag(self, config):
        """Headless runs must bypass per-call approvals and the sandbox prompt."""
        agent = CodexAgent()
        with patch.object(agent, "binary_path", return_value="codex"):
            cmd = agent.build_command("test", config)
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd

    def test_inherits_only_allowed_shell_environment_for_fallback_commands(
        self,
        config,
    ):
        config.env.update(
            {
                "JENKINS_TOKEN": "secret",
                "RCA_AGENT_BACKEND": "codex",
                "COREX_ROBOT_URL": "https://corex.example",
                "WEBEX_BOT_TOKEN": "do-not-include",
                "REMOTE_PASS": "do-not-include",
            }
        )
        agent = CodexAgent()
        with patch.object(agent, "binary_path", return_value="codex"):
            cmd = agent.build_command("test", config)

        config_values = [
            value for idx, value in enumerate(cmd) if idx > 0 and cmd[idx - 1] == "-c"
        ]
        assert 'shell_environment_policy.inherit="all"' not in config_values
        assert "shell_environment_policy.inherit=none" in config_values
        include_only = next(
            value
            for value in config_values
            if value.startswith("shell_environment_policy.include_only=")
        )
        assert "JENKINS_TOKEN" in include_only
        assert "RCA_AGENT_BACKEND" in include_only
        assert "COREX_ROBOT_URL" in include_only
        assert "CODEX_HOME" in include_only
        assert "WEBEX_BOT_TOKEN" not in include_only
        assert "REMOTE_PASS" not in include_only


class TestCodexAgentAvailability:
    def test_available_when_codex_exists(self):
        agent = CodexAgent()
        with patch(
            "shutil.which",
            side_effect=lambda n: "/usr/bin/codex" if n == "codex" else None,
        ):
            assert agent.available() is True
            assert agent.binary_path() == "/usr/bin/codex"

    def test_not_available(self):
        agent = CodexAgent()
        with patch("shutil.which", return_value=None):
            assert agent.available() is False
            assert agent.binary_path() is None


# -- Telemetry JSON parsing --------------------------------------------------

# Realistic single-turn sample. Trimmed copy of an actual
# ``codex exec --json`` run; only the events the parser cares about
# are retained.
SAMPLE_CODEX_JSON_LINES = [
    {"type": "thread.started", "thread_id": "019e422e-e3f5-73c1-a9f8-bb80d5b77f62"},
    {"type": "turn.started"},
    {"type": "item.started", "item": {
        "id": "item_0", "type": "command_execution",
        "command": "sed -n '1,10p' foo.md", "status": "in_progress",
    }},
    {"type": "item.completed", "item": {
        "id": "item_0", "type": "command_execution",
        "command": "sed -n '1,10p' foo.md",
        "aggregated_output": "...", "exit_code": 0, "status": "completed",
    }},
    {"type": "item.started", "item": {
        "id": "item_1", "type": "command_execution",
        "command": "rg pattern .", "status": "in_progress",
    }},
    {"type": "item.completed", "item": {
        "id": "item_1", "type": "command_execution",
        "command": "rg pattern .",
        "aggregated_output": "match", "exit_code": 0, "status": "completed",
    }},
    {"type": "item.completed", "item": {
        "id": "item_2", "type": "agent_message", "text": "Done.",
    }},
    {"type": "turn.completed", "usage": {
        "input_tokens": 49976,
        "cached_input_tokens": 30976,
        "output_tokens": 327,
        "reasoning_output_tokens": 157,
    }},
]

SAMPLE_OUTPUT = "\n".join(json.dumps(line) for line in SAMPLE_CODEX_JSON_LINES)


class TestParseCodexTelemetry:

    def test_extracts_usage_from_turn_completed(self):
        run = AgentRun.create(backend="codex", prompt="test")
        parse_codex_telemetry(SAMPLE_OUTPUT, run)

        assert run.llm_input_tokens == 49976
        assert run.llm_cache_read_tokens == 30976
        assert run.llm_output_tokens == 327

    def test_counts_tool_calls_from_item_started(self):
        run = AgentRun.create(backend="codex", prompt="test")
        parse_codex_telemetry(SAMPLE_OUTPUT, run)

        # Two command_execution starts; agent_message item is not a tool call.
        assert run.tool_call_count == 2

    def test_counts_llm_requests_from_turn_completed(self):
        run = AgentRun.create(backend="codex", prompt="test")
        parse_codex_telemetry(SAMPLE_OUTPUT, run)
        assert run.llm_request_count == 1

    def test_captures_thread_id_as_session_id(self):
        run = AgentRun.create(backend="codex", prompt="test")
        parse_codex_telemetry(SAMPLE_OUTPUT, run)
        assert run.session_id == "019e422e-e3f5-73c1-a9f8-bb80d5b77f62"

    def test_sums_usage_across_multiple_turns(self):
        """Each turn.completed is per-turn, not a running total. Sum them."""
        lines = [
            {"type": "turn.completed", "usage": {
                "input_tokens": 100, "cached_input_tokens": 40,
                "output_tokens": 20,
            }},
            {"type": "turn.completed", "usage": {
                "input_tokens": 200, "cached_input_tokens": 90,
                "output_tokens": 50,
            }},
        ]
        output = "\n".join(json.dumps(line) for line in lines)
        run = AgentRun.create(backend="codex", prompt="test")
        parse_codex_telemetry(output, run)

        assert run.llm_input_tokens == 300
        assert run.llm_cache_read_tokens == 130
        assert run.llm_output_tokens == 70
        assert run.llm_request_count == 2

    def test_handles_mixed_json_and_text(self):
        """Real output may contain stderr lines mixed with JSONL."""
        output = (
            "Reading additional input from stdin...\n"
            "2026-05-19T21:40:29Z ERROR codex_core::session: warning text\n"
            + json.dumps(SAMPLE_CODEX_JSON_LINES[0]) + "\n"
            + json.dumps(SAMPLE_CODEX_JSON_LINES[-1]) + "\n"
        )
        run = AgentRun.create(backend="codex", prompt="test")
        parse_codex_telemetry(output, run)

        assert run.llm_input_tokens == 49976
        assert run.session_id == "019e422e-e3f5-73c1-a9f8-bb80d5b77f62"

    def test_empty_output(self):
        run = AgentRun.create(backend="codex", prompt="test")
        parse_codex_telemetry("", run)
        assert run.tool_call_count == 0
        assert run.llm_input_tokens == 0
        assert run.session_id is None

    def test_no_json_at_all(self):
        run = AgentRun.create(backend="codex", prompt="test")
        parse_codex_telemetry("just plain text\nnothing here\n", run)
        assert run.tool_call_count == 0
        assert run.llm_input_tokens == 0

    def test_partial_usage_block(self):
        """Usage block may omit some fields; absent fields stay zero."""
        output = json.dumps({
            "type": "turn.completed",
            "usage": {"input_tokens": 42},
        })
        run = AgentRun.create(backend="codex", prompt="test")
        parse_codex_telemetry(output, run)

        assert run.llm_input_tokens == 42
        assert run.llm_output_tokens == 0
        assert run.llm_cache_read_tokens == 0


class TestCodexAgentParseOutputHook:
    """CodexAgent.parse_output must delegate to parse_codex_telemetry."""

    def test_parse_output_called_on_run(self):
        agent = CodexAgent()
        run = AgentRun.create(backend="codex", prompt="test")
        agent.parse_output(SAMPLE_OUTPUT, run)

        assert run.llm_input_tokens == 49976
        assert run.tool_call_count == 2
        assert run.session_id == "019e422e-e3f5-73c1-a9f8-bb80d5b77f62"
