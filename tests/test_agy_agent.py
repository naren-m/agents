import json
from pathlib import Path

from agents.cli.agy import AgyAgent, parse_agy_telemetry
from agents.types import AgentConfig, AgentRun, Capability


def _run() -> AgentRun:
    return AgentRun.create(backend="agy", prompt="p")


def _config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(workspace=tmp_path)


def test_capabilities_include_mcp():
    assert AgyAgent.capabilities == frozenset(
        {Capability.WORKSPACE, Capability.TOOLS, Capability.MCP}
    )


def test_build_command_includes_mandatory_permission_flag(tmp_path):
    # Verified: without this flag, headless agy auto-denies every tool that
    # needs approval and returns an empty response.
    agent = AgyAgent()
    agent.binary_path = lambda: "/usr/bin/agy"
    cmd = agent.build_command("do a thing", _config(tmp_path))
    assert "--dangerously-skip-permissions" in cmd
    assert "--print" in cmd
    assert "do a thing" in cmd


def test_build_command_uses_stream_json(tmp_path):
    # stream-json is a strict superset of json: it alone carries tool counts
    # and the resolved model.
    agent = AgyAgent()
    agent.binary_path = lambda: "/usr/bin/agy"
    cmd = agent.build_command("t", _config(tmp_path))
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"


def test_parse_stream_json_maps_all_telemetry():
    stream = "\n".join([
        json.dumps({"event": "init", "conversation_id": "c1",
                    "init": {"model": "gemini-3.8-flash-high", "cwd": "/w"}}),
        json.dumps({"event": "step_update",
                    "step_update": {"step_type": "tool", "tool_name": "read"}}),
        json.dumps({"event": "step_update",
                    "step_update": {"step_type": "tool", "tool_name": "grep"}}),
        json.dumps({"event": "result", "result": {
            "conversation_id": "c1", "status": "SUCCESS", "response": "the answer",
            "duration_seconds": 51.745, "num_turns": 3,
            "usage": {"input_tokens": 101932, "output_tokens": 3829,
                      "thinking_tokens": 2324, "cache_read_tokens": 320050,
                      "total_tokens": 105761}}}),
    ])
    run = _run()
    parse_agy_telemetry(stream, run)

    assert run.session_id == "c1"
    assert run.model == "gemini-3.8-flash-high"
    assert run.llm_input_tokens == 101932
    assert run.llm_output_tokens == 3829
    assert run.llm_cache_read_tokens == 320050
    assert run.llm_request_count == 3
    assert run.duration_ms == 51745
    assert run.tool_call_count == 2
    assert run.metadata["thinking_tokens"] == 2324
    assert run.metadata["response"] == "the answer"
    assert run.llm_total_cost_usd == 0.0
    assert not run.is_terminal  # success path leaves the verdict to wait()


def test_denied_actions_marks_run_failed_despite_success_status():
    # THE TRAP, verified live: agy exits 0 with status SUCCESS and an empty
    # response when a tool was auto-denied. exit_code == 0 is not success.
    payload = json.dumps({"event": "result", "result": {
        "conversation_id": "c2", "status": "SUCCESS", "response": "",
        "duration_seconds": 2.95, "num_turns": 1,
        "usage": {"input_tokens": 16451, "output_tokens": 494,
                  "thinking_tokens": 424, "cache_read_tokens": 0},
        "denied_actions": [{"action": "command", "display_name": "RunCommand"}]}})
    run = _run()
    parse_agy_telemetry(payload, run)

    assert run.status == "failed"
    assert run.is_terminal
    assert run.metadata["denied_actions"] == [
        {"action": "command", "display_name": "RunCommand"}
    ]


def test_error_status_marks_run_failed():
    payload = json.dumps({"event": "result", "result": {
        "conversation_id": "c3", "status": "ERROR", "response": "",
        "error": "timeout waiting for response", "duration_seconds": 118.9,
        "num_turns": 1, "usage": {"input_tokens": 1, "output_tokens": 0}}})
    run = _run()
    parse_agy_telemetry(payload, run)

    assert run.status == "failed"
    assert run.metadata["error"] == "timeout waiting for response"


def test_sandboxed_write_claim_is_annotated():
    # Verified: sandboxed agy reports "has been overwritten" while the real
    # file is untouched and the write went to its scratch dir.
    payload = json.dumps({"event": "result", "result": {
        "conversation_id": "c4", "status": "SUCCESS",
        "response": "[a.py](file:///Users/x/.gemini/antigravity-cli/scratch/a.py) "
                    "has been overwritten with `X`.",
        "duration_seconds": 1.0, "num_turns": 1,
        "usage": {"input_tokens": 1, "output_tokens": 1}}})
    run = _run()
    parse_agy_telemetry(payload, run)

    assert run.metadata["sandbox_write_claim"] is True


def test_single_object_json_format_still_parses():
    # --output-format json emits one object, not NDJSON. Support it so a
    # caller who overrides the format still gets telemetry.
    payload = json.dumps({
        "conversation_id": "c5", "status": "SUCCESS", "response": "hi",
        "duration_seconds": 2.0, "num_turns": 1,
        "usage": {"input_tokens": 10, "output_tokens": 2}})
    run = _run()
    parse_agy_telemetry(payload, run)

    assert run.metadata["response"] == "hi"
    assert run.llm_input_tokens == 10


def test_empty_output_is_safe():
    run = _run()
    parse_agy_telemetry("", run)
    assert run.llm_input_tokens == 0
