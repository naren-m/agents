from agents.console.envelope import (
    EXIT_AGENT_FAILED, EXIT_OK, EXIT_UNAVAILABLE, EXIT_USAGE,
    build_envelope, usage_error,
)
from agents.types import AgentResult, AgentRun


def _result(success=True, output="answer", **fields):
    run = AgentRun.create(backend="ollama", prompt="p")
    run.model = "gemma4:12b"
    run.llm_input_tokens = 4249
    run.llm_output_tokens = 287
    run.duration_ms = 34120
    for k, v in fields.items():
        setattr(run, k, v)
    run.mark_completed() if success else run.mark_failed()
    return AgentResult(success=success, output=output, run=run)


def test_exit_codes_are_distinct():
    assert {EXIT_OK, EXIT_AGENT_FAILED, EXIT_USAGE, EXIT_UNAVAILABLE} == {0, 1, 2, 3}


def test_envelope_shape():
    env = build_envelope(_result())
    assert env["success"] is True
    assert env["backend"] == "ollama"
    assert env["model"] == "gemma4:12b"
    assert env["output"] == "answer"
    assert env["error"] is None
    assert env["run_id"]
    assert env["telemetry"] == {
        "input_tokens": 4249,
        "output_tokens": 287,
        "cache_read_tokens": 0,
        "cost_usd": 0.0,
        "duration_ms": 34120,
        "tool_calls": 0,
        "requests": 0,
    }


def test_envelope_prefers_metadata_response():
    # agy's raw stdout is NDJSON; the clean answer travels in metadata.
    result = _result(output='{"event":"result"}')
    result.run.metadata["response"] = "the real answer"
    assert build_envelope(result)["output"] == "the real answer"


def test_failed_envelope_carries_error():
    result = _result(success=False, output="")
    result.run.metadata["error"] = "timeout waiting for response"
    env = build_envelope(result)
    assert env["success"] is False
    assert env["error"] == "timeout waiting for response"


def test_usage_error_writes_stderr_and_returns_2(capsys):
    code = usage_error("backend 'ollama' cannot explore a workspace.",
                       "agents run --backend ollama -f a.py --task '...'")
    captured = capsys.readouterr()
    assert code == EXIT_USAGE
    assert captured.out == ""          # never pollute stdout
    assert "Error:" in captured.err
    assert "agents run --backend ollama" in captured.err
