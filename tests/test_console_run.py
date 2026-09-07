import json

import pytest

from agents.console.main import main
from agents.console.registry import build_registry, validate_flags
from agents.types import AgentResult, AgentRun


def test_registry_has_both_backends():
    reg = build_registry()
    assert set(reg) >= {"agy", "ollama"}


def test_validate_rejects_workspace_for_ollama():
    reg = build_registry()
    problem = validate_flags(reg["ollama"], workspace="/tmp", files=[])
    assert problem is not None
    message, example = problem
    assert "cannot explore a workspace" in message
    assert "-f" in example


def test_validate_rejects_files_for_agy():
    reg = build_registry()
    problem = validate_flags(reg["agy"], workspace=None, files=["a.py"])
    assert problem is not None
    assert "--workspace" in problem[1]


def test_validate_accepts_correct_combinations():
    reg = build_registry()
    assert validate_flags(reg["ollama"], workspace=None, files=["a.py"]) is None
    assert validate_flags(reg["agy"], workspace="/tmp", files=[]) is None


def test_backends_list_prints_capabilities(capsys):
    code = main(["backends", "list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "agy" in out and "ollama" in out
    assert "context_files" in out


def test_run_unknown_backend_is_usage_error(capsys):
    code = main(["run", "--backend", "nope", "--task", "t"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "Error:" in captured.err


def test_run_capability_mismatch_is_usage_error(capsys):
    code = main(["run", "--backend", "ollama", "--workspace", "/tmp", "--task", "t"])
    captured = capsys.readouterr()
    assert code == 2
    assert "-f" in captured.err


def test_run_prints_envelope_on_success(monkeypatch, capsys, tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")

    async def fake_run(self, prompt, config, mcp_client=None, on_progress=None):
        run = AgentRun.create(backend="ollama", prompt=prompt)
        run.model = "gemma4:12b"
        run.llm_input_tokens = 12
        run.llm_output_tokens = 3
        run.mark_completed()
        return AgentResult(success=True, output="answer", run=run)

    from agents.inprocess.ollama import OllamaAgent
    monkeypatch.setattr(OllamaAgent, "run", fake_run)

    code = main(["run", "--backend", "ollama", "-f", str(f), "--task", "what?"])
    env = json.loads(capsys.readouterr().out)

    assert code == 0
    assert env["success"] is True
    assert env["output"] == "answer"
    assert env["telemetry"]["input_tokens"] == 12


def test_run_failed_agent_exits_1(monkeypatch, capsys, tmp_path):
    async def fake_run(self, prompt, config, mcp_client=None, on_progress=None):
        run = AgentRun.create(backend="ollama", prompt=prompt)
        run.mark_failed()
        run.metadata["error"] = "connection refused"
        return AgentResult(success=False, output="", run=run)

    from agents.inprocess.ollama import OllamaAgent
    monkeypatch.setattr(OllamaAgent, "run", fake_run)

    code = main(["run", "--backend", "ollama", "--task", "t"])
    env = json.loads(capsys.readouterr().out)

    assert code == 1
    assert env["success"] is False
    assert env["error"] == "connection refused"
