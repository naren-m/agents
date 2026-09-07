import json
import types

import pytest

from agents.console.lifecycle import cancel_run, show_logs, start_run, status_run
from agents.console.store import RunStore


def _args(**kw):
    base = dict(backend="ollama", task="t", workspace=None, file=[],
                model=None, timeout=300, idempotency_key=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_start_returns_run_id(monkeypatch, tmp_path, capsys):
    store = RunStore(root=tmp_path)
    monkeypatch.setattr(
        "agents.console.lifecycle._spawn_detached", lambda cmd, log: 4242
    )
    code = start_run(_args(), store)
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["run_id"]
    assert store.load(payload["run_id"])["pid"] == 4242


def test_start_is_idempotent_with_key(monkeypatch, tmp_path, capsys):
    store = RunStore(root=tmp_path)
    monkeypatch.setattr(
        "agents.console.lifecycle._spawn_detached", lambda cmd, log: 1
    )

    def _start_and_read():
        start_run(_args(idempotency_key="k"), store)
        return json.loads(capsys.readouterr().out)["run_id"]

    assert _start_and_read() == _start_and_read()


def test_status_unknown_run_is_usage_error(tmp_path, capsys):
    code = status_run("nope", RunStore(root=tmp_path))
    assert code == 2
    assert "Error:" in capsys.readouterr().err


def test_status_prints_stored_state(monkeypatch, tmp_path, capsys):
    store = RunStore(root=tmp_path)
    monkeypatch.setattr(
        "agents.console.lifecycle._spawn_detached", lambda cmd, log: 7
    )
    start_run(_args(), store)
    run_id = json.loads(capsys.readouterr().out)["run_id"]

    code = status_run(run_id, store)
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["run_id"] == run_id
    assert payload["status"] == "running"


def test_cancel_finished_run_is_noop(monkeypatch, tmp_path, capsys):
    store = RunStore(root=tmp_path)
    monkeypatch.setattr(
        "agents.console.lifecycle._spawn_detached", lambda cmd, log: 9
    )
    start_run(_args(), store)
    run_id = json.loads(capsys.readouterr().out)["run_id"]

    data = store.load(run_id)
    data["status"] = "completed"
    (store.runs_dir / f"{run_id}.json").write_text(json.dumps(data))

    code = cancel_run(run_id, store)
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "completed"
    assert payload["cancelled"] is False


def test_logs_returns_captured_output(monkeypatch, tmp_path, capsys):
    store = RunStore(root=tmp_path)
    monkeypatch.setattr(
        "agents.console.lifecycle._spawn_detached", lambda cmd, log: 11
    )
    start_run(_args(), store)
    run_id = json.loads(capsys.readouterr().out)["run_id"]

    log_path = store.root / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("line one\nline two\n")

    code = show_logs(run_id, store, follow=False)
    assert code == 0
    assert "line two" in capsys.readouterr().out


def test_logs_missing_run_is_usage_error(tmp_path, capsys):
    code = show_logs("nope", RunStore(root=tmp_path), follow=False)
    assert code == 2
    assert "Error:" in capsys.readouterr().err


def test_rebuild_command_passes_run_id():
    # Found by an end-to-end run: without this the detached child has no way
    # to report completion, so the store record stays 'running' forever and
    # `logs --follow` never terminates.
    from agents.console.lifecycle import _rebuild_run_command
    cmd = _rebuild_run_command(_args(), "abc123")
    assert "--run-id" in cmd
    assert cmd[cmd.index("--run-id") + 1] == "abc123"


def test_run_persists_terminal_state_when_given_run_id(monkeypatch, tmp_path, capsys):
    from agents.console.main import main
    from agents.inprocess.ollama import OllamaAgent
    from agents.types import AgentResult, AgentRun

    async def fake_run(self, prompt, config, mcp_client=None, on_progress=None):
        run = AgentRun.create(backend="ollama", prompt=prompt)
        run.mark_completed()
        return AgentResult(success=True, output="done", run=run)

    monkeypatch.setattr(OllamaAgent, "run", fake_run)
    monkeypatch.setenv("AGENTS_HOME", str(tmp_path))

    code = main(["run", "--backend", "ollama", "--task", "t", "--run-id", "fixed1"])
    capsys.readouterr()

    assert code == 0
    stored = RunStore(root=tmp_path).load("fixed1")
    assert stored is not None
    assert stored["status"] == "completed"


def test_follow_stops_when_process_is_gone(monkeypatch, tmp_path, capsys):
    # Safety net: if the child is killed it can never mark itself terminal,
    # so following must not hang forever on a stale 'running' record.
    store = RunStore(root=tmp_path)
    monkeypatch.setattr(
        "agents.console.lifecycle._spawn_detached", lambda cmd, log: 999999
    )
    start_run(_args(), store)
    run_id = json.loads(capsys.readouterr().out)["run_id"]

    log_path = store.root / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("partial\n")

    monkeypatch.setattr("agents.console.lifecycle._pid_alive", lambda pid: False)
    code = show_logs(run_id, store, follow=True)

    assert code == 0
    assert "partial" in capsys.readouterr().out
