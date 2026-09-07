import json

from agents.console.stats import stats_command, summarise
from agents.console.store import RunStore
from agents.types import AgentRun


def _entry(backend, status, tin, tout, cost, ms):
    run = AgentRun.create(backend=backend, prompt="p")
    run.llm_input_tokens = tin
    run.llm_output_tokens = tout
    run.llm_total_cost_usd = cost
    run.duration_ms = ms
    run.mark_completed() if status == "completed" else run.mark_failed()
    return run


def test_summarise_groups_by_backend():
    entries = [
        {"backend": "ollama", "status": "completed", "llm_input_tokens": 100,
         "llm_output_tokens": 10, "llm_total_cost_usd": 0.0, "duration_ms": 1000},
        {"backend": "ollama", "status": "failed", "llm_input_tokens": 50,
         "llm_output_tokens": 0, "llm_total_cost_usd": 0.0, "duration_ms": 500},
        {"backend": "agy", "status": "completed", "llm_input_tokens": 1000,
         "llm_output_tokens": 100, "llm_total_cost_usd": 0.0, "duration_ms": 5000},
    ]

    summary = summarise(entries)

    assert summary["ollama"]["runs"] == 2
    assert summary["ollama"]["input_tokens"] == 150
    assert summary["ollama"]["success_rate"] == 0.5
    assert summary["ollama"]["median_duration_ms"] == 750
    assert summary["agy"]["runs"] == 1
    assert summary["agy"]["cost_usd"] == 0.0


def test_summarise_empty():
    assert summarise([]) == {}


def test_stats_prints_json(tmp_path, capsys):
    store = RunStore(root=tmp_path)
    store.append_ledger(_entry("ollama", "completed", 100, 10, 0.0, 1000))

    code = stats_command(store, since=None, backend=None, fmt="json")
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["ollama"]["runs"] == 1


def test_stats_filters_by_backend(tmp_path, capsys):
    store = RunStore(root=tmp_path)
    store.append_ledger(_entry("ollama", "completed", 100, 10, 0.0, 1000))
    store.append_ledger(_entry("agy", "completed", 900, 90, 0.0, 9000))

    stats_command(store, since=None, backend="agy", fmt="json")
    payload = json.loads(capsys.readouterr().out)

    assert set(payload) == {"agy"}


def test_stats_table_format(tmp_path, capsys):
    store = RunStore(root=tmp_path)
    store.append_ledger(_entry("ollama", "completed", 100, 10, 0.0, 1000))

    code = stats_command(store, since=None, backend=None, fmt="table")
    out = capsys.readouterr().out

    assert code == 0
    assert "ollama" in out
    assert "runs" in out.lower()


def test_stats_bad_since_is_usage_error(tmp_path, capsys):
    code = stats_command(RunStore(root=tmp_path), since="banana", backend=None,
                         fmt="json")
    assert code == 2
    assert "Error:" in capsys.readouterr().err
