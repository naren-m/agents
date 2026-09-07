from agents.console.store import RunStore
from agents.types import AgentRun


def _run():
    r = AgentRun.create(backend="agy", prompt="p")
    r.llm_input_tokens = 10
    return r


def test_save_and_load_roundtrip(tmp_path):
    store = RunStore(root=tmp_path)
    run = _run()
    store.save(run)
    loaded = store.load(run.run_id)
    assert loaded["run_id"] == run.run_id
    assert loaded["backend"] == "agy"
    assert loaded["status"] == "running"


def test_load_missing_returns_none(tmp_path):
    assert RunStore(root=tmp_path).load("nope") is None


def test_ledger_appends(tmp_path):
    store = RunStore(root=tmp_path)
    a, b = _run(), _run()
    a.mark_completed()
    b.mark_failed()
    store.append_ledger(a)
    store.append_ledger(b)
    entries = store.read_ledger()
    assert len(entries) == 2
    assert {e["status"] for e in entries} == {"completed", "failed"}


def test_read_ledger_empty(tmp_path):
    assert RunStore(root=tmp_path).read_ledger() == []


def test_idempotency_key_roundtrip(tmp_path):
    store = RunStore(root=tmp_path)
    assert store.find_by_key("k1") is None
    store.save_key("k1", "r_abc")
    assert store.find_by_key("k1") == "r_abc"
