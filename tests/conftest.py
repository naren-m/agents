import pytest


@pytest.fixture(autouse=True)
def isolate_agents_home(tmp_path, monkeypatch):
    """Keep every test's run state out of the developer's real ~/.agents.

    ``agents run`` appends each finished run to the ledger via ``RunStore()``,
    which defaults to ``~/.agents``. Without this fixture the console tests
    write into the home directory of whoever runs the suite.
    """
    monkeypatch.setenv("AGENTS_HOME", str(tmp_path / "agents-home"))
