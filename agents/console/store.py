"""Persistence for background runs, so status/cancel/logs/stats can work."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from agents.types import AgentRun

_DEFAULT_ROOT = Path.home() / ".agents"


def _serialise(run: AgentRun) -> dict:
    data = asdict(run)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
        elif hasattr(value, "isoformat"):
            data[key] = value.isoformat()
    return data


class RunStore:
    """Run state under ``~/.agents``: one file per run, plus a ledger."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else _DEFAULT_ROOT
        self.runs_dir = self.root / "runs"
        self.ledger_path = self.root / "runs.jsonl"
        self.keys_path = self.root / "keys.json"

    def save(self, run: AgentRun) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / f"{run.run_id}.json"
        path.write_text(json.dumps(_serialise(run), indent=2), encoding="utf-8")

    def load(self, run_id: str) -> dict | None:
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def append_ledger(self, run: AgentRun) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_serialise(run)) + "\n")

    def read_ledger(self) -> list[dict]:
        if not self.ledger_path.exists():
            return []
        entries = []
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def _keys(self) -> dict:
        if not self.keys_path.exists():
            return {}
        try:
            return json.loads(self.keys_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def find_by_key(self, key: str) -> str | None:
        return self._keys().get(key)

    def save_key(self, key: str, run_id: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        keys = self._keys()
        keys[key] = run_id
        self.keys_path.write_text(json.dumps(keys, indent=2), encoding="utf-8")
