"""Aggregate the run ledger, so delegation savings are a query not a guess."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from statistics import median

from agents.console.envelope import EXIT_OK, usage_error
from agents.console.store import RunStore

_UNITS = {"d": "days", "h": "hours", "m": "minutes"}


def _parse_since(since: str) -> timedelta | None:
    if not since or since[-1] not in _UNITS:
        return None
    try:
        value = int(since[:-1])
    except ValueError:
        return None
    return timedelta(**{_UNITS[since[-1]]: value})


def summarise(entries: list[dict]) -> dict:
    """Group ledger entries by backend into totals."""
    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        grouped.setdefault(entry.get("backend", "unknown"), []).append(entry)

    summary = {}
    for backend, rows in grouped.items():
        durations = [int(r.get("duration_ms", 0)) for r in rows]
        successes = sum(1 for r in rows if r.get("status") == "completed")
        summary[backend] = {
            "runs": len(rows),
            "input_tokens": sum(int(r.get("llm_input_tokens", 0)) for r in rows),
            "output_tokens": sum(int(r.get("llm_output_tokens", 0)) for r in rows),
            "cost_usd": round(
                sum(float(r.get("llm_total_cost_usd", 0.0)) for r in rows), 6
            ),
            "median_duration_ms": int(median(durations)) if durations else 0,
            "success_rate": round(successes / len(rows), 4) if rows else 0.0,
        }
    return summary


def stats_command(store: RunStore, since, backend, fmt: str) -> int:
    entries = store.read_ledger()

    if since:
        delta = _parse_since(since)
        if delta is None:
            return usage_error(
                f"could not parse --since value '{since}'.",
                "agents stats --since 7d      # d=days, h=hours, m=minutes",
            )
        cutoff = datetime.now(timezone.utc) - delta
        kept = []
        for entry in entries:
            raw = entry.get("started_at")
            if not raw:
                continue
            try:
                if datetime.fromisoformat(raw) >= cutoff:
                    kept.append(entry)
            except ValueError:
                continue
        entries = kept

    if backend:
        entries = [e for e in entries if e.get("backend") == backend]

    summary = summarise(entries)

    if fmt == "table":
        print(f"{'backend':<10}{'runs':>6}{'in':>12}{'out':>10}"
              f"{'cost_usd':>12}{'median_ms':>12}{'success':>10}")
        for name, row in sorted(summary.items()):
            print(f"{name:<10}{row['runs']:>6}{row['input_tokens']:>12}"
                  f"{row['output_tokens']:>10}{row['cost_usd']:>12.4f}"
                  f"{row['median_duration_ms']:>12}{row['success_rate']:>10.2f}")
    else:
        print(json.dumps(summary, indent=2))
    return EXIT_OK
