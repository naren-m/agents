"""Shared data types for all agent backends.

AgentConfig, AgentRun, and AgentResult are the common currency between
the abstraction layer and its consumers (rca-server, inspector, dashboard).
Both CLI and in-process backends produce the same output types so callers
never need to know which backend ran.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.event_bus import RunEventBus


def _generate_run_id() -> str:
    """12-char hex ID, matching task-manager convention."""
    return uuid.uuid4().hex[:12]


@dataclass
class AgentConfig:
    """Common configuration passed to every agent backend."""

    workspace: Path
    mcp_server_url: str = ""
    timeout_seconds: int = 900
    transcript_dir: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    # Optional live event channel. When set, CLI backends stream raw
    # stdout into the bus ring buffer and structured run/phase events
    # into its event history. Leaving this None preserves the original
    # non-streaming path.
    event_bus: "RunEventBus | None" = None


@dataclass
class AgentRun:
    """Tracks a single agent execution from spawn to completion."""

    run_id: str
    backend: str
    prompt: str
    started_at: datetime
    status: str = "running"
    pid: int | None = None
    exit_code: int | None = None
    transcript_path: Path | None = None
    finished_at: datetime | None = None

    tool_call_count: int = 0
    llm_request_count: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cache_read_tokens: int = 0
    llm_cache_write_tokens: int = 0
    llm_total_cost_usd: float = 0.0
    duration_ms: int = 0
    model: str = ""
    session_id: str | None = None
    agent_name: str | None = None
    metadata: dict = field(default_factory=dict)

    VALID_STATUSES = frozenset(
        {"running", "completed", "failed", "cancelled", "timed_out"}
    )

    def __post_init__(self):
        if self.status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status {self.status!r}, "
                f"must be one of {sorted(self.VALID_STATUSES)}"
            )

    @classmethod
    def create(
        cls,
        backend: str,
        prompt: str,
        pid: int | None = None,
        transcript_path: Path | None = None,
        run_id: str | None = None,
    ) -> AgentRun:
        """Factory that generates run_id and timestamps automatically.

        ``run_id`` may be supplied by callers that need to reserve the
        id up-front (for example to bind an ``EventBusRegistry`` entry
        before the run exists).  When omitted, a fresh 12-char hex id
        is generated.
        """
        return cls(
            run_id=run_id if run_id is not None else _generate_run_id(),
            backend=backend,
            prompt=prompt,
            started_at=datetime.now(timezone.utc),
            pid=pid,
            transcript_path=transcript_path,
        )

    def mark_completed(self, exit_code: int = 0) -> None:
        self.status = "completed"
        self.exit_code = exit_code
        self.finished_at = datetime.now(timezone.utc)

    def mark_failed(self, exit_code: int = 1) -> None:
        self.status = "failed"
        self.exit_code = exit_code
        self.finished_at = datetime.now(timezone.utc)

    def mark_cancelled(self) -> None:
        self.status = "cancelled"
        self.finished_at = datetime.now(timezone.utc)

    def mark_timed_out(self) -> None:
        self.status = "timed_out"
        self.finished_at = datetime.now(timezone.utc)

    @property
    def is_terminal(self) -> bool:
        return self.status in {"completed", "failed", "cancelled", "timed_out"}

    @property
    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


@dataclass
class AgentResult:
    """Final result from a completed agent run."""

    success: bool
    output: str
    run: AgentRun
