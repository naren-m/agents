"""CursorAgent -- CLI agent backend for Cursor (cursor-agent / cursor CLI).

Extracted from inspector/inspector/auto.py. Handles both the standalone
``cursor-agent`` binary and the IDE-injected ``cursor`` binary with its
``agent`` subcommand.

Uses ``--output-format json`` so the output contains structured JSON lines
including a final ``{"type": "usage", ...}`` block with token consumption
and cost data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agents.cli.base import BaseCLIAgent
from agents.types import AgentConfig, AgentRun

logger = logging.getLogger(__name__)


def parse_cursor_telemetry(output: str, run: AgentRun) -> None:
    """Extract telemetry from cursor's JSON-lines output into the AgentRun.

    Cursor with ``--output-format json`` emits one JSON object per line.
    We look for:
    - ``{"type": "result", "usage": {...}, "duration_ms": N}`` -- final summary
      with camelCase token fields (inputTokens, outputTokens, etc.)
    - ``{"type": "system", "subtype": "init", "model": "..."}`` -- model info
    - ``{"type": "usage", ...}`` -- legacy summary block with snake_case fields
    - ``{"type": "tool_use", ...}`` -- individual tool call events

    Priority: ``result`` > ``usage`` > tool_use count fallback.
    When multiple blocks of the same type exist, the last one wins.
    """
    if not output:
        return

    tool_use_count = 0
    last_usage: dict | None = None
    last_result: dict | None = None
    init_model: str = ""

    for line in output.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        if not isinstance(obj, dict):
            continue

        event_type = obj.get("type", "")

        if event_type == "result":
            last_result = obj
        elif event_type == "usage":
            last_usage = obj
        elif event_type == "tool_use":
            tool_use_count += 1
        elif event_type == "system" and obj.get("subtype") == "init":
            init_model = obj.get("model", "")

    if last_result:
        usage = last_result.get("usage", {})
        run.llm_input_tokens = int(usage.get("inputTokens", 0))
        run.llm_output_tokens = int(usage.get("outputTokens", 0))
        run.llm_cache_read_tokens = int(usage.get("cacheReadTokens", 0))
        run.llm_cache_write_tokens = int(usage.get("cacheWriteTokens", 0))
        run.duration_ms = int(last_result.get("duration_ms", 0))
        run.tool_call_count = tool_use_count
    elif last_usage:
        run.llm_total_cost_usd = float(last_usage.get("total_cost_usd", 0.0))
        run.llm_input_tokens = int(last_usage.get("total_input_tokens", 0))
        run.llm_output_tokens = int(last_usage.get("total_output_tokens", 0))
        run.llm_cache_read_tokens = int(last_usage.get("total_cache_read_tokens", 0))
        run.llm_request_count = int(last_usage.get("num_requests", 0))
        run.tool_call_count = int(last_usage.get("num_tool_calls", tool_use_count))
    else:
        run.tool_call_count = tool_use_count

    if init_model:
        run.model = init_model


class CursorAgent(BaseCLIAgent):
    """Cursor headless agent via ``cursor-agent -p`` or ``cursor agent -p``.

    The RCA agent runs unattended, so the spawn command enables the full
    set of non-interactive ("yolo") flags exposed by the Cursor CLI:

    - ``--yolo``  alias for ``--force``; auto-approves shell/command
      execution without per-call prompts.
    - ``--trust`` marks the workspace as trusted so Cursor does not block
      waiting for an interactive "Trust this folder?" confirmation. Only
      valid in headless (``--print``) mode.
    - ``--approve-mcps`` auto-approves every configured MCP server. The
      RCA pipeline depends on many MCP servers (inspector, corex,
      logparser, memory-bank, rca-agent, task-manager), so without this
      flag each tool call would stall on approval prompts.

    Also uses ``--output-format json`` so stdout is structured and
    telemetry (tokens, cost, tool_use events, duration, model) can be
    parsed by :func:`parse_cursor_telemetry`.
    """

    name = "cursor"
    _binary_names = ["cursor-agent", "cursor"]

    def build_command(self, prompt: str, config: AgentConfig) -> list[str]:
        binary = self.binary_path()
        base = Path(binary).name

        # Keep all non-interactive / permissive flags grouped together so
        # the intent (unattended execution) is obvious at call sites.
        headless_flags = ["-p", "--yolo", "--trust", "--approve-mcps"]
        tail = [
            "--output-format", "json",
            "--workspace", str(config.workspace),
            prompt,
        ]

        if base == "cursor-agent":
            return [binary, *headless_flags, *tail]
        return [binary, "agent", *headless_flags, *tail]

    def parse_output(self, output: str, run: AgentRun) -> None:
        """Parse cursor JSON output to extract telemetry into the run."""
        parse_cursor_telemetry(output, run)
