"""AgyAgent -- CLI agent backend for the Google Antigravity CLI (``agy``).

Verified against agy v1.1.27. Three behaviours drive this implementation:

1. ``--dangerously-skip-permissions`` is MANDATORY in print mode. Headless agy
   cannot prompt for tool approval, so any tool needing one is auto-denied and
   the run produces no output.
2. Success cannot be read from the exit code. On tool denial agy returns
   ``status: "SUCCESS"``, exit code 0 and an empty ``response``, signalling the
   failure only via a ``denied_actions`` array.
3. Sandboxed agy misreports writes: it claims a file "has been overwritten"
   while the real file is untouched and the write went to its scratch dir.
"""

from __future__ import annotations

import json
import logging

from agents.cli.base import BaseCLIAgent
from agents.types import AgentConfig, AgentRun, Capability

logger = logging.getLogger(__name__)

_SCRATCH_MARKER = "antigravity-cli/scratch"


def _apply_result(result: dict, run: AgentRun, tool_calls: int) -> None:
    """Map one agy ``result`` payload onto the run."""
    usage = result.get("usage", {}) or {}

    run.session_id = result.get("conversation_id") or run.session_id
    run.llm_input_tokens = int(usage.get("input_tokens", 0))
    run.llm_output_tokens = int(usage.get("output_tokens", 0))
    run.llm_cache_read_tokens = int(usage.get("cache_read_tokens", 0))
    run.llm_request_count = int(result.get("num_turns", 0))
    run.duration_ms = int(float(result.get("duration_seconds", 0.0)) * 1000)
    run.tool_call_count = tool_calls
    # agy reports no cost: it bills Google's quota, not the caller's.
    run.llm_total_cost_usd = 0.0

    if "thinking_tokens" in usage:
        run.metadata["thinking_tokens"] = int(usage["thinking_tokens"])

    response = result.get("response", "") or ""
    # AgentResult.output holds truncated raw NDJSON, so the clean answer has to
    # travel separately for the console to print it.
    run.metadata["response"] = response

    if _SCRATCH_MARKER in response:
        run.metadata["sandbox_write_claim"] = True

    denied = result.get("denied_actions")
    if denied:
        run.metadata["denied_actions"] = denied

    error = result.get("error")
    if error:
        run.metadata["error"] = error

    status = result.get("status", "")
    failed = status != "SUCCESS" or bool(denied) or not response.strip()
    if failed and not run.is_terminal:
        run.mark_failed()


def parse_agy_telemetry(output: str, run: AgentRun) -> None:
    """Extract telemetry from agy output into the run.

    Handles both ``--output-format stream-json`` (NDJSON keyed on ``event``,
    terminal event ``result``) and ``--output-format json`` (a single object).
    """
    if not output or not output.strip():
        return

    tool_calls = 0
    result_payload: dict | None = None

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

        event = obj.get("event")
        if event == "init":
            model = (obj.get("init") or {}).get("model", "")
            if model:
                run.model = model
        elif event == "step_update":
            if (obj.get("step_update") or {}).get("step_type") == "tool":
                tool_calls += 1
        elif event == "result":
            result_payload = obj.get("result") or {}
        elif event is None and "status" in obj:
            # Single-object --output-format json.
            result_payload = obj

    if result_payload is not None:
        _apply_result(result_payload, run, tool_calls)


class AgyAgent(BaseCLIAgent):
    """Google Antigravity CLI in headless print mode."""

    name = "agy"
    _binary_names = ["agy"]
    capabilities = frozenset(
        {Capability.WORKSPACE, Capability.TOOLS, Capability.MCP}
    )

    def build_command(self, prompt: str, config: AgentConfig) -> list[str]:
        binary = self.binary_path()
        cmd = [
            binary,
            "--print", prompt,
            "--output-format", "stream-json",
            # Mandatory: headless agy cannot prompt, so unapproved tools are
            # auto-denied and the run returns nothing.
            "--dangerously-skip-permissions",
        ]
        model = config.env.get("AGY_MODEL", "")
        if model:
            cmd += ["--model", model]
        if config.timeout_seconds:
            cmd += ["--print-timeout", f"{config.timeout_seconds}s"]
        return cmd

    def parse_output(self, output: str, run: AgentRun) -> None:
        parse_agy_telemetry(output, run)
