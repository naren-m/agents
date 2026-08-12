"""CodexAgent -- CLI agent backend for the OpenAI Codex CLI.

Drives ``codex exec --json`` as a non-interactive, unattended subprocess
and parses its JSONL event stream into ``AgentRun`` telemetry fields.

The flags mirror the unattended-headless intent of ``CursorAgent``:

- ``--json`` emits a line-delimited event stream we can parse for tokens,
  tool calls, and session id.
- ``--skip-git-repo-check`` lets the agent run in workspaces that are
  not git repos (transcript dirs, ephemeral scratch dirs, etc.).
- ``--dangerously-bypass-approvals-and-sandbox`` skips the per-call
  approval prompts that would otherwise stall an unattended run. The
  caller is responsible for choosing a safe workspace; the flag name is
  inherited from upstream and not editorialised here.
- ``--cd <workspace>`` makes the agent's working root deterministic and
  matches ``config.workspace`` exactly.

Token usage in ``codex exec --json`` is reported per-turn via
``turn.completed`` events. We sum across turns to get the run total,
matching what an operator would bill for.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable

from agents.cli.base import BaseCLIAgent
from agents.types import AgentConfig, AgentRun

logger = logging.getLogger(__name__)


_TOOL_ITEM_TYPES = frozenset({
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "web_search",
})

_BASE_SHELL_ENV_KEYS = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "CODEX_HOME",
)


def _allowed_shell_env_keys(config_env: dict[str, str]) -> list[str]:
    keys = list(_BASE_SHELL_ENV_KEYS)
    for key in sorted(config_env):
        if (
            key.startswith("JENKINS")
            or key.startswith("RCA_")
            or key.startswith("LOGPARSER_")
            or key == "COREX_ROBOT_URL"
        ):
            if key not in keys:
                keys.append(key)
    return keys


def parse_codex_telemetry(output: str, run: AgentRun) -> None:
    """Extract telemetry from ``codex exec --json`` output into the AgentRun.

    Parses one JSON object per line. Recognises:

    - ``{"type": "thread.started", "thread_id": "..."}`` -- session id.
    - ``{"type": "item.started", "item": {"type": "<tool>"}}`` -- one
      tool invocation. Counted only for tool-shaped item types
      (``command_execution``, ``file_change``, ``mcp_tool_call``,
      ``web_search``); ``agent_message``/``reasoning`` items are model
      output and not tools.
    - ``{"type": "turn.completed", "usage": {...}}`` -- per-turn token
      usage. ``input_tokens``, ``cached_input_tokens``, ``output_tokens``
      are summed across every turn.completed event in the stream.

    Non-JSON lines (stderr noise, banner text) are skipped silently.
    """
    if not output:
        return

    tool_call_count = 0
    request_count = 0
    input_tokens = 0
    cache_read_tokens = 0
    output_tokens = 0
    session_id: str | None = None

    for raw in _iter_json_lines(output):
        event_type = raw.get("type", "")

        if event_type == "thread.started" and session_id is None:
            tid = raw.get("thread_id")
            if isinstance(tid, str):
                session_id = tid

        elif event_type == "item.started":
            item = raw.get("item")
            if isinstance(item, dict) and item.get("type") in _TOOL_ITEM_TYPES:
                tool_call_count += 1

        elif event_type == "turn.completed":
            usage = raw.get("usage")
            if isinstance(usage, dict):
                request_count += 1
                input_tokens += int(usage.get("input_tokens", 0) or 0)
                cache_read_tokens += int(
                    usage.get("cached_input_tokens", 0) or 0
                )
                output_tokens += int(usage.get("output_tokens", 0) or 0)

    run.tool_call_count = tool_call_count
    run.llm_request_count = request_count
    run.llm_input_tokens = input_tokens
    run.llm_cache_read_tokens = cache_read_tokens
    run.llm_output_tokens = output_tokens
    if session_id is not None:
        run.session_id = session_id


def _iter_json_lines(output: str) -> Iterable[dict]:
    """Yield parsed JSON objects from a mixed stdout stream.

    Skips blank lines, lines that do not start with ``{``, and lines
    that fail to parse. Non-dict JSON values (lists, scalars) are
    skipped because the codex event protocol is always objects.
    """
    for line in output.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            yield obj


class CodexAgent(BaseCLIAgent):
    """OpenAI Codex headless agent via ``codex exec --json``."""

    name = "codex"
    _binary_names = ["codex"]

    def build_command(self, prompt: str, config: AgentConfig) -> list[str]:
        binary = self.binary_path()
        include_only = _allowed_shell_env_keys(config.env or {})
        headless_flags = [
            "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-c",
            "shell_environment_policy.inherit=none",
            "-c",
            f"shell_environment_policy.include_only={json.dumps(include_only)}",
        ]
        return [
            binary, "exec",
            *headless_flags,
            "--cd", str(config.workspace),
            prompt,
        ]

    def parse_output(self, output: str, run: AgentRun) -> None:
        parse_codex_telemetry(output, run)
