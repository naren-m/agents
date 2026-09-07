# Agent Console CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the `agents` library a console entry point so an orchestrator can hand a task to a backend, get the finished output back as a JSON envelope, and move on — with two new backends, `agy` (Google Antigravity CLI) and `ollama` (local models).

**Architecture:** Two new backends plug into the existing abstractions unchanged — `agy` extends `BaseCLIAgent` (a CLI that is an agent), `ollama` implements the `InProcessAgent` Protocol (an HTTP completion endpoint, no subprocess). A new `agents/console/` subpackage provides the `agents` command: `run` blocks and prints an envelope, `start` detaches and returns a run_id, and `status`/`cancel`/`logs`/`stats` read persisted run state from `~/.agents/`.

**Tech Stack:** Python 3.10+, stdlib `argparse`, `pytest` + `pytest-asyncio` (`asyncio_mode = "auto"`), `httpx` (optional extra, ollama only).

**Spec:** `docs/superpowers/specs/2026-09-06-agent-console-cli-design.md`

## Global Constraints

- **Python floor is 3.10** (`requires-python = ">=3.10"`). `enum.StrEnum` is 3.11+ — the spec's `class Capability(StrEnum)` MUST be written `class Capability(str, Enum)` instead. Do not use `match` statements or PEP 604 in runtime-evaluated positions without `from __future__ import annotations`.
- **Zero runtime dependencies.** `dependencies = []` in `pyproject.toml` is deliberate. `httpx` goes in a new `ollama` optional extra, never the base dependency list. `OllamaAgent.available()` must return `False` when `httpx` is not importable rather than raising at import time.
- **Existing backends must not break.** `cursor`, `codex`, and `copilot` subclass `BaseCLIAgent`. Any new attribute on the base class needs a default that preserves their current behaviour.
- **`AgentRun.VALID_STATUSES`** is `{"running", "completed", "failed", "cancelled", "timed_out"}`. `__post_init__` raises on anything else. Use the `mark_*()` methods, never assign `.status` directly.
- **Existing hook, do not re-invent:** `BaseCLIAgent.wait()` calls `self.parse_output(output, run)` at `agents/cli/base.py:288`, then sets terminal status only `if not run.is_terminal` (`agents/cli/base.py:292-296`), and returns `AgentResult(success=run.status == "completed", ...)`. A `parse_output` that calls `run.mark_failed()` therefore overrides the exit-code verdict. This is how the agy silent-failure trap is handled — no base-class change.
- **`AgentResult.output` is truncated** to the last 4000 chars (`agents/cli/base.py:301`). For agy the raw stream is NDJSON, not the answer, so `parse_output` MUST store the clean response in `run.metadata["response"]` and the console MUST prefer that over `result.output`.
- Commit after every task. Never commit a failing test suite.

---

### Task 1: Capability model

Backends declare what kind of input they accept, so the console can reject a nonsensical flag combination with a corrected invocation instead of silently doing the wrong thing.

**Files:**
- Modify: `agents/types.py` (append after the `AgentResult` dataclass)
- Modify: `agents/cli/base.py:123-133` (add class attribute to `BaseCLIAgent`)
- Modify: `agents/__init__.py` (export `Capability`)
- Test: `tests/test_capabilities.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `agents.types.Capability` — `str`-valued enum with members `WORKSPACE = "workspace"`, `TOOLS = "tools"`, `MCP = "mcp"`, `CONTEXT = "context_files"`
  - `BaseCLIAgent.capabilities: frozenset[Capability]` — defaults to `frozenset({Capability.WORKSPACE, Capability.TOOLS})`

- [ ] **Step 1: Write the failing test**

Create `tests/test_capabilities.py`:

```python
from agents.types import Capability
from agents.cli.base import BaseCLIAgent
from agents.cli.cursor import CursorAgent


def test_capability_is_str_valued():
    # Must be str-valued, not StrEnum: the project floor is Python 3.10.
    assert Capability.WORKSPACE == "workspace"
    assert Capability.CONTEXT == "context_files"
    assert isinstance(Capability.TOOLS, str)


def test_base_cli_agent_defaults_to_workspace_and_tools():
    assert BaseCLIAgent.capabilities == frozenset(
        {Capability.WORKSPACE, Capability.TOOLS}
    )


def test_existing_backends_inherit_the_default():
    # cursor/codex/copilot must keep working with no edits.
    assert Capability.WORKSPACE in CursorAgent.capabilities
    assert Capability.TOOLS in CursorAgent.capabilities
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_capabilities.py -v`
Expected: FAIL with `ImportError: cannot import name 'Capability' from 'agents.types'`

- [ ] **Step 3: Write minimal implementation**

Append to `agents/types.py`:

```python
class Capability(str, Enum):
    """What kind of input a backend accepts.

    ``str``-valued rather than ``enum.StrEnum`` because the project floor is
    Python 3.10 and ``StrEnum`` landed in 3.11.
    """

    WORKSPACE = "workspace"          # can be pointed at a directory and explore it
    TOOLS = "tools"                  # has its own file/shell tools
    MCP = "mcp"                      # can attach MCP servers
    CONTEXT = "context_files"        # has no tools; content must be inlined
```

Add `from enum import Enum` to the imports at the top of `agents/types.py` if absent.

In `agents/cli/base.py`, inside `class BaseCLIAgent`, next to `name` and `_binary_names`:

```python
    # Backends that explore a workspace with their own tools. Subclasses that
    # work differently (e.g. completion-only) override this.
    capabilities: frozenset[Capability] = frozenset(
        {Capability.WORKSPACE, Capability.TOOLS}
    )
```

Import it: `from agents.types import AgentConfig, AgentResult, AgentRun, Capability`.

Export from `agents/__init__.py` by adding `Capability` to the imports from `agents.types` and to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_capabilities.py -v && pytest -q`
Expected: new tests PASS, full suite still passes (existing backends unaffected).

- [ ] **Step 5: Commit**

```bash
git add agents/types.py agents/cli/base.py agents/__init__.py tests/test_capabilities.py
git commit -m "feat: add Capability model for backend input declaration"
```

---

### Task 2: AgyAgent backend

Wraps the Antigravity CLI. Everything verified against a live `agy` v1.1.27 run — see the spec's "Verified output shape" section.

**Files:**
- Create: `agents/cli/agy.py`
- Test: `tests/test_agy_agent.py`

**Interfaces:**
- Consumes: `Capability` (Task 1); `BaseCLIAgent`, `AgentConfig`, `AgentRun` from the existing library
- Produces:
  - `agents.cli.agy.AgyAgent` — `name = "agy"`, `_binary_names = ["agy"]`, `capabilities = frozenset({WORKSPACE, TOOLS, MCP})`
  - `agents.cli.agy.parse_agy_telemetry(output: str, run: AgentRun) -> None`
  - `AgyAgent.build_command(prompt, config) -> list[str]`
  - After parsing, `run.metadata["response"]` holds the clean answer text; `run.metadata["thinking_tokens"]` holds the thinking token count; `run.metadata["denied_actions"]` holds the denied-action list when present.

- [ ] **Step 1: Write the failing test**

Create `tests/test_agy_agent.py`:

```python
import json
from pathlib import Path

from agents.cli.agy import AgyAgent, parse_agy_telemetry
from agents.types import AgentConfig, AgentRun, Capability


def _run() -> AgentRun:
    return AgentRun.create(backend="agy", prompt="p")


def _config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(workspace=tmp_path)


def test_capabilities_include_mcp():
    assert AgyAgent.capabilities == frozenset(
        {Capability.WORKSPACE, Capability.TOOLS, Capability.MCP}
    )


def test_build_command_includes_mandatory_permission_flag(tmp_path):
    # Verified: without this flag, headless agy auto-denies every tool that
    # needs approval and returns an empty response.
    agent = AgyAgent()
    agent.binary_path = lambda: "/usr/bin/agy"
    cmd = agent.build_command("do a thing", _config(tmp_path))
    assert "--dangerously-skip-permissions" in cmd
    assert "--print" in cmd
    assert "do a thing" in cmd


def test_build_command_uses_stream_json(tmp_path):
    # stream-json is a strict superset of json: it alone carries tool counts
    # and the resolved model.
    agent = AgyAgent()
    agent.binary_path = lambda: "/usr/bin/agy"
    cmd = agent.build_command("t", _config(tmp_path))
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"


def test_parse_stream_json_maps_all_telemetry():
    stream = "\n".join([
        json.dumps({"event": "init", "conversation_id": "c1",
                    "init": {"model": "gemini-3.8-flash-high", "cwd": "/w"}}),
        json.dumps({"event": "step_update",
                    "step_update": {"step_type": "tool", "tool_name": "read"}}),
        json.dumps({"event": "step_update",
                    "step_update": {"step_type": "tool", "tool_name": "grep"}}),
        json.dumps({"event": "result", "result": {
            "conversation_id": "c1", "status": "SUCCESS", "response": "the answer",
            "duration_seconds": 51.745, "num_turns": 3,
            "usage": {"input_tokens": 101932, "output_tokens": 3829,
                      "thinking_tokens": 2324, "cache_read_tokens": 320050,
                      "total_tokens": 105761}}}),
    ])
    run = _run()
    parse_agy_telemetry(stream, run)

    assert run.session_id == "c1"
    assert run.model == "gemini-3.8-flash-high"
    assert run.llm_input_tokens == 101932
    assert run.llm_output_tokens == 3829
    assert run.llm_cache_read_tokens == 320050
    assert run.llm_request_count == 3
    assert run.duration_ms == 51745
    assert run.tool_call_count == 2
    assert run.metadata["thinking_tokens"] == 2324
    assert run.metadata["response"] == "the answer"
    assert run.llm_total_cost_usd == 0.0
    assert not run.is_terminal  # success path leaves the verdict to wait()


def test_denied_actions_marks_run_failed_despite_success_status():
    # THE TRAP, verified live: agy exits 0 with status SUCCESS and an empty
    # response when a tool was auto-denied. exit_code == 0 is not success.
    payload = json.dumps({"event": "result", "result": {
        "conversation_id": "c2", "status": "SUCCESS", "response": "",
        "duration_seconds": 2.95, "num_turns": 1,
        "usage": {"input_tokens": 16451, "output_tokens": 494,
                  "thinking_tokens": 424, "cache_read_tokens": 0},
        "denied_actions": [{"action": "command", "display_name": "RunCommand"}]}})
    run = _run()
    parse_agy_telemetry(payload, run)

    assert run.status == "failed"
    assert run.is_terminal
    assert run.metadata["denied_actions"] == [
        {"action": "command", "display_name": "RunCommand"}
    ]


def test_error_status_marks_run_failed():
    payload = json.dumps({"event": "result", "result": {
        "conversation_id": "c3", "status": "ERROR", "response": "",
        "error": "timeout waiting for response", "duration_seconds": 118.9,
        "num_turns": 1, "usage": {"input_tokens": 1, "output_tokens": 0}}})
    run = _run()
    parse_agy_telemetry(payload, run)

    assert run.status == "failed"
    assert run.metadata["error"] == "timeout waiting for response"


def test_sandboxed_write_claim_is_annotated():
    # Verified: sandboxed agy reports "has been overwritten" while the real
    # file is untouched and the write went to its scratch dir.
    payload = json.dumps({"event": "result", "result": {
        "conversation_id": "c4", "status": "SUCCESS",
        "response": "[a.py](file:///Users/x/.gemini/antigravity-cli/scratch/a.py) "
                    "has been overwritten with `X`.",
        "duration_seconds": 1.0, "num_turns": 1,
        "usage": {"input_tokens": 1, "output_tokens": 1}}})
    run = _run()
    parse_agy_telemetry(payload, run)

    assert run.metadata["sandbox_write_claim"] is True


def test_single_object_json_format_still_parses():
    # --output-format json emits one object, not NDJSON. Support it so a
    # caller who overrides the format still gets telemetry.
    payload = json.dumps({
        "conversation_id": "c5", "status": "SUCCESS", "response": "hi",
        "duration_seconds": 2.0, "num_turns": 1,
        "usage": {"input_tokens": 10, "output_tokens": 2}})
    run = _run()
    parse_agy_telemetry(payload, run)

    assert run.metadata["response"] == "hi"
    assert run.llm_input_tokens == 10


def test_empty_output_is_safe():
    run = _run()
    parse_agy_telemetry("", run)
    assert run.llm_input_tokens == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agy_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.cli.agy'`

- [ ] **Step 3: Write minimal implementation**

Create `agents/cli/agy.py`:

```python
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
from pathlib import Path

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
    failed = (
        status != "SUCCESS"
        or bool(denied)
        or not response.strip()
    )
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agy_agent.py -v && pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/cli/agy.py tests/test_agy_agent.py
git commit -m "feat: add agy CLI backend with verified telemetry and failure detection"
```

---

### Task 3: OllamaAgent backend

An in-process completion backend. Not a `BaseCLIAgent` — ollama has no tools, no workspace and no subprocess; forking a process to make an HTTP request would misrepresent the lifecycle.

**Files:**
- Create: `agents/inprocess/ollama.py`
- Modify: `pyproject.toml` (add `ollama` optional extra)
- Test: `tests/test_ollama_agent.py`

**Interfaces:**
- Consumes: `Capability` (Task 1); `InProcessAgent` Protocol, `AgentConfig`, `AgentResult`, `AgentRun`
- Produces:
  - `agents.inprocess.ollama.OllamaAgent` — `name` property returns `"ollama"`, `capabilities = frozenset({Capability.CONTEXT})`
  - `OllamaAgent(host: str = "http://127.0.0.1:11434", model: str = "qwen2.5-coder:32b")`
  - `OllamaAgent.available() -> bool`
  - `async OllamaAgent.run(prompt, config, mcp_client=None, on_progress=None) -> AgentResult`
  - `async OllamaAgent.cancel() -> bool`
  - `agents.inprocess.ollama.size_num_ctx(char_count: int) -> int`
  - `agents.inprocess.ollama.embed_files(paths: list[Path]) -> str`
  - `agents.inprocess.ollama.TASK_MODELS: dict[str, str]` — `{"code": "qwen2.5-coder:32b", "summary": "gemma3:27b", "fast": "gemma4:12b"}`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ollama_agent.py`:

```python
import pytest

from agents.inprocess.ollama import (
    OllamaAgent, TASK_MODELS, embed_files, size_num_ctx,
)
from agents.types import AgentConfig, Capability


def test_capabilities_are_context_only():
    assert OllamaAgent.capabilities == frozenset({Capability.CONTEXT})


def test_task_model_presets():
    assert TASK_MODELS["code"] == "qwen2.5-coder:32b"
    assert TASK_MODELS["summary"] == "gemma3:27b"
    assert TASK_MODELS["fast"] == "gemma4:12b"


@pytest.mark.parametrize("chars,expected", [
    (100, 4096),
    (30_000, 16384),
    (200_000, 131072),
])
def test_num_ctx_grows_with_input(chars, expected):
    # Ollama defaults to 4096 and SILENTLY truncates past it, so the context
    # has to be sized explicitly from the input.
    assert size_num_ctx(chars) == expected


def test_num_ctx_never_below_default():
    assert size_num_ctx(0) == 4096


def test_embed_files_includes_line_numbers(tmp_path):
    # Without line numbers the model estimates citations and drifts 5-15 lines.
    f = tmp_path / "a.py"
    f.write_text("first\nsecond\n")
    embedded = embed_files([f])
    assert "1\tfirst" in embedded
    assert "2\tsecond" in embedded
    assert str(f) in embedded


def test_embed_files_empty_list():
    assert embed_files([]) == ""


async def test_run_maps_telemetry(monkeypatch, tmp_path):
    agent = OllamaAgent(model="gemma4:12b")

    async def fake_post(self, prompt, num_ctx, model):
        return {
            "response": "the answer",
            "prompt_eval_count": 4249,
            "eval_count": 287,
        }

    monkeypatch.setattr(OllamaAgent, "_post_generate", fake_post)
    result = await agent.run("q", AgentConfig(workspace=tmp_path))

    assert result.success is True
    assert result.output == "the answer"
    assert result.run.llm_input_tokens == 4249
    assert result.run.llm_output_tokens == 287
    # The zero is the point: it is what makes savings legible in stats.
    assert result.run.llm_total_cost_usd == 0.0
    assert result.run.model == "gemma4:12b"
    assert result.run.status == "completed"


async def test_run_marks_failed_on_error(monkeypatch, tmp_path):
    agent = OllamaAgent()

    async def boom(self, prompt, num_ctx, model):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(OllamaAgent, "_post_generate", boom)
    result = await agent.run("q", AgentConfig(workspace=tmp_path))

    assert result.success is False
    assert result.run.status == "failed"
    assert "connection refused" in result.output


async def test_cancel_before_run_is_false():
    assert await OllamaAgent().cancel() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ollama_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.inprocess.ollama'`

- [ ] **Step 3: Write minimal implementation**

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
ollama = ["httpx"]
```

Create `agents/inprocess/ollama.py`:

```python
"""OllamaAgent -- in-process backend for local models served by ollama.

Not a CLI agent: ollama has no tools and no workspace, so it implements the
in-process protocol and talks HTTP directly rather than forking a process.

Two behaviours drive this implementation:

1. ``num_ctx`` MUST be set explicitly. Ollama defaults to 4096 and silently
   truncates anything longer, returning a confident answer about the first
   part of the input with no indication the rest was dropped.
2. Files are embedded with line numbers. A model cannot count lines in a blob;
   without the numbers present it estimates citations and drifts 5-15 lines.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from agents.types import AgentConfig, AgentResult, AgentRun, Capability

logger = logging.getLogger(__name__)

TASK_MODELS = {
    "code": "qwen2.5-coder:32b",
    "summary": "gemma3:27b",
    "fast": "gemma4:12b",
}

_CTX_STEPS = [4096, 8192, 16384, 32768, 65536, 131072]
_CHARS_PER_TOKEN = 3
_ANSWER_HEADROOM_TOKENS = 1024

_SYSTEM = (
    "You are a code analysis assistant answering a senior engineer. Be terse "
    "and factual. File content is given with line numbers in the left column. "
    "When you cite code, copy the line number from that column verbatim - "
    "never estimate it. Format citations as path:line. State plainly if the "
    "context does not contain the answer - never guess. No preamble."
)


def size_num_ctx(char_count: int) -> int:
    """Pick the smallest supported context that fits the input."""
    needed = char_count // _CHARS_PER_TOKEN + _ANSWER_HEADROOM_TOKENS
    for step in _CTX_STEPS:
        if needed <= step:
            return step
    return _CTX_STEPS[-1]


def embed_files(paths: list[Path]) -> str:
    """Inline files with line numbers so citations are copied, not estimated."""
    chunks = []
    for path in paths:
        numbered = "\n".join(
            f"{i}\t{line}"
            for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
        )
        chunks.append(f"===== FILE: {path} =====\n{numbered}")
    return "\n".join(chunks)


class OllamaAgent:
    """Local model completion via the ollama HTTP API."""

    capabilities = frozenset({Capability.CONTEXT})

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = TASK_MODELS["code"],
    ):
        self._host = host.rstrip("/")
        self._model = model
        self._task: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return "ollama"

    def available(self) -> bool:
        # httpx is an optional extra; absence means unavailable, not a crash.
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return True

    async def _post_generate(self, prompt: str, num_ctx: int, model: str) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=900) as client:
            resp = await client.post(
                f"{self._host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": _SYSTEM,
                    "stream": False,
                    "options": {"num_ctx": num_ctx, "temperature": 0.2},
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def run(
        self,
        prompt: str,
        config: AgentConfig,
        mcp_client=None,
        on_progress=None,
    ) -> AgentResult:
        model = config.env.get("OLLAMA_MODEL") or self._model
        run = AgentRun.create(backend="ollama", prompt=prompt)
        run.model = model

        num_ctx = size_num_ctx(len(prompt))
        try:
            payload = await self._post_generate(prompt, num_ctx, model)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            logger.warning("ollama run failed: %s", exc)
            run.mark_failed()
            return AgentResult(success=False, output=str(exc), run=run)

        output = (payload.get("response") or "").strip()
        run.llm_input_tokens = int(payload.get("prompt_eval_count", 0))
        run.llm_output_tokens = int(payload.get("eval_count", 0))
        # Local inference is free. This zero is what makes savings legible.
        run.llm_total_cost_usd = 0.0
        run.metadata["num_ctx"] = num_ctx
        run.mark_completed()

        return AgentResult(success=True, output=output, run=run)

    async def cancel(self) -> bool:
        if self._task is None or self._task.done():
            return False
        self._task.cancel()
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e ".[test,ollama]" && pytest tests/test_ollama_agent.py -v && pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/inprocess/ollama.py tests/test_ollama_agent.py pyproject.toml
git commit -m "feat: add ollama in-process backend with explicit num_ctx sizing"
```

---

### Task 4: Envelope and run store

The console's two shared primitives: the JSON output contract, and persistence for background runs.

**Files:**
- Create: `agents/console/__init__.py` (empty)
- Create: `agents/console/envelope.py`
- Create: `agents/console/store.py`
- Test: `tests/test_console_envelope.py`
- Test: `tests/test_console_store.py`

**Interfaces:**
- Consumes: `AgentResult`, `AgentRun`
- Produces:
  - `agents.console.envelope.build_envelope(result: AgentResult) -> dict`
  - `agents.console.envelope.EXIT_OK = 0`, `EXIT_AGENT_FAILED = 1`, `EXIT_USAGE = 2`, `EXIT_UNAVAILABLE = 3`
  - `agents.console.envelope.usage_error(message: str, example: str) -> int` — prints to stderr, returns `EXIT_USAGE`
  - `agents.console.store.RunStore(root: Path | None = None)` with `save(run: AgentRun) -> None`, `load(run_id: str) -> dict | None`, `append_ledger(run: AgentRun) -> None`, `read_ledger() -> list[dict]`, `find_by_key(key: str) -> str | None`, `save_key(key: str, run_id: str) -> None`
  - `RunStore` defaults its root to `~/.agents`

- [ ] **Step 1: Write the failing test**

Create `tests/test_console_envelope.py`:

```python
from agents.console.envelope import (
    EXIT_AGENT_FAILED, EXIT_OK, EXIT_UNAVAILABLE, EXIT_USAGE,
    build_envelope, usage_error,
)
from agents.types import AgentResult, AgentRun


def _result(success=True, output="answer", **fields):
    run = AgentRun.create(backend="ollama", prompt="p")
    run.model = "gemma4:12b"
    run.llm_input_tokens = 4249
    run.llm_output_tokens = 287
    run.duration_ms = 34120
    for k, v in fields.items():
        setattr(run, k, v)
    run.mark_completed() if success else run.mark_failed()
    return AgentResult(success=success, output=output, run=run)


def test_exit_codes_are_distinct():
    assert {EXIT_OK, EXIT_AGENT_FAILED, EXIT_USAGE, EXIT_UNAVAILABLE} == {0, 1, 2, 3}


def test_envelope_shape():
    env = build_envelope(_result())
    assert env["success"] is True
    assert env["backend"] == "ollama"
    assert env["model"] == "gemma4:12b"
    assert env["output"] == "answer"
    assert env["error"] is None
    assert env["run_id"]
    assert env["telemetry"] == {
        "input_tokens": 4249,
        "output_tokens": 287,
        "cache_read_tokens": 0,
        "cost_usd": 0.0,
        "duration_ms": 34120,
        "tool_calls": 0,
        "requests": 0,
    }


def test_envelope_prefers_metadata_response():
    # agy's raw stdout is NDJSON; the clean answer travels in metadata.
    result = _result(output='{"event":"result"}')
    result.run.metadata["response"] = "the real answer"
    assert build_envelope(result)["output"] == "the real answer"


def test_failed_envelope_carries_error():
    result = _result(success=False, output="")
    result.run.metadata["error"] = "timeout waiting for response"
    env = build_envelope(result)
    assert env["success"] is False
    assert env["error"] == "timeout waiting for response"


def test_usage_error_writes_stderr_and_returns_2(capsys):
    code = usage_error("backend 'ollama' cannot explore a workspace.",
                       "agents run --backend ollama -f a.py --task '...'")
    captured = capsys.readouterr()
    assert code == EXIT_USAGE
    assert captured.out == ""          # never pollute stdout
    assert "Error:" in captured.err
    assert "agents run --backend ollama" in captured.err
```

Create `tests/test_console_store.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_console_envelope.py tests/test_console_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.console'`

- [ ] **Step 3: Write minimal implementation**

Create empty `agents/console/__init__.py`.

Create `agents/console/envelope.py`:

```python
"""JSON output contract for the ``agents`` console command."""

from __future__ import annotations

import sys

from agents.types import AgentResult

EXIT_OK = 0
EXIT_AGENT_FAILED = 1
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3


def build_envelope(result: AgentResult) -> dict:
    """Render an AgentResult as the machine-readable envelope."""
    run = result.run
    # agy's stdout is raw NDJSON and AgentResult.output is truncated, so a
    # parsed response in metadata always wins.
    output = run.metadata.get("response", result.output)
    return {
        "run_id": run.run_id,
        "backend": run.backend,
        "model": run.model,
        "success": result.success,
        "output": output,
        "error": run.metadata.get("error"),
        "telemetry": {
            "input_tokens": run.llm_input_tokens,
            "output_tokens": run.llm_output_tokens,
            "cache_read_tokens": run.llm_cache_read_tokens,
            "cost_usd": run.llm_total_cost_usd,
            "duration_ms": run.duration_ms,
            "tool_calls": run.tool_call_count,
            "requests": run.llm_request_count,
        },
    }


def usage_error(message: str, example: str) -> int:
    """Report a usage error on stderr with a corrected invocation.

    No envelope: no run happened, so there is no run_id or telemetry to report
    and emitting one would fabricate a run.
    """
    print(f"Error: {message}", file=sys.stderr)
    print(f"  {example}", file=sys.stderr)
    return EXIT_USAGE
```

Create `agents/console/store.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_console_envelope.py tests/test_console_store.py -v && pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/console/ tests/test_console_envelope.py tests/test_console_store.py
git commit -m "feat: add console envelope contract and run store"
```

---

### Task 5: `agents run` and `agents backends list`

The first usable command, plus the entry point wiring.

**Files:**
- Create: `agents/console/registry.py`
- Create: `agents/console/main.py`
- Modify: `pyproject.toml` (add `[project.scripts]`)
- Test: `tests/test_console_run.py`

**Interfaces:**
- Consumes: `build_envelope`, `usage_error`, exit codes (Task 4); `AgyAgent` (Task 2); `OllamaAgent` (Task 3); `Capability` (Task 1)
- Produces:
  - `agents.console.registry.build_registry() -> dict[str, object]` — maps `"agy"`/`"ollama"` to backend instances
  - `agents.console.registry.validate_flags(backend, workspace, files) -> tuple[str, str] | None` — returns `(message, example)` on mismatch, else `None`
  - `agents.console.main.main(argv: list[str] | None = None) -> int`
  - Console script `agents = "agents.console.main:main"`

- [ ] **Step 1: Write the failing test**

Create `tests/test_console_run.py`:

```python
import json

import pytest

from agents.console.main import main
from agents.console.registry import build_registry, validate_flags
from agents.types import AgentResult, AgentRun


def test_registry_has_both_backends():
    reg = build_registry()
    assert set(reg) >= {"agy", "ollama"}


def test_validate_rejects_workspace_for_ollama():
    reg = build_registry()
    problem = validate_flags(reg["ollama"], workspace="/tmp", files=[])
    assert problem is not None
    message, example = problem
    assert "cannot explore a workspace" in message
    assert "-f" in example


def test_validate_rejects_files_for_agy():
    reg = build_registry()
    problem = validate_flags(reg["agy"], workspace=None, files=["a.py"])
    assert problem is not None
    assert "--workspace" in problem[1]


def test_validate_accepts_correct_combinations():
    reg = build_registry()
    assert validate_flags(reg["ollama"], workspace=None, files=["a.py"]) is None
    assert validate_flags(reg["agy"], workspace="/tmp", files=[]) is None


def test_backends_list_prints_capabilities(capsys):
    code = main(["backends", "list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "agy" in out and "ollama" in out
    assert "context_files" in out


def test_run_unknown_backend_is_usage_error(capsys):
    code = main(["run", "--backend", "nope", "--task", "t"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "Error:" in captured.err


def test_run_capability_mismatch_is_usage_error(capsys):
    code = main(["run", "--backend", "ollama", "--workspace", "/tmp", "--task", "t"])
    captured = capsys.readouterr()
    assert code == 2
    assert "-f" in captured.err


def test_run_prints_envelope_on_success(monkeypatch, capsys, tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")

    async def fake_run(self, prompt, config, mcp_client=None, on_progress=None):
        run = AgentRun.create(backend="ollama", prompt=prompt)
        run.model = "gemma4:12b"
        run.llm_input_tokens = 12
        run.llm_output_tokens = 3
        run.mark_completed()
        return AgentResult(success=True, output="answer", run=run)

    from agents.inprocess.ollama import OllamaAgent
    monkeypatch.setattr(OllamaAgent, "run", fake_run)

    code = main(["run", "--backend", "ollama", "-f", str(f), "--task", "what?"])
    env = json.loads(capsys.readouterr().out)

    assert code == 0
    assert env["success"] is True
    assert env["output"] == "answer"
    assert env["telemetry"]["input_tokens"] == 12


def test_run_failed_agent_exits_1(monkeypatch, capsys, tmp_path):
    async def fake_run(self, prompt, config, mcp_client=None, on_progress=None):
        run = AgentRun.create(backend="ollama", prompt=prompt)
        run.mark_failed()
        run.metadata["error"] = "connection refused"
        return AgentResult(success=False, output="", run=run)

    from agents.inprocess.ollama import OllamaAgent
    monkeypatch.setattr(OllamaAgent, "run", fake_run)

    code = main(["run", "--backend", "ollama", "--task", "t"])
    env = json.loads(capsys.readouterr().out)

    assert code == 1
    assert env["success"] is False
    assert env["error"] == "connection refused"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_console_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.console.registry'`

- [ ] **Step 3: Write minimal implementation**

Create `agents/console/registry.py`:

```python
"""Backend registry and flag validation for the console."""

from __future__ import annotations

from agents.cli.agy import AgyAgent
from agents.inprocess.ollama import OllamaAgent
from agents.types import Capability


def build_registry() -> dict:
    """All backends the console can drive, by name."""
    return {"agy": AgyAgent(), "ollama": OllamaAgent()}


def validate_flags(backend, workspace, files) -> tuple[str, str] | None:
    """Check flags against the backend's declared capabilities.

    Returns ``(message, example)`` describing the problem and a corrected
    invocation, or ``None`` when the combination is valid.
    """
    caps = backend.capabilities
    name = backend.name

    if workspace and Capability.WORKSPACE not in caps:
        return (
            f"backend '{name}' has no tools and cannot explore a workspace.",
            f"Inline the files instead:\n"
            f"  agents run --backend {name} -f src/a.py -f src/b.py --task \"...\"",
        )

    if files and Capability.CONTEXT not in caps:
        return (
            f"backend '{name}' explores a workspace with its own tools; "
            f"-f/--file does not apply.",
            f"agents run --backend {name} --workspace . --task \"...\"",
        )

    return None
```

Create `agents/console/main.py`:

```python
"""``agents`` console entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from agents.console.envelope import (
    EXIT_AGENT_FAILED, EXIT_OK, EXIT_UNAVAILABLE,
    build_envelope, usage_error,
)
from agents.console.registry import build_registry, validate_flags
from agents.inprocess.ollama import embed_files
from agents.types import AgentConfig, Capability


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", required=True, help="Backend name (see: agents backends list)")
    parser.add_argument("--task", required=True, help="The task description")
    parser.add_argument("--workspace", help="Directory for backends that explore with their own tools")
    parser.add_argument("-f", "--file", action="append", default=[], help="Inline a file (context-only backends)")
    parser.add_argument("--model", help="Model override")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agents",
        description="Hand a task to an agent backend and get the result back.",
        epilog=(
            "Examples:\n"
            "  agents run --backend ollama -f src/a.py --task \"list the public functions\"\n"
            "  agents run --backend agy --workspace . --task \"where is auth handled?\"\n"
            "  agents backends list\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run",
        help="Run a task to completion and print the result envelope",
        epilog=(
            "Examples:\n"
            "  agents run --backend ollama -f a.py -f b.py --task \"what changed?\"\n"
            "  agents run --backend agy --workspace . --task \"find the retry logic\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_flags(run_p)

    backends_p = sub.add_parser("backends", help="Inspect available backends")
    backends_sub = backends_p.add_subparsers(dest="backends_command", required=True)
    backends_sub.add_parser(
        "list",
        help="List backends and their capabilities",
        epilog="Examples:\n  agents backends list\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser


def _cmd_backends_list(registry: dict) -> int:
    for name, backend in sorted(registry.items()):
        caps = ", ".join(sorted(c.value for c in backend.capabilities))
        state = "available" if backend.available() else "unavailable"
        print(f"{name:<8} {state:<12} {caps}")
    return EXIT_OK


def _cmd_run(args, registry: dict) -> int:
    backend = registry.get(args.backend)
    if backend is None:
        known = ", ".join(sorted(registry))
        return usage_error(
            f"unknown backend '{args.backend}'. Known backends: {known}.",
            f"agents run --backend {sorted(registry)[0]} --task \"...\"",
        )

    problem = validate_flags(backend, args.workspace, args.file)
    if problem is not None:
        return usage_error(*problem)

    if not backend.available():
        print(
            f"Error: backend '{args.backend}' is not available on this machine.",
            file=sys.stderr,
        )
        return EXIT_UNAVAILABLE

    env = {}
    if args.model:
        env["AGY_MODEL" if args.backend == "agy" else "OLLAMA_MODEL"] = args.model

    config = AgentConfig(
        workspace=Path(args.workspace or "."),
        timeout_seconds=args.timeout,
        env=env,
    )

    prompt = args.task
    if args.file:
        context = embed_files([Path(p) for p in args.file])
        prompt = f"{context}\n\n===== QUESTION =====\n{args.task}"

    if Capability.CONTEXT in backend.capabilities:
        result = asyncio.run(backend.run(prompt, config))
    else:
        result = asyncio.run(_run_cli_backend(backend, prompt, config))

    print(json.dumps(build_envelope(result), indent=2))
    return EXIT_OK if result.success else EXIT_AGENT_FAILED


async def _run_cli_backend(backend, prompt: str, config: AgentConfig):
    run = await backend.spawn(prompt, config)
    return await backend.wait(run)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    registry = build_registry()

    if args.command == "backends":
        return _cmd_backends_list(registry)
    if args.command == "run":
        return _cmd_run(args, registry)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
```

Add to `pyproject.toml`:

```toml
[project.scripts]
agents = "agents.console.main:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e ".[test,ollama]" && pytest tests/test_console_run.py -v && pytest -q`
Expected: all PASS. Then confirm the entry point: `agents backends list`

- [ ] **Step 5: Commit**

```bash
git add agents/console/registry.py agents/console/main.py pyproject.toml tests/test_console_run.py
git commit -m "feat: add agents run and backends list commands"
```

---

### Task 6: Background lifecycle — `start`, `status`, `cancel`, `logs`

**Files:**
- Create: `agents/console/lifecycle.py`
- Modify: `agents/console/main.py` (register the four subcommands)
- Test: `tests/test_console_lifecycle.py`

**Interfaces:**
- Consumes: `RunStore` (Task 4); `build_registry` (Task 5)
- Produces:
  - `agents.console.lifecycle.start_run(args, store: RunStore) -> int`
  - `agents.console.lifecycle.status_run(run_id: str, store: RunStore) -> int`
  - `agents.console.lifecycle.cancel_run(run_id: str, store: RunStore) -> int`
  - `agents.console.lifecycle.show_logs(run_id: str, store: RunStore, follow: bool) -> int`
  - Log files live at `<store.root>/logs/<run_id>.log`

- [ ] **Step 1: Write the failing test**

Create `tests/test_console_lifecycle.py`:

```python
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
    log_path.write_text("line one\nline two\n")

    code = show_logs(run_id, store, follow=False)
    assert code == 0
    assert "line two" in capsys.readouterr().out


def test_logs_missing_run_is_usage_error(tmp_path, capsys):
    code = show_logs("nope", RunStore(root=tmp_path), follow=False)
    assert code == 2
    assert "Error:" in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_console_lifecycle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.console.lifecycle'`

- [ ] **Step 3: Write minimal implementation**

Create `agents/console/lifecycle.py`:

```python
"""Background run lifecycle: start, status, cancel, logs.

``start`` re-invokes ``agents run`` as a detached process. One mechanism for
every backend, so the in-process ollama backend gets background execution
without a second code path.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from agents.console.envelope import EXIT_OK, usage_error
from agents.console.store import RunStore
from agents.types import AgentRun


def _spawn_detached(cmd: list[str], log_path: Path) -> int:
    """Start the command detached, redirecting output to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
    )
    return process.pid


def _rebuild_run_command(args) -> list[str]:
    cmd = [sys.executable, "-m", "agents.console.main", "run",
           "--backend", args.backend, "--task", args.task]
    if args.workspace:
        cmd += ["--workspace", args.workspace]
    for f in args.file:
        cmd += ["-f", f]
    if args.model:
        cmd += ["--model", args.model]
    cmd += ["--timeout", str(args.timeout)]
    return cmd


def start_run(args, store: RunStore) -> int:
    key = getattr(args, "idempotency_key", None)
    if key:
        existing = store.find_by_key(key)
        if existing:
            print(json.dumps({"run_id": existing, "reused": True}, indent=2))
            return EXIT_OK

    run = AgentRun.create(backend=args.backend, prompt=args.task)
    log_path = store.root / "logs" / f"{run.run_id}.log"
    run.pid = _spawn_detached(_rebuild_run_command(args), log_path)
    run.transcript_path = log_path
    store.save(run)

    if key:
        store.save_key(key, run.run_id)

    print(json.dumps({"run_id": run.run_id, "reused": False}, indent=2))
    return EXIT_OK


def status_run(run_id: str, store: RunStore) -> int:
    data = store.load(run_id)
    if data is None:
        return usage_error(
            f"no run with id '{run_id}'.",
            "agents start --backend ollama --task \"...\"   # returns a run_id",
        )
    print(json.dumps(data, indent=2))
    return EXIT_OK


def cancel_run(run_id: str, store: RunStore) -> int:
    data = store.load(run_id)
    if data is None:
        return usage_error(
            f"no run with id '{run_id}'.",
            "agents status <run_id>",
        )

    # Cancelling a finished run is a no-op that reports the terminal status,
    # because orchestrators retry and a hard error there is noise.
    if data.get("status") in {"completed", "failed", "cancelled", "timed_out"}:
        print(json.dumps({"run_id": run_id, "status": data["status"],
                          "cancelled": False}, indent=2))
        return EXIT_OK

    pid = data.get("pid")
    cancelled = False
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            cancelled = True
        except (ProcessLookupError, PermissionError):
            cancelled = False

    data["status"] = "cancelled"
    (store.runs_dir / f"{run_id}.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "status": "cancelled",
                      "cancelled": cancelled}, indent=2))
    return EXIT_OK


def show_logs(run_id: str, store: RunStore, follow: bool) -> int:
    data = store.load(run_id)
    if data is None:
        return usage_error(
            f"no run with id '{run_id}'.",
            "agents status <run_id>",
        )

    log_path = store.root / "logs" / f"{run_id}.log"
    if not log_path.exists():
        print("")
        return EXIT_OK

    if not follow:
        print(log_path.read_text(encoding="utf-8"), end="")
        return EXIT_OK

    # agy streams step_update events, so following shows progress live.
    # ollama emits nothing until it finishes, so this simply blocks to the end.
    with log_path.open("r", encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if line:
                print(line, end="")
                continue
            current = store.load(run_id) or {}
            if current.get("status") in {"completed", "failed", "cancelled", "timed_out"}:
                print(fh.read(), end="")
                return EXIT_OK
            time.sleep(0.2)
```

In `agents/console/main.py`, register the subcommands inside `_build_parser()` before the `return`:

```python
    start_p = sub.add_parser(
        "start",
        help="Start a task in the background and return a run_id",
        epilog=(
            "Examples:\n"
            "  agents start --backend agy --workspace . --task \"audit error handling\"\n"
            "  agents start --backend ollama -f a.py --task \"...\" --idempotency-key job-42\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_run_flags(start_p)
    start_p.add_argument(
        "--idempotency-key",
        help="Reuse the existing run_id if this key was already started",
    )

    status_p = sub.add_parser(
        "status", help="Show the state of a run",
        epilog="Examples:\n  agents status r_8f3a\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    status_p.add_argument("run_id")

    cancel_p = sub.add_parser(
        "cancel", help="Cancel a running task",
        epilog="Examples:\n  agents cancel r_8f3a\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    cancel_p.add_argument("run_id")

    logs_p = sub.add_parser(
        "logs", help="Show captured output for a run",
        epilog="Examples:\n  agents logs r_8f3a\n  agents logs r_8f3a --follow\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    logs_p.add_argument("run_id")
    logs_p.add_argument("--follow", action="store_true", help="Stream until the run finishes")
```

And dispatch in `main()`, before the final `return EXIT_OK`:

```python
    if args.command == "start":
        return start_run(args, RunStore())
    if args.command == "status":
        return status_run(args.run_id, RunStore())
    if args.command == "cancel":
        return cancel_run(args.run_id, RunStore())
    if args.command == "logs":
        return show_logs(args.run_id, RunStore(), args.follow)
```

with `from agents.console.lifecycle import cancel_run, show_logs, start_run, status_run` and `from agents.console.store import RunStore` added to the imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_console_lifecycle.py -v && pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/console/lifecycle.py agents/console/main.py tests/test_console_lifecycle.py
git commit -m "feat: add background run lifecycle commands"
```

---

### Task 7: `agents stats` and documentation

Closes the loop the whole project exists for: making delegation savings measurable.

**Files:**
- Create: `agents/console/stats.py`
- Modify: `agents/console/main.py` (register `stats`; write the completed run to the ledger in `_cmd_run`)
- Modify: `README.md` (console section)
- Test: `tests/test_console_stats.py`

**Interfaces:**
- Consumes: `RunStore` (Task 4)
- Produces:
  - `agents.console.stats.summarise(entries: list[dict]) -> dict`
  - `agents.console.stats.stats_command(store: RunStore, since: str | None, backend: str | None, fmt: str) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_console_stats.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_console_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.console.stats'`

- [ ] **Step 3: Write minimal implementation**

Create `agents/console/stats.py`:

```python
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
```

In `agents/console/main.py`, register the subcommand in `_build_parser()`:

```python
    stats_p = sub.add_parser(
        "stats",
        help="Summarise past runs from the ledger",
        epilog=(
            "Examples:\n"
            "  agents stats\n"
            "  agents stats --since 7d --backend ollama\n"
            "  agents stats --format table\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    stats_p.add_argument("--since", help="Window, e.g. 7d, 12h, 30m")
    stats_p.add_argument("--backend", help="Only this backend")
    stats_p.add_argument("--format", dest="fmt", choices=["json", "table"], default="json")
```

Dispatch in `main()`:

```python
    if args.command == "stats":
        return stats_command(RunStore(), args.since, args.backend, args.fmt)
```

Record completed runs. At the end of `_cmd_run`, immediately before the `print(...)`:

```python
    RunStore().append_ledger(result.run)
```

Add imports: `from agents.console.stats import stats_command`.

Add to `README.md` after the install section:

````markdown
## Console

`pip install agents` provides an `agents` command for handing a task to a
backend and getting the result back as a JSON envelope.

```bash
agents backends list
agents run --backend ollama -f src/a.py --task "list the public functions"
agents run --backend agy --workspace . --task "where is retry handled?"

agents start --backend agy --workspace . --task "audit error handling"
agents status <run_id>
agents logs <run_id> --follow
agents cancel <run_id>

agents stats --since 7d --format table
```

Backends declare what input they accept. `agy` explores a workspace with its
own tools; `ollama` has no tools, so files must be inlined with `-f`. Passing
the wrong flag fails immediately with a corrected invocation.

Exit codes: `0` success, `1` the agent ran and failed, `2` usage error,
`3` backend unavailable. On a usage error nothing is written to stdout.

The `ollama` backend needs its extra: `pip install "agents[ollama]"`.
````

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_console_stats.py -v && pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/console/stats.py agents/console/main.py README.md tests/test_console_stats.py
git commit -m "feat: add agents stats and console documentation"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Command surface | 5, 6, 7 |
| Module layout | 4, 5 |
| Capability model | 1 (enum + default), 5 (validation) |
| Output contract / exit codes | 4, 5 |
| Background lifecycle | 6 |
| `stats` format | 7 |
| `logs` per-backend semantics | 6 |
| agy verified shape + telemetry mapping | 2 |
| agy three traps | 2 |
| agy timeout headroom | 2 (`--print-timeout` from `timeout_seconds`); base class already truncates |
| ollama backend + two traps | 3 |
| Testing plan | every task |

**Deviations from the spec, deliberate:**
- Spec wrote `class Capability(StrEnum)`. Changed to `class Capability(str, Enum)` — `StrEnum` is 3.11+ and the project floor is 3.10.
- Spec did not mention `httpx`. It is added as an `ollama` optional extra to preserve `dependencies = []`.
- The agy silent-failure verdict is implemented inside `parse_output` rather than as a base-class change, exploiting the existing `if not run.is_terminal` guard at `agents/cli/base.py:292`.
