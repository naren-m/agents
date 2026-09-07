# agents

`agents` is a lightweight Python wrapper for invoking command-line AI agents from
agentic workflows. It gives orchestrators one async API to discover installed
agent CLIs, start tasks, stream output, await results, cancel runs, and collect
telemetry without coupling workflow code to one vendor.

Built-in adapters support Cursor Agent, Codex CLI, and GitHub Copilot CLI. Custom
CLI and in-process backends can use the same lifecycle and result types.

## Install

From GitHub:

```bash
pip install "agents @ git+https://github.com/naren-m/agents.git"
```

From local checkout:

```bash
pip install .
```

Editable for development:

```bash
pip install -e ".[test]"
```

> **Troubleshooting: URL parse error on install**
>
> If you get `ValueError: 'github.com)' does not appear to be an IPv4 or IPv6 address`,
> the URL was corrupted by rich-text copy-paste. Slack, email clients, and some browsers
> auto-link `git@github.com` as a mailto, turning it into
> `[git@github.com](mailto:git@github.com)` when pasted. **Type or re-copy the command
> from a plain-text source** (raw markdown, not rendered).
>
> Alternatively, clone and install locally to avoid the URL entirely:
> ```bash
> git clone https://github.com/naren-m/agents.git /tmp/agents
> cd /tmp/agents && pip install .
> ```

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
`3` backend unavailable. On a usage error nothing is written to stdout, so a
caller can always parse stdout as the envelope when it is non-empty.

Every finished run is appended to a ledger under `~/.agents` (override with
`AGENTS_HOME`), which is what `agents stats` reads. Local ollama runs record a
cost of `0.0` against real token counts, which is what makes the saving from
delegating work legible rather than assumed.

The `ollama` backend needs its extra: `pip install "agents[ollama]"`.

## Quick Start

```python
from pathlib import Path
from agents import AgentConfig, AgentManager
from agents.cli.codex import CodexAgent
from agents.cli.cursor import CursorAgent

manager = AgentManager(
    config=AgentConfig(
        workspace=Path.cwd(),
        timeout_seconds=900,
    ),
    preferred_backend="auto",
)

manager.register_cli(CursorAgent())
manager.register_cli(CodexAgent())

# Spawn an agent
run = await manager.spawn("Analyze this project")
print(f"Started: {run.run_id}, backend={run.backend}, pid={run.pid}")

# Check status
run = manager.get_run(run.run_id)
print(f"Status: {run.status}")

# Wait for completion
result = await manager.wait(run.run_id)
print(f"Success: {result.success}, tokens: {result.run.llm_input_tokens}")

# Or cancel
cancelled = await manager.cancel(run.run_id)
```

## Architecture

Two plugin categories, one unified API:

```
AgentManager
  |
  +-- CLI agents (subprocess-based)
  |     +-- CursorAgent    (cursor-agent / agent binary)
  |     +-- CopilotAgent   (copilot binary)
  |     +-- CodexAgent     (codex binary)
  |
  +-- In-process agents (Python-native)
        +-- LangGraphAgent  (stub)
```

### Backend Selection

| Value | Behavior |
|-------|----------|
| `auto` | First available CLI backend, then first in-process (default) |
| `cursor` | Use CursorAgent. Fails if binary not on PATH |
| `copilot` | Use CopilotAgent. Fails if binary not on PATH |
| `codex` | Use CodexAgent. Fails if binary not on PATH |
| `langgraph` | Use LangGraphAgent. Fails if deps not installed |

## Key Types

- **`AgentConfig`** -- workspace, timeout, transcript dir, env vars, event bus
- **`AgentRun`** -- tracks a run: id, backend, status, PID, timestamps, telemetry
- **`AgentResult`** -- final output: success, text, run with filled telemetry
- **`AgentManager`** -- registry, backend selection, spawn/wait/cancel, concurrency (max 2)

## Custom Backends

Register your own CLI or in-process backend:

```python
from agents.cli.base import BaseCLIAgent

class MyAgent(BaseCLIAgent):
    name = "my-agent"
    _binary_names = ["my-agent-bin"]

    def build_command(self, prompt, config):
        return [self.binary_path(), "--prompt", prompt,
                "--workspace", str(config.workspace)]

manager.register_cli(MyAgent())
```

## Tests

```bash
cd agents
uv run --extra test pytest tests/ -v
```

Tests cover types, CLI protocol, agent backends, streaming, HTTP routing, and AgentManager.

## Zero Dependencies

Pure Python stdlib only (asyncio, subprocess, shutil, dataclasses, pathlib). In-process backends import their frameworks conditionally in `available()`.
