# agents

Pluggable agent backend abstraction for the RCA pipeline. Lets the RCA system spawn and manage different AI agent backends through a single unified API.

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

## Quick Start

```python
from pathlib import Path
from agents import AgentConfig, AgentManager
from agents.cli.cursor import CursorAgent

manager = AgentManager(
    config=AgentConfig(
        workspace=Path.cwd(),
        mcp_server_url="http://localhost:8400/mcp",
        timeout_seconds=900,
    ),
    preferred_backend="auto",
)
manager.register_cli(CursorAgent())

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
  |     +-- CopilotAgent   (stub)
  |     +-- CodexAgent      (stub)
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

- **`AgentConfig`** -- workspace, MCP URL, timeout, transcript dir, env vars
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
