# Agent console CLI — design

Date: 2026-09-06
Status: approved, not yet implemented

## Purpose

Give `agents` a console entry point so an orchestrator (Claude Code, a script, CI) can hand a task
to a backend, get the finished output back, and move on. Two new backends come with it: `agy`
(Google Antigravity CLI) and `ollama` (local models).

The motivating problem is measurement. Delegating work to a cheaper backend is currently an act of
faith — nothing records what a delegation cost or saved. `AgentRun` already carries
`llm_input_tokens`, `llm_output_tokens`, `llm_total_cost_usd` and `duration_ms`. Routing
delegation through this library turns "did that help?" into a query.

## Non-goals

- **No routing policy.** Which task goes to which backend is the caller's decision and lives in the
  caller's config. This repo provides the skeleton, not the judgement.
- **No auto-router.** Explicit `--backend` only. Route on evidence once telemetry accumulates.
- **No daemon.** Background runs are detached processes, not a managed service.

## Command surface

```
agents run      --backend <name> --task <text> [--workspace DIR | -f FILE ...] [--model M]
agents start    --backend <name> --task <text> [...]        -> {"run_id": "..."}
agents status   <run_id>
agents cancel   <run_id>
agents logs     <run_id> [--follow]
agents backends list
agents stats    [--since 7d] [--backend name]
```

`resource + verb` shape, every subcommand carries `--help` with real examples, every input is a
flag. Nothing prompts.

## Module layout

```
agents/
  cli/          agent backends that ARE CLIs      + agy.py
  inprocess/    Python-native backends            + ollama.py
  console/      the `agents` command              (new)
    main.py         dispatch
    commands/       run, start, status, cancel, logs, backends, stats
    envelope.py     JSON output contract
    store.py        run-state persistence
```

`agents/cli/` keeps its existing meaning (backends that are themselves CLIs). The console entry
point is deliberately named `console/` to avoid that collision. `[project.scripts]` gains
`agents = "agents.console.main:main"`.

`manager.py`, `types.py` and `event_bus.py` are untouched apart from the additive capability field
below.

## Capability model

Backends declare what kind of input they accept:

```python
class Capability(StrEnum):
    WORKSPACE = "workspace"      # can be pointed at a directory and explore it
    TOOLS     = "tools"          # has its own file/shell tools
    MCP       = "mcp"            # can attach MCP servers
    CONTEXT   = "context_files"  # has no tools; content must be inlined
```

| Backend | Capabilities |
|---|---|
| `BaseCLIAgent` (default) | `WORKSPACE, TOOLS` |
| `agy` | `WORKSPACE, TOOLS, MCP` |
| `ollama` | `CONTEXT` |

The default on `BaseCLIAgent` keeps cursor/codex/copilot working with no change. The CLI validates
flags against the selected backend and fails fast with a corrected invocation:

```
$ agents run --backend ollama --workspace ~/p --task "..."
Error: backend 'ollama' has no tools and cannot explore a workspace.
  Inline the files instead:
  agents run --backend ollama -f src/a.py -f src/b.py --task "..."
```

This is deliberately not papered over. ollama is a completion endpoint, not an agent; pretending
otherwise would mean silently walking a repo and truncating it to fit `num_ctx`.

## Output contract

`run` emits a JSON envelope on stdout:

```json
{
  "run_id": "r_8f3a",
  "backend": "ollama",
  "model": "qwen2.5-coder:32b",
  "success": true,
  "output": "...",
  "error": null,
  "telemetry": {
    "input_tokens": 4249,
    "output_tokens": 287,
    "cache_read_tokens": 0,
    "cost_usd": 0.0,
    "duration_ms": 34120,
    "tool_calls": 0,
    "requests": 1
  }
}
```

Failure classes are deliberately distinct:

| Case | stdout | exit |
|---|---|---|
| Agent ran, succeeded | envelope, `success: true` | 0 |
| Agent ran, failed | envelope, `success: false`, `error` set | 1 |
| Usage error (bad flag, capability mismatch) | nothing — human text + example on stderr | 2 |
| Backend unavailable (binary missing) | nothing — install hint on stderr | 3 |

A usage error means no run happened: there is no `run_id` and no telemetry, so emitting an envelope
would fabricate a run. At that moment the caller needs a copy-pasteable fix, which is a line of
text, not JSON.

## Background lifecycle

`start` spawns `agents run …` as a detached subprocess and returns `{"run_id": ...}`. One mechanism
for both backends, so in-process ollama gets background execution without a second code path.

State lives at `~/.agents/runs/<run_id>.json`. Completed runs append to `~/.agents/runs.jsonl`,
which is what `stats` reads. Global rather than per-workspace so `stats` sees everything.

`stats` prints a JSON object by default (totals per backend: runs, input/output tokens, cost,
median duration, success rate) and a table under `--format table` for humans.

`logs` means different things per backend, and the spec is explicit rather than silently uneven:
agy runs stream `step_update` events, so `logs --follow` tails them live. ollama is a single
completion with no intermediate events, so `logs` returns the captured stdout/stderr of the run and
`--follow` blocks until the run reaches a terminal state, then returns. `backends list` reports
which behaviour a backend gives.

`start` is not idempotent by nature (each call is a new run), so it takes `--idempotency-key`; a
repeat with the same key returns the existing `run_id` instead of starting a second run. (This flag
was not in the design discussed in chat — it comes from the `cli-for-agents` idempotency rule, on
the reasoning that orchestrators retry. Drop it if you would rather keep `start` minimal.) `cancel`
on an already-finished run is a no-op that reports the terminal status rather than erroring.

## agy backend

### Verified output shape

`--output-format json` returns a **single JSON object**, not JSON-lines (unlike cursor):

```json
{"conversation_id":"8057e198-...","status":"SUCCESS","response":"...",
 "duration_seconds":51.745,"num_turns":1,
 "usage":{"input_tokens":101932,"output_tokens":3829,"thinking_tokens":2324,
          "cache_read_tokens":320050,"total_tokens":105761}}
```

`--output-format stream-json` returns NDJSON keyed on `event`:

| event | contents |
|---|---|
| `init` | `conversation_id`, `init.model`, `init.cwd` |
| `step_update` | `step_index`, `state`, `step_type` (`user_input`\|`agent_response`\|`tool`), `tool_name`, `tool_info`, `duration_seconds`, `usage` |
| `result` | terminal; same payload as the single-object json format, plus `error` when status is ERROR |

**Use `stream-json` as the default for agy.** It is a strict superset: the terminal `result` event
carries everything the `json` format does, and it additionally provides `tool_call_count` (count of
`step_update` with `step_type == "tool"`) and the resolved model from `init` — neither of which the
`json` format reports at all. It also gives `logs --follow` for free.

### Telemetry mapping

| `AgentRun` field | Source |
|---|---|
| `session_id` | `conversation_id` |
| `llm_input_tokens` | `usage.input_tokens` |
| `llm_output_tokens` | `usage.output_tokens` |
| `llm_cache_read_tokens` | `usage.cache_read_tokens` |
| `llm_request_count` | `num_turns` |
| `duration_ms` | `int(duration_seconds * 1000)` |
| `model` | `init.model` (stream-json); else the requested model |
| `tool_call_count` | count of `step_type == "tool"` (stream-json only) |
| `metadata["thinking_tokens"]` | `usage.thinking_tokens` — no `AgentRun` field exists |
| `llm_total_cost_usd` | `0.0` — agy reports no cost; it bills Google's quota |

### Three traps, verified, to be encoded as behaviour with tests

**1. `--dangerously-skip-permissions` is mandatory in print mode.** Headless agy cannot prompt for
tool approval, so any tool needing one is auto-denied. Verified: without the flag even reading a
file in the cwd fails.

**2. Success cannot be derived from the exit code.** On tool denial agy returns
`status: "SUCCESS"`, exit code 0, and an empty `response`, with the only signals being a
`denied_actions` array and a stderr message. The rule must be:

```
success = status == "SUCCESS" and response.strip() != "" and not denied_actions
```

A naive `exit_code == 0` check reports silent failures as wins.

**3. Sandboxed agy misreports writes.** `--sandbox` means "terminal restrictions", not read-only.
Verified: reads hit real files, writes are silently redirected to
`~/.gemini/antigravity-cli/scratch/` — and the agent then reports "the file has been overwritten"
with a `file://` link while the real file is untouched. `parse_output` annotates this so an
orchestrator cannot be fooled into believing an edit landed.

### Cost note

A trivial one-file task consumed 101,932 input tokens and 320,050 cache-read tokens. agy is not
cheap in absolute terms; its value is that it bills Google's quota rather than the caller's. That
is quota arbitrage, not a token saving — worth keeping straight when reading `stats`.

### Reliability note

One observed stream-json run spent 118s and 54 tool steps searching
`~/.gemini/antigravity-cli/brain` for a file that was in the working directory, then hit
`status: "ERROR"`, `error: "timeout waiting for response"`. Sandboxed agy can flail. Timeouts must
be treated as expected, not exceptional, and `--print-timeout` needs the subprocess timeout to sit
above it (see below).

### Timeout

`--print-timeout Ns` and the subprocess timeout must not be equal, or they fire together and agy's
own error message is lost. Subprocess gets `+30s` headroom.

## ollama backend

`InProcessAgent` (a Protocol — implement the shape, no inheritance): `name`, `available()`,
`async run()`, `async cancel()`. An async POST to `/api/generate`; no subprocess, since forking a
process to make an HTTP request would be dishonest about the lifecycle.

Telemetry: `prompt_eval_count` → `llm_input_tokens`, `eval_count` → `llm_output_tokens`,
`llm_total_cost_usd = 0.0`. That zero is the point — it is what makes savings legible.

Two traps, both already learned in `~/.claude/bin/local-ask`:

**1. `num_ctx` must be set explicitly.** Ollama defaults to 4096 and silently truncates. Sized from
input length, with a warning above the largest supported context.

**2. Files are embedded with line numbers.** Without them the model estimates citations and drifts
5–15 lines. With them, citing is a copy rather than a calculation. Measured: drift before, 10/10
exact after.

Model presets carried over: `code` → `qwen2.5-coder:32b`, `summary` → `gemma3:27b`,
`fast` → `gemma4:12b`.

## Testing

Mirrors the existing per-backend pattern (`test_cursor_agent.py`):

- `test_agy_agent.py` — `build_command` includes the mandatory permission flag; `parse_output` maps
  every telemetry field; the denied-actions case yields `success: false`; the write-claim
  annotation fires on sandboxed runs.
- `test_ollama_agent.py` — `num_ctx` sizing across input sizes, line-numbered embedding, telemetry
  mapping, cancel.
- `test_console_run.py` / `test_console_lifecycle.py` — envelope shape, the four exit codes,
  capability-mismatch error text, idempotency key behaviour.

Fakes throughout; no live network in the test suite.

## Open questions

None blocking. The agy output shape was the one assumption in the original design and has now been
verified against a live run.
