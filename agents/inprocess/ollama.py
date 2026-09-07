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
            for i, line in enumerate(
                path.read_text(errors="replace").splitlines(), 1
            )
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
