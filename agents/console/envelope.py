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
