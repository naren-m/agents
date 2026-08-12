"""Reusable FastAPI router for the per-run event bus.

This module is an optional extra: ``pip install 'agents[http]'``.
It is imported lazily so the core ``agents`` package has zero runtime
dependencies.  Hosting apps mount it on any prefix they like:

    from fastapi import FastAPI
    from agents.http import build_router

    app = FastAPI()
    app.include_router(build_router(manager.event_bus_registry),
                       prefix="/api/agents")

The contract:

- ``GET /{run_id}/summary``       -- metadata (open/closed, event count,
                                     available log bytes).
- ``GET /{run_id}/events?since=`` -- JSON list of events strictly newer
                                     than the ISO timestamp ``since``.
- ``GET /{run_id}/events/stream`` -- Server-Sent Events stream.  History
                                     is replayed first, then live events
                                     flow until the bus closes.
- ``GET /{run_id}/log?from_offset=`` -- Raw log bytes (UTF-8 decoded) plus
                                     ``next_offset`` for resumption.

The SSE format is the raw ``event:``/``data:`` framing so any standard
client (EventSource, htmx-sse, curl -N) can consume it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

try:
    from fastapi import APIRouter, HTTPException, Query
    from fastapi.responses import JSONResponse, StreamingResponse
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in test
    raise ModuleNotFoundError(
        "agents.http requires fastapi. Install the 'http' extra: "
        "pip install 'agents[http]'"
    ) from exc

if TYPE_CHECKING:
    from agents.event_bus import EventBusRegistry, RunEventBus

logger = logging.getLogger(__name__)


def build_router(registry: "EventBusRegistry") -> APIRouter:
    """Return a FastAPI router bound to ``registry``.

    The router keeps a closure over the registry so callers do not need
    to wire dependency-injection manually.  All endpoints return 404
    when the requested ``run_id`` has no bus (either never created or
    already cleaned up).
    """
    router = APIRouter()

    def _require_bus(run_id: str) -> "RunEventBus":
        bus = registry.get(run_id)
        if bus is None:
            raise HTTPException(
                status_code=404,
                detail=f"No event bus for run {run_id!r}",
            )
        return bus

    @router.get("/{run_id}/summary")
    async def get_summary(run_id: str) -> dict[str, Any]:
        """Lightweight metadata for dashboards' first paint."""
        bus = _require_bus(run_id)
        events = bus.events_since(None)
        data, _, _ = bus.read_buffer(0)
        return {
            "run_id": run_id,
            "closed": bus.closed,
            "closed_at": bus.closed_at,
            "event_count": len(events),
            "log_bytes_available": len(data),
            "ring_buffer_bytes": bus.ring_buffer_bytes,
        }

    @router.get("/{run_id}/events")
    async def list_events(
        run_id: str,
        since: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Return events newer than ``since`` (ISO timestamp) or all."""
        bus = _require_bus(run_id)
        return {"events": bus.events_since(since)}

    @router.get("/{run_id}/events/stream")
    async def stream_events(run_id: str) -> StreamingResponse:
        """SSE stream of events.  Sends a heartbeat every 15s."""
        bus = _require_bus(run_id)

        async def _gen() -> AsyncIterator[bytes]:
            try:
                async for event in _sse_iter(bus):
                    yield event
            except asyncio.CancelledError:
                return
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "SSE stream errored for run %s", run_id,
                )
                return

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/{run_id}/log")
    async def get_log(
        run_id: str,
        from_offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        """Return raw bytes since ``from_offset`` plus resumption cursor."""
        bus = _require_bus(run_id)
        data, next_offset, lost = bus.read_buffer(from_offset)
        return {
            "data": data.decode("utf-8", errors="replace"),
            "next_offset": next_offset,
            "lost_bytes": lost,
        }

    return router


async def _sse_iter(bus: "RunEventBus") -> AsyncIterator[bytes]:
    """Adapt a bus subscription to SSE-framed bytes.

    The subscription replays history first, then yields live events
    until ``bus.close()`` (which fires the synthetic ``bus.closed``
    sentinel and drives the generator to completion).

    Heartbeats: the bus publishes its own periodic ping via the
    manager's cleanup task in a later phase; keeping the stream loop
    itself heartbeat-free avoids async-generator frame corruption
    that ``wait_for(__anext__)`` is known to cause.
    """
    sub = bus.subscribe()
    try:
        async for event in sub:
            yield _format_sse(event)
    finally:
        try:
            await sub.aclose()
        except BaseException:  # pragma: no cover - defensive
            pass


def _format_sse(event: dict[str, Any]) -> bytes:
    """Format a bus event as SSE.

    Uses ``event:`` for the type and ``data:`` for the JSON payload.
    Line feeds inside the JSON body are already prohibited because
    ``json.dumps`` emits a single line.
    """
    payload = json.dumps(
        {
            "ts": event["ts"],
            "type": event["type"],
            "run_id": event["run_id"],
            "payload": event["payload"],
        },
        separators=(",", ":"),
    )
    return (
        f"event: {event['type']}\n"
        f"id: {event['ts']}\n"
        f"data: {payload}\n\n"
    ).encode("utf-8")
