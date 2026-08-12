"""Per-run event bus: structured events + raw stdout ring buffer.

Producers (``AgentManager``, the subprocess stdout pump, RCA phase
watcher) publish into the bus.  Consumers (SSE push, REST pull)
subscribe or poll.

The bus is purely in-memory and ephemeral -- durable state lives in
the task-manager DB (events), the transcript file (stdout), and the
memory-bank (RCAs).  Closed buses linger for a configurable grace
period so late consumers (reloaded browser tabs, curl retries) can
still pick up the tail of a run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Deque

logger = logging.getLogger(__name__)


Event = dict[str, Any]

# Default ring-buffer capacity (64 KiB).  Large enough for a minute or
# two of typical agent chatter; overflow is signalled via lost_bytes.
_DEFAULT_RING_BUFFER_BYTES = 64 * 1024

# Default registry retention after close(), in seconds.
_DEFAULT_RETENTION_SECONDS = 300.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RunEventBus:
    """Per-run event + raw-log channel.

    - Structured events (``run.*``, ``phase.*``, ``subagent.*``, ...)
      are retained in memory; subscribers receive live pushes via
      ``asyncio.Queue``.  A closed bus emits a synthetic ``bus.closed``
      event so streaming consumers can terminate cleanly.
    - Raw stdout is retained in a bounded ring buffer plus a monotonic
      byte-offset, so REST clients poll with ``from_offset`` semantics
      and detect dropped ranges via ``lost_bytes``.
    """

    def __init__(
        self,
        run_id: str,
        ring_buffer_bytes: int = _DEFAULT_RING_BUFFER_BYTES,
    ) -> None:
        self.run_id = run_id
        self._events: list[Event] = []
        self._subscribers: list[asyncio.Queue[Event]] = []
        # Each deque entry is one logical line (or the raw chunk we
        # received); the sum of their lengths is ``_ring_size``.
        self._ring: Deque[bytes] = deque()
        self._ring_size = 0
        self._ring_cap = ring_buffer_bytes
        self._byte_offset = 0
        self._closed = False
        self._closed_at: float | None = None

    @property
    def ring_buffer_bytes(self) -> int:
        """Maximum retained bytes in the raw log ring buffer."""
        return self._ring_cap

    async def publish_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append an event and fan it out to live subscribers."""
        event: Event = {
            "ts": _now_iso(),
            "type": event_type,
            "run_id": self.run_id,
            "payload": payload,
        }
        self._events.append(event)
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "event-bus subscriber queue full, dropping event %s",
                    event_type,
                )

    async def append_log(self, line: bytes) -> None:
        """Append raw stdout bytes to the ring buffer."""
        if not line:
            return
        self._ring.append(line)
        self._ring_size += len(line)
        self._byte_offset += len(line)
        while self._ring_size > self._ring_cap and self._ring:
            dropped = self._ring.popleft()
            self._ring_size -= len(dropped)

    def read_buffer(self, from_offset: int = 0) -> tuple[bytes, int, int]:
        """Return ``(data, next_offset, lost_bytes)`` starting at offset.

        If ``from_offset`` points before the ring head (data has aged
        out), ``lost_bytes`` is non-zero so the caller can surface a
        gap marker.
        """
        current_tail = self._byte_offset
        current_head = current_tail - self._ring_size
        if from_offset < current_head:
            lost = current_head - from_offset
            from_offset = current_head
        else:
            lost = 0

        skip = from_offset - current_head
        collected = bytearray()
        for chunk in self._ring:
            if skip >= len(chunk):
                skip -= len(chunk)
                continue
            collected.extend(chunk[skip:])
            skip = 0
        return bytes(collected), current_tail, lost

    def events_since(self, ts: str | None = None) -> list[Event]:
        """Return events strictly newer than ``ts`` (None -> all)."""
        if ts is None:
            return list(self._events)
        return [e for e in self._events if e["ts"] > ts]

    async def subscribe(self) -> AsyncIterator[Event]:
        """Yield events as they are published.

        Historical events are replayed first, then live events flow
        via an internal queue until ``bus.closed`` is observed.
        """
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=1024)
        for event in self._events:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                break
        if self._closed:
            # Ensure a late subscriber on a closed bus always terminates.
            try:
                queue.put_nowait({
                    "ts": _now_iso(),
                    "type": "bus.closed",
                    "run_id": self.run_id,
                    "payload": {},
                })
            except asyncio.QueueFull:
                pass
        self._subscribers.append(queue)
        try:
            while True:
                event = await queue.get()
                yield event
                if event["type"] == "bus.closed":
                    break
        finally:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    async def close(self) -> None:
        """Mark the bus closed and broadcast a sentinel event.

        Safe to call multiple times.  Subscribers stop iteration when
        they observe ``bus.closed``.
        """
        if self._closed:
            return
        self._closed = True
        self._closed_at = time.monotonic()
        sentinel: Event = {
            "ts": _now_iso(),
            "type": "bus.closed",
            "run_id": self.run_id,
            "payload": {},
        }
        self._events.append(sentinel)
        for q in list(self._subscribers):
            try:
                q.put_nowait(sentinel)
            except asyncio.QueueFull:
                pass

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def closed_at(self) -> float | None:
        return self._closed_at


class EventBusRegistry:
    """App-level registry of ``RunEventBus`` instances keyed by run_id.

    Buses live for ``retention_seconds`` after ``close()`` so a browser
    tab loaded just after the run finished can still pick up the tail
    from history.  ``cleanup_expired()`` is intended to be called from
    a periodic background task.
    """

    def __init__(
        self,
        retention_seconds: float = _DEFAULT_RETENTION_SECONDS,
        ring_buffer_bytes: int = _DEFAULT_RING_BUFFER_BYTES,
    ) -> None:
        self._buses: dict[str, RunEventBus] = {}
        self._retention = retention_seconds
        self._default_ring_buffer_bytes = ring_buffer_bytes
        self._lock = asyncio.Lock()

    def get_or_create(
        self,
        run_id: str,
        ring_buffer_bytes: int | None = None,
    ) -> RunEventBus:
        bus = self._buses.get(run_id)
        if bus is None:
            size = (
                ring_buffer_bytes
                if ring_buffer_bytes is not None
                else self._default_ring_buffer_bytes
            )
            bus = RunEventBus(
                run_id=run_id,
                ring_buffer_bytes=size,
            )
            self._buses[run_id] = bus
        return bus

    def get(self, run_id: str) -> RunEventBus | None:
        return self._buses.get(run_id)

    async def cleanup_expired(self) -> int:
        """Drop buses that have been closed longer than the retention window.

        Returns the number of buses removed so callers can log or
        instrument cleanup cadence.
        """
        now = time.monotonic()
        removed = 0
        async with self._lock:
            for run_id, bus in list(self._buses.items()):
                if bus.closed_at is None:
                    continue
                if (now - bus.closed_at) >= self._retention:
                    self._buses.pop(run_id, None)
                    removed += 1
        return removed
