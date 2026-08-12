"""Tests for ``agents.event_bus``: RunEventBus and EventBusRegistry.

RunEventBus is the per-run structured-events + raw-log channel feeding
SSE push and REST pull consumers.  Tests cover event history, live
subscription, ring-buffer overflow, and registry lifecycle.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest


@pytest.mark.asyncio
async def test_publish_event_appends_to_history():
    from agents.event_bus import RunEventBus

    bus = RunEventBus(run_id="r1")
    await bus.publish_event("run.started", {"pid": 1})

    events = bus.events_since()
    assert len(events) == 1
    assert events[0]["type"] == "run.started"
    assert events[0]["run_id"] == "r1"
    assert events[0]["payload"] == {"pid": 1}
    assert "ts" in events[0]


@pytest.mark.asyncio
async def test_events_since_returns_only_newer():
    from agents.event_bus import RunEventBus

    bus = RunEventBus(run_id="r1")
    await bus.publish_event("a", {})
    mid = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    await asyncio.sleep(0.01)
    await bus.publish_event("b", {})

    newer = bus.events_since(mid)
    assert [e["type"] for e in newer] == ["b"]


@pytest.mark.asyncio
async def test_subscribe_receives_live_events():
    from agents.event_bus import RunEventBus

    bus = RunEventBus(run_id="r1")
    received: list[str] = []

    async def consumer():
        async for event in bus.subscribe():
            received.append(event["type"])
            if event["type"] == "done":
                break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)

    await bus.publish_event("a", {})
    await bus.publish_event("done", {})

    await asyncio.wait_for(task, timeout=1.0)
    assert received == ["a", "done"]


@pytest.mark.asyncio
async def test_subscribe_receives_historical_events_on_join():
    """A late subscriber must see events published before subscribe()."""
    from agents.event_bus import RunEventBus

    bus = RunEventBus(run_id="r1")
    await bus.publish_event("first", {})
    await bus.publish_event("second", {})

    received: list[str] = []

    async def consumer():
        async for event in bus.subscribe():
            received.append(event["type"])
            if event["type"] == "third":
                break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    await bus.publish_event("third", {})

    await asyncio.wait_for(task, timeout=1.0)
    assert received == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_subscribe_after_close_gets_synthetic_final_event():
    from agents.event_bus import RunEventBus

    bus = RunEventBus(run_id="r1")
    await bus.publish_event("run.started", {})
    await bus.close()

    async def consumer():
        collected: list[str] = []
        async for event in bus.subscribe():
            collected.append(event["type"])
        return collected

    events = await asyncio.wait_for(consumer(), timeout=1.0)
    assert "run.started" in events
    assert events[-1] == "bus.closed"


@pytest.mark.asyncio
async def test_append_log_records_bytes_and_offset():
    from agents.event_bus import RunEventBus

    bus = RunEventBus(run_id="r1", ring_buffer_bytes=1024)
    await bus.append_log(b"hello\n")

    data, offset, lost = bus.read_buffer(0)
    assert data == b"hello\n"
    assert offset == 6
    assert lost == 0


@pytest.mark.asyncio
async def test_read_buffer_from_offset_returns_only_new():
    from agents.event_bus import RunEventBus

    bus = RunEventBus(run_id="r1", ring_buffer_bytes=1024)
    await bus.append_log(b"abc\n")
    _, offset, _ = bus.read_buffer(0)
    await bus.append_log(b"def\n")

    data, next_offset, lost = bus.read_buffer(offset)
    assert data == b"def\n"
    assert next_offset == offset + 4
    assert lost == 0


@pytest.mark.asyncio
async def test_ring_buffer_overflow_signals_lost_bytes():
    from agents.event_bus import RunEventBus

    bus = RunEventBus(run_id="r1", ring_buffer_bytes=8)
    await bus.append_log(b"0123456789\n")

    data, _, lost = bus.read_buffer(0)
    assert len(data) <= 8
    assert lost >= 3


def test_registry_get_or_create_is_idempotent():
    from agents.event_bus import EventBusRegistry

    reg = EventBusRegistry()
    b1 = reg.get_or_create("r1")
    b2 = reg.get_or_create("r1")
    assert b1 is b2


def test_registry_get_returns_none_for_unknown():
    from agents.event_bus import EventBusRegistry

    reg = EventBusRegistry()
    assert reg.get("missing") is None


@pytest.mark.asyncio
async def test_registry_cleanup_retains_for_grace_period():
    from agents.event_bus import EventBusRegistry

    reg = EventBusRegistry(retention_seconds=0.05)
    bus = reg.get_or_create("r1")
    await bus.close()
    # Within grace period the bus must still be reachable.
    assert reg.get("r1") is bus

    await asyncio.sleep(0.1)
    removed = await reg.cleanup_expired()
    assert removed == 1
    assert reg.get("r1") is None


@pytest.mark.asyncio
async def test_registry_cleanup_ignores_still_open_buses():
    """Only closed buses past the grace period should be removed."""
    from agents.event_bus import EventBusRegistry

    reg = EventBusRegistry(retention_seconds=0.0)
    reg.get_or_create("r1")

    removed = await reg.cleanup_expired()
    assert removed == 0
    assert reg.get("r1") is not None


def test_registry_default_ring_buffer_propagates_to_new_buses():
    """Hosts (the dashboard) configure ring buffer size once at the registry.

    Passing a per-app default at construction time means spawned runs
    pick up that size via ``get_or_create(run_id)`` without the caller
    having to plumb the knob through every call site.
    """
    from agents.event_bus import EventBusRegistry

    reg = EventBusRegistry(ring_buffer_bytes=1024)
    bus = reg.get_or_create("r1")
    assert bus.ring_buffer_bytes == 1024


def test_registry_per_call_ring_buffer_overrides_default():
    """An explicit override at the call site wins over the registry default.

    This keeps existing single-knob call sites working unchanged while
    still letting hosts configure a fleet-wide default.
    """
    from agents.event_bus import EventBusRegistry

    reg = EventBusRegistry(ring_buffer_bytes=1024)
    bus = reg.get_or_create("r1", ring_buffer_bytes=4096)
    assert bus.ring_buffer_bytes == 4096
