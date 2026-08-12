"""Tests for ``agents.http`` -- the optional FastAPI router.

The router is deliberately framework-agnostic at the surface: a caller
hands it an :class:`EventBusRegistry` and mounts the returned
``APIRouter`` wherever it wants (the dashboard, a standalone service,
or a unit test).  These tests hit the router via FastAPI's test
client so they exercise the HTTP contract end-to-end, not the Python
API.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture
def registry():
    from agents.event_bus import EventBusRegistry

    return EventBusRegistry()


@pytest.fixture
def app(registry):
    from fastapi import FastAPI

    from agents.http import build_router

    app = FastAPI()
    app.include_router(build_router(registry), prefix="/api/agents")
    return app


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


class TestSummaryEndpoint:
    def test_summary_unknown_run_returns_404(self, client):
        resp = client.get("/api/agents/unknown-run/summary")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_summary_returns_metadata(self, registry, client):
        bus = registry.get_or_create("run-abc", ring_buffer_bytes=512)
        await bus.publish_event("run.started", {"backend": "fake"})
        await bus.append_log(b"hello\n")

        resp = client.get("/api/agents/run-abc/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "run-abc"
        assert body["closed"] is False
        assert body["event_count"] >= 1
        assert body["log_bytes_available"] >= len(b"hello\n")


class TestEventsEndpoint:
    @pytest.mark.asyncio
    async def test_events_returns_history(self, registry, client):
        bus = registry.get_or_create("run-e1")
        await bus.publish_event("run.started", {"backend": "f"})
        await bus.publish_event("task.progress", {"step": 1})

        resp = client.get("/api/agents/run-e1/events")
        assert resp.status_code == 200
        events = resp.json()["events"]
        types = [e["type"] for e in events]
        assert "run.started" in types
        assert "task.progress" in types

    @pytest.mark.asyncio
    async def test_events_since_filters(self, registry, client):
        bus = registry.get_or_create("run-e2")
        await bus.publish_event("a", {})
        await asyncio.sleep(0.001)
        cutoff_event = bus.events_since(None)[-1]
        await bus.publish_event("b", {})

        resp = client.get(
            "/api/agents/run-e2/events",
            params={"since": cutoff_event["ts"]},
        )
        events = resp.json()["events"]
        assert [e["type"] for e in events] == ["b"]

    def test_events_unknown_run_returns_404(self, client):
        resp = client.get("/api/agents/missing/events")
        assert resp.status_code == 404


class TestEventStream:
    """SSE live push is exercised at two layers:

    1. :func:`agents.http._sse_iter` is unit-tested directly against a
       bus -- single event loop, zero HTTP plumbing.  This is where
       live-push behaviour actually matters.
    2. The FastAPI route is smoke-tested against a *closed* bus so the
       stream completes quickly.  That exercises status/content-type/
       framing without depending on the transport layer's streaming
       semantics, which differ widely (``TestClient`` uses a thread
       bridge, ``httpx.ASGITransport`` buffers until the generator
       completes).
    """

    @pytest.mark.asyncio
    async def test_sse_iter_formats_live_events(self, registry):
        from agents.http import _sse_iter

        bus = registry.get_or_create("run-s1")
        await bus.publish_event("run.started", {"backend": "f"})

        chunks: list[bytes] = []

        async def _drain() -> None:
            async for chunk in _sse_iter(bus):
                chunks.append(chunk)

        consumer = asyncio.create_task(_drain())
        await asyncio.sleep(0)  # let subscribe() replay history

        await bus.publish_event("task.progress", {"n": 1})
        await asyncio.sleep(0)
        await bus.close()
        await asyncio.wait_for(consumer, timeout=1.0)

        body = b"".join(chunks).decode("utf-8")
        assert "event: run.started" in body
        assert "event: task.progress" in body
        # ``bus.closed`` sentinel is the stream's terminator.
        assert "event: bus.closed" in body

    @pytest.mark.asyncio
    async def test_stream_endpoint_replays_history_on_closed_bus(
        self, registry, client,
    ):
        """Route smoke test: headers + SSE framing for historical events.

        The bus is closed *before* the request, so the response is
        deterministic and completes without live push.  Live push is
        covered by :meth:`test_sse_iter_formats_live_events`.
        """
        bus = registry.get_or_create("run-s2")
        await bus.publish_event("run.started", {"backend": "f"})
        await bus.publish_event("task.progress", {"n": 1})
        await bus.close()

        with client.stream("GET", "/api/agents/run-s2/events/stream") as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith(
                "text/event-stream"
            )
            body = b"".join(resp.iter_raw())

        text = body.decode("utf-8")
        assert "event: run.started" in text
        assert "event: task.progress" in text
        assert "event: bus.closed" in text


class TestLogEndpoint:
    @pytest.mark.asyncio
    async def test_log_returns_bytes_and_offset(self, registry, client):
        bus = registry.get_or_create("run-l1")
        await bus.append_log(b"first\n")
        await bus.append_log(b"second\n")

        resp = client.get("/api/agents/run-l1/log")
        assert resp.status_code == 200
        body = resp.json()
        assert b"first" in body["data"].encode()
        assert b"second" in body["data"].encode()
        assert body["next_offset"] >= len(b"first\nsecond\n")

    @pytest.mark.asyncio
    async def test_log_from_offset_skips_consumed(self, registry, client):
        bus = registry.get_or_create("run-l2")
        await bus.append_log(b"aaa\n")
        first = client.get("/api/agents/run-l2/log").json()

        await bus.append_log(b"bbb\n")
        second = client.get(
            "/api/agents/run-l2/log",
            params={"from_offset": first["next_offset"]},
        ).json()
        assert b"bbb" in second["data"].encode()
        assert b"aaa" not in second["data"].encode()

    def test_log_unknown_run_returns_404(self, client):
        resp = client.get("/api/agents/missing/log")
        assert resp.status_code == 404


class TestBuildRouter:
    def test_fastapi_missing_raises_helpful_error(self, monkeypatch):
        """If FastAPI is unavailable, build_router should say so clearly."""
        import builtins
        import sys

        real_import = builtins.__import__

        def _fail_import(name, *a, **kw):
            if name == "fastapi":
                raise ModuleNotFoundError("No module named 'fastapi'")
            return real_import(name, *a, **kw)

        saved = sys.modules.pop("agents.http", None)
        monkeypatch.setattr(builtins, "__import__", _fail_import)
        try:
            with pytest.raises(ModuleNotFoundError, match="fastapi"):
                __import__("agents.http").http  # noqa: F401
        finally:
            if saved is not None:
                sys.modules["agents.http"] = saved
