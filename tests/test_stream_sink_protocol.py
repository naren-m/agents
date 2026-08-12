"""Protocol-level tests for ``agents.stream.base.StreamSink``.

StreamSink is the abstract contract any stream target must implement.
These tests assert shape (not behaviour) so the protocol stays stable
across concrete implementations.
"""

import inspect
from pathlib import Path

import pytest


class _MinimalSink:
    """Smallest concrete implementation that satisfies the protocol."""

    async def open(self, path: Path) -> None:
        self._opened = path

    async def write_line(self, line: bytes) -> None:
        self._line = line

    async def close(self) -> None:
        self._closed = True


def test_streamsink_is_runtime_checkable():
    from agents.stream.base import StreamSink

    assert isinstance(_MinimalSink(), StreamSink)


def test_streamsink_declares_expected_methods():
    from agents.stream.base import StreamSink

    assert "open" in StreamSink.__dict__
    assert "write_line" in StreamSink.__dict__
    assert "close" in StreamSink.__dict__


def test_streamsink_methods_are_async():
    from agents.stream.base import StreamSink

    for name in ("open", "write_line", "close"):
        method = getattr(StreamSink, name)
        assert inspect.iscoroutinefunction(method), (
            f"{name} must be async"
        )


@pytest.mark.asyncio
async def test_minimal_sink_satisfies_contract(tmp_path):
    sink = _MinimalSink()
    path = tmp_path / "t.log"

    await sink.open(path)
    assert sink._opened == path

    await sink.write_line(b"hello\n")
    assert sink._line == b"hello\n"

    await sink.close()
    assert sink._closed is True
