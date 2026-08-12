"""Tests for ``agents.stream.file_sink.FileStreamSink``.

FileStreamSink is the default StreamSink implementation.  These tests
cover unbuffered writes (``tail -f`` visibility), idempotent close,
error handling, and binary-safety.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.stream.base import StreamSink


@pytest.mark.asyncio
async def test_file_stream_sink_is_streamsink():
    from agents.stream.file_sink import FileStreamSink

    assert isinstance(FileStreamSink(), StreamSink)


@pytest.mark.asyncio
async def test_write_line_appends_unbuffered(tmp_path: Path):
    from agents.stream.file_sink import FileStreamSink

    path = tmp_path / "t.log"
    sink = FileStreamSink()
    await sink.open(path)
    try:
        await sink.write_line(b"first line\n")
        # Another reader (or ``tail -f``) must see the line immediately,
        # before close().
        assert path.read_bytes() == b"first line\n"
        await sink.write_line(b"second\n")
        assert path.read_bytes() == b"first line\nsecond\n"
    finally:
        await sink.close()


@pytest.mark.asyncio
async def test_close_is_idempotent(tmp_path: Path):
    from agents.stream.file_sink import FileStreamSink

    sink = FileStreamSink()
    await sink.open(tmp_path / "t.log")
    await sink.close()
    await sink.close()


@pytest.mark.asyncio
async def test_write_before_open_raises():
    from agents.stream.file_sink import FileStreamSink

    sink = FileStreamSink()
    with pytest.raises(RuntimeError, match="not open"):
        await sink.write_line(b"nope\n")


@pytest.mark.asyncio
async def test_writes_binary_safely(tmp_path: Path):
    """Non-UTF-8 bytes must round-trip without transformation."""
    from agents.stream.file_sink import FileStreamSink

    path = tmp_path / "t.log"
    sink = FileStreamSink()
    await sink.open(path)
    try:
        await sink.write_line(b"\xff\xfe\x00\n")
    finally:
        await sink.close()
    assert path.read_bytes() == b"\xff\xfe\x00\n"


@pytest.mark.asyncio
async def test_open_creates_parent_dirs(tmp_path: Path):
    """Nested transcript paths should be created on demand."""
    from agents.stream.file_sink import FileStreamSink

    path = tmp_path / "nested" / "deep" / "t.log"
    sink = FileStreamSink()
    await sink.open(path)
    try:
        await sink.write_line(b"hi\n")
    finally:
        await sink.close()
    assert path.read_bytes() == b"hi\n"


@pytest.mark.asyncio
async def test_open_appends_not_truncates(tmp_path: Path):
    """Re-opening an existing file must preserve prior content."""
    from agents.stream.file_sink import FileStreamSink

    path = tmp_path / "t.log"
    path.write_bytes(b"existing\n")

    sink = FileStreamSink()
    await sink.open(path)
    try:
        await sink.write_line(b"added\n")
    finally:
        await sink.close()
    assert path.read_bytes() == b"existing\nadded\n"
