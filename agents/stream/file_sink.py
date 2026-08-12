"""File-backed ``StreamSink`` implementation.

Writes one line at a time, unbuffered at the OS level, so a concurrent
``tail -f`` sees output the moment the subprocess emits it.

Blocking file I/O is dispatched to the default executor so the event
loop never blocks on disk.  A per-sink ``asyncio.Lock`` serialises
writes, which keeps output uncorrupted when multiple pumps share a
single sink (currently just stdout, but kept general on purpose).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import BinaryIO


class FileStreamSink:
    """``StreamSink`` that appends lines to a local file, flushing each
    write so readers observe output in real time.

    The file is opened in append-binary mode with ``buffering=0`` so the
    kernel does not buffer writes behind stdio; raw bytes are the
    contract.
    """

    def __init__(self) -> None:
        self._path: Path | None = None
        self._fh: BinaryIO | None = None
        self._lock = asyncio.Lock()

    async def open(self, path: Path) -> None:
        """Open ``path`` for appending; create parent dirs as needed."""
        loop = asyncio.get_running_loop()

        def _do_open() -> BinaryIO:
            path.parent.mkdir(parents=True, exist_ok=True)
            return open(path, "ab", buffering=0)

        self._fh = await loop.run_in_executor(None, _do_open)
        self._path = path

    async def write_line(self, line: bytes) -> None:
        """Append ``line`` verbatim; raises if the sink is not open."""
        if self._fh is None:
            raise RuntimeError("FileStreamSink not open")
        fh = self._fh
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(None, fh.write, line)

    async def close(self) -> None:
        """Close the underlying file; safe to call multiple times."""
        if self._fh is None:
            return
        fh, self._fh = self._fh, None
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, fh.close)
