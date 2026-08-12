"""Abstract stream sink protocol.

A ``StreamSink`` receives subprocess output a line at a time and is
responsible for durable storage.  The sink is opened once per run,
written to as bytes arrive, and closed when the subprocess terminates
(or is cancelled).

The protocol is intentionally minimal so alternative implementations --
rotating files, remote uploaders, in-memory buffers -- can plug in
without touching callers.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class StreamSink(Protocol):
    """Destination for a single agent run's raw output stream."""

    async def open(self, path: Path) -> None:
        """Prepare the sink for writing.

        ``path`` is a hint (used by file-based sinks); sinks that do
        not need a filesystem path may ignore it.  Implementations must
        be idempotent on re-open for the same path.
        """
        ...

    async def write_line(self, line: bytes) -> None:
        """Append a single line of output.

        ``line`` includes the trailing newline if the source emitted one.
        Implementations should not strip or transform the payload -- the
        raw bytes are the contract consumers depend on.
        """
        ...

    async def close(self) -> None:
        """Finalise the sink.

        Called exactly once per successful ``open()``.  Implementations
        must be safe to call even if ``open()`` partially failed.
        """
        ...
