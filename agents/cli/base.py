"""CLI agent protocol and shared subprocess lifecycle.

BaseCLIAgent handles the common plumbing: binary discovery, async subprocess
management, transcript capture, timeout watchdog, and cancellation. Subclasses
only need to set ``name``, ``_binary_names``, and implement ``build_command``.

When ``config.event_bus`` is set, ``wait()`` switches into a streaming
path: stdout is read incrementally and fanned out to the bus ring buffer
(for UI consumers) and the transcript file sink (for ``tail -f``
debugging).  When it is unset, the legacy ``communicate()`` behaviour is
preserved byte-for-byte.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agents.stream.file_sink import FileStreamSink
from agents.types import AgentConfig, AgentResult, AgentRun

if TYPE_CHECKING:
    from agents.event_bus import RunEventBus

logger = logging.getLogger(__name__)

_CANCEL_GRACE_SECONDS = 10

# Bound for draining the stdout reader after the direct child process
# exits.  Most agents flush their final bytes within milliseconds; the
# only reason this can stretch is a forked daemon descendant that
# inherited the stdout fd (e.g. cursor-agent's ``worker-server``) and
# never closes it.  Without this bound, ``await reader_task`` blocks
# until the orphan dies and the run stays in ``running`` forever.
_STDOUT_DRAIN_GRACE_SECONDS = 2.0

# Read raw stdout in bounded chunks instead of readline(). Codex JSONL
# events can exceed asyncio's StreamReader line limit.
_STDOUT_READ_CHUNK_BYTES = 64 * 1024

# Strict allow-list for caller-supplied agent_name.
# Rules (codeguard input-validation):
#   - chars limited to [A-Za-z0-9._-]
#   - 1..64 chars
#   - must not start with '.' (no hidden files) or '-' (no flag-lookalikes)
#   - must not contain ".." as a substring (defence-in-depth vs traversal)
_AGENT_NAME_RE = re.compile(r"^(?![.\-])(?!.*\.\.)[A-Za-z0-9._-]{1,64}$")


def _validate_agent_name(name: str) -> None:
    """Raise ``ValueError`` if ``name`` is not safe to use as a filename
    component.

    Caller must handle ``None`` separately -- this function expects a
    real string. ``None`` means "no name supplied" and should not flow
    through here.
    """
    if not isinstance(name, str) or not _AGENT_NAME_RE.match(name):
        raise ValueError(f"Invalid agent_name: {name!r}")


_DEFAULT_TRANSCRIPT_DIR = Path("/tmp/agent_logs")


def _build_transcript_path(
    config: AgentConfig,
    run: AgentRun,
    agent_name: str | None,
) -> Path:
    """Compute the absolute transcript path for a spawn.

    Resolution order:
        directory = config.transcript_dir or _DEFAULT_TRANSCRIPT_DIR
        filename  = f"{agent_name}_{run.run_id}.log" if agent_name else
                    f"{run.run_id}.log"

    The parent directory is created (``mkdir(parents=True,
    exist_ok=True)``) so the caller can immediately open the path for
    writing. ``agent_name`` must already be validated by
    :func:`_validate_agent_name` -- this helper does no checking.
    """
    directory = config.transcript_dir or _DEFAULT_TRANSCRIPT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    if agent_name:
        filename = f"{agent_name}_{run.run_id}.log"
    else:
        filename = f"{run.run_id}.log"
    return directory / filename


@runtime_checkable
class CLIAgent(Protocol):
    """Interface for subprocess-based agent backends."""

    @property
    def name(self) -> str: ...

    def available(self) -> bool: ...

    def binary_path(self) -> str | None: ...

    def build_command(self, prompt: str, config: AgentConfig) -> list[str]: ...

    async def spawn(
        self,
        prompt: str,
        config: AgentConfig,
        run_id_hint: str | None = None,
    ) -> AgentRun: ...

    async def cancel(self, run: AgentRun) -> bool: ...

    async def wait(self, run: AgentRun) -> AgentResult: ...


class BaseCLIAgent:
    """Shared subprocess lifecycle for all CLI agents.

    Subclasses override: ``name``, ``_binary_names``, ``build_command``.
    Everything else -- spawn, cancel, wait, transcript capture, timeout --
    is handled here.
    """

    name: str = ""
    _binary_names: list[str] = []

    _processes: dict[str, asyncio.subprocess.Process]

    def __init__(self):
        self._processes = {}

    def available(self) -> bool:
        return self.binary_path() is not None

    def binary_path(self) -> str | None:
        for bin_name in self._binary_names:
            path = shutil.which(bin_name)
            if path:
                return path
        return None

    def build_command(self, prompt: str, config: AgentConfig) -> list[str]:
        raise NotImplementedError

    def parse_output(self, output: str, run: AgentRun) -> None:
        """Hook for subclasses to extract telemetry from process output.

        Called by ``wait()`` after the process finishes, before building the
        AgentResult. Subclasses override this to parse structured output
        (e.g. JSON lines) and populate the AgentRun's telemetry fields.
        """

    async def spawn(
        self,
        prompt: str,
        config: AgentConfig,
        run_id_hint: str | None = None,
        agent_name: str | None = None,
    ) -> AgentRun:
        """Launch the agent as a background subprocess.

        ``run_id_hint`` lets callers (typically ``AgentManager``) reserve
        a run id before the backend starts -- useful for pre-binding an
        ``EventBusRegistry`` entry so subscribers never miss early events.

        ``agent_name`` is an optional human-friendly label that becomes
        part of the transcript filename (``{agent_name}_{run_id}.log``).
        It is validated against a strict allow-list before any subprocess
        is started; invalid names raise ``ValueError`` and leave no
        partial state behind.
        """
        if agent_name is not None:
            _validate_agent_name(agent_name)

        if not self.available():
            raise RuntimeError(
                f"Agent backend {self.name!r} is not available. "
                f"Looked for: {self._binary_names}"
            )

        cmd = self.build_command(prompt, config)
        run = AgentRun.create(
            backend=self.name, prompt=prompt, run_id=run_id_hint,
        )
        run.agent_name = agent_name

        transcript_path = _build_transcript_path(config, run, agent_name)
        run.transcript_path = transcript_path
        logger.info(
            "Transcript: %s (run_id=%s, agent_name=%s)",
            transcript_path, run.run_id, agent_name or "<none>",
        )

        env = os.environ.copy()
        env.update(config.env)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(config.workspace) if config.workspace.exists() else None,
            env=env,
        )

        run.pid = process.pid
        # Stash timeout, event bus, and sink factory on the run so wait()
        # can pick them up without having config passed in again. These
        # are private attributes; not part of the public dataclass.
        run._timeout = config.timeout_seconds  # type: ignore[attr-defined]
        run._event_bus = config.event_bus  # type: ignore[attr-defined]
        run._transcript_sink_factory = FileStreamSink  # type: ignore[attr-defined]
        self._processes[run.run_id] = process

        logger.info(
            "Spawned %s agent (pid=%d, run_id=%s)",
            self.name, process.pid, run.run_id,
        )

        # Publish run.started as soon as the subprocess is alive. Doing
        # it here (rather than in wait()) guarantees the event fires even
        # for callers that never await wait() directly (for example the
        # AgentManager background monitor).
        if config.event_bus is not None:
            await config.event_bus.publish_event(
                "run.started",
                {
                    "backend": self.name,
                    "pid": process.pid,
                    "prompt_len": len(prompt),
                    "transcript_path": (
                        str(run.transcript_path)
                        if run.transcript_path
                        else None
                    ),
                },
            )

        return run

    async def wait(self, run: AgentRun) -> AgentResult:
        """Wait for the agent to finish. Enforces timeout from config.

        Dispatches to the streaming pump when an event bus was attached
        during ``spawn()``; otherwise falls through to the legacy
        ``communicate()`` path for backwards compatibility.
        """
        process = self._processes.get(run.run_id)
        if process is None:
            if run.is_terminal:
                return AgentResult(
                    success=run.status == "completed",
                    output="",
                    run=run,
                )
            raise RuntimeError(f"No process tracked for run {run.run_id}")

        bus = getattr(run, "_event_bus", None)
        if bus is not None:
            return await self._wait_streaming(run, process, bus)

        # Legacy non-streaming path (preserved byte-for-byte).
        try:
            stdout_bytes, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=getattr(run, "_timeout", 900),
            )
        except asyncio.TimeoutError:
            await self._kill_process(process)
            run.mark_timed_out()
            output = ""
            if run.transcript_path:
                output = self._read_transcript_tail(run.transcript_path)
            self._processes.pop(run.run_id, None)
            return AgentResult(success=False, output=output, run=run)

        output = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""

        if run.transcript_path:
            run.transcript_path.write_text(output, encoding="utf-8")

        self.parse_output(output, run)

        # Guard: cancel() may have already set a terminal status while we
        # were waiting on communicate().  Honour the earlier transition.
        if not run.is_terminal:
            if process.returncode == 0:
                run.mark_completed(exit_code=0)
            else:
                run.mark_failed(exit_code=process.returncode or 1)

        self._processes.pop(run.run_id, None)

        # Truncate output for the result (keep last 4000 chars)
        truncated = output[-4000:] if len(output) > 4000 else output

        return AgentResult(
            success=run.status == "completed",
            output=truncated,
            run=run,
        )

    async def _wait_streaming(
        self,
        run: AgentRun,
        process: asyncio.subprocess.Process,
        bus: RunEventBus,
    ) -> AgentResult:
        """Streaming variant of ``wait()`` used when an event bus is present.

        Reads stdout incrementally so:
        - the bus ring buffer gets bytes as soon as they arrive,
        - the transcript sink flushes to disk for ``tail -f`` viewers,
        - an unbounded local accumulator preserves the full stdout for
          ``parse_output()`` (telemetry extraction must not be truncated
          by ring-buffer overflow).
        """
        accum = bytearray()
        sink = None
        sink_factory = getattr(run, "_transcript_sink_factory", FileStreamSink)
        if run.transcript_path is not None:
            sink = sink_factory()
            await sink.open(run.transcript_path)

        assert process.stdout is not None  # stdout was PIPEd in spawn()

        async def _reader() -> None:
            while True:
                chunk = await process.stdout.read(_STDOUT_READ_CHUNK_BYTES)
                if not chunk:
                    break
                accum.extend(chunk)
                try:
                    await bus.append_log(chunk)
                except Exception:
                    logger.exception(
                        "append_log failed for run %s", run.run_id,
                    )
                if sink is not None:
                    try:
                        await sink.write_line(chunk)
                    except Exception:
                        logger.exception(
                            "sink.write_line failed for run %s", run.run_id,
                        )

        reader_task = asyncio.create_task(_reader())

        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=getattr(run, "_timeout", 900),
            )
        except asyncio.TimeoutError:
            await self._kill_process(process)
            if not run.is_terminal:
                run.mark_timed_out()
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception):
                pass
            if sink is not None:
                await sink.close()
            self._processes.pop(run.run_id, None)
            output = bytes(accum).decode("utf-8", errors="replace")
            return AgentResult(
                success=False, output=output[-4000:], run=run,
            )

        # Process exited; give the reader a brief drain window for any
        # remaining buffered lines, then force-cancel it.  Bounding the
        # drain matters when the child forked a daemon that inherited
        # stdout (cursor-agent's worker-server is the canonical
        # example) -- without this, ``read()`` blocks waiting for
        # an EOF that only arrives when the orphan dies.
        try:
            await asyncio.wait_for(
                reader_task, timeout=_STDOUT_DRAIN_GRACE_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "stdout drain timed out for run %s after %.1fs; an "
                "orphan child likely inherited stdout. Cancelling reader.",
                run.run_id, _STDOUT_DRAIN_GRACE_SECONDS,
            )
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception):
                pass
        except Exception:
            logger.exception("stdout pump errored for run %s", run.run_id)

        if sink is not None:
            await sink.close()

        output = bytes(accum).decode("utf-8", errors="replace")
        self.parse_output(output, run)

        if not run.is_terminal:
            if process.returncode == 0:
                run.mark_completed(exit_code=0)
            else:
                run.mark_failed(exit_code=process.returncode or 1)

        self._processes.pop(run.run_id, None)
        truncated = output[-4000:] if len(output) > 4000 else output
        return AgentResult(
            success=run.status == "completed",
            output=truncated,
            run=run,
        )

    async def cancel(self, run: AgentRun) -> bool:
        """Send SIGTERM to the process. Returns True if signal was sent."""
        process = self._processes.get(run.run_id)
        if process is None or process.returncode is not None:
            return False

        try:
            process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return False

        try:
            await asyncio.wait_for(
                process.wait(), timeout=_CANCEL_GRACE_SECONDS
            )
        except asyncio.TimeoutError:
            await self._kill_process(process)

        run.mark_cancelled()
        self._processes.pop(run.run_id, None)
        return True

    async def _kill_process(self, process: asyncio.subprocess.Process) -> None:
        """Force-kill a process that didn't respond to SIGTERM."""
        try:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass

    def _read_transcript_tail(self, path: Path, max_bytes: int = 4000) -> str:
        """Read the tail of a transcript file, best-effort."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return text[-max_bytes:]
        except OSError:
            return ""
