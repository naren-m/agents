"""Stream sinks for capturing agent subprocess output.

A ``StreamSink`` is the abstract destination any CLI agent can write
subprocess stdout/stderr into.  The default implementation is
``FileStreamSink`` which writes to the transcript file incrementally
so operators can ``tail -f`` in real time.

Custom sinks (for example an S3 uploader or a stdout echo) can plug in
without the agent caring about destination specifics.
"""

from agents.stream.base import StreamSink
from agents.stream.file_sink import FileStreamSink

__all__ = ["FileStreamSink", "StreamSink"]
