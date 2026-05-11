"""Subprocess-backed live matplotlib window.

A matplotlib window from a worker thread fights with whatever framework
owns the main thread (Cocoa under mjpython on macOS, X under some Linux
backends). Putting it in a separate process gives it its own main thread
and decouples rendering from the eval loop.

The caller supplies a top-level `worker(queue, *args)` function that
owns the figure and reads items from the queue. None is the sentinel.
"""

from __future__ import annotations

import queue
from multiprocessing import Process, Queue, get_context
from typing import Any, Callable


class LivePlotProcess:
    """Lifecycle wrapper around a spawn-based plotting subprocess."""

    def __init__(self, worker: Callable[..., None], *worker_args: Any, max_queue: int = 500) -> None:
        """worker is a top-level function with signature `(queue, *worker_args)`."""
        self._worker = worker
        self._worker_args = worker_args
        self._max_queue = max_queue
        self._proc: Process | None = None
        self._queue: Queue | None = None

    def start(self) -> None:
        ctx = get_context("spawn")
        self._queue = ctx.Queue(maxsize=self._max_queue)
        self._proc = ctx.Process(
            target=self._worker,
            args=(self._queue, *self._worker_args),
            daemon=True,
        )
        self._proc.start()

    def publish(self, item: Any) -> None:
        """Best-effort send. Drops silently if the queue is full."""
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass

    def stop(self, timeout: float = 2.0) -> None:
        """Send sentinel, join, terminate if still alive."""
        if self._proc is None:
            return
        if self._proc.is_alive():
            self.publish(None)
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                self._proc.terminate()
        self._proc = None
        self._queue = None
