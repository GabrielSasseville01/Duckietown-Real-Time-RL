"""Stopwatch."""

import time


class Stopwatch:
    """Stopwatch."""

    _time: float | None

    def __init__(self) -> None:
        """Initialize stopwatch."""
        self._time = None

    def _clear(self) -> None:
        """Clear."""
        self._time = None

    @property
    def elapsed(self) -> float:
        """Return time elapsed."""
        if self.running and self._time is not None:
            return time.time() - self._time
        message = "Start the stopwatch first!"
        raise ValueError(message)

    def restart(self) -> None:
        """Restart."""
        self.stop()
        self.start()

    @property
    def running(self) -> bool:
        """Return `True` if running, `False` otherwise."""
        return self._time is not None

    def start(self) -> None:
        """Start."""
        self._time = time.time()

    def stop(self) -> float:
        """Stop."""
        if self.running and self._time is not None:
            stime = self._time
            self._clear()
            return time.time() - stime
        return 0
