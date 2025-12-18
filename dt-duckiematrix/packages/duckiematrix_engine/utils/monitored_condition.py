"""Monitored condition."""

from threading import Condition


class MonitoredCondition(Condition):
    """Monitored condition."""

    _has_changes: bool

    def __init__(self) -> None:
        """Initialize monitored condition."""
        super().__init__()
        self._has_changes = False

    def clear(self) -> None:
        """Clear."""
        self._has_changes = False

    def notify(self, n: int = 1) -> None:
        """Wake up thread(s) waiting on this condition."""
        self._has_changes = True
        super().notify(n)

    def notify_all(self) -> None:
        """Wake up all threads waiting on this condition."""
        self._has_changes = True
        super().notify_all()

    def notifyAll(self) -> None:  # noqa: N802
        """Run `notify_all` function."""
        self.notify_all()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until notified or until a timeout occurs."""
        if self._has_changes:
            self._has_changes = False
            return True
        return super().wait(timeout)
