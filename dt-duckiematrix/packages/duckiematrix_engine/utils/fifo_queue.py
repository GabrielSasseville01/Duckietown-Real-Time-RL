"""FIFO queue."""

from queue import Empty, SimpleQueue
from threading import Semaphore
from typing import Generic, TypeVar

T = TypeVar("T")


class FIFOQueue(Generic[T]):
    """FIFO queue."""

    _count: int
    _lock: Semaphore
    _max_size: int
    _queue: SimpleQueue[T]

    def __init__(self, max_size: int) -> None:
        """Initialize FIFO queue."""
        self._max_size = max_size
        self._count = 0
        self._lock = Semaphore()
        self._queue = SimpleQueue()

    def _get(
        self,
        *,
        block: bool = False,
        timeout: float | None = None,
    ) -> T:
        result = self._queue.get(block=block, timeout=timeout)
        self._count -= 1
        return result

    def _put(self, item: T) -> None:
        self._count += 1
        self._queue.put(item)

    @property
    def bounded(self) -> bool:
        """Return `True` if bounded, `False` otherwise."""
        return self._max_size > 0

    @property
    def empty(self) -> bool:
        """Return `True` if empty, `False` otherwise."""
        return self._count == 0

    @property
    def full(self) -> bool:
        """Return `True` if full, `False` otherwise."""
        return self.bounded and self._count >= self._max_size

    def get(
        self,
        *,
        block: bool = False,
        timeout: float | None = None,
    ) -> T:
        """Return item from queue."""
        with self._lock:
            if self.empty:
                raise Empty
            return self._get(block=block, timeout=timeout)

    def put(self, item: T) -> None:
        """Put item in queue."""
        with self._lock:
            # remove oldest if full
            if self.full:
                _ = self._get()
            # execute put
            self._put(item)
