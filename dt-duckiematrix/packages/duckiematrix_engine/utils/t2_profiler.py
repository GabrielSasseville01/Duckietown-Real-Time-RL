"""T2 profiler."""

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from types import TracebackType
from typing import Any, ClassVar

# | x | 13 | 13 | 13 |
TABLE_SEPARATOR = "+-{0}-+-------------+-------------+-------------+\n"
TABLE_HEADER_FMT = "| {0} |   # Calls   |  Time (ms)  |  Freq (Hz)  |\n"
TABLE_ROW_FMT = "| {key} | {count} | {duration} | {frequency} |\n"


class T2Profiler:
    """T2 profiler."""

    __enabled__: ClassVar[bool] = False
    _buffer: ClassVar[dict[str, float]] = {}
    _count: ClassVar[dict[str, int]] = defaultdict(lambda: 0)
    _data: ClassVar[dict[str, float]] = defaultdict(lambda: 0)

    class ProfilingContext:
        """Profiling context."""

        _key: str

        def __enter__(self) -> None:
            """Enter profiling context."""
            T2Profiler.tick(self._key)

        def __exit__(
            self,
            _: type[BaseException] | None,
            __: BaseException | None,
            ___: TracebackType | None,
        ) -> None:
            """Exit profiling context."""
            T2Profiler.tock(self._key)

        def __init__(self, key: str) -> None:
            """Initialize profiling context."""
            self._key = key

    @staticmethod
    def enabled(*, status: bool) -> None:
        """Set status."""
        T2Profiler.__enabled__ = status

    @staticmethod
    def print(logger: logging.Logger) -> None:
        """Log information."""
        if not T2Profiler.__enabled__:
            return
        # find longest key
        column_size = max(*[len(k) for k in T2Profiler._count])
        # compile table
        table = ""
        table += TABLE_SEPARATOR.format("-" * column_size)
        table += TABLE_HEADER_FMT.format("Key".ljust(column_size, " "))
        table += TABLE_SEPARATOR.format("-" * column_size)
        for key in sorted(T2Profiler._count.keys()):
            cumulative_duration = T2Profiler._data[key]
            count = T2Profiler._count[key]
            # compute stats
            avg_duration = cumulative_duration / float(count)
            avg_frequency = 1 / (avg_duration / 1000)
            row = TABLE_ROW_FMT.format(
                key=key.ljust(column_size, " "),
                count=str(count).rjust(11, " "),
                duration=str(round(avg_duration, 2)).rjust(11, " "),
                frequency=str(round(avg_frequency, 1)).rjust(11, " "),
            )
            table += row
        table += TABLE_SEPARATOR.format("-" * column_size)
        logger.info("\nEngine Profiling Information:\n%s\n", table)

    @staticmethod
    def profile(key: str) -> ProfilingContext:
        """Return profiling context."""
        return T2Profiler.ProfilingContext(key)

    @staticmethod
    def profiled(key_or_function: str | Callable | None = None) -> Any:
        """Return profiler wrapper or wrapper factory."""
        if isinstance(key_or_function, str):

            def wrapper_factory(function: Any) -> Any:
                if key_or_function is None:
                    key = function.__name__
                else:
                    key = key_or_function

                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    with T2Profiler.profile(key):
                        return function(*args, **kwargs)

                return wrapper

            return wrapper_factory
        if callable(key_or_function):

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with T2Profiler.profile(key_or_function.__name__):
                    return key_or_function(*args, **kwargs)

            return wrapper
        return None

    @staticmethod
    def tick(key: str) -> None:
        """Set time."""
        if not T2Profiler.__enabled__:
            return
        now = time.time() * 1000
        T2Profiler._buffer[key] = now

    @staticmethod
    def tock(key: str) -> None:
        """Set time difference."""
        if not T2Profiler.__enabled__:
            return
        now = time.time() * 1000
        try:
            tick = T2Profiler._buffer[key]
            del T2Profiler._buffer[key]
        except KeyError:
            return
        T2Profiler._data[key] += now - tick
        T2Profiler._count[key] += 1
