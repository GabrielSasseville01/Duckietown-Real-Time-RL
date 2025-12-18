"""Abstract side."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from packages.duckiematrix_engine.utils.monitored_condition import (
    MonitoredCondition,
)


class SideAbs(ABC):
    """Abstract side."""

    if TYPE_CHECKING:
        from packages.duckiematrix_engine.engine import MatrixEngine

    _flush_inputs_event: MonitoredCondition
    _flushed_inputs_event: MonitoredCondition
    _logger: logging.Logger
    empty_dict: dict
    engine: "MatrixEngine | None"
    name: str

    def __init__(self, name: str) -> None:
        """Initialize abstract side."""
        self.empty_dict = {}
        self.name = name
        self.engine = None
        self._flush_inputs_event = MonitoredCondition()
        self._flushed_inputs_event = MonitoredCondition()
        self._logger = logging.getLogger(name)

    @abstractmethod
    def clear_session(self) -> None:
        """Clear session."""

    def flush_inputs(
        self,
        *,
        block: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Flush inputs."""
        # clear 'sent' event
        with self._flushed_inputs_event:
            self._flushed_inputs_event.clear()
        # set the 'to-send' event
        with self._flush_inputs_event:
            self._flush_inputs_event.notify_all()
        # block (if needed) until the 'sent' event occurs
        if block:
            with self._flushed_inputs_event:
                self._flushed_inputs_event.wait(timeout)

    def mark_as_flushed_inputs(self) -> None:
        """Mark as flushed inputs."""
        with self._flushed_inputs_event:
            self._flushed_inputs_event.notify_all()

    def notify_engine(self) -> None:
        """Notify engine."""
        with self.engine.events:
            self.engine.events.notify()

    def start(self) -> None:
        """Start."""
        from packages.duckiematrix_engine.engine import MatrixEngine

        self.engine = MatrixEngine.get_instance()
        self._logger.setLevel(self.engine.logger.level)

    def wait_for_inputs(self, timeout: float | None = None) -> bool:
        """Wait for inputs."""
        with self._flush_inputs_event:
            return self._flush_inputs_event.wait(timeout)

    def wait_until_flushed(self, timeout: float | None = None) -> bool:
        """Wait until flushed."""
        with self._flushed_inputs_event:
            return self._flushed_inputs_event.wait(timeout)
