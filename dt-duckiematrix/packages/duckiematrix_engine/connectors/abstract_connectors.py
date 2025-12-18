"""Abstract connectors."""

import logging
from abc import ABC, abstractmethod
from threading import Thread
from typing import TYPE_CHECKING

from dtps import DTPSContext

from packages.duckiematrix_engine.containers.data_containers import (
    SideDataContainer,
)


class DataConnectorPublisher(Thread):
    """Data connector publisher."""

    if TYPE_CHECKING:
        from packages.duckiematrix_engine.engine import MatrixEngine

    _connector: "AsyncDataConnectorAbs | DataConnectorAbs"
    _container: SideDataContainer
    _engine: "MatrixEngine | None"

    def __init__(
        self,
        connector: "AsyncDataConnectorAbs | DataConnectorAbs",
        container: SideDataContainer,
    ) -> None:
        """Initialize data connector publisher."""
        super().__init__(daemon=True)
        self._connector = connector
        self._container = container
        self._engine = None

    def run(self) -> None:
        """Run."""
        changes = False
        while not self._engine.is_shutdown:
            for key, data in self._container.iterate_updated_queue("input"):
                self._connector.send(key, data)
            if changes:
                self._container.mark_as_flushed("input")
            changes = self._container.wait_for_queue("input", 1)

    def start(self) -> None:
        """Start."""
        from packages.duckiematrix_engine.engine import MatrixEngine

        self._engine = MatrixEngine.get_instance()
        super().start()


class ConnectorAbs(ABC, Thread):
    """Abstract connector."""

    if TYPE_CHECKING:
        from packages.duckiematrix_engine.engine import MatrixEngine

    _engine: "MatrixEngine | None"
    _logger: logging.Logger

    def __init__(self, name: str) -> None:
        """Initialize abstract connector."""
        super().__init__(daemon=True)
        self._engine = None
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)

    @abstractmethod
    async def _arun(self) -> None:
        pass

    def start(self) -> None:
        """Start."""
        from packages.duckiematrix_engine.engine import MatrixEngine

        self._engine = MatrixEngine.get_instance()
        self._logger.setLevel(self._engine.logger.level)
        super().start()


class AsyncConnectorAbs(ABC):
    """Abstract asynchronous connector."""

    if TYPE_CHECKING:
        from packages.duckiematrix_engine.engine import MatrixEngine

    _engine: "MatrixEngine | None"
    _logger: logging.Logger

    def __init__(self, name: str) -> None:
        """Initialize abstract asynchronous connector."""
        self._engine = None
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)

    @abstractmethod
    async def astart(self, context: DTPSContext) -> None:
        """Start asynchronously."""

    def start(self) -> None:
        """Start."""
        from packages.duckiematrix_engine.engine import MatrixEngine

        self._engine = MatrixEngine.get_instance()
        self._logger.setLevel(self._engine.logger.level)


class DataConnectorAbs(ConnectorAbs):
    """Abstract data connector."""

    _container: SideDataContainer
    _publisher: DataConnectorPublisher
    group: str

    def __init__(self, group: str, container: SideDataContainer) -> None:
        """Initialize abstract data connector."""
        super().__init__(f"DataConnector[{group}]")
        self.group = group
        self._container = container
        self._publisher = DataConnectorPublisher(self, self._container)

    def _mark_as_sent(self, key: str) -> None:
        self._container.mark_as_sent(key)

    def _mark_as_received(self, key: str) -> None:
        self._container.mark_as_received(key)

    @abstractmethod
    def receive(self, key: str, data: bytes) -> None:
        """Receive."""

    @abstractmethod
    def send(self, key: str, data: bytes) -> None:
        """Send."""


class AsyncDataConnectorAbs(AsyncConnectorAbs, DataConnectorAbs):
    """Abstract asynchronous data connector."""

    _container: SideDataContainer
    _publisher: DataConnectorPublisher
    group: str

    def __init__(self, group: str, container: SideDataContainer) -> None:
        """Initialize abstract asynchronous data connector."""
        super().__init__(f"DataConnector[{group}]")
        self.group = group
        self._container = container
        self._publisher = DataConnectorPublisher(self, self._container)

    def _mark_as_received(self, key: str) -> None:
        self._container.mark_as_received(key)

    def _mark_as_sent(self, key: str) -> None:
        self._container.mark_as_sent(key)

    @abstractmethod
    def receive(self, key: str, data: bytes) -> None:
        """Receive."""

    @abstractmethod
    def send(self, key: str, data: bytes) -> None:
        """Send."""
