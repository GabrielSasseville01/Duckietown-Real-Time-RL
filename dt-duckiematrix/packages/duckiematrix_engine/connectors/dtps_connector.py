"""DTPS connector."""

import asyncio
import traceback
from asyncio import AbstractEventLoop, Event
from collections.abc import Coroutine

from dtps import DTPSContext

from packages.duckiematrix_engine.connectors.abstract_connectors import (
    ConnectorAbs,
)


class DTPSConnector(ConnectorAbs):
    """DTPS connector."""

    _context: DTPSContext | None
    _context_ready: Event
    _host: str
    _loop: AbstractEventLoop | None
    _port: int

    def __init__(self, name: str, host: str, port: int) -> None:
        """Initialize DTPS connector."""
        super().__init__(name)
        self._host = host
        self._port = port
        self._loop = None
        self._context = None
        self._context_ready = Event()

    async def _task(self, coroutine: Coroutine, name: str) -> None:
        try:
            await coroutine
        except Exception:
            self._logger.exception("Exception in task: %s", name)
            traceback.print_exc()

    async def context(self) -> DTPSContext:
        """Context."""
        await self._context_ready.wait()
        return self._context

    def run(self) -> None:
        """Run."""
        main = self._arun()
        asyncio.run(main)
