"""World connector."""

import asyncio

import dtps
from dtps_http import RawData, TopicProperties

from packages.duckiematrix_engine.connectors.data_connectors import (
    DTPSDataConnector,
)
from packages.duckiematrix_engine.connectors.dtps_connector import (
    DTPSConnector,
)


class WorldConnector(DTPSConnector):
    """World connector."""

    _PROTOCOL_VERSION = "0.1.0"

    _data_connectors: list[DTPSDataConnector]

    def __init__(
        self,
        host: str,
        port: int,
        data_connectors: list[DTPSDataConnector],
    ) -> None:
        """Initialize world connector."""
        super().__init__("WorldConnector", host, port)
        # data connectors
        self._data_connectors = data_connectors

    async def _arun(self) -> None:
        self._loop = asyncio.get_event_loop()
        # create context
        self._logger.info("Opening DTPS World connector...")
        self._context = await dtps.context(
            urls=[f"create:http://{self._host}:{self._port}/"],
        )
        self._context_ready.set()
        # expose data
        topic_properties = TopicProperties.readonly()
        version_queue = await (
            self._context / "protocol" / "version"
        ).queue_create(topic_properties=topic_properties)
        version_raw_data = RawData.json_from_native_object(
            self._PROTOCOL_VERSION,
        )
        await version_queue.publish(version_raw_data)
        # start data connectors
        for data_connector in self._data_connectors:
            await data_connector.astart(self._context)
        # keep loop alive
        while not self._engine.is_shutdown:
            await asyncio.sleep(0.1)
