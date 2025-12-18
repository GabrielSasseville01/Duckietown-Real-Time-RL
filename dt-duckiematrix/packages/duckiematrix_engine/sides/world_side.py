"""World side."""

import argparse
import time
from typing import TYPE_CHECKING

from dtps_http import RawData
from duckietown_messages.base import BaseMessage
from duckietown_messages.standard.dictionary import Dictionary
from duckietown_messages.standard.header import Header

from packages.duckiematrix_engine.connectors.data_connectors import (
    DTPSDataConnector,
    ZMQDataConnector,
)
from packages.duckiematrix_engine.connectors.world_connector import (
    WorldConnector,
)
from packages.duckiematrix_engine.containers.data_containers import (
    SideDataContainer,
)
from packages.duckiematrix_engine.exceptions import SideDataDecoderError
from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)
from packages.duckiematrix_engine.sides.side_abs import SideAbs

DESIRED_LAYERS = ("frames", "tile_maps", "tiles", "traffic_signs")

if TYPE_CHECKING:
    from packages.duckiematrix_engine.map import Map


class WorldSide(SideAbs):
    """World side."""

    _connectors: dict[str, DTPSDataConnector | ZMQDataConnector]
    _groups: dict[str, SideDataContainer]
    _layer_connector: ZMQDataConnector
    _map_connector: DTPSDataConnector
    _side_connector: WorldConnector
    layer: SideDataContainer
    robot: SideDataContainer
    robot_connector: DTPSDataConnector

    def __init__(self, parsed: argparse.Namespace) -> None:
        """Initialize world side."""
        super().__init__("WorldSide")
        # data containers
        self.layer = SideDataContainer("layer", None, dict, 0, 0)
        self.robot = SideDataContainer("robot", BaseMessage, BaseMessage, 1, 1)
        # data connectors
        self._layer_connector = ZMQDataConnector(
            "layer",
            parsed.hostname,
            self.layer,
            outbound=True,
            inbound=False,
            port_out=parsed.layer_data_out_port,
            port_in=None,
        )
        self._map_connector = DTPSDataConnector(
            "map",
            self.layer,
            outbound=True,
            inbound=False,
        )
        self.robot_connector = DTPSDataConnector(
            "robot",
            self.robot,
            outbound=True,
            inbound=True,
        )
        # main connector
        self._side_connector = WorldConnector(
            parsed.hostname,
            parsed.world_control_out_port,
            [self.robot_connector, self._map_connector],
        )
        # groups
        self._groups = {
            "layer": self.layer,
            "map": self.layer,
            "robot": self.robot,
        }
        # connectors
        self._connectors = {
            "layer": self._layer_connector,
            "map": self._map_connector,
            "robot": self.robot_connector,
        }

    def _publish_layer(self, layer: str, layer_dictionary: dict) -> None:
        """Publish layer."""
        try:
            while layer not in self._map_connector.publishers:
                self._logger.info(
                    "Waiting for the '%s' publisher to be created...",
                    layer,
                )
                time.sleep(1)
            timestamp = time.time()
            header = Header(timestamp=timestamp)
            dictionary = Dictionary(header=header, data=layer_dictionary)
            self._logger.info("Publishing '%s' data...", layer)
            self._map_connector.send(layer, dictionary)
            self._logger.info("Published '%s' data.", layer)
        except Exception:
            self._logger.exception("Failed to publish '%s' data.", layer)

    def _publish_layers(self) -> None:
        """Publish layers."""
        while not self._map_connector.loop:
            self._logger.info(
                "Waiting for the 'map' data connector to start...",
            )
            time.sleep(1)
        map_: Map = self.engine.map_
        for layer, _ in map_.SUPPORTED_LAYERS:
            if layer in DESIRED_LAYERS and layer in map_.layers:
                self.declare_input("map", layer, Dictionary)
                layer_dictionary = map_[layer].as_raw_dict()
                self._publish_layer(layer, layer_dictionary)

    def clear_session(self) -> None:
        """Clear session."""
        # TODO: self.engine.world_data_connector.clear_session()

    def complete_session(self) -> None:
        """Complete session."""
        # TODO: implement this

    def declare_input(
        self,
        group: str,
        key: str,
        encoder: type[BaseMessage],
    ) -> None:
        """Declare input."""
        self._connectors[group].declare_input(key, encoder)

    def declare_output(
        self,
        group: str,
        key: str,
        decoder: type[BaseMessage],
        max_frequency: float | None = None,
    ) -> None:
        """Declare output."""
        self._connectors[group].declare_output(key, decoder, max_frequency)

    def input(
        self,
        group: str,
        key: str,
        data: BaseMessage,
    ) -> None:
        """Input."""
        try:
            decoded_data = self._groups[group].decode_queue("input", data)
            self._groups[group].register_queue("input", key, decoded_data)
        except KeyError:
            self._logger.warning(
                "Trying to set data for unknown group '%s'.",
                group,
            )
        except SideDataDecoderError as error:
            self._logger.exception(
                "Some data could not be decoded.\n\tExpected: %s;\n\tReceived:"
                " %s;\n\tMessage:  %s;\n\tException:\n%s\n"
                "------------------------------------------\n",
                error.expected,
                error.received,
                error.message,
                error.trigger_stack,
            )

    def output(
        self,
        group: str,
        key: str,
        decoder: type,
    ) -> bytes | RawData | InternalCBORMessage | object | None:
        """Return queue."""
        try:
            return self._groups[group].queue("output", key, decoder)
        except KeyError:
            self._logger.warning(
                "Trying to get data for unknown group '%s'.",
                group,
            )
        return None

    def print_startup_info(self) -> None:
        """Log startup information."""

    def start(self) -> None:
        """Start."""
        super().start()
        self._side_connector.start()
        self._layer_connector.start()
        self._publish_layers()
