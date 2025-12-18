"""Network endpoint message."""

import dataclasses
from enum import Enum

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)


@dataclasses.dataclass
class NetworkEndpointMessage(InternalCBORMessage):
    """Network endpoint message."""

    type: str
    protocol: str
    hostname: str | None
    port: int


class NetworkEndpointProtocol(Enum):
    """Network endpoint protocol."""

    TCP = "tcp"
    UDP = "udp"


class NetworkEndpointType(Enum):
    """Network endpoint type."""

    DATA_IN = "data_in"
    DATA_OUT = "data_out"
    CONTROL_IN = "control_in"
    CONTROL_OUT = "control_out"
