"""Network joined message."""

import dataclasses

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)
from packages.duckiematrix_engine.messages.network_endpoint_message import (
    NetworkEndpointMessage,
)


@dataclasses.dataclass
class NetworkJoinedMessage(InternalCBORMessage):
    """Network joined message."""

    id: str
    endpoints: list[NetworkEndpointMessage]
