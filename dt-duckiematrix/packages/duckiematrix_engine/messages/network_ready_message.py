"""Network ready message."""

import dataclasses

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)


@dataclasses.dataclass
class NetworkReadyMessage(InternalCBORMessage):
    """Network ready message."""

    id: str
    role: int
    key: str
