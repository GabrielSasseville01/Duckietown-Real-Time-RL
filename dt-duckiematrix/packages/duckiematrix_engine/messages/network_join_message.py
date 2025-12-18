"""Network join message."""

import dataclasses

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)


@dataclasses.dataclass
class NetworkJoinMessage(InternalCBORMessage):
    """Network join message."""

    id: str
    role: int
    key: str
    location: str
