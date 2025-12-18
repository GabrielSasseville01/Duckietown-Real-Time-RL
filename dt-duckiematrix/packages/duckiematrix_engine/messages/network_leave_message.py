"""Network leave message."""

import dataclasses

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)


@dataclasses.dataclass
class NetworkLeaveMessage(InternalCBORMessage):
    """Network leave message."""

    id: str
