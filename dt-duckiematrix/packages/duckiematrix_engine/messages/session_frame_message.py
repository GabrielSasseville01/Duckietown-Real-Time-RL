"""Session frame message."""

import dataclasses

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)


@dataclasses.dataclass
class SessionFrameMessage(InternalCBORMessage):
    """Session frame message."""

    session_id: int
    payload: bytes
    timestamp: float
