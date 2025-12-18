"""Context response message."""

import dataclasses

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)


@dataclasses.dataclass
class ContextResponseMessage(InternalCBORMessage):
    """Context response message."""

    name: str
    data: bytes
