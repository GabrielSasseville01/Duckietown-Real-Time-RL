"""String message."""

import dataclasses

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)


@dataclasses.dataclass
class StringMessage(InternalCBORMessage):
    """String message."""

    value: str
