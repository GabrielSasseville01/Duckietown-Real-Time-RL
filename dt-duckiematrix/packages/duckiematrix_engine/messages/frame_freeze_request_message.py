"""Frame freeze request message."""

import dataclasses

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)


@dataclasses.dataclass
class FrameFreezeRequestMessage(InternalCBORMessage):
    """Frame freeze request message."""

    key: str
    frozen: bool
