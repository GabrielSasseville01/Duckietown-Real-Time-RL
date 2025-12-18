"""Drive keyboard action message."""

import dataclasses

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)


@dataclasses.dataclass
class DriveKeyboardActionMessage(InternalCBORMessage):
    """Drive keyboard action message."""

    forward: bool = False
    backward: bool = False
    left: bool = False
    right: bool = False
    reset: bool = False
