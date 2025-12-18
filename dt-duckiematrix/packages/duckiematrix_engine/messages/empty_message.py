"""Empty message."""

import dataclasses

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)


@dataclasses.dataclass
class EmptyMessage(InternalCBORMessage):
    """Empty message."""

    __empty__: str = ""
