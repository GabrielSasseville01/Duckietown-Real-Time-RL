"""May layer message."""

import dataclasses

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)


@dataclasses.dataclass
class MapLayerMessage(InternalCBORMessage):
    """MapLayer message."""

    name: str
    content: str
