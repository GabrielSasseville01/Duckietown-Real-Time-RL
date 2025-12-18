"""Camera frame message."""

import dataclasses

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)


@dataclasses.dataclass
class CameraFrameMessage(InternalCBORMessage):
    """Camera frame message."""

    format: str
    width: int
    height: int
    frame: bytes

    @classmethod
    def from_jpeg(cls, jpeg: bytes) -> "CameraFrameMessage":
        """Return camera frame message from JPEG."""
        return CameraFrameMessage("jpeg", 0, 0, jpeg)
