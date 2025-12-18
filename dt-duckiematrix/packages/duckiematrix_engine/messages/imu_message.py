"""IMU message."""

import dataclasses

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)


@dataclasses.dataclass
class IMUMessage(InternalCBORMessage):
    """IMU message."""

    a_x: float
    a_y: float
    a_z: float
    w_x: float
    w_y: float
    w_z: float
