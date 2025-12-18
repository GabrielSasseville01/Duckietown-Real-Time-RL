"""Types."""

import dataclasses
from enum import Enum, IntEnum


class NetworkRole(IntEnum):
    """Network role."""

    RENDERER = 1
    CLIENT = 2


class Protocol(Enum):
    """Protocol."""

    SENSOR_DATA = "sensor"
    LAYER = "layer"
    FRAME = "frame"
    NETWORK = "network"
    CONTEXT = "context"
    RENDERER = "renderer"
    INPUT = "input"
    COLLISION = "collision"


@dataclasses.dataclass
class Robot:
    """Robot."""

    type: "RobotType"
    data: dict


class RobotType(Enum):
    """Robot type."""

    DUCKIEBOT = "duckiebot"
    WATCHTOWER = "watchtower"
    TRAFFIC_LIGHT = "traffic_light"
    DUCKIECAM = "duckiecam"
    DUCKIETOWN = "duckietown"
    DUCKIEDRONE = "duckiedrone"
    WORKSTATION = "workstation"
