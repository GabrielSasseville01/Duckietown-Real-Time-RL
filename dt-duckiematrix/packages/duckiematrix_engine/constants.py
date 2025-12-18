"""Constants."""

import tempfile
from enum import IntEnum

DEFAULT_DATA_DIR = tempfile.mkdtemp()
DEFAULT_DELTA_T = 0.025
DEFAULT_MAPS_DIR = "/maps"
DEFAULT_MATRIX_CONNECTOR_HOST = "0.0.0.0"
DEFAULT_MATRIX_CONTROL_OUT_CONNECTOR_PORT = 7502
DEFAULT_WORLD_CONTROL_OUT_CONNECTOR_PORT = 7501


class EngineMode(IntEnum):
    """Duckiematrix Engine mode."""

    REALTIME = 0
    GYM = 1

    @classmethod
    def from_string(cls, string: str) -> "EngineMode":
        """Return mode value from string."""
        return cls.REALTIME if string == "realtime" else cls.GYM

    def to_string(self) -> str:
        """Return mode string from value."""
        return "realtime" if self == EngineMode.REALTIME else "gym"
