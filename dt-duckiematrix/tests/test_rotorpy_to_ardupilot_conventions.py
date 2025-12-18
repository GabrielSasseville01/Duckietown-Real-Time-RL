"""RotorPy to ArduPilot conventions test."""

import logging

import numpy as np
from scipy.spatial.transform import Rotation

from packages.duckiematrix_engine.entities.QuadcopterEntity import enu_to_body

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RotationError(Exception):
    """Rotation error."""

    def __init__(self, axis: str | None = None) -> None:
        """Initialize rotation error."""
        if axis is None:
            message = "Identity quaternion failed."
        else:
            message = f"Rotation around the {axis}-axis failed."
        super().__init__(message)


def test_enu_to_body() -> None:
    """Test ENU to body.

    Test the `enu_to_body` function with different rotations.
    """
    # Input acceleration in ENU frame
    acceleration_enu = [0, 0, -9.81]  # Gravity in ENU
    # Test 1: Identity quaternion (no rotation)
    rotation_identity = Rotation.from_quat([0, 0, 0, 1])
    acceleration_body = enu_to_body(acceleration_enu, rotation_identity)
    if not np.allclose(acceleration_body, acceleration_enu):
        raise RotationError
    # Test 2: 90 degree rotation around the x-axis
    axis = "x"
    rotation_x = Rotation.from_euler(axis, 90, degrees=True)
    acceleration_body_x = enu_to_body(acceleration_enu, rotation_x)
    expected_x = [0, -9.81, 0]
    if not np.allclose(acceleration_body_x, expected_x):
        raise RotationError(axis)
    # Test 3: 90 degree rotation around the y-axis
    axis = "y"
    rotation_y = Rotation.from_euler(axis, 90, degrees=True)
    acceleration_body_y = enu_to_body(acceleration_enu, rotation_y)
    expected_y = [9.81, 0, 0]
    if not np.allclose(acceleration_body_y, expected_y):
        raise RotationError(axis)
    # Test 4: 90 degree rotation around the z-axis
    axis = "z"
    rotation_z = Rotation.from_euler(axis, 90, degrees=True)
    acceleration_body_z = enu_to_body(acceleration_enu, rotation_z)
    expected_z = [0, 0, -9.81]
    if not np.allclose(acceleration_body_z, expected_z):
        raise RotationError(axis)
    logger.info("All tests passed!")
