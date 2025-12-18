"""Duckiebot dynamics."""

import geometry
import numpy as np
from dt_modeling.dynamics.platform_dynamics import PlatformDynamics
from dt_modeling.dynamics.pwm_dynamics import (
    DynamicModel,
    PWMCommands,
    get_DB18_uncalibrated,
)


def create_db18_entity(
    x_0: float = 0,
    y_0: float = 0,
    phi_0: float = 0,
    delay: float = 0,
    _: float = 0,
) -> PlatformDynamics:
    """Create DB18 entity."""
    last_pose = geometry.SE2_from_xytheta([x_0, y_0, phi_0])
    last_vel = geometry.se2_from_linear_angular(linear=[0, 0], angular=0)
    nominal_duckie = get_DB18_uncalibrated(delay=delay)
    return nominal_duckie.initialize(c0=(last_pose, last_vel))


def get_pose(vehicle: PlatformDynamics) -> tuple[float, float, float]:
    """Return pose."""
    pose_se2, _ = vehicle.TSE2_from_state()
    return geometry.xytheta_from_SE2(pose_se2)


def get_linear_angular(vehicle: PlatformDynamics) -> tuple[float, float]:
    """Return linear angular."""
    __, twist = vehicle.TSE2_from_state()
    v, w = geometry.linear_angular_from_se2(twist)
    return v[0], w


def get_ticks_from_dynamics(model: DynamicModel) -> tuple[int, int]:
    """Return ticks."""
    ticks_left = int(
        model.axis_left_obs_rad / model.parameters.encoder_resolution_rad,
    )
    ticks_right = int(
        model.axis_right_obs_rad / model.parameters.encoder_resolution_rad,
    )
    return ticks_left, ticks_right


def get_wheel_rotation_from_dynamics(
    model: DynamicModel,
) -> tuple[float, float]:
    """Return wheel rotation."""
    return model.axis_left_obs_rad, model.axis_right_obs_rad


def pwm_commands_from_wheel_speed(
    omega_left: float = 0,
    omega_right: float = 0,
    motor_constant_left: float = 27,
    motor_constant_right: float = 27,
    limit: float = 1,
) -> PWMCommands:
    """Return PWM commands."""
    # This gives us the duty cycle input to each motor
    # (clipped to [-1, 1])
    u_left = omega_left / motor_constant_left
    u_right = omega_right / motor_constant_right
    u_left = np.clip(u_left, -limit, limit)
    u_right = np.clip(u_right, -limit, limit)
    return PWMCommands(motor_left=u_left, motor_right=u_right)
