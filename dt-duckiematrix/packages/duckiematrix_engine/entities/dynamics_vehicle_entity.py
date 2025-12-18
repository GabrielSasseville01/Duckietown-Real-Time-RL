"""Dynamics vehicle entity."""

from typing import Any

from packages.duckiematrix_engine.entities.differential_drive_entity_abs import (
    DifferentialDriveEntityAbs,
)
from packages.duckiematrix_engine.utils.duckiebot_dynamics import (
    create_db18_entity,
    get_linear_angular,
    get_pose,
    get_ticks_from_dynamics,
    get_wheel_rotation_from_dynamics,
    pwm_commands_from_wheel_speed,
)

DEFAULT_DYNAMICS_PARAMETERS = {
    "commands_delay": 0,
    "motor_constant_left": 27,
    "motor_constant_right": 27,
    "trim": 0,
}


class DynamicsVehicleEntity(DifferentialDriveEntityAbs):
    """Dynamics vehicle entity."""

    _dynamic_vehicle: Any
    _vehicle_dynamics_parameters: Any

    def __init__(
        self,
        matrix_key: str,
        baseline: float,
        world_key: str | None = None,
    ) -> None:
        """Initialze dynamics vehicle entity."""
        super().__init__(matrix_key, baseline, world_key)
        if self.state is None:
            self._logger.warning("The state is 'None'.")
        self._vehicle_dynamics_parameters = (
            self._engine.map_.vehicle_dynamics.get(
                self.matrix_key,
                DEFAULT_DYNAMICS_PARAMETERS,
            )
        )
        delay = self._vehicle_dynamics_parameters.get(
            "commands_delay",
            DEFAULT_DYNAMICS_PARAMETERS["commands_delay"],
        )
        trim = self._vehicle_dynamics_parameters.get(
            "trim",
            DEFAULT_DYNAMICS_PARAMETERS["trim"],
        )
        self._dynamic_vehicle = create_db18_entity(
            self.state.x,
            self.state.y,
            self.state.yaw,
            delay,
            trim,
        )

    def _compute_ticks(self) -> tuple[int, int]:
        return get_ticks_from_dynamics(self._dynamic_vehicle)

    def _integrate_dynamics(
        self,
        w_left: float,
        w_right: float,
        delta_time: float,
    ) -> None:
        motor_constant_left = self._vehicle_dynamics_parameters.get(
            "motor_constant_left",
            DEFAULT_DYNAMICS_PARAMETERS["motor_constant_left"],
        )
        motor_constant_right = self._vehicle_dynamics_parameters.get(
            "motor_constant_right",
            DEFAULT_DYNAMICS_PARAMETERS["motor_constant_right"],
        )
        commands = pwm_commands_from_wheel_speed(
            w_left,
            w_right,
            motor_constant_left,
            motor_constant_right,
        )
        self._dynamic_vehicle = self._dynamic_vehicle.integrate(
            commands=commands,
            dt=delta_time,
        )

    def _set_chassis(
        self,
        x: float,
        y: float,
        yaw: float,
        v: float,
        w: float,
    ) -> None:
        self.state.x = x
        self.state.y = y
        self.state.yaw = yaw
        self.state.v_x = v
        self.state.w_z = w
        # update frames
        self.state.commit()

    def _step_physics(
        self,
        w_left: float,
        w_right: float,
        delta_time: float,
    ) -> None:
        if self._reset:
            delay = self._vehicle_dynamics_parameters.get(
                "commands_delay",
                DEFAULT_DYNAMICS_PARAMETERS["commands_delay"],
            )
            trim = self._vehicle_dynamics_parameters.get(
                "trim",
                DEFAULT_DYNAMICS_PARAMETERS["trim"],
            )
            self._dynamic_vehicle = create_db18_entity(
                self.state.x,
                self.state.y,
                self.state.yaw,
                delay,
                trim,
            )
        self._integrate_dynamics(w_left, w_right, delta_time)
        x, y, yaw = get_pose(self._dynamic_vehicle)
        v, w = get_linear_angular(self._dynamic_vehicle)
        # wheel rotation
        self._rotate_wheels(w_left, w_right, delta_time)
        self._set_chassis(x, y, yaw, v, w)

    def _update_wheels_state(
        self,
        _: float,
        __: float,
        delta_time: float,
    ) -> None:
        # angular displacements
        theta_left, theta_right = get_wheel_rotation_from_dynamics(
            self._dynamic_vehicle,
        )
        if delta_time > 0:
            self._wheels_state["left"].w_x = (
                theta_left - self._wheels_state["left"].roll
            ) / delta_time
            self._wheels_state["right"].w_x = (
                theta_right - self._wheels_state["right"].roll
            ) / delta_time
        self._wheels_state["left"].roll = theta_left
        self._wheels_state["right"].roll = theta_right
