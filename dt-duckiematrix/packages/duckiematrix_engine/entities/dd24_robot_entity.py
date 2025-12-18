"""DD24 robot entity."""

import logging
import time
from datetime import UTC, datetime

import numpy as np
from rotorpy.vehicles.ardupilot_multirotor import Ardupilot
from scipy.spatial.transform import Rotation

from packages.duckiematrix_engine.entities.quadcopter_entity import (
    INITIAL_STATE,
    QuadcopterEntity,
    QuadParams,
)

MOTOR_DISTANCE = 0.16  # m
Z = 0.05  # m

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DD24RobotEntity(QuadcopterEntity):
    """DD24 robot entity."""

    PARAMS = QuadParams(
        mass=0.820,
        inertia_xx=QuadParams.compute_inertia_moments(MOTOR_DISTANCE)[0],
        inertia_yy=QuadParams.compute_inertia_moments(MOTOR_DISTANCE)[1],
        inertia_zz=QuadParams.compute_inertia_moments(MOTOR_DISTANCE)[2],
        inertia_xy=0,
        inertia_yz=0,
        inertia_xz=0,
        num_rotors=4,
        rotor_radius=0.05,
        rotor_position={
            "r1": 0.5 * MOTOR_DISTANCE * np.array([1, 1, 0]),
            "r2": 0.5 * MOTOR_DISTANCE * np.array([1, -1, 0]),
            "r3": 0.5 * MOTOR_DISTANCE * np.array([-1, -1, 0]),
            "r4": 0.5 * MOTOR_DISTANCE * np.array([-1, 1, 0]),
        },
        rotor_directions=np.array([1, -1, 1, -1]),
        r_i=np.array([0, 0, 0]),
        c_d_x=0.5e-2,
        c_d_y=0.5e-2,
        c_d_z=1e-2,
        # TODO: identify ACTUAL k_eta value
        # (and normalized pwm to rad/s, hardcoded in
        # rotorpy.vehicles.Ardupilot=838) from the DD24
        k_eta=5e-6,
        k_m=1.36e-07,
        k_d=1.19e-04,
        k_z=2.32e-04,
        k_flap=0,
        tau_m=0.005,
        rotor_speed_min=0,
        rotor_speed_max=1500,
        motor_noise_std=0,
    )

    def __init__(
        self,
        matrix_key: str,
        world_key: str | None = None,
    ) -> None:
        """Initialize DD24 robot entity."""
        super().__init__(
            matrix_key,
            quadparams=self.PARAMS,
            world_key=world_key,
        )

    def update(self, delta_time: float) -> None:
        """Update."""
        super().update(delta_time)
        # ---
        collision, timestamp = self.matrix.collision()
        # TODO: do something here
        if collision is not None:
            date_time = datetime.fromtimestamp(timestamp, UTC)
            self._logger.warning(
                "%s collided with %s at %s on %s.",
                self.world_key,
                collision,
                date_time.time(),
                date_time.date(),
            )


def ramp_input(
    elapsed_time: float,
    max_speed: float = 838,
    ramp_time: float = 10,
) -> list[float]:
    """Return ramp input.

    Generate a ramp input for motor speeds.
    """
    if elapsed_time > ramp_time:
        return [0] * 4
    speed = min(max_speed, max_speed * elapsed_time / ramp_time)
    logger.info("Ramp command ON: %.2f%", elapsed_time * 100 / ramp_time)
    return [speed] * 4


if __name__ == "__main__":
    quad_params = DD24RobotEntity.DD24_PARAMS.to_dict()
    vehicle = Ardupilot(
        initial_state=INITIAL_STATE,
        quad_params=quad_params,
        aero=True,
        ardupilot_control=True,
        enable_imu_noise=False,
        enable_ground=True,
    )
    state = INITIAL_STATE
    time_step = 0.01
    elapsed_time = 0
    ramp_time = 10
    liftoff = False
    while True:
        cmd_motor_speeds = ramp_input(elapsed_time, ramp_time=ramp_time)
        state = vehicle.step(
            state,
            {"cmd_motor_speeds": cmd_motor_speeds},
            time_step,
        )
        rotation = Rotation.from_quat(state["q"], scalar_first=True)
        attitude_angles = rotation.as_euler(
            "zyx",
            degrees=True,
        )
        x, y, z = state["x"].tolist()
        # # Format the output with explicit precision
        logger.info(
            "\nAttitude angles: %7.2f, %7.2f, %7.2f\n"
            "Position [x, y, z] (m): %.3f, %.3f, %.3f\n",
            attitude_angles[0],
            attitude_angles[1],
            attitude_angles[2],
            x,
            y,
            z,
        )
        if z > Z and not liftoff:
            logger.info(
                "Drone took off at %.2f%% of throttle",
                elapsed_time * 100 / ramp_time,
            )
            liftoff = True
        time.sleep(time_step)
        elapsed_time += time_step
