"""Quadcopter entity."""

from dataclasses import asdict, dataclass, field

import numpy as np
from rotorpy.vehicles.ardupilot_multirotor import Ardupilot
from scipy.spatial.transform import Rotation

from packages.duckiematrix_engine.entities.robot_entity_abs import (
    RobotEntityAbs,
)

PWM_MAX = 1900
PWM_MIN = 1100
INITIAL_STATE = {
    "x": np.array([0, 0, 0]),
    "v": np.zeros(3),
    "q": [0, 0, 0, 1],  # [i,j,k,w]
    "w": np.zeros(3),
    "wind": np.array(
        [0, 0, 0],
    ),  # Since wind is handled elsewhere, this value is overwritten
    "rotor_speeds": np.array([0, 0, 0, 0]),
}
SEND_RATE_HZ = 400


@dataclass
class QuadParams:
    """Quadcopter parameters."""

    # Inertial properties
    mass: float
    inertia_xx: float
    inertia_yy: float
    inertia_zz: float
    inertia_xy: float
    inertia_yz: float
    inertia_xz: float
    # Geometric properties
    num_rotors: int
    rotor_radius: float
    rotor_position: dict[str, np.ndarray]
    rotor_directions: np.ndarray
    r_i: np.ndarray
    # Frame aerodynamic properties
    c_d_x: float
    c_d_y: float
    c_d_z: float
    # Rotor properties
    k_eta: float
    k_m: float
    k_d: float
    k_z: float
    k_flap: float
    # Motor properties
    tau_m: float
    rotor_speed_min: float
    rotor_speed_max: float
    motor_noise_std: float

    @staticmethod
    def compute_inertia_moments(motors_distance: float) -> tuple:
        """Return inertia moments.

        Compute the inertia moments of the quadcopter from the weight
        of the robot's relevant components.

        Args:
            motors_distance (float): the distance between the motors

        Returns:
            tuple: the inertia moments of the robot

        """
        motor_weight = 0.030  # kg
        feet_weight = 0.026  # kg
        battery_weight = 0.178  # kg
        battery_distance_from_com = 0.0425  # m
        inertia_xx = (
            4 * (motor_weight + feet_weight) * motors_distance**2
            + battery_weight * battery_distance_from_com**2
        )
        inertia_yy = inertia_xx
        inertia_zz = 4 * (motor_weight + feet_weight) * motors_distance**2
        return inertia_xx, inertia_yy, inertia_zz

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return as dictionary."""
        return asdict(self)


@dataclass
class ControlCommand:
    """Control command."""

    cmd_motor_speeds: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def to_dict(self) -> dict[str, list]:
        """Return as dictionary."""
        return asdict(self)


class QuadcopterEntity(RobotEntityAbs):
    """Quadcopter entity."""

    _rotorpy_vehicle_state: dict[str, np.ndarray]

    def __init__(
        self,
        matrix_key: str,
        quadparams: QuadParams,
        world_key: str | None = None,
    ) -> None:
        """Initialize quadcopter entity."""
        super().__init__(matrix_key, world_key)
        # Instatiating the Quadrotor physics simulation
        state = self.state
        self._rotorpy_vehicle_state = INITIAL_STATE
        self._rotorpy_vehicle_state["x"] = np.array(
            [state.x, state.y, state.z],
        )
        rotation = Rotation.from_euler(
            "zyx",
            [state.yaw, state.pitch, state.roll],
        )
        self._rotorpy_vehicle_state["q"] = np.array(rotation.as_quat())
        # TODO: if port 9002 is being used this fails quietly, only
        # printing an error to STDOUT. This should raise an exception.
        self._quadrotor = Ardupilot(
            quad_params=quadparams.to_dict(),
            initial_state=self._rotorpy_vehicle_state,
            ardupilot_control=True,
            enable_ground=True,
            enable_imu_noise=False,
        )
        self._update_state(
            self._rotorpy_vehicle_state["x"],
            self._rotorpy_vehicle_state["q"],
        )
        self._engine.ensure_frequency(1000)
        self._static = False

    def _update_state(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> None:
        """Update state.

        Update the state of the entity based on the observation from the
        environment.
        """
        self.state.x, self.state.y, self.state.z = position
        # Convert the quaternion to Euler angles (radians)
        rotation = Rotation.from_quat(quaternion)
        self.state.yaw, self.state.pitch, self.state.roll = rotation.as_euler(
            "zyx",
            degrees=False,
        )
        self.state.commit()

    def update(self, delta_time: float) -> None:
        """Update."""
        super().update(delta_time)
        # The integration step is only performed if the time step is
        # positive
        if delta_time > 0:
            self._rotorpy_vehicle_state = self._quadrotor.step(
                self._rotorpy_vehicle_state,
                None,
                delta_time,
            )
        self._update_state(
            position=self._rotorpy_vehicle_state["x"],
            quaternion=self._rotorpy_vehicle_state["q"],
        )
