"""Differential drive vehicle entity."""

import math
import time
from pathlib import Path

from dt_modeling.dynamics.pwm_dynamics import wheel_speed_from_pwm_commands
from duckietown_messages.actuators import DifferentialPWM
from duckietown_messages.standard import Header, Integer

from packages.duckiematrix_engine.entities.matrix_entity import State
from duckietown_messages.geometry_3d import Transformation
from packages.duckiematrix_engine.entities.robot_entity_abs import (
    RobotEntityAbs,
    Wheel,
)
from packages.duckiematrix_engine.messages.drive_keyboard_action_message import (
    DriveKeyboardActionMessage,
)
from packages.duckiematrix_engine.types_ import Protocol

type Side = str
type Key = str


class DifferentialDriveEntityAbs(RobotEntityAbs):
    """Abstract differential drive vehicle entity."""

    KEYBOARD_COMMAND_EXPIRATION_SECS = 0.5
    KEYBOARD_SPEED_V = 0.41
    KEYBOARD_SPEED_W = 8.3
    WHEELS_COMMAND_EXPIRATION_SECS = 0.5

    _baseline: float
    _encoders: dict[Side, Key]
    _has_encoders: bool
    _keyboard_command: RobotEntityAbs.StickyMatrixOutput
    _pwm_wheels_command: RobotEntityAbs.StickyWorldOutput
    _radius: float
    _wheels: dict[str, Wheel]
    _wheels_state: dict[str, State]
    world: RobotEntityAbs.ProxiedWorldSide

    def __init__(
        self,
        matrix_key: str,
        baseline: float,
        world_key: str | None = None,
    ) -> None:
        """Initialze abstract differential drive vehicle entity."""
        super().__init__(matrix_key, world_key)
        self._baseline = baseline
        self._wheels = {
            "left": self._get_wheel("left"),
            "right": self._get_wheel("right"),
        }
        if self._wheels["left"]:
            self._radius = self._wheels["left"].radius
        else:
            self._radius = 0
        self._has_encoders = (
            self._wheels["left"] is not None
            and self._wheels["left"].encoder is not None
            and self._wheels["right"] is not None
            and self._wheels["right"].encoder is not None
        )
        self._wheels_state = {
            "left": self._get_wheel_state("left"),
            "right": self._get_wheel_state("right"),
        }
        # mark the wheels topic as 'wanted' by this entity
        path = (
            Path(self.matrix_key)
            / "actuator"
            / "wheels"
            / "base"
            / "pwm_filtered"
        )
        wheels_key = path.as_posix()
        self.world_outputs.add(wheels_key)
        self._encoders = {}
        # mark the wheels encoder ticks as 'provided' by this entity
        if self._has_encoders:
            path = (
                Path(self.matrix_key)
                / "sensor"
                / "wheel_encoder"
                / "left"
                / "ticks"
            )
            wheel_encoder_left_key = path.as_posix()
            path = (
                Path(self.matrix_key)
                / "sensor"
                / "wheel_encoder"
                / "right"
                / "ticks"
            )
            wheel_encoder_right_key = path.as_posix()
            self._encoders["left"] = wheel_encoder_left_key
            self._encoders["right"] = wheel_encoder_right_key
            # declare world inputs
            self.world_inputs.add(wheel_encoder_left_key)
            self.world_inputs.add(wheel_encoder_right_key)
            # remap the wheel encoder ticks
            path = (
                Path(self.world_key)
                / "sensor"
                / "wheel_encoder"
                / "left"
                / "ticks"
            )
            out_wheel_encoder_left_key = path.as_posix()
            path = (
                Path(self.world_key)
                / "sensor"
                / "wheel_encoder"
                / "right"
                / "ticks"
            )
            out_wheel_encoder_right_key = path.as_posix()
            self.world.remap(
                wheel_encoder_left_key,
                out_wheel_encoder_left_key,
            )
            self.world.remap(
                wheel_encoder_right_key,
                out_wheel_encoder_right_key,
            )
            # tell the world side that this is something we want to send
            self.world.declare_input(out_wheel_encoder_left_key, Integer)
            self.world.declare_input(out_wheel_encoder_right_key, Integer)
        # create sticky world outputs
        # - wheels pwm
        path = (
            Path(self.world_key)
            / "actuator"
            / "wheels"
            / "base"
            / "pwm_filtered"
        )
        out_wheels_pwm = path.as_posix()
        self._pwm_wheels_command = RobotEntityAbs.StickyWorldOutput(
            self.world,
            wheels_key,
            self.WHEELS_COMMAND_EXPIRATION_SECS,
            DifferentialPWM,
        )
        # remap the wheel pwm
        self.world.remap(wheels_key, out_wheels_pwm)
        # keep an eye out for our wheels data coming out of the world
        # side
        self.world.declare_output(out_wheels_pwm, DifferentialPWM)
        # create sticky matrix outputs
        # - keyboard command
        self._keyboard_command = RobotEntityAbs.StickyMatrixOutput(
            self.matrix,
            Protocol.INPUT,
            "keyboard",
            self.KEYBOARD_COMMAND_EXPIRATION_SECS,
            DriveKeyboardActionMessage,
        )
        self._static = False

    def _compute_ticks(self) -> tuple[int, int]:
        ticks_left = int(
            (0.5 * self._wheels_state["left"].roll / math.pi)
            * self._wheels["left"].encoder.resolution,
        )
        ticks_right = int(
            (0.5 * self._wheels_state["right"].roll / math.pi)
            * self._wheels["right"].encoder.resolution,
        )
        return ticks_left, ticks_right

    def _move_chassis(self, v: float, w: float, delta_time: float) -> None:
        # linear and angular displacement
        self.state.x += v * math.cos(self.state.yaw) * delta_time
        self.state.y += v * math.sin(self.state.yaw) * delta_time
        self.state.yaw += w * delta_time
        self.state.v_x = v
        self.state.w_z = w
        # update frames
        self.state.commit()

    def _rotate_wheels(
        self,
        w_left: float,
        w_right: float,
        delta_time: float,
    ) -> None:
        self._update_wheels_state(w_left, w_right, delta_time)
        # update frames
        self._wheels_state["left"].commit()
        self._wheels_state["right"].commit()
        # encoder ticks
        if self._has_encoders:
            self._update_encoder_ticks()

    def _update_encoder_ticks(self) -> None:
        ticks_left, ticks_right = self._compute_ticks()
        # update world
        timestamp = time.time()
        header = Header(timestamp=timestamp)
        integer = Integer(header=header, data=ticks_left)
        self.world.input("", self._encoders["left"], integer)
        integer = Integer(header=header, data=ticks_right)
        self.world.input("", self._encoders["right"], integer)

    def _update_wheels_state(
        self,
        w_left: float,
        w_right: float,
        delta_time: float,
    ) -> None:
        # angular displacements
        theta_left = w_left * delta_time
        theta_right = w_right * delta_time
        self._wheels_state["left"].roll += theta_left
        self._wheels_state["left"].w_x = w_left
        self._wheels_state["right"].roll += theta_right
        self._wheels_state["right"].w_x = w_right

    def _get_wheel(self, name: str) -> Wheel | None:
        wheel_name = f"{self.matrix_key}/wheel_{name}"
        wheel_dict = self._engine.map_.wheels.get(wheel_name)
        if wheel_dict is not None:
            return Wheel.from_dict(wheel_dict)
        return None

    def _get_wheel_speeds(self, v: float, w: float) -> tuple[float, float]:
        # linear velocities
        v_left = v - (w * self._baseline) / 2
        v_right = v + (w * self._baseline) / 2
        # angular velocities
        w_left = v_left / self._radius
        w_right = v_right / self._radius
        return w_left, w_right

    def _get_wheel_state(self, name: str) -> State | None:
        wheel_key = f"{self.matrix_key}/wheel_{name}"
        state = State(wheel_key)
        state.relative_to = self.matrix_key
        return state

    def _step_physics(
        self,
        w_left: float,
        w_right: float,
        delta_time: float,
    ) -> None:
        # linear velocities of the wheels' center
        v_left = w_left * self._radius
        v_right = w_right * self._radius
        # linear and angular velocity of the chassis
        v = (v_right + v_left) / 2
        w = (v_right - v_left) / self._baseline
        # wheel rotation
        self._rotate_wheels(w_left, w_right, delta_time)
        self._move_chassis(v, w, delta_time)

    def update(self, delta_time: float) -> None:
        """Update."""
        super().update(delta_time)
        pose_reset_cmd = self.world.output("", self._pose_reset_key, Transformation)
        if pose_reset_cmd is not None:
            self._apply_pose_reset(pose_reset_cmd)
        w_left = 0
        w_right = 0
        # PWM wheels
        pwm_wheels_cmd = self._pwm_wheels_command.value()
        if pwm_wheels_cmd is not None:
            pwm_left = pwm_wheels_cmd.left
            pwm_right = pwm_wheels_cmd.right
            # robot's kinematics
            if self._baseline > 0:
                w_left, w_right = wheel_speed_from_pwm_commands(
                    pwm_left,
                    pwm_right,
                )
        # keyboard command
        keyboard_command = self._keyboard_command.value()
        if not self._reset and keyboard_command is not None:
            if keyboard_command.reset:
                self._reset = True
            else:
                v = (
                    int(keyboard_command.forward)
                    - int(keyboard_command.backward)
                ) * self.KEYBOARD_SPEED_V
                w = (
                    int(keyboard_command.left) - int(keyboard_command.right)
                ) * self.KEYBOARD_SPEED_W
                # wheel rotation
                if self._radius > 0:
                    w_left, w_right = self._get_wheel_speeds(v, w)
        if self._reset:
            for state in (
                self.state,
                self._wheels_state["left"],
                self._wheels_state["right"],
            ):
                state.reset()
        if self._baseline > 0:
            self._step_physics(w_left, w_right, delta_time)
        self._reset = False
