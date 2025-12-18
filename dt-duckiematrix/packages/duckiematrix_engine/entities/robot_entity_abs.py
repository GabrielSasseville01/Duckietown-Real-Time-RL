"""Abstract robot entity."""

import dataclasses
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from dtps_http import RawData
from duckietown_messages.actuators import CarLights
from duckietown_messages.base import BaseMessage
from duckietown_messages.geometry_3d import (
    Position,
    Quaternion,
    Transformation,
    Twist,
    Vector3,
)
from duckietown_messages.sensors import (
    AngularVelocities,
    Camera,
    CompressedImage,
    LinearAccelerations,
    Range,
)
from duckietown_messages.sensors.imu import Imu
from duckietown_messages.standard import Boolean, Empty, Header
from lens_distortion_utils.lens_map import (
    compute_field_of_view,
    compute_lens_uv_textures,
    compute_pinhole_camera_matrix,
)
from scipy.spatial.transform import Rotation

from packages.duckiematrix_engine.entities.fixed_frequency_sensor_entity import (
    FixedFrequencySensorEntity,
)
from packages.duckiematrix_engine.entities.matrix_entity import MatrixEntity
from packages.duckiematrix_engine.exceptions import BringUpError
from packages.duckiematrix_engine.messages.empty_message import EmptyMessage
from packages.duckiematrix_engine.messages.float_message import FloatMessage
from packages.duckiematrix_engine.messages.imu_message import IMUMessage
from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)
from packages.duckiematrix_engine.sides.world_side import WorldSide
from packages.duckiematrix_engine.types_ import Protocol

if TYPE_CHECKING:
    from duckietown_messages.colors import RGBA


type MatrixKey = str
type WorldKey = str
type LightName = str
type LightKey = str

CAMERA_NAME = "front_center"


@dataclasses.dataclass
class WheelEncoder:
    """Wheel encoder."""

    resolution: int

    @staticmethod
    def from_dict(data: dict | None) -> "WheelEncoder | None":
        """Return from dictionary."""
        return WheelEncoder(**data) if data else None


@dataclasses.dataclass
class Wheel:
    """Wheel."""

    radius: float
    encoder: WheelEncoder | None = None

    def as_dict(self) -> dict:
        """Return as dictionary."""
        return {
            "radius": self.radius,
            "encoder": self.encoder.__dict__ if self.encoder else None,
        }

    @staticmethod
    def from_dict(data: dict) -> "Wheel":
        """Return from dictionary."""
        wheel_encoder = WheelEncoder.from_dict(data["encoder"])
        return Wheel(data["radius"], wheel_encoder)


class RobotEntityAbs(MatrixEntity):
    """Abstract robot entity."""

    # this is used to compute the underlying pinhole camera that the
    # lens distortion is applied to
    CAMERA_SCALE = 2

    _light_key: str
    _reset: bool
    _reset_key: str
    _pose_reset_key: str
    camera_info: Camera | None
    camera_key: str | None
    imus: list[tuple[str, dict]]
    lights: dict[LightName, LightKey]
    tofs: list[tuple[str, dict]]
    world: "RobotEntityAbs.ProxiedWorldSide"

    class ProxiedWorldSide(MatrixEntity.ProxiedWorldSide):
        """Proxied world side."""

        _remap_m2w: dict[MatrixKey, WorldKey]
        _remap_w2m: dict[WorldKey, MatrixKey]

        def __init__(
            self,
            matrix_key: str,
            world_key: str,
            proxied: WorldSide,
        ) -> None:
            """Initialize proxied world side."""
            super().__init__(matrix_key, world_key, proxied)
            self._remap_m2w = {}
            self._remap_w2m = {}

        def _to_matrix(self, world_key: str) -> str:
            return self._remap_w2m.get(world_key, world_key)

        def _to_world(self, matrix_key: str) -> str:
            return self._remap_m2w.get(matrix_key, matrix_key)

        def declare_input(
            self,
            world_key: str,
            encoder: type[BaseMessage],
        ) -> None:
            """Declare input."""
            self.proxied.declare_input("robot", world_key, encoder)

        def declare_output(
            self,
            world_key: str,
            decoder: type[BaseMessage],
            max_frequency: float | None = None,
        ) -> None:
            """Declare output."""
            self.proxied.declare_output(
                "robot",
                world_key,
                decoder,
                max_frequency,
            )

        def input(
            self,
            _: str,
            resource_or_key: str,
            data: BaseMessage,
        ) -> None:
            """Return input."""
            matrix_key = self.resource_key(resource_or_key, self._matrix_key)
            key = self._to_world(matrix_key)
            self.proxied.input("robot", key, data)

        def output(
            self,
            _: str,
            resource_or_key: str,
            decoder: type,
        ) -> bytes | RawData | InternalCBORMessage | object | None:
            """Return output."""
            world_key = self.resource_key(resource_or_key, self._world_key)
            key = self._to_matrix(world_key)
            return self.proxied.output("robot", key, decoder)

        def remap(self, matrix_key: str, world_key: str) -> None:
            """Remap."""
            self._remap_m2w[matrix_key] = world_key
            self._remap_w2m[world_key] = matrix_key

    class StickyWorldOutput:
        """Sticky world output.

        Maintains a world output valid over time, according to a
        given 'expiration' value.

        When the world produces an output, that value will be kept
        available until it expires or a new value is received, in which
        case the expiration resets.
        """

        _decoder: type
        _expiration: float
        _last_update: float
        _resource: str
        _value: bytes | RawData | InternalCBORMessage | object | None
        _world: "RobotEntityAbs.ProxiedWorldSide"

        def __init__(
            self,
            world: "RobotEntityAbs.ProxiedWorldSide",
            resource: str,
            expiration: float,
            decoder: type,
        ) -> None:
            """Initialize sticky world output."""
            self._world = world
            self._resource = resource
            self._expiration = expiration
            self._decoder = decoder
            self._last_update = 0
            self._value = None

        @property
        def _value_expired(self) -> bool:
            return time.time() - self._last_update > self._expiration

        def value(
            self,
        ) -> bytes | RawData | InternalCBORMessage | object | None:
            """Return value."""
            new_value = self._world.output("", self._resource, self._decoder)
            if new_value is not None:
                self._value = new_value
                self._last_update = time.time()
            return self._value if not self._value_expired else None

    class StickyMatrixOutput:
        """Sticky Duckiematrix output.

        Maintains a matrix output valid over time, according to a
        given 'expiration' value.

        When the matrix produces an output, that value will be kept
        available until it expires or a new value is received, in which
        case the expiration resets.
        """

        _decoder: type[InternalCBORMessage]
        _expiration: float
        _last_update: float
        _matrix: "RobotEntityAbs.ProxiedMatrixSide"
        _protocol: Protocol
        _resource: str
        _value: bytes | InternalCBORMessage | None

        def __init__(
            self,
            matrix: "RobotEntityAbs.ProxiedMatrixSide",
            protocol: Protocol,
            resource: str,
            expiration: float,
            decoder: type[InternalCBORMessage],
        ) -> None:
            """Initialize sticky world output."""
            self._matrix = matrix
            self._protocol = protocol
            self._resource = resource
            self._expiration = expiration
            self._decoder = decoder
            self._last_update = 0
            self._value = None

        @property
        def _value_expired(self) -> bool:
            current_time = time.time()
            return current_time - self._last_update > self._expiration

        def value(self) -> bytes | InternalCBORMessage | None:
            """Return value."""
            new_value, _ = self._matrix.output(
                self._protocol,
                self._resource,
                self._decoder,
            )
            if new_value is not None:
                self._value = new_value
                self._last_update = time.time()
            return self._value if not self._value_expired else None

    def __init__(self, matrix_key: str, world_key: str | None = None) -> None:
        """Initialize sticky Duckiematrix output."""
        from packages.duckiematrix_engine.engine import MatrixEngine

        # if the world_key is not given (no override), try and get it
        # from the map layers
        if world_key is None:
            engine = MatrixEngine.get_instance()
            world_key = engine.map_.physical_robots.get(matrix_key, None)
        # if no key is provided, just reuse the matrix key
        if world_key is None:
            world_key = matrix_key
        # create entity
        super().__init__(matrix_key, world_key)
        # replace proxied world with a simpler one
        self.world = RobotEntityAbs.ProxiedWorldSide(
            self.matrix_key,
            self.world_key,
            self._engine.world,
        )
        # robot's presence
        self.matrix.input(
            "sensor_gates",
            "presence",
            {
                "open": False,
            },
        )
        # mark the presence topic as 'wanted' by this entity
        presence_key = self.matrix.resource_key("presence")
        self.matrix_outputs.add(presence_key)
        self._initialize_camera()
        self._initialize_tofs()
        self._initialize_imus()
        self._initialize_lights()
        path = Path(self.world_key) / "state" / "pose"
        world_key = path.as_posix()
        self.world.declare_input(world_key, Transformation)
        path = Path(self.world_key) / "state" / "twist"
        world_key = path.as_posix()
        self.world.declare_input(world_key, Twist)
        path = Path(self.matrix_key) / "state" / "reset"
        self._reset_key = path.as_posix()
        path = Path(self.world_key) / "state" / "reset"
        out_reset_key = path.as_posix()
        self.world.remap(self._reset_key, out_reset_key)
        self.world_outputs.add(self._reset_key)
        self.world.declare_output(out_reset_key, Boolean)
        self._reset = False
        # pose reset/teleport
        path = Path(self.matrix_key) / "state" / "pose_reset"
        self._pose_reset_key = path.as_posix()
        path = Path(self.world_key) / "state" / "pose_reset"
        out_pose_reset_key = path.as_posix()
        self.world.remap(self._pose_reset_key, out_pose_reset_key)
        self.world_outputs.add(self._pose_reset_key)
        self.world.declare_output(out_pose_reset_key, Transformation)

    def _apply_pose_reset(self, pose_reset: Transformation) -> None:
        """Apply a requested pose reset to the robot state."""
        if self.state is None:
            return
        position = pose_reset.position
        rotation = pose_reset.rotation
        # scipy expects quaternions in (x, y, z, w) order
        quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
        yaw, pitch, roll = Rotation.from_quat(quaternion).as_euler("zyx", degrees=False)
        self.state.x = position.x
        self.state.y = position.y
        self.state.z = position.z
        self.state.yaw = yaw
        self.state.pitch = pitch
        self.state.roll = roll
        # zero velocities on teleport
        self.state.v_x = 0
        self.state.v_y = 0
        self.state.v_z = 0
        self.state.w_x = 0
        self.state.w_y = 0
        self.state.w_z = 0
        # update the reference pose so subsequent resets reuse it
        self.state.initial_pose.update(
            {
                "x": position.x,
                "y": position.y,
                "z": position.z,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
            }
        )
        self.state.initial_twist.update(
            {
                "v_x": 0,
                "v_y": 0,
                "v_z": 0,
                "w_x": 0,
                "w_y": 0,
                "w_z": 0,
            }
        )
        self.state.commit()
        self._reset = True

    def _initialize_camera(self) -> None:
        camera_key, camera = self.map_.find(
            "cameras",
            self.matrix_key,
            ("name", CAMERA_NAME),
        )
        camera_info_key = None
        self.camera_info = None
        if camera_key is not None:
            # robot's camera lens
            rendering_config = self.map_.get(
                "rendering_configuration",
                camera_key,
                {},
            )
            if rendering_config.get("lens_distortion", False):
                # get camera parameters
                camera_matrix = np.array(camera["camera_matrix"])
                distortion_parameters = np.array(
                    camera["distortion_parameters"],
                )
                width, height = camera["width"], camera["height"]
                # scale the camera matrix
                camera_matrix[0, 0] *= self.CAMERA_SCALE
                camera_matrix[0, 2] *= self.CAMERA_SCALE
                camera_matrix[1, 1] *= self.CAMERA_SCALE
                camera_matrix[1, 2] *= self.CAMERA_SCALE
                width *= int(self.CAMERA_SCALE)
                height *= int(self.CAMERA_SCALE)
                # compute camera matrix of underlying pinhole camera
                camera_matrix_rect = compute_pinhole_camera_matrix(
                    camera_matrix,
                    distortion_parameters,
                    width,
                    height,
                )
                # create physical camera layers entry
                horizontal_fov, vertical_fov = compute_field_of_view(
                    camera_matrix_rect,
                    width,
                    height,
                )
                self.map_.set(
                    "physical_cameras",
                    camera_key,
                    {
                        "horizontal_fov": horizontal_fov,
                        "vertical_fov": vertical_fov,
                        "width": width,
                        "height": height,
                    },
                )
                # create (or reuse) assets
                path = Path(self.matrix_key) / CAMERA_NAME
                camera_asset_dir = path.as_posix()
                lens_texture_u_asset = self._engine.map_.asset(
                    camera_asset_dir,
                    "lens_texture_u.png",
                )
                lens_texture_v_asset = self._engine.map_.asset(
                    camera_asset_dir,
                    "lens_texture_v.png",
                )
                # reuse existing assets (if available)
                lens_texture_u_asset_path = Path(lens_texture_u_asset.fpath)
                lens_texture_v_asset_path = Path(lens_texture_v_asset.fpath)
                if (
                    lens_texture_u_asset_path.is_file()
                    and lens_texture_v_asset_path.is_file()
                ):
                    # make sure the files are valid PNGs
                    for png_fpath in (
                        lens_texture_u_asset.fpath,
                        lens_texture_v_asset.fpath,
                    ):
                        if cv2.imread(png_fpath) is None:
                            message = (
                                "Invalid lens distortion map, not an image: "
                                f"{png_fpath}"
                            )
                            self._logger.error(message)
                            raise BringUpError(message)
                    self._logger.info(
                        "Using 'lens_distortion' UV texture assets from disk.",
                    )
                else:
                    self._logger.info(
                        "Generating 'lens_distortion' UV texture assets.",
                    )
                    # create lens textures
                    lens_texture_u, lens_texture_v = compute_lens_uv_textures(
                        camera_matrix,
                        camera_matrix_rect,
                        distortion_parameters,
                        width,
                        height,
                    )
                    # save lens textures to disk
                    lens_texture_u_asset.make_dirs()
                    lens_texture_v_asset.make_dirs()
                    cv2.imwrite(lens_texture_u_asset.fpath, lens_texture_u)
                    cv2.imwrite(lens_texture_v_asset.fpath, lens_texture_v)
            # robot's camera gate
            camera_entity = FixedFrequencySensorEntity(
                camera_key,
                camera["frame_rate"],
            )
            self._engine.register_entity(camera_key, camera_entity)
            # mark the camera topic as 'wanted' by this entity
            self.matrix_outputs.add(camera_key)
            # remap the camera topic
            path = Path(self.world_key) / "sensor" / "camera" / CAMERA_NAME
            out_camera_key = (path / "jpeg").as_posix()
            self.world.remap(camera_key, out_camera_key)
            # declare the camera info topic
            camera_info_key = f"{camera_key}/info"
            out_camera_info_key = (path / "info").as_posix()
            self.world.remap(camera_info_key, out_camera_info_key)
            # tell the world side that these are things we want to send
            self.world.declare_input(out_camera_key, CompressedImage)
            self.world.declare_input(out_camera_info_key, Camera)
            frame_id = f"/{self.world_key}/camera/{CAMERA_NAME}/optical_frame"
            # define the camera info
            timestamp = time.time()
            header = Header(timestamp=timestamp)
            fov = np.deg2rad(160)
            self.camera_info = Camera(
                # -- base
                header=header,
                # -- sensor
                name=CAMERA_NAME,
                type="camera",
                simulated=True,
                description="RGB8/JPEG Camera",
                frame_id=frame_id,
                frequency=camera["frame_rate"],
                maker="Duckietown",
                model="Virtual",
                # -- camera
                width=camera["width"],
                height=camera["height"],
                # TODO: should be computed from hFOV and vFOV
                fov=fov,
            )
            path = Path(camera_key) / "info"
            self.camera_info_key = path.as_posix()
        self.camera_key = camera_key
        self.camera_info_key = camera_info_key

    def _initialize_imus(self) -> None:
        self.imus = self.map_.find_all(
            "imus",
            self.matrix_key,
            return_key=True,
        )
        for imu_key, imu in self.imus:
            imu_entity = FixedFrequencySensorEntity(imu_key, imu["frequency"])
            self._engine.register_entity(imu_key, imu_entity)
            # mark the imu topic as 'wanted' by this entity
            self.matrix_outputs.add(imu_key)
            # remap the imu topic
            imu_path: Path = (
                Path(self.world_key) / "sensor" / "imu" / imu["name"]
            )
            out_imu_linear_acceleration_key = (
                imu_path / "acceleration" / "linear"
            ).as_posix()
            out_imu_angular_velocity_key = (
                imu_path / "velocity" / "angular"
            ).as_posix()
            out_imu_all_key = (imu_path / "all").as_posix()
            self.world.remap(
                f"{imu_key}/acceleration/linear",
                out_imu_linear_acceleration_key,
            )
            self.world.remap(
                f"{imu_key}/velocity/angular",
                out_imu_angular_velocity_key,
            )
            self.world.remap(f"{imu_key}/all", out_imu_all_key)
            # tell the world side that this is something we want to send
            self.world.declare_input(
                out_imu_linear_acceleration_key,
                LinearAccelerations,
            )
            self.world.declare_input(
                out_imu_angular_velocity_key,
                AngularVelocities,
            )
            self.world.declare_input(out_imu_all_key, Imu)

    def _initialize_lights(self) -> None:
        lights = self.map_.find_all("lights", self.matrix_key, return_key=True)
        self.lights = {}
        for light_key, light in lights:
            self.lights[light["name"]] = light_key
        # remap the lights topic
        path = (
            Path(self.matrix_key) / "actuator" / "lights" / "base" / "pattern"
        )
        self._light_key = path.as_posix()
        path = (
            Path(self.world_key) / "actuator" / "lights" / "base" / "pattern"
        )
        out_light_key = path.as_posix()
        self.world.remap(self._light_key, out_light_key)
        # mark the light topic as 'wanted' by this entity
        self.world_outputs.add(self._light_key)
        # tell the world side that this is something we want to receive
        # TODO: [DTSW-6530] this crashes the engine due to a NoneType
        # error when there are no Cameras (or no sensors in general?)
        self.world.declare_output(out_light_key, CarLights)

    def _initialize_tofs(self) -> None:
        self.tofs = self.map_.find_all(
            "time_of_flights",
            self.matrix_key,
            return_key=True,
        )
        for tof_key, tof in self.tofs:
            tof_entity = FixedFrequencySensorEntity(tof_key, tof["frequency"])
            self._engine.register_entity(tof_key, tof_entity)
            # mark the tof topic as 'wanted' by this entity
            self.matrix_outputs.add(tof_key)
            # remap the tof topic
            path: Path = (
                Path(self.world_key)
                / "sensor"
                / "time_of_flight"
                / tof["name"]
                / "range"
            )
            out_tof_key = path.as_posix()
            self.world.remap(tof_key, out_tof_key)
            # tell the world side that this is something we want to send
            self.world.declare_input(out_tof_key, Range)

    def flush(self) -> None:
        """Flush."""
        # passthrough presence sensor
        presence, timestamp = self.matrix.output(
            Protocol.SENSOR_DATA,
            "presence",
            EmptyMessage,
        )
        if presence is not None:
            header = Header(timestamp=timestamp)
            empty = Empty(header=header)
            self.world.input("", "presence", empty)
        # passthrough camera frame
        if self.camera_key is not None:
            # publish camera jpeg
            jpeg, timestamp = self.matrix.output(
                Protocol.SENSOR_DATA,
                self.camera_key,
                bytes,
            )
            if jpeg is not None:
                header = Header(timestamp=timestamp)
                compressed_image = CompressedImage(
                    header=header,
                    format="jpeg",
                    data=jpeg,
                )
                self.world.input("", self.camera_key, compressed_image)
            # publish camera info
            if self.camera_info is not None:
                self.world.input("", self.camera_info_key, self.camera_info)
        # lights
        lights: CarLights | None = self.world.output(
            "",
            self._light_key,
            CarLights,
        )
        if lights is not None:
            for light_name, light_key in self.lights.items():
                rgba: RGBA = getattr(lights, light_name, None)
                if rgba is not None:
                    self.matrix.input(
                        "lights",
                        light_key,
                        {
                            "intensity": rgba.a,
                            "color": rgba.hex,
                            "type": "spot",
                            "range": 2,
                            "angle": 0,
                        },
                    )
        # time of flight sensors
        for tof_key, _ in self.tofs:
            tof_range, timestamp = self.matrix.output(
                Protocol.SENSOR_DATA,
                tof_key,
                FloatMessage,
            )
            if tof_range is not None:
                header = Header(timestamp=timestamp)
                range_ = Range(header=header, data=tof_range.value)
                self.world.input("", tof_key, range_)
        # IMUs
        for imu_key, _ in self.imus:
            imu_data, timestamp = self.matrix.output(
                Protocol.SENSOR_DATA,
                imu_key,
                IMUMessage,
            )
            if imu_data is not None:
                header = Header(timestamp=timestamp)
                linear_accelerations = LinearAccelerations(
                    header=header,
                    x=imu_data.a_x,
                    y=imu_data.a_y,
                    z=imu_data.a_z,
                )
                angular_velocities = AngularVelocities(
                    header=header,
                    x=imu_data.w_x,
                    y=imu_data.w_y,
                    z=imu_data.w_z,
                )
                imu = Imu(
                    header=header,
                    linear_acceleration=linear_accelerations,
                    angular_velocity=angular_velocities,
                )
                self.world.input(
                    "",
                    f"{imu_key}/acceleration/linear",
                    linear_accelerations,
                )
                self.world.input(
                    "",
                    f"{imu_key}/velocity/angular",
                    angular_velocities,
                )
                self.world.input("", f"{imu_key}/all", imu)
        # reset
        reset: Boolean | None = self.world.output("", self._reset_key, Boolean)
        if reset is not None:
            data: bool = getattr(reset, "data", None)
            if data is not None:
                self._reset = data

    def update(self, _: float) -> None:
        """Update."""
        if self.state and (not self._state_published or not self._static):
            timestamp = time.time()
            header = Header(timestamp=timestamp)
            position = Position(
                header=header,
                x=self.state.x,
                y=self.state.y,
                z=self.state.z,
            )
            rotation = Rotation.from_euler(
                "zyx",
                [self.state.yaw, self.state.pitch, self.state.roll],
                degrees=False,
            )
            quaternion = rotation.as_quat(scalar_first=True)
            rotation = Quaternion(
                header=header,
                w=quaternion[0],
                x=quaternion[1],
                y=quaternion[2],
                z=quaternion[3],
            )
            pose = Transformation(
                header=header,
                source=self.state.relative_to,
                target=self.matrix_key,
                position=position,
                rotation=rotation,
            )
            linear_velocity = Vector3(
                header=header,
                x=self.state.v_x,
                y=self.state.v_y,
                z=self.state.v_z,
            )
            angular_velocity = Vector3(
                header=header,
                x=self.state.w_x,
                y=self.state.w_y,
                z=self.state.w_z,
            )
            twist = Twist(
                header=header,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
            )
            path = Path(self.world_key) / "state" / "pose"
            key = path.as_posix()
            self.world.proxied.robot_connector.send(key, pose)
            path = Path(self.world_key) / "state" / "twist"
            key = path.as_posix()
            self.world.proxied.robot_connector.send(key, twist)
            self._state_published = True
        # open the presence gate (gym mode only)
        if self._engine.gym_mode:
            self.matrix.input(
                "sensor_gates",
                "presence",
                {
                    "open": True,
                },
            )
