"""Matrix entity."""

import copy
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from dtps_http import RawData
from duckietown_messages.base import BaseMessage

from packages.duckiematrix_engine.exceptions import (
    InvalidMapConfigurationError,
)
from packages.duckiematrix_engine.map import Map
from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)
from packages.duckiematrix_engine.messages.string_message import StringMessage
from packages.duckiematrix_engine.sides.matrix_side import MatrixSide
from packages.duckiematrix_engine.sides.world_side import WorldSide
from packages.duckiematrix_engine.types_ import Protocol
from packages.duckiematrix_engine.utils import BUILTIN_TYPES


class State:
    """State."""

    if TYPE_CHECKING:
        from packages.duckiematrix_engine.engine import MatrixEngine

    _engine: "MatrixEngine | None"
    _key: str
    empty_dict: dict
    initial_pose: dict[str, float]
    initial_twist: dict[str, float]

    def __init__(self, key: str) -> None:
        """Initialize state."""
        from packages.duckiematrix_engine.engine import MatrixEngine

        self.empty_dict = {}
        self._engine = MatrixEngine.get_instance()
        self._key = key
        # make frame if it does not exist
        if self._key not in self._engine.map_.frames:
            self._engine.map_.frames[self._key] = {
                "relative_to": None,
                "pose": {
                    "x": 0,
                    "y": 0,
                    "z": 0,
                    "roll": 0,
                    "pitch": 0,
                    "yaw": 0,
                },
                "twist": {
                    "v_x": 0,
                    "v_y": 0,
                    "v_z": 0,
                    "w_x": 0,
                    "w_y": 0,
                    "w_z": 0,
                },
            }
        elif "twist" not in self._engine.map_.frames:
            self._engine.map_.frames[self._key]["twist"] = {
                "v_x": 0,
                "v_y": 0,
                "v_z": 0,
                "w_x": 0,
                "w_y": 0,
                "w_z": 0,
            }
        frame = self.as_frame()
        self.initial_pose = copy.deepcopy(frame["pose"])
        self.initial_twist = copy.deepcopy(frame["twist"])

    def _get_property(self, type_: str, field: str) -> float | None:
        return self._engine.map_.frames[self._key][type_].get(field, None)

    def _set_property(
        self,
        type_: str,
        field: str,
        value: float | None,
    ) -> None:
        # make sure the frame is not frozen
        frozen_frames = self._engine.map_.frozen_frames
        frozen_frames_dictionary = frozen_frames.get(
            self._key,
            self.empty_dict,
        )
        if frozen_frames_dictionary.get("frozen", False):
            return
        # make sure the value is YAML-serializable
        if value is not None and value.__class__ not in BUILTIN_TYPES:
            # Numpy float values
            if isinstance(value, np.float32 | np.float64):
                value = float(value)
            # Numpy int values
            elif isinstance(
                value,
                np.int8 | np.int16 | np.int32 | np.int64 | np.uint,
            ):
                value = int(value)
            # unknown value
            else:
                value_class = type(value)
                self._engine.logger.error(
                    "You cannot set the property '%s' to an object of type "
                    "'%s'",
                    field,
                    value_class,
                )
                return
        self._engine.map_.frames[self._key][type_][field] = value

    def as_frame(self) -> dict:
        """Return state as frame."""
        return self._engine.map_.frames.get(self._key, None)

    def commit(self) -> None:
        """Commit."""
        data = self.as_frame()
        self._engine.matrix.layer("frames", self._key, data)

    @property
    def pitch(self) -> float:
        """Return pitch."""
        return self._get_property("pose", "pitch")

    @pitch.setter
    def pitch(self, value: float) -> None:
        self._set_property("pose", "pitch", value)

    @property
    def relative_to(self) -> str | None:
        """Return frame this state is relative to."""
        return self._engine.map_.frames[self._key].get("relative_to", None)

    @relative_to.setter
    def relative_to(self, value: str) -> None:
        self._engine.map_.frames[self._key]["relative_to"] = value

    def reset(self) -> None:
        """Reset."""
        self.x = self.initial_pose["x"]
        self.y = self.initial_pose["y"]
        self.z = self.initial_pose["z"]
        self.roll = self.initial_pose["roll"]
        self.pitch = self.initial_pose["pitch"]
        self.yaw = self.initial_pose["yaw"]
        self.v_x = self.initial_twist["v_x"]
        self.v_y = self.initial_twist["v_y"]
        self.v_z = self.initial_twist["v_z"]
        self.w_x = self.initial_twist["w_x"]
        self.w_y = self.initial_twist["w_y"]
        self.w_z = self.initial_twist["w_z"]

    @property
    def roll(self) -> float:
        """Return roll."""
        return self._get_property("pose", "roll")

    @roll.setter
    def roll(self, value: float) -> None:
        self._set_property("pose", "roll", value)

    @property
    def v_x(self) -> float:
        """Return v_x."""
        return self._get_property("twist", "v_x")

    @v_x.setter
    def v_x(self, value: float) -> None:
        self._set_property("twist", "v_x", value)

    @property
    def v_y(self) -> float:
        """Return v_y."""
        return self._get_property("twist", "v_y")

    @v_y.setter
    def v_y(self, value: float) -> None:
        self._set_property("twist", "v_y", value)

    @property
    def v_z(self) -> float:
        """Return v_z."""
        return self._get_property("twist", "v_z")

    @v_z.setter
    def v_z(self, value: float) -> None:
        self._set_property("twist", "v_z", value)

    @property
    def w_x(self) -> float:
        """Return w_x."""
        return self._get_property("twist", "w_x")

    @w_x.setter
    def w_x(self, value: float) -> None:
        self._set_property("twist", "w_x", value)

    @property
    def w_y(self) -> float:
        """Return w_y."""
        return self._get_property("twist", "w_y")

    @w_y.setter
    def w_y(self, value: float) -> None:
        self._set_property("twist", "w_y", value)

    @property
    def w_z(self) -> float:
        """Return w_z."""
        return self._get_property("twist", "w_z")

    @w_z.setter
    def w_z(self, value: float) -> None:
        self._set_property("twist", "w_z", value)

    @property
    def x(self) -> float:
        """Return x."""
        return self._get_property("pose", "x")

    @x.setter
    def x(self, value: float) -> None:
        self._set_property("pose", "x", value)

    @property
    def y(self) -> float:
        """Return y."""
        return self._get_property("pose", "y")

    @y.setter
    def y(self, value: float) -> None:
        self._set_property("pose", "y", value)

    @property
    def yaw(self) -> float:
        """Return yaw."""
        return self._get_property("pose", "yaw")

    @yaw.setter
    def yaw(self, value: float) -> None:
        self._set_property("pose", "yaw", value)

    @property
    def z(self) -> float:
        """Return z."""
        return self._get_property("pose", "z")

    @z.setter
    def z(self, value: float) -> None:
        self._set_property("pose", "z", value)


class MatrixEntity:
    """Matrix entity."""

    if TYPE_CHECKING:
        from packages.duckiematrix_engine.engine import MatrixEngine

    _engine: "MatrixEngine | None"
    _logger: logging.Logger
    _state_published: bool
    _static: bool
    map_: "MatrixEntity.ProxiedMap"
    matrix: "MatrixEntity.ProxiedMatrixSide"
    matrix_inputs: set[str]
    matrix_key: str
    matrix_outputs: set[str]
    state: State | None
    world: "MatrixEntity.ProxiedWorldSide"
    world_inputs: set[str]
    world_key: str | None
    world_outputs: set[str]

    class ProxiedMap:
        """Proxied map."""

        _key: str
        _proxied: Map

        def __init__(self, key: str, proxied: Map) -> None:
            """Initialize proxied map."""
            self._key = key
            self._proxied = proxied

        def find(
            self,
            layer: str,
            prefix: str,
            field_value: tuple[str, str] | None = None,
        ) -> tuple[str | None, dict]:
            """Return match."""
            matches = self.find_all(
                layer,
                prefix,
                field_value,
                max_matches=1,
                return_key=True,
            )
            return matches[0] if matches else (None, {})

        def find_all(
            self,
            layer: str,
            prefix: str,
            field_value: tuple[str, str] | None = None,
            max_matches: int = 0,
            *,
            return_key: bool = False,
        ) -> list[dict] | list[tuple[str, dict]]:
            """Return matches."""
            # prefix must have an ending slash to avoid aliasing
            prefix = prefix.rstrip("/") + "/"
            layer_dictionary = self._proxied.__getattr__(layer)
            matches = []
            for key, entity in layer_dictionary.items():
                # filter by prefix
                if not key.startswith(prefix):
                    continue
                # filter by field=value
                if field_value is not None:
                    field, value = field_value
                    if entity.get(field) != value:
                        continue
                match_ = (key, entity) if return_key else entity
                matches.append(match_)
                # stop once the wanted number of matches are found
                if 0 < max_matches <= len(matches):
                    break
            return matches

        def get(
            self,
            layer: str,
            resource_or_key: str,
            default: str = "",
        ) -> dict:
            """Return layer."""
            layer_dictionary = self._proxied.__getattr__(layer)
            resource_key = self.resource_key(resource_or_key)
            return layer_dictionary.get(resource_key, default)

        # TODO: lru cache for this method
        def resource_key(self, resource: str) -> str:
            """Return resource key."""
            if resource.startswith(f"{self._key}/"):
                return resource
            return f"{self._key}/{resource}".rstrip("/")

        def set(self, layer: str, resource_or_key: str, value: dict) -> None:
            """Set."""
            layer_dictionary = self._proxied.__getattr__(layer)
            resource_key = self.resource_key(resource_or_key)
            layer_dictionary[resource_key] = value

    class ProxiedMatrixSide:
        """Proxied Duckiematrix side."""

        _key: str
        _proxied: MatrixSide

        def __init__(self, key: str, proxied: MatrixSide) -> None:
            """Initialize proxied Duckiematrix side."""
            self._key = key
            self._proxied = proxied

        def collision(self) -> tuple[str | None, float | None]:
            """Return collision."""
            collision, timestamp = self._proxied.output(
                Protocol.COLLISION,
                self._key,
                StringMessage,
            )
            if collision is None:
                return (None, None)
            return collision.value, timestamp

        def input(self, layer: str, resource_or_key: str, data: dict) -> None:
            """Input."""
            resource_key = self.resource_key(resource_or_key)
            self._proxied.layer(layer, resource_key, data)

        def output(
            self,
            protocol: Protocol,
            resource_or_key: str,
            decoder: type[InternalCBORMessage] | bytes,
        ) -> tuple[bytes | InternalCBORMessage | None, float | None]:
            """Return output."""
            resource_key = self.resource_key(resource_or_key)
            return self._proxied.output(protocol, resource_key, decoder)

        # TODO: lru cache for this method
        def resource_key(self, resource: str) -> str:
            """Return resource key."""
            if resource.startswith(f"{self._key}/"):
                return resource
            return f"{self._key}/{resource}".rstrip("/")

    class ProxiedWorldSide:
        """Proxied world side."""

        _matrix_key: str
        _world_key: str | None
        proxied: WorldSide

        def __init__(
            self,
            matrix_key: str,
            world_key: str | None,
            proxied: WorldSide,
        ) -> None:
            """Initialize proxied world side."""
            self._matrix_key = matrix_key
            self._world_key = world_key
            self.proxied = proxied

        def input(
            self,
            group: str,
            resource_or_key: str,
            data: BaseMessage,
        ) -> None:
            """Input."""
            resource_key = self.resource_key(resource_or_key)
            self.proxied.input(group, resource_key, data)

        def output(
            self,
            group: str,
            resource_or_key: str,
            decoder: type,
        ) -> bytes | RawData | InternalCBORMessage | object | None:
            """Return output."""
            resource_key = self.resource_key(resource_or_key)
            return self.proxied.output(group, resource_key, decoder)

        # TODO: lru cache for this method
        def resource_key(self, resource: str, prefix: str = "") -> str:
            """Return resource key."""
            prefix = prefix or self._world_key
            if resource.startswith(f"{prefix}/"):
                # full key already
                return resource
            # partial key
            return f"{prefix}/{resource}".rstrip("/")

    def __init__(self, matrix_key: str, world_key: str | None) -> None:
        """Initialize matrix entity."""
        from packages.duckiematrix_engine.engine import MatrixEngine

        self.matrix_key = matrix_key
        self.world_key = world_key
        self.matrix_inputs = set()
        self.matrix_outputs = set()
        self.world_inputs = set()
        self.world_outputs = set()
        self._engine = MatrixEngine.get_instance()
        self._logger = logging.getLogger(f"Entity[{matrix_key}]")
        self._logger.level = self._engine.logger.level
        self.matrix = MatrixEntity.ProxiedMatrixSide(
            self.matrix_key,
            self._engine.matrix,
        )
        self.world = MatrixEntity.ProxiedWorldSide(
            self.matrix_key,
            self.world_key,
            self._engine.world,
        )
        self.map_ = MatrixEntity.ProxiedMap(self.matrix_key, self._engine.map_)
        # verify components' uniqueness
        self._check_components_uniqueness()
        frame = self._engine.map_.frames.get(self.matrix_key, None)
        self.state = State(self.matrix_key) if frame is not None else None
        self._state_published = False

    def _check_components_uniqueness(self) -> None:
        # we are looking for multiple components attached to this entity
        # of the same type and with the same name
        # - for every layer name
        for layer in self._engine.map_.layers:
            # find components that are attached to this entity
            components = self.map_.find_all(
                layer,
                self.matrix_key,
                return_key=True,
            )
            # keep only those with names
            components = [
                component for component in components if "name" in component[1]
            ]
            # find overlaps
            name_to_keys = defaultdict(list)
            for component in components:
                name_to_keys[component[1]["name"]].append(component[0])
            # report conflicts
            messages = []
            for name, keys in name_to_keys.items():
                if len(keys) > 1:
                    message = (
                        f"CONFLICT: components with keys {keys} belong to the "
                        f"same entity '{self.matrix_key}' but share the same "
                        f"name '{name}'. This is not allowed, names of "
                        "components must be unique within the same entity."
                    )
                    self._logger.error(message)
                    messages.append(message)
            if messages:
                raise InvalidMapConfigurationError(messages)

    def early_update(self, _: float) -> None:
        """Early update."""
        message = (
            "The method `early_update` on the base class `MatrixEntity` was "
            "called, this should not have happened.",
        )
        raise RuntimeError(message)

    def flush(self) -> None:
        """Flush."""
        message = (
            "The method `flush` on the base class `MatrixEntity` was called, "
            "this should not have happened.",
        )
        raise RuntimeError(message)

    def late_update(self, _: float) -> None:
        """Late update."""
        message = (
            "The method `late_update` on the base class `MatrixEntity` was "
            "called, this should not have happened.",
        )
        raise RuntimeError(message)

    def matrix_inputs_wanted(self) -> set[str] | None:
        """Return Duckiematrix inputs."""
        return self.matrix_inputs

    def matrix_outputs_wanted(self) -> set[str] | None:
        """Return Duckiematrix outputs."""
        return self.matrix_outputs

    def update(self, _: float) -> None:
        """Update."""
        message = (
            "The method `update` on the base class `MatrixEntity` was called, "
            "this should not have happened.",
        )
        raise RuntimeError(message)

    def world_inputs_wanted(self) -> set[str] | None:
        """Return world inputs."""
        return self.world_inputs

    def world_outputs_wanted(self) -> set[str] | None:
        """Return world outputs."""
        return self.world_outputs


class MatrixEntityBehavior(MatrixEntity):
    """Duckiematrix entity behavior."""

    def __init__(
        self,
        matrix_key: str,
        world_key: str | None,
    ) -> None:
        """Initialize Duckiematrix entity behavior."""
        super().__init__(matrix_key, world_key)
