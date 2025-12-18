"""Duckiematrix side."""

import dataclasses
import uuid
from collections.abc import Generator
from enum import IntEnum
from pathlib import Path
from threading import Semaphore
from typing import TYPE_CHECKING, Any, Literal

import yaml

from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)
from packages.duckiematrix_engine.messages.map_layer_message import (
    MapLayerMessage,
)
from packages.duckiematrix_engine.sides.side_abs import SideAbs
from packages.duckiematrix_engine.types_ import Protocol


class RendererStatus(IntEnum):
    """Renderer status."""

    UNASSIGNED = 0
    JOINING = 1
    READY = 10
    ERROR = 20


@dataclasses.dataclass
class Renderer:
    """Renderer."""

    id: str
    key: str
    location: str | None = None
    status: RendererStatus = RendererStatus.UNASSIGNED

    def clear(self) -> None:
        """Clear renderer."""
        self.location = None
        self.status = RendererStatus.UNASSIGNED


class MatrixSide(SideAbs):
    """Duckiematrix side."""

    if TYPE_CHECKING:
        from packages.duckiematrix_engine.engine import MatrixEngine

    _context_id: str | None
    _context_path: str | None
    _inputs: dict[Protocol, dict[str, bytes]]
    _inputs_lock: Semaphore
    _layers_lock: Semaphore
    _outputs: dict[Protocol, dict[str, tuple[bytes, float]]]
    _renderers: dict[str, Renderer]
    _session_completed: bool
    _updated_inputs: dict[Protocol, list[str]]
    _updated_layers: dict[str, list[str]]
    engine: "MatrixEngine | None"
    secure: bool

    def __init__(self, num_renderers: int, *, secure: bool) -> None:
        """Initialize Duckiematrix side."""
        super().__init__("matrix")
        uuid4 = uuid.uuid4()
        key = str(uuid4) if secure else None
        self._renderers = {}
        for i in range(num_renderers):
            self._renderers[f"renderer_{i}"] = Renderer(
                f"renderer_{i}",
                key,
            )
        self.secure = secure
        # layer-based inputs
        self._updated_layers = {}
        # generic inputs/outputs
        self._inputs = {}
        self._outputs = {}
        for protocol in Protocol:
            self._inputs[protocol] = {}
            self._outputs[protocol] = {}
        self._updated_inputs = {}
        self._context_path = None
        self._context_id = None
        self._session_completed = False
        # locks
        self._layers_lock = Semaphore()
        self._inputs_lock = Semaphore()
        # initialize data structures
        self.clear_session()
        self.engine = None

    def all_renderers_joined(self) -> bool:
        """Return `True` if all renderers joined, `False` otherwise."""
        for renderer in self._renderers.values():
            if renderer.status != RendererStatus.READY:
                return False
        return True

    def authenticate_renderer(self, key: str) -> Renderer | None:
        """Authenticate renderer."""
        # find renderer with given key
        for renderer in self._renderers.values():
            if key == renderer.key:
                return renderer
        return None

    def clear_session(self) -> None:
        """Clear session."""
        if self.engine is not None:
            self.engine.matrix_data_connector.clear_session()
        # clear data structures
        with self._inputs_lock:
            self._updated_inputs.clear()
        with self._layers_lock:
            self._updated_layers.clear()

    def complete_session(self) -> None:
        """Complete session."""
        self._session_completed = True
        self.flush_inputs()

    def get_context(self) -> bytes:
        """Return context."""
        with Path(self._context_path).open("rb") as fin:
            return fin.read()

    def get_renderer(
        self,
        renderer_id: str | None = None,
        *,
        unassigned: bool = False,
    ) -> Renderer | None:
        """Return renderer."""
        if renderer_id is not None:
            return self._renderers.get(renderer_id, None)
        # filter renderers
        for renderer in self._renderers.values():
            if unassigned and renderer.status == RendererStatus.UNASSIGNED:
                return renderer
        return None

    def input(self, protocol: Protocol, key: str, data: bytes) -> None:
        """Input."""
        with self._inputs_lock:
            # apply update:
            self._inputs[protocol][key] = data
            # mark changes:
            # - protocol -> list[str]
            if protocol not in self._updated_inputs:
                self._updated_inputs[protocol] = []
            self._updated_inputs[protocol].append(key)

    @property
    def is_session_complete(self) -> bool:
        """Return `True` if session completed, `False` otherwise."""
        value = self._session_completed
        self._session_completed = False
        return value

    def iterate_updated_inputs(
        self,
    ) -> Generator[tuple[Protocol, dict[str, bytes]], Any, None]:
        """Iterate updated inputs."""
        with self._inputs_lock:
            for protocol, keys in self._updated_inputs.items():
                yield (
                    protocol,
                    {key: self._inputs[protocol][key] for key in keys},
                )
                # clear changes
                # TODO: this might not be necessary, we are clearing
                # everything anyway
                keys.clear()
            self._updated_inputs.clear()

    def iterate_updated_layers(
        self,
    ) -> Generator[
        tuple[Literal["session_updates", "updates"], bytes],
        Any,
        None,
    ]:
        """Iterate updated layers."""
        key = "session_updates" if self.engine.gym_mode else "updates"
        with self._layers_lock:
            for layer, keys in self._updated_layers.items():
                layer_updates = {
                    layer: {key: self.engine.map_[layer][key] for key in keys},
                }
                # TODO: this is a test for CBOR encoded map layers
                """
                t = SensorGatesElement
                cbor_layer_updates = {}
                if layer == "sensor_gates":
                    for key, value in layer_updates[layer].items():
                        cbor_layer_updates[key] = t(**value).to_bytes()
                raw: bytes = MapLayerMessage(
                    layer, yaml.safe_dump(layer_updates),
                    cbor2.dumps(cbor_layer_updates)).to_bytes()
                """
                content = yaml.safe_dump(layer_updates)
                map_layer_message = MapLayerMessage(layer, content)
                data = map_layer_message.to_bytes()
                yield key, data
            self._updated_layers.clear()

    def layer(self, layer: str, key: str, data: dict) -> None:
        """Layer."""
        with self._layers_lock:
            # apply update:
            self.engine.map_[layer][key].update(data)
            # mark changes:
            # - layer -> list[str]
            if layer not in self._updated_layers:
                self._updated_layers[layer] = []
            self._updated_layers[layer].append(key)

    def notify_engine(self) -> None:
        """Notify engine."""
        with self.engine.events:
            self.engine.events.notify()

    def output(
        self,
        protocol: Protocol,
        key: str,
        decoder: type,
    ) -> tuple[bytes | InternalCBORMessage | None, float | None]:
        """Return output."""
        data_dictionary = self._outputs.get(protocol, self.empty_dict)
        data, timestamp = data_dictionary.pop(key, (None, None))
        # decode (if needed)
        if data is not None and not isinstance(data, decoder):
            if issubclass(decoder, InternalCBORMessage):
                data = decoder.from_buffer(data)
            else:
                message = "Decoder not supported."
                raise TypeError(message)
        return data, timestamp

    def print_startup_info(self) -> None:
        """Log startup information."""
        if self.secure:
            message = (
                "This engine is running in 'secure' mode, renderers will need "
                "to present a key to be accepted.\nDistribute these keys to "
                "the proper renderers:\n"
            )
            for renderer_id, renderer in self._renderers.items():
                message += (
                    f"\n\t -  {renderer_id}".ljust(24, " ") + renderer.key
                )
            message += "\n"
            self._logger.info(message)

    def register_output(
        self,
        session_id: int,
        protocol: Protocol,
        key: str,
        data: bytes,
        timestamp: float,
    ) -> None:
        """Register output."""
        # check session ID
        if session_id != self.engine.session_id:
            self._logger.warning(
                "An output with protocol '%s' and key '%s' belonging to the "
                "session #%s was received while the engine was working on "
                "session #%s. This is a problem. Investigate.",
                protocol,
                key,
                session_id,
                self.engine.session_id,
            )
            return
        # store data
        self._outputs[protocol][key] = (data, timestamp)
        # notify the engine that new stuff was received
        self.notify_engine()

    @property
    def renderers(self) -> dict[str, Renderer]:
        """Return renderers."""
        return self._renderers

    def set_context(self, context: str) -> None:
        """Set context."""
        uuid_ = uuid.uuid4()
        self._context_id = str(uuid_)
        self._context_path = context

    def start(self) -> None:
        """Start."""
        from packages.duckiematrix_engine.engine import MatrixEngine

        self.engine = MatrixEngine.get_instance()
        self._logger.setLevel(self.engine.logger.level)
        # TODO: [DTSW-4405] fix Map.renderer_authentication not existing
        # load renderers' keys from map
        renderer_auth: dict = self.engine.map_.renderer_authentication
        for renderer in self._renderers.values():
            if renderer.id in renderer_auth:
                renderer.key = renderer_auth[renderer.id].get(
                    "key",
                    renderer.key,
                )

    def unplug_renderer(self, renderer: str) -> None:
        """Unplug renderer."""
        if renderer in self._renderers:
            self._renderers[renderer].clear()

    def update_renderer(
        self,
        renderer: str,
        **kwargs: int | str,
    ) -> None:
        """Update renderer."""
        self._renderers[renderer].__dict__.update(kwargs)
