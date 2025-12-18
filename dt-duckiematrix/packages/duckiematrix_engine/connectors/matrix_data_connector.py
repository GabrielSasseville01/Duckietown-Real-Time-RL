"""Duckiematrix data connector."""

import time
import traceback
from threading import Thread

from cbor2 import CBORDecodeError
from zmq import PUB, SUB, SUBSCRIBE, Context, NotDone, Socket

from packages.duckiematrix_engine.connectors.abstract_connectors import (
    ConnectorAbs,
)
from packages.duckiematrix_engine.messages.session_frame_message import (
    SessionFrameMessage,
)
from packages.duckiematrix_engine.types_ import Protocol
from packages.duckiematrix_engine.utils.communication import (
    compile_topic,
    parse_topic,
)


class MatrixDataConnectorPublisher(Thread):
    """Duckiematrix data connector publisher."""

    def __init__(self, connector: "MatrixDataConnector") -> None:
        """Initialize Duckiematrix data connector publisher."""
        super().__init__(daemon=True)
        self._connector = connector
        # engine pointer
        self._engine = None

    def run(self) -> None:
        """Run."""
        matrix = self._engine.matrix
        changes = False
        while not self._engine.is_shutdown:
            timestamp = time.time()
            # push layer updates
            for key, data in matrix.iterate_updated_layers():
                sframe = SessionFrameMessage(
                    self._engine.session_id,
                    data,
                    timestamp,
                )
                message = sframe.to_bytes()
                self._connector.send(Protocol.LAYER, key, message)
            # push generic raw updates
            for protocol, inputs in matrix.iterate_updated_inputs():
                for key, data in inputs.items():
                    sframe = SessionFrameMessage(
                        self._engine.session_id,
                        data,
                        timestamp,
                    )
                    message = sframe.to_bytes()
                    self._connector.send(protocol, key, message)
            # push session closures
            if matrix.is_session_complete:
                sframe = SessionFrameMessage(
                    self._engine.session_id,
                    b"",
                    timestamp,
                )
                message = sframe.to_bytes()
                self._connector.send(Protocol.LAYER, "complete", message)
            if changes:
                self._engine.matrix.mark_as_flushed_inputs()
            changes = matrix.wait_for_inputs(1)

    def start(self) -> None:
        """Start."""
        from packages.duckiematrix_engine.engine import MatrixEngine

        self._engine = MatrixEngine.get_instance()
        super().start()


class MatrixDataConnector(ConnectorAbs):
    """Duckiematrix data connector."""

    _publisher: MatrixDataConnectorPublisher
    _received: dict[Protocol, set[str]]
    _sent: dict[Protocol, set[str]]
    _socket_pub: Socket
    _socket_sub: Socket
    hostname: str
    port_in: int | None
    port_out: int | None

    def __init__(
        self,
        hostname: str,
        port_out: int | None = None,
        port_in: int | None = None,
    ) -> None:
        """Initialize Duckiematrix data connector."""
        super().__init__("DataConnector[Matrix]")
        self.hostname = hostname
        self.port_out = None
        self.port_in = None
        context = Context()
        # socket OUT
        self._socket_sub = context.socket(SUB)
        if port_out is not None:
            uri = f"tcp://{hostname}:{port_out}"
            self._logger.info("Opening link for OUT connector at %s...", uri)
            self._socket_sub.bind(uri)
        else:
            port_out = self._socket_sub.bind_to_random_port(
                f"tcp://{hostname}",
            )
            uri = f"tcp://{hostname}:{port_out}"
            self._logger.info("Opening link for OUT connector at %s...", uri)
        self.port_out = port_out
        self._socket_sub.setsockopt_string(SUBSCRIBE, "")
        self._logger.info("Link for OUT connector now open at %s", uri)
        # socket IN
        self._socket_pub = context.socket(PUB)
        if port_in is not None:
            uri = f"tcp://{hostname}:{port_in}"
            self._logger.info("Opening link for IN connector at %s...", uri)
            self._socket_pub.bind(uri)
        else:
            port_in = self._socket_pub.bind_to_random_port(f"tcp://{hostname}")
            uri = f"tcp://{hostname}:{port_in}"
            self._logger.info("Opening link for IN connector at %s...", uri)
        self.port_in = port_in
        self._logger.info("Link for IN connector now open at %s", uri)
        # keep track of the things we send/receive
        self._sent = {}
        self._received = {}
        # publisher object
        self._publisher = MatrixDataConnectorPublisher(self)

    async def _arun(self) -> None:
        pass

    def _mark_as_received(self, protocol: Protocol, key: str) -> None:
        try:
            self._received[protocol]
        except KeyError:
            self._received[protocol] = set()
        finally:
            self._received[protocol].add(key)

    def _mark_as_sent(self, protocol: Protocol, key: str) -> None:
        try:
            self._sent[protocol]
        except KeyError:
            self._sent[protocol] = set()
        finally:
            self._sent[protocol].add(key)

    def clear_session(self) -> None:
        """Clear session."""
        for struct in self._sent.values():
            struct.clear()
        for struct in self._received.values():
            struct.clear()

    def get_received(self, protocol: Protocol) -> set[str]:
        """Return received."""
        default = set()
        return self._received.get(protocol, default)

    def get_sent(self, protocol: Protocol) -> set[str]:
        """Return sent."""
        default = set()
        return self._sent.get(protocol, default)

    def receive(
        self,
        session_id: int,
        protocol: Protocol,
        key: str,
        blob: bytes,
        timestamp: float,
    ) -> None:
        """Receive."""
        self._engine.matrix.register_output(
            session_id,
            protocol,
            key,
            blob,
            timestamp,
        )
        self._mark_as_received(protocol, key)

    def run(self) -> None:
        """Run."""
        while not self._engine.is_shutdown:
            topic, data = self._socket_sub.recv_multipart(copy=False)
            protocol, key = parse_topic(topic)
            try:
                session_frame: SessionFrameMessage = (
                    SessionFrameMessage.from_buffer(data.buffer)
                )
            except CBORDecodeError:
                self._logger.exception("Exception.")
                traceback.print_exc()
                continue
            self.receive(
                session_frame.session_id,
                protocol,
                key,
                session_frame.payload,
                session_frame.timestamp,
            )

    def send(self, protocol: Protocol, key: str, blob: bytes) -> None:
        """Send."""
        topic = compile_topic(protocol, key)
        message_tracker = self._socket_pub.send_multipart(
            (topic, blob),
            copy=False,
            track=True,
        )
        try:
            message_tracker.wait(1)
        except NotDone:
            self._logger.warning("Message could not be sent within 1 second.")
            return
        self._mark_as_sent(protocol, key)

    def start(self) -> None:
        """Start."""
        # TODO: implement proper shutdown of _publisher
        super().start()
        self._publisher.start()
