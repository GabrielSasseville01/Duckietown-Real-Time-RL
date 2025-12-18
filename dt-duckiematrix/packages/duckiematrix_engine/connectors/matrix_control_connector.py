"""Duckiematrix control connector."""

import logging

from zmq import REP, Context, Frame, NotDone, Socket

from packages.duckiematrix_engine.connectors.abstract_connectors import (
    ConnectorAbs,
)
from packages.duckiematrix_engine.messages.context_response_message import (
    ContextResponseMessage,
)
from packages.duckiematrix_engine.messages.empty_message import EmptyMessage
from packages.duckiematrix_engine.messages.frame_freeze_request_message import (
    FrameFreezeRequestMessage,
)
from packages.duckiematrix_engine.messages.network_endpoint_message import (
    NetworkEndpointMessage,
    NetworkEndpointProtocol,
    NetworkEndpointType,
)
from packages.duckiematrix_engine.messages.network_join_message import (
    NetworkJoinMessage,
)
from packages.duckiematrix_engine.messages.network_joined_message import (
    NetworkJoinedMessage,
)
from packages.duckiematrix_engine.messages.network_leave_message import (
    NetworkLeaveMessage,
)
from packages.duckiematrix_engine.messages.network_ready_message import (
    NetworkReadyMessage,
)
from packages.duckiematrix_engine.sides.matrix_side import RendererStatus
from packages.duckiematrix_engine.types_ import NetworkRole, Protocol
from packages.duckiematrix_engine.utils.communication import (
    compile_topic,
    parse_topic,
)


class MatrixControlConnector(ConnectorAbs):
    """Duckieatrix control connector."""

    _empty_message_data: bytes
    _logger: logging.Logger
    _socket_rep: Socket

    def __init__(self, hostname: str, port_out: int) -> None:
        """Initialize Duckiematrix control connector."""
        super().__init__("MatrixControlConnector")
        context = Context()
        # socket OUT
        uri = f"tcp://{hostname}:{port_out}"
        self._logger.info("Opening link for OUT connector at %s...", uri)
        self._socket_rep = context.socket(REP)
        self._socket_rep.bind(uri)
        self._logger.info("Link for OUT connector now open at %s", uri)
        empty_message = EmptyMessage()
        self._empty_message_data = empty_message.to_bytes()

    async def _arun(self) -> None:
        pass

    def on_context_get(self) -> tuple[bytes, bytes]:
        """Return compiled topic and message on context."""
        context = self._engine.matrix.get_context()
        message = ContextResponseMessage(self._engine.map_.name, context)
        return compile_topic(Protocol.CONTEXT, "set"), message.to_bytes()

    def on_frame_freeze(self, data: Frame) -> tuple[bytes, bytes]:
        """Return compiled topic and message on frame freeze."""
        message: FrameFreezeRequestMessage = (
            FrameFreezeRequestMessage.from_buffer(data.buffer)
        )
        self._engine.matrix.layer(
            "frozen_frames",
            message.key,
            {
                "frozen": message.frozen,
            },
        )
        topic_out = "frozen" if message.frozen else "unfrozen"
        return (
            compile_topic(Protocol.FRAME, topic_out),
            self._empty_message_data,
        )

    def on_network_join(self, data: Frame) -> tuple[bytes, bytes]:
        """Return compiled topic and message on network join."""
        network_entity_desc: NetworkJoinMessage = (
            NetworkJoinMessage.from_buffer(data.buffer)
        )
        # out
        network_entity_id: str | None = None
        # - Role: RENDERER
        if network_entity_desc.role == NetworkRole.RENDERER:
            renderer_desc = network_entity_desc
            if renderer_desc.key is None:
                self._logger.debug(
                    "New renderer requesting to join the network.",
                )
                # missing key?
                if self._engine.matrix.secure:
                    self._logger.debug(
                        "Renderer was denied access. No keys were presented.",
                    )
                    return (
                        compile_topic(Protocol.NETWORK, "unauthenticated"),
                        self._empty_message_data,
                    )
                # required a specific assignment?
                if renderer_desc.id is not None:
                    self._logger.debug(
                        "Renderer with location '%s' is requesting to join the"
                        " network as '%s'.",
                        renderer_desc.location,
                        renderer_desc.id,
                    )
                    # TODO: this can overthrow sitting renderers
                    renderer = self._engine.matrix.get_renderer(
                        renderer_desc.id,
                    )
                    if renderer is None:
                        self._logger.debug(
                            "Renderer was denied access. Renderer ID not "
                            "recognized.",
                        )
                        return (
                            compile_topic(Protocol.NETWORK, "forbidden"),
                            self._empty_message_data,
                        )
                else:
                    # get next unassigned renderer slot
                    self._logger.debug(
                        "Anonymous renderer with location '%s' is requesting "
                        "to join the network.",
                        renderer_desc.location,
                    )
                    renderer = self._engine.matrix.get_renderer(
                        renderer_desc.id,
                        unassigned=True,
                    )
                    if renderer is None:
                        self._logger.debug(
                            "Renderer was denied access. Network is full.",
                        )
                        return (
                            compile_topic(Protocol.NETWORK, "full"),
                            self._empty_message_data,
                        )
            else:
                # the renderer has a key, (try to) authenticate it
                self._logger.debug(
                    "Renderer with key '%s' and location '%s' is requesting to"
                    " join the network.",
                    renderer_desc.key,
                    renderer_desc.location,
                )
                renderer = self._engine.matrix.authenticate_renderer(
                    renderer_desc.key,
                )
                if renderer is None:
                    self._logger.debug(
                        "Renderer was denied access. Key is incorrect.",
                    )
                    return (
                        compile_topic(Protocol.NETWORK, "unauthorized"),
                        self._empty_message_data,
                    )
            # assign renderer
            self._logger.debug(
                "Renderer was granted access. Will serve as '%s'.",
                renderer.id,
            )
            self._engine.matrix.update_renderer(
                renderer.id,
                location=renderer_desc.location,
                status=RendererStatus.JOINING,
            )
            # set the renderer id
            network_entity_id = renderer.id
        # send assigned ID and network configuration to the entity
        endpoints = [
            NetworkEndpointMessage(
                type=NetworkEndpointType.DATA_IN.value,
                protocol=NetworkEndpointProtocol.TCP.value,
                hostname=None,
                port=self._engine.matrix_data_connector.port_in,
            ),
            NetworkEndpointMessage(
                type=NetworkEndpointType.DATA_OUT.value,
                protocol=NetworkEndpointProtocol.TCP.value,
                hostname=None,
                port=self._engine.matrix_data_connector.port_out,
            ),
        ]
        message = NetworkJoinedMessage(network_entity_id, endpoints)
        return compile_topic(Protocol.NETWORK, "joined"), message.to_bytes()

    def on_network_leave(self, data: Frame) -> tuple[bytes, bytes]:
        """Return compiled topic and message on network leave."""
        # get renderer ID
        message = NetworkLeaveMessage.from_buffer(data.buffer)
        self._engine.matrix.unplug_renderer(message.id)
        # confirm that the renderer has been unplugged
        return compile_topic(
            Protocol.NETWORK,
            "left",
        ), self._empty_message_data

    @staticmethod
    def on_network_ping(data: Frame) -> tuple[bytes, bytes]:
        """Return compiled topic and message on network ping."""
        # send same message back
        return compile_topic(Protocol.NETWORK, "pong"), data.bytes

    def on_network_ready(self, data: Frame) -> tuple[bytes, bytes]:
        """Return compiled topic and message on network ready."""
        # TODO: this should be verified with a key
        network_entity_desc: NetworkReadyMessage = (
            NetworkReadyMessage.from_buffer(data.buffer)
        )
        # - Role: RENDERER
        if network_entity_desc.role == NetworkRole.RENDERER:
            renderer_desc = network_entity_desc
            # update renderer
            self._logger.debug("Renderer is now ready.")
            self._engine.matrix.update_renderer(
                renderer_desc.id,
                status=RendererStatus.READY,
            )
        return compile_topic(
            Protocol.NETWORK,
            "ready",
        ), self._empty_message_data

    def run(self) -> None:
        """Run."""
        while not self._engine.is_shutdown:
            topic, data = self._socket_rep.recv_multipart(copy=False)
            protocol, key = parse_topic(topic)
            request = (protocol, key)
            # route request
            reply = None
            if request == (Protocol.NETWORK, "join"):
                topic, reply = self.on_network_join(data)
            elif request == (Protocol.NETWORK, "ready"):
                topic, reply = self.on_network_ready(data)
            elif request == (Protocol.NETWORK, "ping"):
                topic, reply = self.on_network_ping(data)
            elif request == (Protocol.CONTEXT, "get"):
                topic, reply = self.on_context_get()
            elif request == (Protocol.FRAME, "freeze"):
                topic, reply = self.on_frame_freeze(data)
            elif request == (Protocol.NETWORK, "leave"):
                topic, reply = self.on_network_leave(data)
            else:
                self._logger.warning("Received unknown request %s.", request)
            if reply is None:
                encoded_string = "empty".encode("ascii")
                message_tracker = self._socket_rep.send_multipart(
                    (encoded_string, self._empty_message_data),
                    copy=False,
                    track=True,
                )
            else:
                message_tracker = self._socket_rep.send_multipart(
                    (topic, reply),
                    copy=False,
                    track=True,
                )
            try:
                message_tracker.wait(1)
            except NotDone:
                self._logger.warning(
                    "Message could not be sent within 1 second.",
                )
