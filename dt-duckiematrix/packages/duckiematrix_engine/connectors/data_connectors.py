"""Data connectors."""

import asyncio
import traceback
from asyncio import Event
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from dtps import DTPSContext, PublisherInterface, SubscriptionInterface
from dtps_http import RawData
from duckietown_messages.base import BaseMessage
from zmq import PUB, SUB, Context, NotDone, Socket

from packages.duckiematrix_engine.connectors.abstract_connectors import (
    AsyncDataConnectorAbs,
    DataConnectorAbs,
)
from packages.duckiematrix_engine.containers.data_containers import (
    SideDataContainer,
)
from packages.duckiematrix_engine.exceptions import SideDataDecoderError


class DTPSDataConnector(AsyncDataConnectorAbs):
    """DTPS data connector."""

    _context: DTPSContext | None
    _context_ready: Event
    _inbound: bool
    _outbound: bool
    _queues: dict[str, DTPSContext]
    loop: asyncio.AbstractEventLoop | None
    publishers: dict[str, PublisherInterface]
    subscriptions: dict[str, SubscriptionInterface]

    def __init__(
        self,
        group: str,
        container: SideDataContainer,
        *,
        outbound: bool = False,
        inbound: bool = False,
    ) -> None:
        """Initialize DTPS data connector."""
        super().__init__(group, container)
        self._outbound = outbound
        self._inbound = inbound
        self._context = None
        self._context_ready = Event()
        self._queues = {}
        self.loop = None
        self.publishers = {}
        self.subscriptions = {}

    async def _arun(self) -> None:
        pass

    async def _asend(self, key: str, raw_data: RawData) -> None:
        publisher = await self._get_publisher(key)
        await publisher.publish(raw_data)
        self._mark_as_sent(key)

    async def _get_publisher(self, key: str) -> PublisherInterface:
        publisher = self.publishers.get(key, None)
        if publisher is None:
            queue = await self._dtps_queue(key)
            publisher = await queue.publisher()
            self.publishers[key] = publisher
        return publisher

    async def _dtps_queue(self, key: str) -> DTPSContext:
        queue = self._queues.get(key, None)
        if queue is None:
            await self._context_ready.wait()
            queue = await (self._context / key).queue_create()
            self._queues[key] = queue
        return queue

    async def _dtps_subscription(
        self,
        key: str,
        callback: Callable[[RawData], Awaitable],
        max_frequency: float | None = None,
    ) -> SubscriptionInterface:
        subscription = self.subscriptions.get(key, None)
        if subscription is None:
            queue = await self._dtps_queue(key)
            subscription = await queue.subscribe(
                callback,
                max_frequency=max_frequency,
            )
            self.subscriptions[key] = subscription
        return subscription

    def _get_callback(self, key: str, decoder: type[BaseMessage]) -> Any:
        async def callback(raw_data: RawData) -> None:
            if not self._inbound:
                # TODO: too silent?
                return
            data = decoder.from_rawdata(raw_data)
            self.receive(key, data)

        return callback

    async def _task(self, coroutine: Coroutine, name: str) -> None:
        try:
            await coroutine
        except Exception:
            self._logger.exception("Exception in task: %s", name)
            traceback.print_exc()

    async def astart(self, context: DTPSContext) -> None:
        """Start asynchronously."""
        self.loop = asyncio.get_event_loop()
        self._context = context / self.group
        self._context_ready.set()
        self._publisher.start()

    def declare_input(self, key: str, _: type[BaseMessage]) -> None:
        """Declare input."""
        if key in self.publishers:
            return
        coroutine = self._get_publisher(key)
        coroutine = self._task(coroutine, "DTPSDataConnector.declare_input")
        asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def declare_output(
        self,
        key: str,
        decoder: type[BaseMessage],
        max_frequency: float | None = None,
    ) -> None:
        """Declare output."""
        if key in self.subscriptions:
            return
        callback = self._get_callback(key, decoder)
        coroutine = self._dtps_subscription(key, callback, max_frequency)
        coroutine = self._task(coroutine, "DTPSDataConnector.declare_output")
        asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def receive(self, key: str, data: BaseMessage) -> None:
        """Receive."""
        self._container.register_queue("output", key, data)
        self._mark_as_received(key)

    def send(self, key: str, data: BaseMessage) -> None:
        """Send."""
        if not self._outbound:
            # TODO: too silent?
            return
        if key not in self.publishers:
            self._logger.error(
                "You are trying to publish data with key '%s' but a publisher "
                "for this data does not exist. You must declare one using the "
                "'declare_input()' method first.",
                key,
            )
            return
        # TODO: check how efficient this is
        raw_data = data.to_rawdata()
        coroutine = self._asend(key, raw_data)
        coroutine = self._task(coroutine, "DTPSDataConnector.send")
        asyncio.run_coroutine_threadsafe(coroutine, self.loop)


class ZMQDataConnector(DataConnectorAbs):
    """ZMQ data connector."""

    _socket_pub: Socket | None
    _socket_sub: Socket | None
    hostname: str
    ports: dict[str, int | None]

    def __init__(
        self,
        group: str,
        hostname: str,
        container: SideDataContainer,
        *,
        outbound: bool = False,
        inbound: bool = False,
        port_out: int | None = None,
        port_in: int | None = None,
    ) -> None:
        """Initialize ZMQ data connector."""
        super().__init__(group, container)
        self.hostname = hostname
        self.ports = {
            "in": port_in,
            "out": port_out,
        }
        self._socket_sub = None
        self._socket_pub = None
        context = Context()
        if outbound:
            self._create_socket("out", context)
        if inbound:
            self._create_socket("in", context)

    async def _arun(self) -> None:
        pass

    def _create_socket(self, direction: str, context: Context) -> None:
        socket_name, socket_type = (
            ("_socket_sub", SUB) if direction == "in" else ("_socket_pub", PUB)
        )
        socket: Socket | None = getattr(self, socket_name)
        socket = context.socket(socket_type)
        port = self.ports[direction]
        if port is not None:
            uri = f"tcp://{self.hostname}:{port}"
            uppercase_direction = direction.upper()
            self._logger.info(
                "Opening link for %s connector at %s...",
                uppercase_direction,
                uri,
            )
            socket.bind(uri)
        else:
            port = socket.bind_to_random_port(f"tcp://{self.hostname}")
            uri = f"tcp://{self.hostname}:{port}"
            uppercase_direction = direction.upper()
            self._logger.info(
                "Opening link for %s connector at %s...",
                uppercase_direction,
                uri,
            )
        uppercase_direction = direction.upper()
        self._logger.info(
            "Link for %s connector now open at %s",
            uppercase_direction,
            uri,
        )

    def receive(self, key: str, data: bytes) -> None:
        """Receive."""
        try:
            decoded_data = self._container.decode_queue("output", data)
            self._container.register_queue("output", key, decoded_data)
            self._mark_as_received(key)
        except SideDataDecoderError:
            self._logger.exception("Some data could not be decoded.")

    def run(self) -> None:
        """Run."""
        if self._socket_sub is None:
            return
        while not self._engine.is_shutdown:
            # TODO: except for invalid number of parts
            key, data = self._socket_sub.recv_multipart()
            key = key.decode("ascii")
            # register data
            self.receive(key, data)

    def send(self, key: str, data: bytes) -> None:
        """Send."""
        key_raw = key.encode("ascii")
        message_tracker = self._socket_pub.send_multipart(
            (key_raw, data),
            copy=False,
            track=True,
        )
        try:
            message_tracker.wait(1)
        except NotDone:
            self._logger.warning("Message could not be sent within 1 second.")
            return
        self._mark_as_sent(key)

    def start(self) -> None:
        """Start."""
        super().start()
        if self._socket_pub is not None:
            self._publisher.start()
