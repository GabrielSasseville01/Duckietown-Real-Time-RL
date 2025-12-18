"""Data containers."""

import logging
from collections import defaultdict
from collections.abc import Generator
from functools import partial
from queue import Empty
from threading import Semaphore

import cbor2
from dtps_http import RawData
from duckietown_messages.base import BaseMessage

from packages.duckiematrix_engine.exceptions import (
    SideDataDecoderError,
)
from packages.duckiematrix_engine.messages.internal_cbor_message import (
    InternalCBORMessage,
)
from packages.duckiematrix_engine.utils.fifo_queue import FIFOQueue
from packages.duckiematrix_engine.utils.monitored_condition import (
    MonitoredCondition,
)


class SideDataContainer:
    """Side data container."""

    _flushed_queues: dict[str, MonitoredCondition]
    _logger: logging.Logger
    _name: str
    _new_queues: dict[str, MonitoredCondition]
    _queue_dictionaries: dict[str, defaultdict[str, FIFOQueue]]
    _queues_lock: dict[str, Semaphore]
    _queues_type: dict[str, type]
    _received: set[str]
    _sent: set[str]
    _updated_queues: dict[str, set[str]]
    queues: dict[str, partial[FIFOQueue]]

    def __init__(
        self,
        name: str,
        inputs_type: type,
        outputs_type: type,
        inputs_queue_size: int,
        outputs_queue_size: int,
    ) -> None:
        """Initialize side data container."""
        self._name = name
        # types
        self._queues_type = {
            "input": inputs_type,
            "output": outputs_type,
        }
        # containers
        self._queue_dictionaries = {
            "input": defaultdict(lambda: FIFOQueue(inputs_queue_size)),
            "output": defaultdict(lambda: FIFOQueue(outputs_queue_size)),
        }
        # updates
        self._updated_queues = {
            "input": set(),
            "output": set(),
        }
        # sent / received
        self._sent = set()
        self._received = set()
        # locks
        self._queues_lock = {
            "input": Semaphore(),
            "output": Semaphore(),
        }
        # events
        self._new_queues = {
            "input": MonitoredCondition(),
            "output": MonitoredCondition(),
        }
        self._flushed_queues = {
            "input": MonitoredCondition(),
            "output": MonitoredCondition(),
        }
        self._logger = logging.getLogger(name)

    @staticmethod
    def _decode(
        data: bytes | BaseMessage,
        decoder: type,
    ) -> bytes | InternalCBORMessage | object:
        if isinstance(data, decoder):
            return data
        if isinstance(data, RawData) and issubclass(decoder, BaseMessage):
            try:
                decoded_data = decoder.from_rawdata(data)
            except BaseException as error:
                data_class = type(data)
                raise SideDataDecoderError(
                    error,
                    data_class,
                    decoder,
                ) from error
        elif issubclass(decoder, InternalCBORMessage):
            try:
                decoded_data = decoder.from_buffer(data)
            except cbor2.CBORDecodeError as error:
                data_class = type(data)
                raise SideDataDecoderError(
                    error,
                    data_class,
                    decoder,
                ) from error
        elif decoder is dict:
            try:
                decoded_data = cbor2.loads(data)
            except cbor2.CBORDecodeError as error:
                data_class = type(data)
                raise SideDataDecoderError(
                    error,
                    data_class,
                    decoder,
                ) from error
        else:
            try:
                decoded_data = decoder(data)
            except BaseException as error:
                data_class = type(data)
                raise SideDataDecoderError(
                    error,
                    data_class,
                    decoder,
                ) from error
        # make sure the object was decoded
        if not isinstance(decoded_data, decoder):
            data_class = type(data)
            raise SideDataDecoderError(
                None,
                data_class,
                decoder,
                f"Object of type {data.__class__.__name__} could not be "
                f"decoded into type {decoder.__name__}.",
            )
        return decoded_data

    def clear(self) -> None:
        """Clear."""
        self._sent.clear()
        self._received.clear()

    # TODO: drop bytes
    def decode_queue(
        self,
        queue_name: str,
        data: bytes | BaseMessage,
    ) -> bytes | InternalCBORMessage | object:
        """Return decoded queue."""
        return self._decode(data, self._queues_type[queue_name])

    def get_received(self) -> set[str]:
        """Return received."""
        return self._received

    def get_sent(self) -> set[str]:
        """Return sent."""
        return self._sent

    def iterate_updated_queue(
        self,
        queue_name: str,
    ) -> Generator[tuple[str, bytes | RawData]]:
        """Iterate updated queue."""
        with self._queues_lock[queue_name]:
            for key in self._updated_queues[queue_name]:
                queue = self._queue_dictionaries[queue_name][key]
                while True:
                    try:
                        value = queue.get()
                    except Empty:
                        break
                    yield key, value
            self._updated_queues[queue_name].clear()

    def mark_as_flushed(self, queue_name: str) -> None:
        """Mark as flushed."""
        flushed_queue = self._flushed_queues[queue_name]
        with flushed_queue:
            flushed_queue.notify_all()

    def mark_as_received(self, key: str) -> None:
        """Mark as received."""
        self._received.add(key)

    def mark_as_sent(self, key: str) -> None:
        """Mark as sent."""
        self._sent.add(key)

    def register_queue(
        self,
        queue_name: str,
        key: str,
        data: bytes | InternalCBORMessage | object | BaseMessage,
    ) -> None:
        """Register queue."""
        queue_type = self._queues_type[queue_name]
        if not isinstance(data, queue_type):
            given = data.__class__.__name__
            expected = queue_type.__name__
            self._logger.warning(
                "You are trying to register an %s of type %s against the %s "
                "data container, which expects %ss of type %s. The data will "
                "be dropped.",
                queue_name,
                given,
                self._name,
                queue_name,
                expected,
            )
            return
        with self._queues_lock[queue_name]:
            self._queue_dictionaries[queue_name][key].put(data)
            # mark changes
            self._updated_queues[queue_name].add(key)
            # notify threads about new data
            new_queue = self._new_queues[queue_name]
            with new_queue:
                new_queue.notify_all()

    def queue(
        self,
        queue_name: str,
        key: str,
        decoder: type,
    ) -> bytes | RawData | InternalCBORMessage | object | None:
        """Return data from queue."""
        queue = self._queue_dictionaries[queue_name][key]
        try:
            data = queue.get()
        except Empty:
            return None
        # decode (if needed)
        if data is not None and not isinstance(data, decoder):
            return self._decode(data, decoder)
        return data

    def wait_for_queue(
        self,
        queue_name: str,
        timeout: float | None = None,
    ) -> None:
        """Wait for queue."""
        new_queue = self._new_queues[queue_name]
        with new_queue:
            new_queue.wait(timeout)
