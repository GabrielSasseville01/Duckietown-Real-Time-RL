"""Communication."""

from zmq import Frame

from packages.duckiematrix_engine.types_ import Protocol


def compile_topic(protocol: Protocol, key: str) -> bytes:
    """Compile topic."""
    return f"{protocol.value}:/{key}".encode("ascii")


def parse_topic(topic: Frame) -> tuple[Protocol, str]:
    """Parse topic."""
    topic_string = topic.bytes.decode("ascii")
    protocol, key = topic_string.split(":/", 1)
    return Protocol(protocol), key
