"""Internal CBOR message."""

import dataclasses
from collections.abc import Buffer
from io import BytesIO
from typing import _GenericAlias

from cbor2 import CBORDecodeError, CBORDecoder, CBOREncoder


@dataclasses.dataclass
class InternalCBORMessage:
    """Internal CBOR message."""

    def _sanitize_value(
        self,
        value: "InternalCBORMessage | list | set | tuple | dict",
    ) -> "InternalCBORMessage | list | set | tuple | dict | bytes":
        if isinstance(value, InternalCBORMessage):
            return value.to_bytes()
        if isinstance(value, list | set | tuple):
            return [self._sanitize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._sanitize_value(v) for k, v in value.items()}
        return value

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> "InternalCBORMessage":
        """Return message from buffer."""
        file = BytesIO(buffer)
        decoder = CBORDecoder(file)
        result = {}
        for field in dataclasses.fields(cls):
            value = decoder.decode()
            if type(field.type) is type:
                if isinstance(value, bytes) and issubclass(
                    field.type,
                    InternalCBORMessage,
                ):
                    value = field.type.from_buffer(value)
                if (
                    value is not None
                    and not isinstance(field.type, _GenericAlias)
                    and not isinstance(value, field.type)
                ):
                    value_class = type(value)
                    message = (
                        f"Could not decode buffer into type '{cls.__name__}', "
                        f"field '{field.name}' is of type "
                        f"'{field.type.__name__}' but the buffer contains an "
                        f"object of type '{value_class.__name__}' in that "
                        "position."
                    )
                    raise CBORDecodeError(message)
            result[field.name] = value
        return cls(**result)

    def to_bytes(self) -> bytes:
        """Return bytes from message."""
        file = BytesIO()
        encoder = CBOREncoder(file)
        for field in dataclasses.fields(self):
            field_value = getattr(self, field.name)
            sanitized_value = self._sanitize_value(field_value)
            encoder.encode(sanitized_value)
        return encoder.fp.getvalue()
