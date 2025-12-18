"""Exceptions."""

import traceback
from collections.abc import Iterable


class BringUpError(Exception):
    """Bring-up error."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        """Initialize bring-up error."""
        super().__init__(message, exit_code)

    @property
    def exit_code(self) -> int:
        """Return exit code."""
        return self.args[1]

    @property
    def message(self) -> str:
        """Return message."""
        return self.args[0]


class DuckiematrixEngineError(Exception):
    """Duckiematrix Engine error."""


class InvalidMapConfigurationError(Exception):
    """Invalid map configuration error."""

    def __init__(self, messages: Iterable[str], exit_code: int = 1) -> None:
        """Initialize invalid map configuration error."""
        super().__init__(messages, exit_code)

    @property
    def exit_code(self) -> int:
        """Return exit code."""
        return self.args[1]

    @property
    def messages(self) -> Iterable[str]:
        """Return message."""
        return self.args[0]


class SideDataDecoderError(DuckiematrixEngineError):
    """Side-data decoder error."""

    expected: type
    message: str | None
    received: type
    trigger: BaseException | None

    def __init__(
        self,
        trigger: BaseException | None,
        received: type,
        expected: type,
        message: str | None = None,
    ) -> None:
        """Initialize side-data decoder error."""
        self.trigger = trigger
        self.received = received
        self.expected = expected
        self.message = message
        super().__init__(trigger)

    @property
    def trigger_stack(self) -> str:
        """Return trigger traceback."""
        if self.trigger is None:
            return ""
        formatted_traceback = traceback.format_tb(self.trigger.__traceback__)
        return "".join(formatted_traceback)


class SideDataEncoderError(DuckiematrixEngineError):
    """Side-data encoder error."""
