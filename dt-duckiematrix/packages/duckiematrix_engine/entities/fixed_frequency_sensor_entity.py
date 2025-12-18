"""Fixed-frequency sensor entity."""

from packages.duckiematrix_engine.entities.matrix_entity import MatrixEntity


class FixedFrequencySensorEntity(MatrixEntity):
    """Fixed-frequency sensor entity."""

    _last_event: float
    _last_gate: bool
    _period: float
    _time: float

    def __init__(
        self,
        matrix_key: str,
        frequency: float,
        world_key: str | None = None,
    ) -> None:
        """Initialize fixed-frequency sensor entity."""
        super().__init__(matrix_key, world_key)
        self._period = (1 / frequency) if frequency > 0 else 0
        self._last_event = 0
        self._last_gate = False
        self._time = 0
        self._engine.ensure_frequency(frequency)

    @property
    def gate(self) -> bool:
        """Return `True` if gate is open, `False` otherwise."""
        return self.map_.get(
            "sensor_gates",
            "",
            {
                "open": False,
            },
        )["open"]

    @gate.setter
    def gate(self, isopen: bool) -> None:
        # remember the last gate value
        self._last_gate = isopen
        # set the matrix input
        self.matrix.input(
            "sensor_gates",
            "",
            {
                "open": isopen,
            },
        )

    def update(self, delta_time: float) -> None:
        """Update."""
        self._time += delta_time
        period = self._time - self._last_event
        if (
            self._time == 0
            or period >= self._period
            or self._period - period < delta_time * 0.5
        ):
            # open sensor gate
            self.gate = True
            self._last_event = self._time
        elif delta_time == 0:
            # use last value
            self.gate = self._last_gate
        else:
            # close sensor gate: gates are closed after being consumed
            # by the renderers
            pass
