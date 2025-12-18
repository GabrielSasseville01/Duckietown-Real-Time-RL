"""Sun script."""

import math

from packages.duckiematrix_engine.entities.matrix_entity import (
    MatrixEntityBehavior,
)


class SunScript(MatrixEntityBehavior):
    """Sun script."""

    _speed: float
    _time: float

    def __init__(
        self,
        matrix_key: str,
        world_key: str | None,
        speed: float = 400,
    ) -> None:
        """Initialize sun script."""
        from packages.duckiematrix_engine.engine import MatrixEngine

        super().__init__(matrix_key, world_key)
        self._speed = 2 * math.pi * speed / (24 * 60 * 60)
        self._time = 0
        engine = MatrixEngine.get_instance()
        engine.ensure_frequency(60)

    def update(self, delta_time: float) -> None:
        """Update."""
        self._time += delta_time
        if self.state:
            self.state.roll = self._time * self._speed + math.pi
            self.state.commit()
