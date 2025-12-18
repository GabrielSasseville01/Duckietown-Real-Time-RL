"""DB21 robot entity."""

from datetime import UTC, datetime

from packages.duckiematrix_engine.entities.dynamics_vehicle_entity import (
    DynamicsVehicleEntity,
)


class DB21RobotEntity(DynamicsVehicleEntity):
    """DB21 robot entity."""

    BASELINE = 0.11

    def __init__(self, matrix_key: str, world_key: str | None = None) -> None:
        """Initialize DB21 robot entity."""
        super().__init__(matrix_key, self.BASELINE, world_key)

    def update(self, delta_time: float) -> None:
        """Update."""
        super().update(delta_time)
        collision, timestamp = self.matrix.collision()
        # TODO: do something here
        if collision is not None:
            date_time = datetime.fromtimestamp(timestamp, UTC)
            self._logger.warning(
                "%s collided with %s at %s on %s.",
                self.world_key,
                collision,
                date_time.time(),
                date_time.date(),
            )
