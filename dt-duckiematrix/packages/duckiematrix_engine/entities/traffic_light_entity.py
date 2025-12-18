"""Traffic light entity."""

from packages.duckiematrix_engine.entities.matrix_entity import (
    MatrixEntityBehavior,
)
from packages.duckiematrix_engine.entities.robot_entity_abs import (
    RobotEntityAbs,
)


class TrafficLightEntity(RobotEntityAbs):
    """Traffic light entity."""

    def __init__(self, matrix_key: str, world_key: str | None = None) -> None:
        """Initialize traffic light entity."""
        super().__init__(matrix_key, world_key)
        self._static = True


class TrafficLightEntityBehavior(TrafficLightEntity, MatrixEntityBehavior):
    """Traffic light entity behavior."""

    def __init__(
        self,
        matrix_key: str,
        world_key: str | None = None,
    ) -> None:
        """Initialize traffic light entity behavior."""
        super().__init__(matrix_key, world_key)
