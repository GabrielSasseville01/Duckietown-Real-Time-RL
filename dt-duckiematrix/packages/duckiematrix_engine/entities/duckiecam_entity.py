"""Duckiecam entity."""

from packages.duckiematrix_engine.entities.robot_entity_abs import (
    RobotEntityAbs,
)


class DuckiecamEntity(RobotEntityAbs):
    """Duckiecam entity."""

    def __init__(self, matrix_key: str, world_key: str | None = None) -> None:
        """Initialize Duckiecam entity."""
        super().__init__(matrix_key, world_key)
        self._static = True
