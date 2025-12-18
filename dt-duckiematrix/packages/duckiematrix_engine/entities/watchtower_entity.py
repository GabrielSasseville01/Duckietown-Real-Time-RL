"""Watchtower entity."""

from packages.duckiematrix_engine.entities.robot_entity_abs import (
    RobotEntityAbs,
)


class WatchtowerEntity(RobotEntityAbs):
    """Watchtower entity."""

    def __init__(self, matrix_key: str, world_key: str | None = None) -> None:
        """Initialize watchtower entity."""
        super().__init__(matrix_key, world_key)
        self._static = True
