"""TL robot entity."""

from packages.duckiematrix_engine.entities.traffic_light_entity import (
    TrafficLightEntity,
)


class TLRobotEntity(TrafficLightEntity):
    """TL robot entity."""

    def __init__(self, matrix_key: str, world_key: str | None = None) -> None:
        """Initialize TL robot entity."""
        super().__init__(matrix_key, world_key)
