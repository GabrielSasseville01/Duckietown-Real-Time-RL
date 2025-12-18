"""Duckiematrix Engine entities."""

from packages.duckiematrix_engine.entities.db18_robot_entity import (
    DB18RobotEntity,
)
from packages.duckiematrix_engine.entities.db19_robot_entity import (
    DB19RobotEntity,
)
from packages.duckiematrix_engine.entities.db21_robot_entity import (
    DB21RobotEntity,
)
from packages.duckiematrix_engine.entities.dc2x_robot_entity import (
    DC2XRobotEntity,
)
from packages.duckiematrix_engine.entities.dd24_robot_entity import (
    DD24RobotEntity,
)
from packages.duckiematrix_engine.entities.matrix_entity import MatrixEntity
from packages.duckiematrix_engine.entities.tl_robot_entity import TLRobotEntity
from packages.duckiematrix_engine.entities.wt1x_robot_entity import (
    WT1XRobotEntity,
)
from packages.duckiematrix_engine.entities.wt2x_robot_entity import (
    WT2XRobotEntity,
)

_ENTITY_NAME_TO_CLASS: dict[str, type[MatrixEntity]] = {
    "Robot/DB18": DB18RobotEntity,
    "Robot/DB19": DB19RobotEntity,
    "Robot/DB21M": DB21RobotEntity,
    "Robot/DB21J": DB21RobotEntity,
    "Robot/DD24": DD24RobotEntity,
    "Robot/DBR": DB21RobotEntity,
    "Robot/WT18": WT1XRobotEntity,
    "Robot/WT19A": WT1XRobotEntity,
    "Robot/WT19B": WT1XRobotEntity,
    "Robot/WT21A": WT2XRobotEntity,
    "Robot/WT21B": WT2XRobotEntity,
    "Robot/DC21": DC2XRobotEntity,
    "Robot/TL18": TLRobotEntity,
    "Robot/TL19": TLRobotEntity,
    "Robot/TL21": TLRobotEntity,
}


def instantiate_entity(
    type_name: str,
    matrix_key: str,
    world_key: str | None = None,
) -> MatrixEntity | None:
    """Instantiate_entity."""
    if type_name in _ENTITY_NAME_TO_CLASS:
        cls = _ENTITY_NAME_TO_CLASS[type_name]
        return cls(matrix_key, world_key)
    return None
