"""Layers passthrough entity helper."""

from typing import TYPE_CHECKING

from packages.duckiematrix_engine.entities.matrix_entity import MatrixEntity


class LayersPassthroughEntity(MatrixEntity):
    """Layers passthrough entity."""

    def __init__(self) -> None:
        """Initialize layers passthrough entity."""
        super().__init__(
            "layers-passthrough-helper",
            "layers-passthrough-helper",
        )

    def start(self) -> None:
        """Start."""
        # register myself as matrix entity
        self._engine.register_entity(self.matrix_key, self, quiet=True)

    def update(self, _: float) -> None:
        """Update."""
        layers = self._engine.world.layer.iterate_updated_queue("output")
        for layer, layer_content in layers:
            # iterate over keys
            for key, update in layer_content.items():
                # update
                self._engine.matrix.layer(layer, key, update)


class LayersPassthroughHelper:
    """Layers passthrough helper."""

    if TYPE_CHECKING:
        from packages.duckiematrix_engine.engine import MatrixEngine

    # TODO: remap world robot names to matrix names
    _entity: LayersPassthroughEntity | None
    _engine: "MatrixEngine | None"

    def __init__(self) -> None:
        """Initialize layers passthrough helper."""
        # entity pointer
        self._entity = None
        # engine pointer
        self._engine = None

    def start(self) -> None:
        """Start."""
        from packages.duckiematrix_engine.engine import MatrixEngine

        self._engine = MatrixEngine.get_instance()
        # create entity
        self._entity = LayersPassthroughEntity()
        # initialize entity
        self._entity.start()
