"""Scriptable map helper."""

import dataclasses
import importlib.util
import inspect
import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from packages.duckiematrix_engine.entities.matrix_entity import (
    MatrixEntity,
    MatrixEntityBehavior,
)

MatrixEntityBehaviorClass = type[MatrixEntityBehavior]


@dataclasses.dataclass
class MapScript:
    """Map script."""

    id: str
    name: str
    path: str


class ScriptableMapHelper:
    """Scriptable map helper."""

    if TYPE_CHECKING:
        from packages.duckiematrix_engine.engine import MatrixEngine

    _engine: "MatrixEngine | None"
    _logger: logging.Logger
    _map_dir: str | None
    _scripts: dict[str, MatrixEntityBehaviorClass]
    _scripts_dir: str | None

    def __init__(self) -> None:
        """Initialize scriptable map helper."""
        self._map_dir = None
        self._scripts_dir = None
        self._scripts = {}
        self._engine = None
        self._logger = logging.getLogger("Helper[ScriptableMap]")

    def _attach_scripts_to_entities(self) -> None:
        for matrix_key, scripts_obj in self._engine.map_.scripts.items():
            self._logger.debug("Attaching scripts to '%s'...", matrix_key)
            # get list of scripts to attach
            scripts = scripts_obj.get("scripts", None)
            if scripts is None or not isinstance(scripts, dict | list):
                self._logger.warning(
                    "Scripts list not found for entity '%s'",
                    matrix_key,
                )
                continue
            # turn list into dictionary where each scripts has a default
            # configuration
            if isinstance(scripts, list):
                scripts = {script: {} for script in scripts}
            # attach scripts
            for script_id, script_desc in scripts.items():
                script_class: MatrixEntityBehaviorClass = self._scripts.get(
                    script_id,
                    None,
                )
                if script_class is None:
                    self._logger.warning(
                        " x - Script '%s' not registered",
                        script_id,
                    )
                    continue
                entity: MatrixEntity | None = (
                    self._engine.get_registered_entity(matrix_key)
                )
                world_key = entity.world_key if entity is not None else None
                entity_key = f"{matrix_key}/{script_id}"
                entity_script = script_class(
                    matrix_key,
                    world_key,
                    **script_desc,
                )
                self._engine.register_entity(
                    entity_key,
                    entity_script,
                    quiet=True,
                )
                self._logger.debug(" o - Script '%s' attached", script_id)

    def _list_scripts(self) -> list[MapScript]:
        # list scripts from disk
        path = Path(self._scripts_dir)
        python_scripts = path.rglob("*.py")
        scripts: list[Path] = [
            python_script
            for python_script in python_scripts
            if not python_script.name.startswith("_")
        ]
        scripts_length = len(scripts)
        self._logger.info("Found %s scripts.", scripts_length)
        # create script descriptors
        map_scripts = []
        for script in scripts:
            relative_path = script.relative_to(self._scripts_dir)
            relative_path_without_suffix = relative_path.with_suffix("")
            script_id = str(relative_path_without_suffix)
            script_without_suffix = script.with_suffix("")
            script_name = str(script_without_suffix.name)
            absolute_path = script.absolute()
            script_path = str(absolute_path)
            map_script = MapScript(script_id, script_name, script_path)
            map_scripts.append(map_script)
        return map_scripts

    def _load_script(
        self,
        script: MapScript,
    ) -> MatrixEntityBehaviorClass | None:
        script_class: MatrixEntityBehaviorClass | None = None
        # (try to) load script from file
        try:
            spec = importlib.util.spec_from_file_location(
                script.name,
                script.path,
            )
            # load module from file
            if spec is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module_name = module.__name__
                for _, member_object in inspect.getmembers(module):
                    if (
                        inspect.isclass(member_object)
                        and issubclass(
                            member_object,
                            MatrixEntityBehavior,
                        )
                        and member_object.__module__ == module_name
                    ):
                        script_class = member_object
                        break
                if script_class is None:
                    self._logger.warning(
                        "Script '%s' failed to load.",
                        script.id,
                    )
                    return None
            else:
                self._logger.warning(
                    "Script '%s' failed to load. Module cannot be found.",
                    script.id,
                )
                return None
        except Exception:
            formatted_traceback = traceback.format_exc()
            split_formatted_traceback = formatted_traceback.splitlines()
            error = "\n\t".join(split_formatted_traceback)
            message = (
                f"An error occurred while loading the script \n'{script.id}'."
                f"\nThe error is:\n\t\n{error}"
            )
            self._logger.exception(message)
        return script_class

    def _load_scripts(self) -> None:
        for script in self._list_scripts():
            entity_script_class: MatrixEntityBehaviorClass | None = (
                self._load_script(script)
            )
            if entity_script_class is not None:
                self._scripts[script.id] = entity_script_class
                self._logger.info("Script '%s' loaded.", script.id)

    def start(self) -> None:
        """Start."""
        from packages.duckiematrix_engine.engine import MatrixEngine

        self._engine = MatrixEngine.get_instance()
        # setup logger
        self._logger.level = self._engine.logger.level
        # get map dir
        self._map_dir = self._engine.map_dir
        path = Path(self._map_dir) / "scripts"
        self._scripts_dir = path.as_posix()
        self._logger.info("Initialized on %s", self._scripts_dir)
        # load scripts
        self._load_scripts()
        # attach scripts
        self._attach_scripts_to_entities()
