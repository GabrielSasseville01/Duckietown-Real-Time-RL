"""Engine."""

import argparse
import concurrent
import dataclasses
import logging
import math
import multiprocessing
import shutil
import tempfile
import time
import traceback
from collections.abc import Callable
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from threading import Thread

import netifaces
from dt_class_utils import DTProcess
from dt_module_utils import set_module_healthy, set_module_unhealthy

from packages.duckiematrix_engine.connectors.matrix_control_connector import (
    MatrixControlConnector,
)
from packages.duckiematrix_engine.connectors.matrix_data_connector import (
    MatrixDataConnector,
)
from packages.duckiematrix_engine.constants import EngineMode
from packages.duckiematrix_engine.entities import instantiate_entity
from packages.duckiematrix_engine.entities.matrix_entity import MatrixEntity
from packages.duckiematrix_engine.exceptions import DuckiematrixEngineError
from packages.duckiematrix_engine.helpers.layers_passthrough_helper import (
    LayersPassthroughHelper,
)
from packages.duckiematrix_engine.helpers.scriptable_map_helper import (
    ScriptableMapHelper,
)
from packages.duckiematrix_engine.map import Map
from packages.duckiematrix_engine.sides.matrix_side import MatrixSide
from packages.duckiematrix_engine.sides.world_side import WorldSide
from packages.duckiematrix_engine.types_ import Protocol, Robot, RobotType
from packages.duckiematrix_engine.utils.file_system import (
    get_ownership,
    set_ownership,
)
from packages.duckiematrix_engine.utils.monitored_condition import (
    MonitoredCondition,
)
from packages.duckiematrix_engine.utils.stopwatch import Stopwatch
from packages.duckiematrix_engine.utils.t2_profiler import T2Profiler


@dataclasses.dataclass
class EngineConfiguration:
    """Duckiematrix Engine configuration."""

    mode: EngineMode
    delta_time: float
    data_dir: str

    @classmethod
    def from_args(cls, parsed: argparse.Namespace) -> "EngineConfiguration":
        """Return Duckiematrix Engine configuration from arguments."""
        return EngineConfiguration(
            EngineMode.from_string(parsed.mode),
            parsed.delta_t,
            parsed.data_dir,
        )


class MatrixEngine(DTProcess):
    """Duckiematrix Engine."""

    _TRIGGER_RESEND_DELAY_SECONDS = 5
    _UPDATE_TYPES = ("early_update", "update", "late_update")

    __instance__: "MatrixEngine | None"
    _early_updates: list[MatrixEntity]
    _entities: dict[str, MatrixEntity]
    _executor: ThreadPoolExecutor
    _flushes: list[MatrixEntity]
    _late_updates: list[MatrixEntity]
    _manager: Thread
    _markers_helper: LayersPassthroughHelper
    _max_period: float | None
    _original_map_dir: str
    _parsed: argparse.Namespace
    _scriptable_map_helper: ScriptableMapHelper
    _started: bool
    _updates: list[MatrixEntity]
    configuration: EngineConfiguration
    events: MonitoredCondition
    logger: logging.Logger
    map_: Map
    map_dir: str
    matrix: MatrixSide
    matrix_control_connector: MatrixControlConnector
    matrix_data_connector: MatrixDataConnector
    robots: dict[str, Robot]
    session_id: int
    temp_dir: str
    world: WorldSide

    def __init__(self, parsed: argparse.Namespace) -> None:
        """Initialize Duckiematrix Engine."""
        if self.__instance__ is not None:
            message = (
                "You cannot instantiate the class `MatrixEngine` directly, "
                "use the singleton method `MatrixEngine.get_instance()` "
                "instead."
            )
            raise DuckiematrixEngineError(message)
        set_module_unhealthy()
        super().__init__("duckiematrix")
        self._parsed = parsed
        # set debug mode
        self.logger.setLevel(logging.INFO)
        if parsed.debug:
            self.logger.setLevel(logging.DEBUG)
        # configure profiler
        T2Profiler.enabled(status=parsed.profiler)
        # create data dir
        path: Path = Path(parsed.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        # configuration
        self.configuration = EngineConfiguration.from_args(parsed)
        # load map (just to make sure everything is there)
        map_ = parsed.map
        Map.from_disk(map_, parsed.maps_dir)
        # copy map then load
        self.temp_dir = tempfile.mkdtemp()
        path = Path(self.temp_dir) / map_
        self.map_dir = path.as_posix()
        self.logger.info(
            "Copying map to temporary directory [%s]...",
            self.map_dir,
        )
        path = Path(self.map_dir)
        path.mkdir(parents=True, exist_ok=True)
        path = Path(parsed.maps_dir) / map_
        self._original_map_dir = path.as_posix()
        shutil.copytree(
            self._original_map_dir,
            self.map_dir,
            dirs_exist_ok=True,
        )
        # load map
        self.map_ = Map.from_disk(map_, self.temp_dir)
        self.logger.info("Map copied successfully.")
        # load robots
        self.robots = self._get_robots()
        # create thread pool executor
        cpu_count = multiprocessing.cpu_count()
        self._executor = ThreadPoolExecutor(
            cpu_count,
            "duckiematrix_thread_worker_",
        )
        # event lock
        self.events = MonitoredCondition()
        # create engine manager
        if self.mode == EngineMode.REALTIME:
            self._manager = Thread(target=self._realtime_manager_job)
        else:
            self._manager = Thread(target=self._gym_manager_job)
        mode_string = self.mode.to_string()
        uppercase_mode_string = mode_string.upper()
        self.logger.info("Engine running in %s mode.", uppercase_mode_string)
        # prepare dictionary of entities
        self._entities = {}
        self._flushes = []
        self._early_updates = []
        self._updates = []
        self._late_updates = []
        # maximum period between updates (realtime mode only)
        self._max_period = None
        # other parameters
        self.session_id = 0
        self._started = False
        # connectors
        self.matrix_data_connector = MatrixDataConnector(
            parsed.hostname,
            parsed.matrix_data_out_port,
            parsed.matrix_data_in_port,
        )
        self.matrix_control_connector = MatrixControlConnector(
            parsed.hostname,
            parsed.matrix_control_out_port,
        )
        # helpers
        self._scriptable_map_helper = ScriptableMapHelper()
        self._markers_helper = LayersPassthroughHelper()
        # the two sides
        self.matrix = MatrixSide(parsed.renderers, secure=parsed.secure)
        self.world = WorldSide(parsed)
        # register shutdown callback
        self.register_shutdown_callback(T2Profiler.print, self.logger)
        self.register_shutdown_callback(self._shutdown)

    def _all_renderers_joined(self) -> bool:
        return self.matrix.all_renderers_joined()

    def _flush_entities(self) -> None:
        # submit flush
        with T2Profiler.profile("[engine]:flushes"):
            try:
                futures = []
                for entity in self._flushes:
                    future = self._executor.submit(
                        self._monitored_job,
                        entity,
                        entity.flush,
                    )
                    futures.append(future)
            except RuntimeError:
                if self.is_shutdown:
                    return
                raise
            # wait for flush to complete
            for _ in concurrent.futures.as_completed(futures):
                pass

    @T2Profiler.profiled("[engine]:_gathered_all_matrix_outputs")
    def _gathered_all_matrix_outputs(self) -> bool:
        matrix_out_full = set()
        for sensor_key, gate in self.map_.sensor_gates.items():
            if gate["open"]:
                matrix_out_full.add(sensor_key)
        matrix_out_full_length = len(matrix_out_full)
        if matrix_out_full_length <= 0:
            return True
        matrix_out_partial = self.matrix_data_connector.get_received(
            Protocol.SENSOR_DATA,
        )
        matrix_out_remaining = matrix_out_full.difference(matrix_out_partial)
        matrix_out_remaining_length = len(matrix_out_remaining)
        if matrix_out_remaining_length > 0:
            self.logger.debug(
                " waiting for %s matrix outputs.",
                matrix_out_remaining_length / matrix_out_full_length,
            )
            return False
        self.logger.debug(
            " all %s matrix outputs received.",
            matrix_out_full_length,
        )
        return True

    @T2Profiler.profiled("[engine]:_gathered_all_world_outputs")
    def _gathered_all_world_outputs(self) -> bool:
        world_out_partial = self.world.robot.get_received()
        for matrix_key in self.robots:
            robot = self.get_registered_entity(matrix_key)
            world_out_full = robot.world_outputs_wanted()
            if len(world_out_full) <= 0:
                continue
            world_out_remaining = world_out_full.difference(world_out_partial)
            if len(world_out_remaining) > 0:
                self.logger.debug(
                    "[{matrix_key}]: waiting for %s world outputs.",
                    len(world_out_remaining) / len(world_out_full),
                )
                return False
            self.logger.debug(
                "[%s]: all {len(world_out_full)} world outputs received.",
                matrix_key,
            )
        return True

    def _get_robots(self) -> dict[str, Robot]:
        robots = {}
        # - Duckietown robots (aka vehicles)
        for matrix_key, vehicle in self.map_.vehicles.items():
            # TODO: handle this better
            if vehicle["configuration"] in ("DD24",):
                robot_type = RobotType.DUCKIEDRONE
            else:
                robot_type = RobotType.DUCKIEBOT
            robots[matrix_key] = Robot(robot_type, vehicle)
        # - watchtowers
        for matrix_key, watchtower in self.map_.watchtowers.items():
            robots[matrix_key] = Robot(RobotType.WATCHTOWER, watchtower)
        # - traffic lights
        for matrix_key, traffic_light in self.map_.traffic_lights.items():
            robots[matrix_key] = Robot(RobotType.TRAFFIC_LIGHT, traffic_light)
        # - duckiecams
        for matrix_key, cameraman in self.map_.cameramen.items():
            robots[matrix_key] = Robot(RobotType.DUCKIECAM, cameraman)
        return robots

    def _gym_manager_job(self) -> None:
        num_renderers = len(self.matrix.renderers)
        # stopwatches
        matrix_stopwatch = Stopwatch()
        world_stopwatch = Stopwatch()
        # wait for all renderers to join
        self.logger.info(
            "Waiting for %s renderers to join the network.",
            num_renderers,
        )
        while not self.is_shutdown:
            if self._all_renderers_joined():
                break
            time.sleep(0.5)
        self.logger.info("All renderers joined the network.")
        time.sleep(1)
        # - let's start clean
        self.matrix.clear_session()
        self.world.clear_session()
        self._run_gym_manager_job(matrix_stopwatch, world_stopwatch)
        self.shutdown()

    def _monitored_job(
        self,
        entity: MatrixEntity,
        job: Callable,
        *args: float,
    ) -> None:
        try:
            with T2Profiler.profile(f"[{job.__name__}]:{entity.matrix_key}"):
                job(*args)
        except Exception:
            self.logger.exception(traceback.format_exc())

    def _realtime_manager_job(self) -> None:
        stime = ftime = time.time()
        while not self.is_shutdown:
            delta_time = ftime - stime
            stime = time.time()
            # TODO: send context to newly connected renderers
            # update all the entities
            self._update_entities(delta_time)
            self._flush_entities()
            # tell the connectors to publish what they have
            self.matrix.flush_inputs()
            self.world.flush_inputs()
            # wait for new events
            with self.events:
                self.events.wait(timeout=self._max_period)
            ftime = time.time()

    def _run_gym_manager_job(
        self,
        matrix_stopwatch: Stopwatch,
        world_stopwatch: Stopwatch,
    ) -> None:
        time_paused = True
        while not self.is_shutdown:
            delta_time = self.delta_time if not time_paused else 0
            time_paused = False
            T2Profiler.tick("[engine]:session")
            # - let the entities produce their Matrix commands
            with T2Profiler.profile("[engine]:update-entities"):
                self._update_entities(delta_time)
            # - send inputs to the Matrix
            with T2Profiler.profile("[matrix]:flush-inputs"):
                self.matrix.flush_inputs(block=True)
            # - tell the Matrix to act on those inputs
            with T2Profiler.profile("[matrix]:complete-session"):
                self.matrix.complete_session()
                matrix_stopwatch.restart()
            # - wait for the Matrix to produce all the outputs we need
            next_loop = self._wait_for_outputs("matrix", matrix_stopwatch)
            if self.is_shutdown:
                break
            if next_loop:
                time_paused = True
                continue
            # - we have the Matrix outputs, let's bridge them over to
            # the World
            with T2Profiler.profile("[engine]:flush-entities"):
                self._flush_entities()
            # - Matrix outputs have been bridged, let's clear the Matrix
            # side
            self.matrix.clear_session()
            # - now we can bump the session
            self.session_id += 1
            # - wait for World inputs to be sent
            with T2Profiler.profile("[world]:flush-inputs"):
                self.world.flush_inputs(block=True)
            # - tell the World to act on those inputs
            with T2Profiler.profile("[world]:complete-session"):
                self.world.complete_session()
                world_stopwatch.restart()
            # - World inputs have been sent, let's clear the World side
            self.world.clear_session()
            # - wait for the World to produce all the outputs we need
            next_loop = self._wait_for_outputs("world", world_stopwatch)
            if self.is_shutdown:
                break
            if next_loop:
                time_paused = True
                continue
            # - repeat forever
            T2Profiler.tock("[engine]:session")

    def _show_ip_addresses(self) -> None:
        message = "The Engine can be reached at any of these IP addresses:\n"
        interfaces = netifaces.interfaces()
        for interface in interfaces:
            addresses = netifaces.ifaddresses(interface)
            if netifaces.AF_INET not in addresses:
                continue
            for address in addresses[netifaces.AF_INET]:
                ip_address = address["addr"]
                message += f"\n\t -  {ip_address}".ljust(24, " ")
                # TODO: use info from here instead:
                # https://www.avast.com/c-ip-address-public-vs-private
                if ip_address in (
                    "127.0.0.1",
                    "127.0.1.1",
                ) or ip_address.startswith(
                    "172.17.",
                ):
                    message += "\t(local machine only)"
                elif ip_address.startswith("192.168."):
                    message += "\t(local network only)"
        message += "\n"
        self.logger.info(message)

    def _shutdown(self) -> None:
        if self._executor:
            self._executor.shutdown()
        if self._manager:
            self.join()

    def _update_entities(self, delta_time: float) -> None:
        for update_type in self._UPDATE_TYPES:
            key = f"[engine]:{update_type}s".replace("_", "-")
            with T2Profiler.profile(key):
                futures = []
                for entity in getattr(self, f"_{update_type}s"):
                    entity_update_type = getattr(entity, update_type)
                    future = self._executor.submit(
                        self._monitored_job,
                        entity,
                        entity_update_type,
                        delta_time,
                    )
                    futures.append(future)
                # wait for early-update to complete
                for _ in concurrent.futures.as_completed(futures):
                    pass

    def _wait_for_outputs(self, side_name: str, stopwatch: Stopwatch) -> bool:
        next_loop = False
        with T2Profiler.profile(f"[{side_name}]:wait-for-outputs"):
            gathered_all_outputs = getattr(
                self,
                f"_gathered_all_{side_name}_outputs",
            )
            while not self.is_shutdown:
                # check what we have got
                if gathered_all_outputs():
                    break
                # don't wait forever
                if stopwatch.elapsed > self._TRIGGER_RESEND_DELAY_SECONDS:
                    break
                # wait for events
                with self.events:
                    self.events.wait(1)
            # are we going home?
            if self.is_shutdown:
                return next_loop
            # have we got what we needed
            if not gathered_all_outputs():
                capitalized_side_name = side_name.capitalize()
                self.logger.warning(
                    "Detected %s in a stale state. Restarting session.",
                    capitalized_side_name,
                )
                side: MatrixSide | WorldSide = getattr(self, side_name)
                side.clear_session()
                stopwatch.stop()
                next_loop = True
        return next_loop

    @property
    def delta_time(self) -> float:
        """Return time step."""
        return self.configuration.delta_time

    def ensure_frequency(self, frequency: float) -> None:
        """Ensure frequency."""
        if frequency > 0:
            self.ensure_period(1 / frequency)

    def ensure_period(self, period: float) -> None:
        """Ensure period."""
        # only realtime mode supports frequency/period
        if self.mode != EngineMode.REALTIME:
            return
        old_max_period = self._max_period or 0
        if self._max_period is None:
            self._max_period = period
        else:
            # find the greatest common divisor of the two periods
            max_period_integer = int(self._max_period * 1000)
            period_integer = int(period * 1000)
            greatest_common_divisor = math.gcd(
                max_period_integer,
                period_integer,
            )
            self._max_period = greatest_common_divisor / 1000
        self.logger.info(
            "Updated maximum period. Old=%.3f, Requested=%.3f, New=%.3f",
            old_max_period,
            period,
            self._max_period,
        )

    @classmethod
    def get_instance(cls) -> "MatrixEngine":
        """Return instance."""
        return cls.__instance__

    def get_registered_entity(self, matrix_key: str) -> MatrixEntity | None:
        """Return registered entity."""
        return self._entities.get(matrix_key, None)

    @property
    def gym_mode(self) -> bool:
        """Return `True` if in `gym` mode, `False` otherwise."""
        return self.mode == EngineMode.GYM

    @classmethod
    def initialize(cls, parsed: argparse.Namespace) -> None:
        """Initialize Duckiematrix Engine class."""
        if cls.__instance__ is None:
            cls.__instance__ = MatrixEngine(parsed)
        else:
            message = "Class `MatrixEngine` is already initialized."
            raise DuckiematrixEngineError(message)

    def join(self, timeout: int | None = None) -> None:
        """Join."""
        if self._started:
            self._manager.join(timeout)

    @property
    def mode(self) -> EngineMode:
        """Return mode."""
        return self.configuration.mode

    def register_entity(
        self,
        key: str,
        entity: MatrixEntity,
        *,
        quiet: bool = False,
    ) -> None:
        """Register entity."""
        if key in self._entities:
            self.logger.error(
                "[Entity]: An entity with the same key '%s' is already "
                "registered. You cannot register the same entity multiple "
                "times.",
                key,
            )
            return
        if not quiet:
            if entity.world_key:
                string = f"> [{entity.world_key}]:World"
            else:
                string = "|"
            self.logger.info(
                "[Entity]: Registered new entity with key '%s':\n\tMatrix:[%s]"
                " <-%s",
                key,
                entity.matrix_key,
                string,
            )
        self._entities[key] = entity
        # register functions
        if entity.flush.__code__ != MatrixEntity.flush.__code__:
            self._flushes.append(entity)
        if entity.early_update.__code__ != MatrixEntity.early_update.__code__:
            self._early_updates.append(entity)
        if entity.update.__code__ != MatrixEntity.update.__code__:
            self._updates.append(entity)
        if entity.late_update.__code__ != MatrixEntity.late_update.__code__:
            self._late_updates.append(entity)

    def start(self) -> None:
        """Start."""
        # configure sides
        self.matrix.start()
        self.world.start()
        # add custom links to physical robots (if any)
        for link in self._parsed.links:
            matrix_key, world_key = link
            if matrix_key not in self.robots:
                self.logger.warning(
                    "The key '%s' does not correspond to a robot in this map.",
                    matrix_key,
                )
            else:
                self.logger.info(
                    "Linking robot of type '%s' to a world robot with name "
                    "'%s'.",
                    self.robots[matrix_key].type.value,
                    world_key,
                )
            self.map_.physical_robots[matrix_key] = world_key
        # create robot entities
        for matrix_key, robot in self.robots.items():
            self.logger.info(
                "[Entity]: Creating entity [%s]:%s",
                robot.type.value,
                matrix_key,
            )
            entity = instantiate_entity(
                f"Robot/{robot.data['configuration']}",
                matrix_key,
            )
            if entity:
                self.register_entity(matrix_key, entity)
        # create `engine_configuration` layer
        self.map_.engine_configuration["engine"] = {
            "mode": self.mode.value,
        }
        # update map layers on disk
        self.map_.to_disk()
        # copy assets directory back to original map
        if self._parsed.build_assets:
            self.logger.info("Copying assets back to original map...")
            path = Path(self._original_map_dir) / "assets"
            original_assets_dir = path.as_posix()
            shutil.copytree(
                self.map_.assets_dir,
                original_assets_dir,
                dirs_exist_ok=True,
            )
            self.logger.info(
                "Returning ownership of the generated assets to the user...",
            )
            uid, gid = get_ownership(self._original_map_dir)
            set_ownership(original_assets_dir, uid, gid, recursive=True)
            self.logger.info("Assets built! Exiting.")
            return
        # compress map directory to .zip archive (with all the assets
        # inside)
        context_fpath = self.map_.make_context()
        self.matrix.set_context(context_fpath)
        # run workers
        self.matrix_data_connector.start()
        self.matrix_control_connector.start()
        # run helpers
        self._scriptable_map_helper.start()
        self._markers_helper.start()
        # show list of IP addresses the engine can be reached at
        self._show_ip_addresses()
        # let the sides print out useful info
        self.matrix.print_startup_info()
        self.world.print_startup_info()
        # launch engine manager
        self._manager.start()
        # set the module as healthy
        set_module_healthy()
        # mark it as started
        self._started = True

    def stop(self) -> None:
        """Stop."""
        self._shutdown()
