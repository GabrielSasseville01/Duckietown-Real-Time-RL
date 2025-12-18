"""Traffic light script."""

import time
from pathlib import Path

from duckietown_messages.actuators import CarLights
from duckietown_messages.colors import RGBA
from duckietown_messages.standard import Header

from packages.duckiematrix_engine.entities.robot_entity_abs import (
    RobotEntityAbs,
)
from packages.duckiematrix_engine.entities.traffic_light_entity import (
    TrafficLightEntityBehavior,
)

BLINK_FREQUENCY = 7.8
PERIOD = 1 / (2 * BLINK_FREQUENCY)
TOTAL_GREEN_TIME = 5
TOTAL_RED_TIME = 4
TOTAL_TIME = TOTAL_GREEN_TIME + TOTAL_RED_TIME

color_dictionary = {
    "green": "#00ff00",
    "red": "#ff0000",
}


class TrafficLightScript(TrafficLightEntityBehavior):
    """Traffic light script."""

    _color: str
    _green_time: float
    _intensity: float
    _light_keys: tuple
    _light_index: int
    _light_names: tuple
    _number_of_lights: int
    _rgba_dictionary: dict[str, RGBA]
    _stop: bool
    _time: float
    world: RobotEntityAbs.ProxiedWorldSide
    world_key: str

    def __init__(self, matrix_key: str, world_key: str | None) -> None:
        """Initialize traffic light script."""
        super().__init__(matrix_key, world_key)
        light_keys = self.lights.keys()
        light_values = self.lights.values()
        self._light_names = tuple(light_keys)
        self._light_keys = tuple(light_values)
        self._number_of_lights = len(self.lights)
        self._light_index = 0
        self._color = color_dictionary["green"]
        self._intensity = 0.6
        self._green_time = 0
        self._rgba_dictionary = {
            "front_left": RGBA.zero(),
            "front_right": RGBA.zero(),
            "back_left": RGBA.zero(),
            "back_right": RGBA.zero(),
        }
        self._time = 0
        self._stop = False
        path = (
            Path(self.world_key) / "actuator" / "lights" / "base" / "pattern"
        )
        world_key = path.as_posix()
        self.world.declare_input(world_key, CarLights)

    def update(self, delta_t: float) -> None:
        """Update."""
        self._time += delta_t
        change = False
        if self._time < TOTAL_GREEN_TIME:
            self._green_time += delta_t
            if self._green_time > PERIOD:
                self._intensity = 0 if self._intensity else 0.6
                self._green_time = 0
                change = True
        elif self._time < TOTAL_TIME:
            if not self._stop:
                self._color = color_dictionary["red"]
                self._intensity = 0.6
                self._stop = True
                change = True
        else:
            self._light_index = (
                self._light_index + 1
            ) % self._number_of_lights
            self._color = color_dictionary["green"]
            self._intensity = 0.6
            self._green_time = 0
            self._time = 0
            self._stop = False
            change = True
        if change:
            light_key = self._light_keys[self._light_index]
            light_name = self._light_names[self._light_index]
            data = {
                "name": light_name,
                "type": "spot",
                "intensity": self._intensity,
                "range": 2,
                "angle": 0,
                "color": self._color,
            }
            timestamp = time.time()
            header = Header(timestamp=timestamp)
            self._rgba_dictionary[light_name] = RGBA(
                header=header,
                r=int(self._color[1:3], 16) / 255,
                g=int(self._color[3:5], 16) / 255,
                b=int(self._color[5:7], 16) / 255,
                a=self._intensity,
            )
            lights = CarLights(
                header=header,
                front_left=self._rgba_dictionary["front_left"],
                front_right=self._rgba_dictionary["front_right"],
                back_left=self._rgba_dictionary["back_left"],
                back_right=self._rgba_dictionary["back_right"],
            )
            path = (
                Path(self.world_key)
                / "actuator"
                / "lights"
                / "base"
                / "pattern"
            )
            world_key = path.as_posix()
            self.matrix.input("lights", light_key, data)
            self.world.proxied.robot_connector.send(world_key, lights)
