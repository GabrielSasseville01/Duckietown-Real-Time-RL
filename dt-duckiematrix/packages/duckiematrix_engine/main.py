"""Main function."""

import argparse
import logging
import sys

from packages.duckiematrix_engine.constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_DELTA_T,
    DEFAULT_MAPS_DIR,
    DEFAULT_MATRIX_CONNECTOR_HOST,
    DEFAULT_MATRIX_CONTROL_OUT_CONNECTOR_PORT,
    DEFAULT_WORLD_CONTROL_OUT_CONNECTOR_PORT,
)
from packages.duckiematrix_engine.engine import MatrixEngine
from packages.duckiematrix_engine.exceptions import (
    BringUpError,
    InvalidMapConfigurationError,
)

PARTS_LENGTH = 2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main() -> None:
    """Run Duckiematrix Engine."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["realtime", "gym"],
        help="Agents' synchronization profile",
    )
    parser.add_argument(
        "-d",
        "--maps-dir",
        default=DEFAULT_MAPS_DIR,
        type=str,
        help="Directory containing the maps",
    )
    parser.add_argument(
        "-m",
        "--map",
        required=True,
        type=str,
        help="Name of the map to serve",
    )
    parser.add_argument(
        "-D",
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        type=str,
        help="Directory containing the data connectors (needs to be shared "
        "with the agents)",
    )
    parser.add_argument(
        "-t",
        "-dt",
        "--delta-t",
        default=DEFAULT_DELTA_T,
        type=float,
        help="Time step (gym mode only)",
    )
    parser.add_argument(
        "-H",
        "--hostname",
        default=DEFAULT_MATRIX_CONNECTOR_HOST,
        type=str,
        help="Matrix rendering engine connector hostname",
    )
    # ZMQ Ports
    # TODO: remove shortcuts; add zmq- prefix to the argument
    # name; and add '[ZMQ] ' help prefix to all the ZMQ ports
    parser.add_argument(
        "-mcop",
        "--matrix-control-out-port",
        default=DEFAULT_MATRIX_CONTROL_OUT_CONNECTOR_PORT,
        type=int,
        help="Matrix side OUT (control) connector port",
    )
    parser.add_argument(
        "-mdop",
        "--matrix-data-out-port",
        default=None,
        type=int,
        help="Matrix side OUT (data) connector port",
    )
    parser.add_argument(
        "-mdip",
        "--matrix-data-in-port",
        default=None,
        type=int,
        help="Matrix side IN (data) connector port",
    )
    parser.add_argument(
        "-wdop",
        "--world-data-out-port",
        default=None,
        type=int,
        help="World side OUT (data) connector port",
    )
    parser.add_argument(
        "-wdip",
        "--world-data-in-port",
        default=None,
        type=int,
        help="World side IN (data) connector port",
    )
    parser.add_argument(
        "-rdop",
        "--robot-data-out-port",
        default=None,
        type=int,
        help="World side 'robot' OUT (data) connector port",
    )
    parser.add_argument(
        "-rdip",
        "--robot-data-in-port",
        default=None,
        type=int,
        help="World side 'robot' IN (data) connector port",
    )
    parser.add_argument(
        "-ldop",
        "--layer-data-out-port",
        default=None,
        type=int,
        help="World side 'layer' OUT (data) connector port",
    )
    parser.add_argument(
        "-ldip",
        "--layer-data-in-port",
        default=None,
        type=int,
        help="World side 'layer' IN (data) connector port",
    )
    parser.add_argument(
        "-wcop",
        "--world-control-out-port",
        default=DEFAULT_WORLD_CONTROL_OUT_CONNECTOR_PORT,
        type=int,
        help="World side OUT (control) connector port",
    )
    # DTPS Ports
    parser.add_argument(
        "-rdp",
        "--dtps-robot-data-port",
        default=None,
        type=int,
        help="[DTPS] World side 'robot' IN (data) connector port",
    )
    parser.add_argument(
        "-r",
        "--renderers",
        default=1,
        type=int,
        help="Number of renderers to spawn",
    )
    parser.add_argument(
        "-l",
        dest="links_short",
        action="append",
        default=[],
        metavar=("matrix:world",),
        help="Link robots inside the matrix to robots outside",
    )
    parser.add_argument(
        "--link",
        dest="links",
        nargs=2,
        action="append",
        default=[],
        metavar=("matrix", "world"),
        help="Link robots inside the matrix to robots outside",
    )
    parser.add_argument(
        "--secure",
        default=False,
        action="store_true",
        help="Requires renderers to present a valid key when joining the "
        "network",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Run in Debug mode",
    )
    parser.add_argument(
        "--build-assets",
        default=False,
        action="store_true",
        help="Build map assets and exit",
    )
    parser.add_argument(
        "--profiler",
        default=False,
        action="store_true",
        help="Run in Profiled mode. Shows profiling information.",
    )
    parsed = parser.parse_args()
    # turn links_short into links
    for link in parsed.links_short:
        parts = link.split(":")
        if len(parts) != PARTS_LENGTH:
            message = "Option -l expects the format MATRIX:WORLD."
            raise ValueError(message)
        parsed.links.append(parts)
    parsed.links_short = []
    try:
        MatrixEngine.initialize(parsed)
        app = MatrixEngine.get_instance()
        app.start()
    except BringUpError as error:
        logger.info(
            "An error occurred while bringing up the Engine.\n%s",
            error.message,
        )
        sys.exit(error.exit_code)
    except InvalidMapConfigurationError as error:
        logger.info(
            "Invalid map configuration detected.\nErrors:\n\t- %s\n",
            "\n\t- ".join(error.messages),
        )
        sys.exit(error.exit_code)
    app.join()


if __name__ == "__main__":
    main()
