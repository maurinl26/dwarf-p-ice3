# -*- coding: utf-8 -*-
from drivers.core import core

import logging
import datetime
import sys

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid

from drivers.core import core
from ice3_gt4py.components.ice4_tendencies import Ice4Tendencies

from ice3_gt4py.utils.state_ice4_tendencies import (
    get_constant_state_ice4_tendencies,
)

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


if __name__ == "__main__":

    logging.info(f"With backend gt:cpu_ifirst")
    gt4py_config = GT4PyConfig(
        backend="gt:cpu_ifirst", rebuild=False, validate_args=False, verbose=True
    )

    ############# Grid #####################
    logging.info("Initializing grid ...")
    grid = ComputationalGrid(nx=50, ny=1, nz=15)
    dt = datetime.timedelta(seconds=1)

    output_path = "./data/tests/"
    tracking_file = "track.json"

    ####### Create state for Ice4Tendencies #######
    logging.info("Getting state for Ice4Tendencies")
    state = get_constant_state_ice4_tendencies(grid, gt4py_config=gt4py_config)

    ####### Launch execution ##################
    core(Ice4Tendencies, gt4py_config, grid, state, output_path, dt, tracking_file)
