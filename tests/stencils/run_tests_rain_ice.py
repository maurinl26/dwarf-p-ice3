# -*- coding: utf-8 -*-
import datetime
import logging

import numpy as np
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.framework.components import ComputationalGridComponent
from ifs_physics_common.utils.typingx import (
    NDArrayLikeDict,
)

from stencils.test_ice4_rrhong import Ice4RRHONG
from utils.allocate_state import allocate_state


from ice3_gt4py.initialisation.utils import initialize_field
from ice3_gt4py.phyex_common.phyex import Phyex



###### Default config for tests #######
backend = "gt:cpu_ifirst"
rebuild = True
validate_args = True

phyex = Phyex(program="AROME")

test_grid = ComputationalGrid(50, 1, 15)
dt = datetime.timedelta(seconds=1)

default_gt4py_config = GT4PyConfig(
    backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=False
)

if __name__ == "__main__":

    KPROMA, KSIZE = 50, 50

    # TODO : set in env values
    backend = "gt:cpu_ifirst"
    rebuild = True
    validate_args = True

    phyex = Phyex(program="AROME")

    logging.info("Initializing grid ...")

    # Grid has only 1 dimension since fields are packed in fortran version
    grid = ComputationalGrid(50, 1, 1)
    dt = datetime.timedelta(seconds=1)

    ######## Backend and gt4py config #######
    logging.info(f"With backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=False
    )

    logging.info("Calling ice4_rrhong with dicts")

    test_ice4_rrhong = Ice4RRHONG(
        computational_grid=grid, gt4py_config=gt4py_config, phyex=phyex, 
        fortran_module="mode_ice4_rrhong", fortran_subroutine="ice4_rrhong",
        fortran_script="mode_ice4_rrhong.F90",
        gt4py_stencil="ice4_rrhong"
    ).test()
    
  