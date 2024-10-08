# -*- coding: utf-8 -*-
from functools import partial
import fmodpy
import numpy as np
import logging

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid
import datetime

from ice3_gt4py.phyex_common.phyex import Phyex

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
from ifs_physics_common.utils.typingx import (
    DataArray,
    NDArrayLikeDict,
)

from tests.stencils.test_cloud_fraction import TestCloudFraction
from tests.stencils.test_condensation import TestCondensation

if __name__ == "__main__":
    NIJT = 50
    NKT = 15
    NKB = 15
    NKE = 15
    NKL = 1
    NIJB = 0
    NIJE = 15

    # TODO : set in env values
    backend = "gt:cpu_ifirst"
    rebuild = True
    validate_args = True

    phyex = Phyex(program="AROME")

    # TODO : set init with grid
    logging.info("Initializing grid ...")
    grid = ComputationalGrid(50, 1, 15)
    dt = datetime.timedelta(seconds=1)

    ######## Backend and gt4py config #######
    logging.info(f"With backend {backend}")
    gt4py_config = GT4PyConfig(
        backend=backend, rebuild=rebuild, validate_args=validate_args, verbose=False
    )

    logging.info("TestCloudFraction")
    TestCloudFraction(
        computational_grid=grid, gt4py_config=gt4py_config, phyex=phyex
    ).test()

    logging.info("TestCondensation with dicts")
    TestCondensation(
        computational_grid=grid, gt4py_config=gt4py_config, phyex=phyex
    ).test_fortran()
