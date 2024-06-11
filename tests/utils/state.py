# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
from functools import partial
import logging
from pathlib import Path
import numpy as np
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array
from ifs_physics_common.framework.components import ImplicitTendencyComponent


from ice3_gt4py.components.ice_adjust import IceAdjust
from ice3_gt4py.initialisation.state import get_state
from ice3_gt4py.initialisation.state_ice_adjust import KRR_MAPPING
from ice3_gt4py.initialisation.utils import initialize_field
from ice3_gt4py.phyex_common.phyex import Phyex

from typing import Literal, Tuple

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
from ifs_physics_common.utils.typingx import (
    DataArray,
    DataArrayDict,
    NDArrayLikeDict,
)

from ice3_gt4py.utils.reader import NetCDFReader


if __name__ == "__main__":

    logging.info(f"With backend gt:cpu_ifirst")
    gt4py_config = GT4PyConfig(
        backend="gt:cpu_ifirst", rebuild=False, validate_args=False, verbose=True
    )

    phyex = Phyex("AROME")

    ############# Grid #####################
    logging.info("Initializing grid ...")
    grid = ComputationalGrid(nx=50, ny=1, nz=15)
    dt = datetime.timedelta(seconds=1)

    output_path = "./data/tests/"
    tracking_file = "track.json"
    dataset = "./data/ice_adjust/reference.nc"

    ice_adjust = IceAdjust(grid, gt4py_config, phyex)

    reader = NetCDFReader(Path(dataset))

    ####### Create state for Ice4Tendencies #######
    logging.info("Getting state for Ice4Tendencies")
    state = get_state(
        grid, gt4py_config=gt4py_config, component=ice_adjust, netcdf_reader=reader
    )
