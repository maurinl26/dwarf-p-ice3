# -*- coding: utf-8 -*-
from __future__ import annotations
import logging
from functools import partial
from typing import TYPE_CHECKING
import sys
from datetime import timedelta, datetime

from gt4py.storage import ones
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, I, J, K

from ice3_gt4py.components.ice_adjust import IceAdjust
from ice3_gt4py.initialisation.state import allocate_data_array, allocate_state
from ice3_gt4py.initialisation.utils import initialize_field
from ice3_gt4py.phyex_common.phyex import Phyex
from ice3_gt4py.drivers.config import default_python_config

if TYPE_CHECKING:
    from typing import Literal, Tuple

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
    from ifs_physics_common.utils.typingx import DataArray, DataArrayDict


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


def initialize_state_with_constant(
    state: DataArrayDict, C: float, gt4py_config: GT4PyConfig
) -> None:

    keys = [
        "f_sigqsat",
        "f_exnref",  # ref exner pression
        "f_exn",
        "f_rhodref",
        "f_pabs",  # absolute pressure at t
        "f_sigs",  # Sigma_s at time t
        "f_cf_mf",  # convective mass flux fraction
        "f_rc_mf",  # convective mass flux liquid mixing ratio
        "f_ri_mf",
        "f_th",
        "f_rv",
        "f_rc",
        "f_rr",
        "f_ri",
        "f_rs",
        "f_rg",
    ]

    for name in keys:
        logging.debug(f"{name}, {state[name].shape}")
        state[name][...] = C * ones(state[name].shape, backend=gt4py_config.backend)


def get_state_with_constant(
    computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig, c: float
) -> DataArrayDict:
    """All arrays are filled with a constant between 0 and 1.

    Args:
        computational_grid (ComputationalGrid): _description_
        gt4py_config (GT4PyConfig): _description_

    Returns:
        DataArrayDict: _description_
    """
    state = allocate_state(computational_grid, gt4py_config=gt4py_config)
    initialize_state_with_constant(state, c, gt4py_config)
    return state


if __name__ == "__main__":

    nx = 100
    ny = 1
    nz = 90

    cprogram = "AROME"
    phyex_config = Phyex(cprogram)

    gt4py_config = GT4PyConfig(
        backend="numpy", rebuild=False, validate_args=True, verbose=True
    )

    grid = ComputationalGrid(nx, ny, nz)
    dt = timedelta(seconds=1)

    ice_adjust = IceAdjust(grid, gt4py_config, phyex_config)

    # Test 1
    logging.debug("Test with 0")
    state = get_state_with_constant(grid, gt4py_config, 0)
    tends, diags = ice_adjust(state, dt)
    logging.debug(f"State : {state.keys()}")
    logging.debug(f"Tendencies : {tends.keys()}")
    logging.debug(f"Diagnostics : {diags.keys()}")
    logging.debug("Test passed")

    # Test 2
    logging.debug("Test with 1")
    state = get_state_with_constant(grid, gt4py_config, 1)
    tends, diags = ice_adjust(state, dt)
    logging.debug("Test passed")

    # Test 3
    logging.debug("Test with 0.5")
    state = get_state_with_constant(grid, gt4py_config, 0.5)
    tends, diags = ice_adjust(state, dt)
    logging.debug("Test passed")
