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

from phyex_gt4py.drivers.ice_adjust import IceAdjust
from phyex_gt4py.initialisation.state import allocate_data_array, allocate_state
from phyex_gt4py.initialisation.utils import initialize_field
from phyex_gt4py.phyex_common.phyex import Phyex
from phyex_gt4py.drivers.config import default_python_config

if TYPE_CHECKING:
    from typing import Literal, Tuple

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
    from ifs_physics_common.utils.typingx import DataArray, DataArrayDict


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger()


def allocate_state(
    computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig
) -> DataArrayDict:
    def _allocate(
        grid_id: Tuple[DimSymbol, ...],
        units: str,
        dtype: Literal["bool", "float", "int"],
    ) -> DataArray:
        return allocate_data_array(
            computational_grid, grid_id, units, gt4py_config=gt4py_config, dtype=dtype
        )

    # allocate_b_ij = partial(_allocate, grid_id=(I, J), units="", dtype="bool")
    allocate_f = partial(_allocate, grid_id=(I, J, K), units="", dtype="float")
    # allocate_f_h = partial(
    #     _allocate, grid_id=(I, J, K - 1 / 2), units="", dtype="float"
    # )
    # allocate_f_ij = partial(_allocate, grid_id=(I, J), units="", dtype="float")
    # allocate_i_ij = partial(_allocate, grid_id=(I, J), units="", dtype="int")

    return {
        "time": datetime(year=2024, month=1, day=1),
        "f_sigqsat": allocate_f(),
        "f_exnref": allocate_f(),  # ref exner pression
        "f_exn": allocate_f(),
        "f_rhodref": allocate_f(),
        "f_pabs": allocate_f(),  # absolute pressure at t
        "f_sigs": allocate_f(),  # Sigma_s at time t
        "f_cf_mf": allocate_f(),  # convective mass flux fraction
        "f_rc_mf": allocate_f(),  # convective mass flux liquid mixing ratio
        "f_ri_mf": allocate_f(),
    }


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
    gt4py_config = default_python_config.gt4py_config
    grid = ComputationalGrid(nx, ny, nz)
    dt = timedelta(seconds=1)

    ice_adjust = IceAdjust(grid, gt4py_config, phyex_config)

    # Test 1
    state = get_state_with_constant(grid, gt4py_config, 0)
    tends, diags = ice_adjust(state, dt)

    # Test 2
    state = get_state_with_constant(grid, gt4py_config, 1)
    tends, diags = ice_adjust(state, dt)

    # Test 3
    state = get_state_with_constant(grid, gt4py_config, 0.5)
    tends, diags = ice_adjust(state, dt)
