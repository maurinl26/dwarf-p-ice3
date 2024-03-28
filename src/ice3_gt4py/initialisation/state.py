# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING

from gt4py.storage import ones
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array

from ice3_gt4py.initialisation.utils import initialize_field

if TYPE_CHECKING:
    from typing import Literal, Tuple

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
    from ifs_physics_common.utils.typingx import (
        DataArray,
        DataArrayDict,
        NDArrayLikeDict,
    )


def allocate_state(
    computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig
) -> NDArrayLikeDict:
    """Allocate field to state keys following type (float, int, bool) and dimensions (2D, 3D).

    Args:
        computational_grid (ComputationalGrid): grid indexes
        gt4py_config (GT4PyConfig): gt4py configuration

    Returns:
        NDArrayLikeDict: dictionnary of field with associated keys for field name
    """

    def _allocate(
        grid_id: Tuple[DimSymbol, ...],
        units: str,
        dtype: Literal["bool", "float", "int"],
    ) -> DataArray:
        return allocate_data_array(
            computational_grid, grid_id, units, gt4py_config=gt4py_config, dtype=dtype
        )

    allocate_b_ij = partial(_allocate, grid_id=(I, J), units="", dtype="bool")
    allocate_f = partial(_allocate, grid_id=(I, J, K), units="", dtype="float")
    allocate_f_h = partial(
        _allocate, grid_id=(I, J, K - 1 / 2), units="", dtype="float"
    )
    allocate_f_ij = partial(_allocate, grid_id=(I, J), units="", dtype="float")
    allocate_i_ij = partial(_allocate, grid_id=(I, J), units="", dtype="int")

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
        "f_th": allocate_f(),
        "f_rv": allocate_f(),
        "f_rc": allocate_f(),
        "f_rr": allocate_f(),
        "f_ri": allocate_f(),
        "f_rs": allocate_f(),
        "f_rg": allocate_f(),
        "f_tht": allocate_f(),
        # "f_zzf": allocate_f(),
    }


def initialize_state_with_constant(
    state: DataArrayDict, C: float, gt4py_config: GT4PyConfig
) -> None:
    """Initialize fields of state dictionnary with a constant field.

    Args:
        state (DataArrayDict): dictionnary of state
        C (float): constant value for initialization
        gt4py_config (GT4PyConfig): configuration of gt4py
    """

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
        "f_tht",
    ]

    for name in keys:
        state[name][...] = C * ones(state[name].shape, backend=gt4py_config.backend)


def get_state(
    computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig
) -> DataArrayDict:
    """Create state dictionnary with allocation of tables and setup to a constant value.

    Args:
        computational_grid (ComputationalGrid): grid indexes
        gt4py_config (GT4PyConfig): configuration for gt4py

    Returns:
        DataArrayDict: initialized dictionnary of state
    """
    state = allocate_state(computational_grid, gt4py_config=gt4py_config)
    initialize_state_with_constant(state, 0.5)
    return state
