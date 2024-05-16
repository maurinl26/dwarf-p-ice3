# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array

from ice3_gt4py.initialisation.state import initialize_state_with_constant

if TYPE_CHECKING:
    from typing import Literal, Tuple

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
    from ifs_physics_common.utils.typingx import (
        DataArray,
        DataArrayDict,
        NDArrayLikeDict,
    )

############################## Ice4Tendencies #################################
def allocate_state_ice4_tendencies(
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
    allocate_b = partial(_allocate, grid_id=(I, J, K), units="", dtype="bool")
    allocate_f = partial(_allocate, grid_id=(I, J, K), units="", dtype="float")
    allocate_f_h = partial(
        _allocate, grid_id=(I, J, K - 1 / 2), units="", dtype="float"
    )
    allocate_f_ij = partial(_allocate, grid_id=(I, J), units="", dtype="float")
    allocate_i_ij = partial(_allocate, grid_id=(I, J), units="", dtype="int")

    time_state = {
        "t_micro": allocate_f(),
        "t_soft": allocate_f(),
    }

    masks = {"ldcompute": allocate_b(), "ldmicro": allocate_b()}

    state = {
        "tht": allocate_f(),
        "pabs": allocate_f(),
        "rhodref": allocate_f(),
        "exn": allocate_f(),
        "ls_fact": allocate_f(),
        "lv_fact": allocate_f(),
        "t": allocate_f(),
        "rv_t": allocate_f(),
        "rc_t": allocate_f(),
        "rr_t": allocate_f(),
        "ri_t": allocate_f(),
        "rs_t": allocate_f(),
        "rg_t": allocate_f(),
        "ci_t": allocate_f(),
        "pres": allocate_f(),
        "ssi": allocate_f(),  # supersaturation over ice
        "ka": allocate_f(),  #
        "dv": allocate_f(),
        "ai": allocate_f(),
        "cj": allocate_f(),
        "hlc_hcf": allocate_f(),  # High Cloud Fraction in grid
        "hlc_lcf": allocate_f(),  # Low Cloud Fraction in grid
        "hlc_hrc": allocate_f(),  # LWC that is high in grid
        "hlc_lrc": allocate_f(),
        "hli_hcf": allocate_f(),
        "hli_hri": allocate_f(),
    }

    increments = {
        "theta_increment": allocate_f(),
        "rv_increment": allocate_f(),
        "rc_increment": allocate_f(),
        "rr_increment": allocate_f(),
        "ri_increment": allocate_f(),
        "rs_increment": allocate_f(),
        "rg_increment": allocate_f(),
    }

    # Used in state tendencies update
    tnd_update = {
        "theta_tnd": allocate_f(),
        "rv_tnd": allocate_f(),
        "rc_tnd": allocate_f(),
        "rr_tnd": allocate_f(),
        "ri_tnd": allocate_f(),
        "rs_tnd": allocate_f(),
        "rg_tnd": allocate_f(),
    }

    return {
        **time_state,
        **masks,
        **state,
        **tnd_update,
        **increments,
    }


def get_constant_state_ice4_tendencies(
    computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig
) -> DataArrayDict:
    """Create state dictionnary with allocation of tables and setup to a constant value.

    Args:
        computational_grid (ComputationalGrid): grid indexes
        gt4py_config (GT4PyConfig): configuration for gt4py

    Returns:
        DataArrayDict: initialized dictionnary of state
    """
    state = allocate_state_ice4_tendencies(
        computational_grid, gt4py_config=gt4py_config
    )
    initialize_state_with_constant(state, 0.5, gt4py_config, list(state.keys()))
    return state
