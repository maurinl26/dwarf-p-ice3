# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
from functools import partial
from typing import TYPE_CHECKING, Dict, List

from gt4py.storage import ones
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array

from ice3_gt4py.initialisation.state import initialize_state_with_constant
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

############################## AroAdjust #################################
def allocate_state_aro_adjust(
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

    state = {
        "time": datetime.datetime(year=2024, month=1, day=1),
        "tht": allocate_f(),
        "exn": allocate_f(),
        "exnref": allocate_f(),
        "rhodref": allocate_f(),
        "pabs": allocate_f(),
        "sigs": allocate_f(),
        "cf_mf": allocate_f(),
        "rc_mf": allocate_f(),
        "ri_mf": allocate_f(),
        "th": allocate_f(),
        "rv": allocate_f(),
        "rc": allocate_f(),
        "ri": allocate_f(),
        "rr": allocate_f(),
        "rs": allocate_f(),
        "rg": allocate_f(),
        "sigqsat": allocate_f(),
        "cldfr": allocate_f(),
        "ifr": allocate_f(),
        "hlc_hrc": allocate_f(),
        "hlc_hcf": allocate_f(),
        "hli_hri": allocate_f(),
        "hli_hcf": allocate_f(),
        "sigrc": allocate_f(),
    }

    diagnostics = {
        "ths": allocate_f(),
        "rcs": allocate_f(),
        "rrs": allocate_f(),
        "ris": allocate_f(),
        "rvs": allocate_f(),
        "rgs": allocate_f(),
        "rss": allocate_f(),
    }

    return {**state, **diagnostics}


aro_adjust_fields_keys = [
    "exnref",
    "tht",
    "exn",
    "exnref",
    "rhodref",
    "pabs",
    "sigs",
    "cf_mf",
    "rc_mf",
    "ri_mf",
    "th",
    "rv",
    "rc",
    "ri",
    "rr",
    "rs",
    "rg",
    "sigqsat",
    "cldfr",
    "ifr",
    "hlc_hrc",
    "hlc_hcf",
    "hli_hri",
    "hli_hcf",
    "sigrc",
]


def get_constant_state_aro_adjust(
    computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig, keys: List[str]
) -> DataArrayDict:
    """Create state dictionnary with allocation of tables and setup to a constant value.

    Args:
        computational_grid (ComputationalGrid): grid indexes
        gt4py_config (GT4PyConfig): configuration for gt4py

    Returns:
        DataArrayDict: initialized dictionnary of state
    """
    state = allocate_state_aro_adjust(computational_grid, gt4py_config=gt4py_config)
    initialize_state_with_constant(state, 0.5, gt4py_config, keys)
    return state
