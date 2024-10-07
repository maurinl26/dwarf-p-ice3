# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import partial
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array


from typing import Literal, Tuple

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
from ifs_physics_common.utils.typingx import (
    DataArray,
)


def _allocate(
    gt4py_config: GT4PyConfig,
    computational_grid: ComputationalGrid,
    grid_id: Tuple[DimSymbol, ...],
    units: str,
    dtype: Literal["bool", "float", "int"],
) -> DataArray:
    return allocate_data_array(
        computational_grid, grid_id, units, gt4py_config=gt4py_config, dtype=dtype
    )


allocate_b_ij = partial(_allocate, grid_id=(I, J), units="", dtype="bool")
allocate_f = partial(_allocate, grid_id=(I, J, K), units="", dtype="float")
allocate_h = partial(_allocate, grid_id=(I, J, K - 1 / 2), units="", dtype="float")
allocate_ij = partial(_allocate, grid_id=(I, J), units="", dtype="float")
allocate_i_ij = partial(_allocate, grid_id=(I, J), units="", dtype="int")
allocate_i = partial(_allocate, grid_id=(I, J, K), units="", dtype="int")
