# -*- coding: utf-8 -*-
from functools import partial
from typing import Literal, Tuple

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol, I, J, K
from ifs_physics_common.framework.storage import allocate_data_array
from ifs_physics_common.utils.typingx import DataArray, NDArrayLikeDict


def allocate_state(
    computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig, fields: dict
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
    allocate_i = partial(_allocate, grid_id=(I, J, K), units="", dtype="int")
    allocate_h = partial(_allocate, grid_id=(I, J, K - 1 / 2), units="", dtype="float")
    allocate_ij = partial(_allocate, grid_id=(I, J), units="", dtype="float")
    allocate_i_ij = partial(_allocate, grid_id=(I, J), units="", dtype="int")

    state = dict()
    for field_name, field_attributes in fields.items():
        if field_attributes["dtype"] == "float":
            state.update({field_name: allocate_f()})
        elif field_attributes["dtype"] == "int":
            state.update({field_name: allocate_i()})
        elif field_attributes["dtype"] == "bool":
            state.update({field_name: allocate_b()})

    return state
