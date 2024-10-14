# -*- coding: utf-8 -*-
from ifs_physics_common.framework.storage import allocate_data_array
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid

from typing import Literal, Tuple

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
from ifs_physics_common.utils.typingx import (
    DataArray,
)


def allocate(
    grid_id: Tuple[DimSymbol, ...],
    units: str,
    dtype: Literal["bool", "float", "int"],
    computational_grid: ComputationalGrid,
    gt4py_config: GT4PyConfig,
) -> DataArray:
    """Allocate array given a ComputationalGrid

    Args:
        grid_id (Tuple[DimSymbol, ...]): _description_
        units (str): _description_
        dtype (Literal["bool", "float", "int"]): _description_
        computational_grid (ComputationalGrid): _description_
        gt4py_config (GT4PyConfig): _description_

    Returns:
        DataArray: _description_
    """
    return allocate_data_array(
        computational_grid, grid_id, units, gt4py_config=gt4py_config, dtype=dtype
    )
