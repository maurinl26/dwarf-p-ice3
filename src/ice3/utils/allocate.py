# -*- coding: utf-8 -*-
from ifs_physics_common.framework.storage import allocate_data_array
from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol
from ifs_physics_common.utils.typingx import (
    DataArray,
)

from gt4py.storage import from_array
from typing import Literal, Tuple
import numpy as np


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


def allocate_random_fields(names, gt4py_config, grid, dtype=None):
    dtype = dtype or gt4py_config.dtypes.float
    fields = {name: np.array(np.random.rand(*grid.shape), dtype=dtype, order="F") for name in names}
    gt4py_buffers = {name: from_array(fields[name], dtype=gt4py_config.dtypes.float, backend=gt4py_config.backend) for name in names}
    return fields, gt4py_buffers
