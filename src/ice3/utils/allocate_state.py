# -*- coding: utf-8 -*-
from functools import partial
from typing import TYPE_CHECKING, Literal, Tuple, Dict
import numpy as np

from ifs_physics_common.framework.config import GT4PyConfig
from ifs_physics_common.framework.grid import ComputationalGrid, DimSymbol, I, J, K
from ifs_physics_common.framework.storage import allocate_data_array
from ifs_physics_common.utils.numpyx import assign

if TYPE_CHECKING:
    from ifs_physics_common.utils.typingx import DataArray, NDArrayLike, NDArrayLikeDict
    from numpy.typing import NDArray

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


def allocate_state(
    computational_grid: ComputationalGrid, gt4py_config: GT4PyConfig, fields: Dict[str, NDArray]
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



def initialize_storage_2d(storage: NDArrayLike, buffer: NDArray) -> None:
    """Assign storage for 2D field in buffer

    GPU (cupy) / CPU (numpy) compatible

    Args:
        storage (NDArrayLike): storage slot
        buffer (NDArray): 2D field in buffer
    """
    assign(storage, buffer[:, np.newaxis])


def initialize_storage_3d(storage: NDArrayLike, buffer: NDArray) -> None:
    """Assign storage for 3D field in buffer

    GPU (cupy) / CPU (numpy) compatible

    Args:
        storage (NDArrayLike): storage slot
        buffer (NDArray): 3D field in buffer
    """

    # expand a dimension of size 1 for nj
    assign(storage, buffer[:, np.newaxis, :])


def initialize_field(field: DataArray, buffer: NDArray) -> None:
    """Initialize storage for a given field with dimension descriptor

    Args:
        field (DataArray): field to assign
        buffer (NDArray): buffer

    Raises:
        ValueError: restriction to 2D or 3D fields
    """
    if field.ndim == 2:
        initialize_storage_2d(field.data, buffer)
    elif field.ndim == 3:
        initialize_storage_3d(field.data, buffer)
    else:
        raise ValueError("The field to initialize must be either 2-d or 3-d.")

