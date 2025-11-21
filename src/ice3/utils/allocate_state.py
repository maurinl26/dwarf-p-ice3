# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
from numpy.typing import NDArray

# todo : implement assign
def assign(rhs: NDArray, lhs: NDArray):
    ...

def initialize_storage_2d(storage: NDArray, buffer: NDArray) -> None:
    """Assign storage for 2D field in buffer

    GPU (cupy) / CPU (numpy) compatible

    Args:
        storage (NDArrayLike): storage slot
        buffer (NDArray): 2D field in buffer
    """
    assign(storage, buffer[:, np.newaxis])


def initialize_storage_3d(storage: NDArray, buffer: NDArray) -> None:
    """Assign storage for 3D field in buffer

    GPU (cupy) / CPU (numpy) compatible

    Args:
        storage (NDArrayLike): storage slot
        buffer (NDArray): 3D field in buffer
    """

    # expand a dimension of size 1 for nj
    assign(storage, buffer[:, np.newaxis, :])


def initialize_field(field: xr.DataArray, buffer: NDArray) -> None:
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

