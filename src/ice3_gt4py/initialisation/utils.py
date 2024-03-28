# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ifs_physics_common.utils.numpyx import assign

if TYPE_CHECKING:
    from ifs_physics_common.utils.typingx import DataArray, NDArrayLike
    from numpy.typing import NDArray


def initialize_storage_2d(storage: NDArrayLike, buffer: NDArray) -> None:
    """Assign storage for 2D field in buffer

    GPU (cupy) / CPU (numpy) compatible

    Args:
        storage (NDArrayLike): storage slot
        buffer (NDArray): 2D field in buffer
    """
    ni = storage.shape[0]
    mi = buffer.size
    nb = ni // mi
    for b in range(nb):
        assign(storage[b * mi : (b + 1) * mi, 0:1], buffer[:, np.newaxis])
    assign(storage[nb * mi :, 0:1], buffer[: ni - nb * mi, np.newaxis])


def initialize_storage_3d(storage: NDArrayLike, buffer: NDArray) -> None:
    """Assign storage for 3D field in buffer

    GPU (cupy) / CPU (numpy) compatible

    Args:
        storage (NDArrayLike): storage slot
        buffer (NDArray): 3D field in buffer
    """
    ni, _, nk = storage.shape
    mi, mk = buffer.shape
    lk = min(nk, mk)
    nb = ni // mi
    for b in range(nb):
        assign(storage[b * mi : (b + 1) * mi, 0:1, :lk], buffer[:, np.newaxis, :lk])
    assign(storage[nb * mi :, 0:1, :lk], buffer[: ni - nb * mi, np.newaxis, :lk])


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
