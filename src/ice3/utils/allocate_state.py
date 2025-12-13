# -*- coding: utf-8 -*-
"""Utilities for initializing GT4Py storage from buffer data.

This module provides functions to initialize GT4Py storage objects from NumPy arrays,
handling the conversion between different dimensional representations. It's particularly
useful for setting up initial conditions from existing data (e.g., NetCDF files).

The functions handle the expansion of dimensions to match GT4Py's expected 3D structure
(ni, nj, nk) even when the input data is 2D (ni, nk) with nj=1.
"""

import numpy as np
import xarray as xr
try:
    import cupy as cp
except ImportError:
    cp = None
from numpy.typing import NDArray
from typing import Union, Any


def assign(lhs: Any, rhs: Any) -> None:
    """Assign array data from source to destination, handling NumPy/CuPy cross-assignments
    and memoryview objects.
    
    Args:
        lhs: Left-hand side (destination storage) - modified in place
        rhs: Right-hand side (source data) - not modified
    """
    is_rhs_cp = cp is not None and isinstance(rhs, cp.ndarray)
    is_lhs_cp = cp is not None and isinstance(lhs, cp.ndarray)

    if is_lhs_cp:
        if is_rhs_cp:
            lhs[:] = rhs
        else:
            # lhs is cupy, rhs is numpy/memoryview
            lhs.set(np.asarray(rhs))
    else:
        # lhs is numpy/memoryview
        target = np.asarray(lhs)
        if is_rhs_cp:
            # rhs is cupy
            target[:] = rhs.get()
        else:
            # both are numpy/memoryview
            target[:] = np.asarray(rhs)

def get_allocator(backend: str) -> Any:
    """Return the appropriate allocator (numpy or cupy) for the given backend.
    
    Args:
        backend (str): GT4Py backend name
        
    Returns:
        module: numpy or cupy module
    """
    if "gpu" in backend and cp is not None:
        return cp
    return np

def initialize_storage_2d(storage: NDArray, buffer: NDArray) -> None:
    """Initialize GT4Py storage from 2D buffer data by adding a singleton j-dimension.
    
    Converts 2D data of shape (ni, nk) into 3D storage of shape (ni, 1, nk) by
    inserting a dimension of size 1 for the j-axis (horizontal y-direction). This is
    necessary because GT4Py stencils expect 3D fields even for column or 2D slice data.

    GPU (CuPy) / CPU (NumPy) compatible.

    Args:
        storage (NDArray): GT4Py storage object to populate, expected shape (ni, 1, nk)
        buffer (NDArray): Source 2D data array, shape (ni, nk)
    """
    assign(storage, buffer[:, np.newaxis])


def initialize_storage_3d(storage: NDArray, buffer: NDArray) -> None:
    """Initialize GT4Py storage from 3D buffer data by adding a singleton j-dimension.
    
    Converts 3D data of shape (ni, nk) into 3D storage of shape (ni, 1, nk) by
    inserting a dimension of size 1 for the j-axis (horizontal y-direction). This
    handles the common case where input data doesn't include the j-dimension but
    GT4Py stencils require it.

    GPU (CuPy) / CPU (NumPy) compatible.

    Args:
        storage (NDArray): GT4Py storage object to populate, expected shape (ni, 1, nk)
        buffer (NDArray): Source 3D data array, shape (ni, nk) - despite the name, 
            this is typically 2D data that will be expanded to 3D
    """

    # expand a dimension of size 1 for nj
    assign(storage, buffer[:, np.newaxis, :])


def initialize_field(field: Union[xr.DataArray, NDArray, Any], buffer: NDArray) -> None:
    """Initialize a field (DataArray or array) with data from a buffer.
    
    Automatically detects whether the field is 2D or 3D and calls the appropriate
    initialization function. This provides a convenient high-level interface for
    setting up field data from external sources (e.g., NetCDF files, initial conditions).

    Args:
        field (Union[xr.DataArray, NDArray, Any]): Field to initialize. Can be xarray DataArray,
            numpy/cupy array, or GT4Py storage.
        buffer (NDArray): Source data array to copy into the field, shape should match
            the field dimensions
    """
    data = getattr(field, "data", field)
    
    if data.shape == buffer.shape:
        assign(data, buffer)
    elif data.ndim == 3 and buffer.ndim == 2:
        initialize_storage_3d(data, buffer)
    elif data.ndim == 2 and buffer.ndim == 1:
        initialize_storage_2d(data, buffer)
    else:
         assign(data, buffer) # Try direct assignment, broadcasting might work
