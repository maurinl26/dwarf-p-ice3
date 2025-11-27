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
from numpy.typing import NDArray

def assign(lhs: NDArray, rhs: NDArray) -> None:
    """Assign array data from source to destination.
    
    Provides a unified interface for array assignment that works across both CPU
    (NumPy) and GPU (CuPy) arrays. This function performs an in-place copy of data
    from the right-hand side (source) to the left-hand side (destination) using
    slice assignment, which is compatible with both NumPy and CuPy.
    
    Args:
        lhs (NDArray): Left-hand side array (destination storage) - modified in place
        rhs (NDArray): Right-hand side array (source data) - not modified
    
    Note:
        The destination array (lhs) must be pre-allocated and have compatible shape
        with the source array (rhs). Broadcasting rules apply.
    """
    lhs[:] = rhs

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


def initialize_field(field: xr.DataArray, buffer: NDArray) -> None:
    """Initialize an xarray DataArray field with data from a buffer.
    
    Automatically detects whether the field is 2D or 3D and calls the appropriate
    initialization function. This provides a convenient high-level interface for
    setting up field data from external sources (e.g., NetCDF files, initial conditions).

    Args:
        field (xr.DataArray): xarray DataArray with GT4Py storage in its .data attribute,
            must be either 2D or 3D
        buffer (NDArray): Source data array to copy into the field, shape should match
            the field dimensions

    Raises:
        ValueError: If the field is neither 2D nor 3D
    """
    if field.ndim == 2:
        initialize_storage_2d(field.data, buffer)
    elif field.ndim == 3:
        initialize_storage_3d(field.data, buffer)
    else:
        raise ValueError("The field to initialize must be either 2-d or 3-d.")
