# -*- coding: utf-8 -*-
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple, List, Dict
    from numpy.typing import NDArray


############# Utils #############
def absolute_differences(
    fortran_fields: Dict[str, NDArray],
    gt4py_fields: Dict[str, NDArray],
    fields_to_compare: List[str]
) -> Dict[str, NDArray]:
    """Compute mean absolute differences between Fortran and GT4Py field implementations.
    
    This function is used for validation and testing, comparing outputs from Fortran
    reference implementations against GT4Py stencil implementations. It only computes
    differences for fields with matching shapes, returning None for mismatched fields.

    Args:
        fortran_fields (Dict[str, NDArray]): Dictionary of fields from Fortran implementation,
            typically output from f2py-wrapped subroutines
        gt4py_fields (Dict[str, NDArray]): Dictionary of fields from GT4Py stencils,
            typically xarray DataArrays or GT4Py storage objects
        fields_to_compare (List[str]): List of field names to compare between the two
            implementations

    Returns:
        Dict[str, float | None]: Dictionary mapping field names to their mean absolute
            differences. Returns None for fields with shape mismatches.
    """

    return {
        field_name : abs(gt4py_fields[field_name] - fortran_fields[field_name]).values.mean()
        if (gt4py_fields[field_name].shape == fortran_fields[field_name].shape)
        else None
        for field_name in fields_to_compare.keys()
    }


def remove_y_axis(fields: Dict[str, NDArray]) -> Dict[str, NDArray]:
    """Remove the y-axis (axis=1) from 3D fields, reducing them to 2D.
    
    This is useful when working with column-like data structures where the horizontal
    y-dimension is singular (nj=1). Squeezing out this dimension simplifies visualization
    and analysis of vertical profiles or 2D slices.

    Args:
        fields (Dict[str, NDArray]): Dictionary of 3D arrays with shape (ni, nj, nk)
            where nj is typically 1

    Returns:
        Dict[str, NDArray]: Dictionary with same keys but 2D arrays of shape (ni, nk)
    """
    return {key: np.squeeze(array, axis=1) for key, array in fields.items()}


def unpack(fields: Dict[str, NDArray], domain: Tuple[int, int, int]) -> Dict[str, NDArray]:
    """Reshape 3D fields to 2D by flattening the horizontal dimensions.
    
    Converts fields from 3D structure (ni, nj, nk) to 2D (ni*nj, nk), effectively
    treating each horizontal grid point as an independent column. This is commonly
    used for interfacing with 1D column physics schemes or for efficient vectorized
    operations over horizontal points.

    Args:
        fields (Dict[str, NDArray]): Dictionary of 3D arrays to reshape
        domain (Tuple[int, int, int]): The 3D domain shape (ni, nj, nk) used for reshaping

    Returns:
        Dict[str, NDArray]: Dictionary with same keys but 2D arrays of shape (ni*nj, nk)
    """
    return {
        key: array.reshape(domain[0] * domain[1], domain[2])
        for key, array in fields.items()
    }
