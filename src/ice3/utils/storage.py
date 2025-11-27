# -*- coding: utf-8 -*-
"""Context manager for temporary GT4Py storage allocation.

This module provides utilities for managing temporary storage in GT4Py stencil computations.
Temporary fields are automatically allocated at context entry and can be used within the
context, promoting clean resource management.
"""

from contextlib import contextmanager
from typing import List, Tuple, Union

from gt4py.storage import zeros
from gt4py.cartesian.gtscript import IJ, IJK, K, Axis
from ..utils.env import DTYPES, BACKEND


@contextmanager
def managed_temporaries(
    temporaries: List[Tuple[Tuple[int, ...], str]],
    domain: Tuple[Axis, ...],
    backend: str = BACKEND,
    dtypes: dict = DTYPES,
    aligned_index: Tuple[int, ...] = (0, 0, 0),
):
    """Context manager for allocating temporary GT4Py storage fields.
    
    Provides a convenient way to allocate temporary storage for intermediate computations
    in GT4Py stencils. The storage is allocated based on the specified dimensions (IJ for
    2D horizontal, IJK for 3D) and automatically yields the allocated fields for use.
    
    The context manager handles varying dimensional requirements - 2D horizontal fields
    (IJ) use only the first two dimensions of the domain, while 3D fields (IJK) use the
    full domain shape.

    Args:
        temporaries (List[Tuple[Tuple[int, ...], str]]): List of tuples specifying temporary
            fields to create. Each tuple contains:
            - dimensions: Tuple specifying field dimensions (e.g., IJ, IJK)
            - dtype_key: String key for the data type (e.g., "float")
        domain (Tuple[int, int, int]): 3D domain shape as (ni, nj, nk)
        backend (str, optional): GT4Py backend to use. Defaults to BACKEND from environment.
        dtypes (dict, optional): Dictionary mapping dtype keys to numpy types. 
            Defaults to DTYPES from environment.
        aligned_index (Tuple[int, ...], optional): Starting index for aligned memory access.
            Defaults to (0, 0, 0).

    Yields:
        GT4Py storage objects: Each temporary field as specified in the temporaries list
        
    Example:
        >>> temporaries = [(IJK, "float"), (IJ, "float")]
        >>> domain = (64, 1, 80)
        >>> with managed_temporaries(temporaries, domain) as (temp_3d, temp_2d):
        ...     # Use temp_3d and temp_2d in computations
        ...     pass
    """

    def _allocate_temporary(dims, dtype):

        match dims:
            case _ if dims == K:
                return zeros(
                    shape=(domain[2],),
                    dtype=dtypes[dtype],
                    aligned_index=(aligned_index[2],),
                    backend=backend,
                )
            case _ if dims == IJ:
                return zeros(
                    shape=domain[:2],
                    dtype=dtypes[dtype],
                    aligned_index=aligned_index[:2],
                    backend=backend,
                )
            case _ if dims == IJK:
                return zeros(
                    shape=domain,
                    dtype=dtypes[dtype],
                    aligned_index=aligned_index,
                    backend=backend,
                )

    yield tuple(_allocate_temporary(dims, dtype) for dims, dtype in temporaries)
