# -*- coding: utf-8 -*-
"""
Field backup functions for iterative microphysics calculations.

This module provides GT4Py implementations of functions to save and restore
field states during iterative microphysics computations. These are used in
schemes that require multiple sub-steps within a model time step.
"""
from __future__ import annotations

from typing import Tuple

from gt4py.cartesian.gtscript import Field, function


@function
def backup(
    rv_tmp: Field["float"],
    ri_tmp: Field["float"],
    rc_tmp: Field["float"],
    rv_out: Field["float"],
    ri_out: Field["float"],
    rc_out: Field["float"],
) -> Tuple[Field["float"]]:
    """
    Store output fields into temporary fields for iterative calculations.
    
    This function copies the current state of mixing ratio fields into
    temporary storage fields. This is used in iterative schemes where
    the same computations are applied multiple times, requiring the
    preservation of intermediate states.
    
    Parameters
    ----------
    rv_tmp : Field[float]
        Temporary storage for water vapor mixing ratio (kg/kg).
    ri_tmp : Field[float]
        Temporary storage for ice mixing ratio (kg/kg).
    rc_tmp : Field[float]
        Temporary storage for cloud liquid water mixing ratio (kg/kg).
    rv_out : Field[float]
        Current water vapor mixing ratio to be backed up (kg/kg).
    ri_out : Field[float]
        Current ice mixing ratio to be backed up (kg/kg).
    rc_out : Field[float]
        Current cloud liquid water mixing ratio to be backed up (kg/kg).
        
    Returns
    -------
    Tuple[Field[float], Field[float], Field[float]]
        The temporary fields (rv_tmp, ri_tmp, rc_tmp) containing
        the backed-up values.
        
    Notes
    -----
    This function performs a point-wise copy operation:
    - rv_tmp = rv_out[0, 0, 0]
    - ri_tmp = ri_out[0, 0, 0]
    - rc_tmp = rc_out[0, 0, 0]
    
    The [0, 0, 0] indexing indicates local grid point access in the
    GT4Py stencil context.
    
    This is commonly used in saturation adjustment schemes where
    multiple iterations are performed to reach thermodynamic equilibrium,
    and the state must be saved before each iteration to check for
    convergence or to restore in case of numerical issues.
    """
    rv_tmp = rv_out[0, 0, 0]
    ri_tmp = ri_out[0, 0, 0]
    rc_tmp = rc_out[0, 0, 0]

    return rv_tmp, ri_tmp, rc_tmp
