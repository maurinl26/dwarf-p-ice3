# -*- coding: utf-8 -*-
"""
Sign function for microphysics calculations.

This module provides a GT4Py implementation of the sign function that
returns the absolute value of a scalar with the sign preserved.
"""
from __future__ import annotations

from gt4py.cartesian.gtscript import function

@function
def sign(a: "float", b: "float") -> "float":
    """
    Return the absolute value of a with the sign of a.
    
    This function implements a sign transfer operation commonly used in
    Fortran-style scientific codes. It returns abs(a) with the sign of a
    itself (positive if a >= 0, negative if a < 0).
    
    Parameters
    ----------
    a : float
        Scalar input whose absolute value and sign determine the output.
    b : float
        Second parameter (currently not used in the implementation).
        
    Returns
    -------
    float
        The absolute value of a multiplied by the sign of a:
        - Returns +|a| if a >= 0
        - Returns -|a| if a < 0
        
    Notes
    -----
    The parameter b appears in the signature but is not used in the current
    implementation. This may be a placeholder for future functionality or
    compatibility with other sign function variants.
    
    Examples
    --------
    sign(5.0, _) returns 5.0
    sign(-5.0, _) returns -5.0
    sign(0.0, _) returns 0.0
    """
    if a >= 0.0:
        sign_b = 1 * abs(a)
    elif a < 0.0:
        sign_b = -1 * abs(a)

    return sign_b
