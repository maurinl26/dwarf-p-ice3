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
    Return the absolute value of b with the sign of a.
    
    This function implements the Fortran SIGN intrinsic function commonly used in
    scientific codes. It returns abs(b) with the sign transferred from a:
    - If a >= 0, returns +|b|
    - If a < 0, returns -|b|
    
    Parameters
    ----------
    a : float
        Scalar input whose sign determines the sign of the output.
    b : float
        Scalar input whose absolute value determines the magnitude of the output.
        
    Returns
    -------
    float
        The absolute value of b with the sign of a:
        - Returns +|b| if a >= 0
        - Returns -|b| if a < 0
        
    Notes
    -----
    This is equivalent to the Fortran intrinsic function SIGN(b, a), which returns
    the value of b with the sign of a. The implementation follows the standard
    definition where SIGN(x, y) = |x| if y >= 0, and -|x| if y < 0.
    
    Examples
    --------
    sign(5.0, 3.0) returns 3.0 (positive sign from a, magnitude from b)
    sign(-5.0, 3.0) returns -3.0 (negative sign from a, magnitude from b)
    sign(0.0, 7.0) returns 7.0 (zero or positive is treated as positive)
    sign(2.0, -4.0) returns 4.0 (magnitude is absolute value of b)
    """
    if a >= 0.0:
        sign_b = 1 * abs(b)
    elif a < 0.0:
        sign_b = -1 * abs(b)

    return sign_b
