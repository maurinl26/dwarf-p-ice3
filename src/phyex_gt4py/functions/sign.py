# -*- coding: utf-8 -*-
from gt4py.cartesian.gtscript import function


@function
def sign(
    x: float,
) -> float:
    """Compute sign function 
    
    sign = 1 if x > 0
    sign = 0 if x = 0
    sign = -1 if x < 0

    Args:
        x (float): value 

    Returns:
        float: value of sign function at x
    """
    if x > 0:
        sign = 1
    elif x == 0:
        sign = 0
    elif x < 0:
        sign = -1

    return sign
