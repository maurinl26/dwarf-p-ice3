# -*- coding: utf-8 -*-
from gt4py.cartesian.gtscript import function


@function
def sign(a: float, b: float) -> float:
    """Returns the value of a with the sign of b

    sign(a,b) = abs(a) if b >= 0 else -abs(a)

    Args:
        a (float): value to take the sign from
        b (float): value to change with the sign of a

    Returns:
        float: _description_
    """
    return abs(a) if b >= 0 else -abs(a)
