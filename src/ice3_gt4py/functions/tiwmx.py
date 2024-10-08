# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, function, log

from ice3_gt4py.functions.sign import sign


@function
def e_sat_w(t: Field["float"]) -> Field["float"]:
    """Saturation vapor pressure over liquid water

    Args:
        t (Field[float]): temperature

    Returns:
        Field[float]: saturation vapor pressure
    """

    from __externals__ import ALPW, BETAW, GAMW

    return exp(ALPW - BETAW / t - GAMW * log(t))


@function
def e_sat_i(t: Field["float"]):
    """Saturation vapor pressure over ice

    Args:
        t (Field[float]): temperature

    Returns:
        Field[float]: saturation vapor pressure
    """
    from __externals__ import ALPI, BETAI, GAMI

    return exp(ALPI - BETAI / t - GAMI * log(t))


# @function
# def esati(cst_tt: float, alpw: float, betaw: float, tt: Field["float"]):

#     from __externals__ import ALPW, BETAW, GAMW

#     return (0.5 + sign(0.5, tt - cst_tt)) * esatw(tt) - (
#         sign(0.5, tt - cst_tt) - 0.5
#     ) * esatw(tt)
