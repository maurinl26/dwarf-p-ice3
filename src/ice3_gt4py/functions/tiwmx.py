# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, function, log

from ice3_gt4py.functions.sign import sign


@function
def esatw(alpw: float, betaw: float, tt: Field["float"]) -> Field["float"]:
    """Saturation function over liquid water

    Args:
        alpw (float): alpha parameter for exponential - liquid water
        betaw (float): beta parameter for exponential - liquid water
        tt (Field[float]): temperature

    Returns:
        Field["float"]: vapour content at saturation
    """
    esatw = exp(alpw - betaw / tt[0, 0, 0] - log(tt[0, 0, 0]))
    return esatw


@function
def esati(cst_tt: float, alpw: float, betaw: float, tt: Field["float"]):
    """Saturation function over ice

    Args:
        cst_tt (float): Temperature at triple point for water
        alpw (float): alpha coefficient for ice
        betaw (float): beta coefficient for ice
        tt (Field[float]): temperature

    Returns:
        _type_: _description_
    """
    esati = (0.5 + sign(0.5, tt - cst_tt)) * esatw(alpw, betaw, tt) - (
        sign(0.5, tt - cst_tt) - 0.5
    ) * esatw(alpw, betaw, tt)

    return esati
