# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, function, log

from ice3_gt4py.functions.sign import sign


@function
def esatw(alpw: float, betaw: float, tt: Field["float"]):
    
    from __externals__ import ALPW, BETAW, GAMW

    return exp(ALPW - BETAW / tt[0, 0, 0] - GAMW * log(tt[0, 0, 0]))


@function
def esati(cst_tt: float, alpw: float, betaw: float, tt: Field["float"]):
    
    from __externals__ import ALPW, BETAW, GAMW
    
    return (0.5 + sign(0.5, tt - cst_tt)) * esatw(tt) - (
        sign(0.5, tt - cst_tt) - 0.5
    ) * esatw(tt)

