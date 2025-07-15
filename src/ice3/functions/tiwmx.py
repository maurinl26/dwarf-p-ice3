# -*- coding: utf-8 -*-
from __future__ import annotations

import dace
from ice3_gt4py.utils.typingx import dtype_float, dtype_int


def e_sat_w(
        t: dtype_float,
        pv: dtype_float,
        ext: dace.compiletime
):
    """Saturation vapor pressure over liquid water

    Args:
        t (Field[float]): temperature

    Returns:
        Field[float]: saturation vapor pressure
    """

    pv = exp(ext.ALPW - ext.BETAW / t - ext.GAMW * log(t))


def e_sat_i(
    t: dtype_float,
    piv: dtype_float,
    ext: dace.compiletime
):
    """Saturation vapor pressure over ice

    Args:
        t (Field[float]): temperature

    Returns:
        Field[float]: saturation vapor pressure
    """

    piv = exp(ext.ALPI - ext.BETAI / t - ext.GAMI * log(t))

