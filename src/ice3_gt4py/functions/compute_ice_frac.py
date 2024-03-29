# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

from gt4py.cartesian.gtscript import Field, function


@function
def compute_frac_ice(
    t: Field["float"],
) -> Field["float"]:
    """Compute ice fraction based on temperature

    frac_ice_adjust is the mode of calculation

    Args:
        t (Field[float]): temperatue

    Returns:
        Field[float]: ice fraction with respect to ice + liquid
    """

    from __externals__ import FRAC_ICE_ADJUST, TMAXMIX, TMINMIX, TT

    frac_ice = 0

    # using temperature
    # FracIceAdujst.T.value
    if FRAC_ICE_ADJUST == 0:
        frac_ice = max(0, min(1, ((TMAXMIX - t[0, 0, 0]) / (TMAXMIX - TMINMIX))))

    # using temperature with old formula
    # FracIceAdujst.O.value
    elif FRAC_ICE_ADJUST == 1:
        frac_ice = max(0, min(1, ((TT - t[0, 0, 0]) / 40)))

    # no ice
    # FracIceAdujst.N.value
    elif FRAC_ICE_ADJUST == 2:
        frac_ice = 0

    # same as previous
    # FracIceAdujst.S.value
    elif FRAC_ICE_ADJUST == 3:
        frac_ice = max(0, min(1, frac_ice))

    return frac_ice
