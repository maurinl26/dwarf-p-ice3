# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function, __INLINED


@function
def compute_frac_ice(
    frac_ice: Field["float"],
    t: Field["float"],  # type: ignore
) -> Field["float"]:  # type: ignore
    """Compute ice fraction based on temperature

    FRAC_ICE_ADJUST is the mode of calculation

    Args:
        t (Field[float]): temperature

    Returns:
        Field[float]: ice fraction with respect to ice + liquid
    """

    from __externals__ import FRAC_ICE_ADJUST, TMAXMIX, TMINMIX, TT  # type: ignore

    frac_ice = 0

    # using temperature
    # FracIceAdjust.T.value
    if __INLINED(FRAC_ICE_ADJUST == 0):
        frac_ice = max(0, min(1, ((TMAXMIX - t) / (TMAXMIX - TMINMIX))))

    # using temperature with old formula
    # FracIceAdjust.O.value
    elif __INLINED(FRAC_ICE_ADJUST == 1):
        frac_ice = max(0, min(1, ((TT - t) / 40)))

    # no ice
    # FracIceAdjust.N.value
    elif __INLINED(FRAC_ICE_ADJUST == 2):
        frac_ice = 0

    # same as previous
    # FracIceAdjust.S.value
    elif __INLINED(FRAC_ICE_ADJUST == 3):
        frac_ice = max(0, min(1, frac_ice))

    return frac_ice
