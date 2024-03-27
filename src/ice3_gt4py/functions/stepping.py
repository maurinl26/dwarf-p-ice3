# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function


@function
def mixing_ratio_step_limiter(
    r_a_tnd: Field["float"],
    r_b: Field["float"],
    r_t: Field["float"],
    delta_t_micro: Field["float"],
    RTMIN: "float",
    MNH_TINY: "float",
) -> Field["float"]:
    """Computes the step  heat of vaporisation

    Returns:
        Field[float]: delta t micro
    """

    if r_a_tnd < -1e20 and r_t > RTMIN:
        delta_t_micro = min(delta_t_micro, -(r_b + r_t) / r_a_tnd)
        delta_t_micro = max(delta_t_micro, MNH_TINY)

    return delta_t_micro
