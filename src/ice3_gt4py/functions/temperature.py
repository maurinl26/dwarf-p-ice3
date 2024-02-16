# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function


@function
def update_temperature(
    t: Field["float"],
    rc_in: Field["float"],
    rc_out: Field["float"],
    ri_in: Field["float"],
    ri_out: Field["float"],
    lv: Field["float"],
    ls: Field["float"],
    cpd: float,
) -> Field["float"]:
    """Compute temperature given a change of mixing ratio in ice and liquid

    Args:
        t (Field[float]): temperature to update
        rc_in (Field[float]): previous cloud droplets m.r.
        rc_out (Field[float]): updated cloud droplets m.r.
        ri_in (Field[float]): previous ice m.r.
        ri_out (Field[float]): updated ice m.r.
        lv (Field[float]): latent heat of vaporisation
        ls (Field[float]): latent heat of sublimation
        cpd (float): specific heat at constant pressure for dry air

    Returns:
        Field[float]: updated temperature
    """
    t = (
        t[0, 0, 0]
        + (
            (rc_out[0, 0, 0] - rc_in[0, 0, 0]) * lv[0, 0, 0]
            + (ri_out[0, 0, 0] - ri_in[0, 0, 0]) * ls[0, 0, 0]
        )
        / cpd
    )

    return t
