# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function


@function
def update_potential_temperature(
    theta: Field["float"],
    transfo_mixing_ratio: Field["float"],
    ls_fact: Field["float"],
    lv_fact: Field["float"],
):
    """Update theta along a phase transformation given a mixing ratio
    of transformation.

    The transformation is defined from liquid to ice

    Args:
        theta (Field[float]): potential temperature to update
        transfo_mixing_ration (Field[float]): mixing ratio in transformation
        ls_fact (Field[float]): latent heat of sublimation over heat capacity
        lv_fact (Field[float]): latent heat of vaporisation over heat capacity

    Returns:
        Field[float]: updated theta
    """
    return theta + transfo_mixing_ratio * (ls_fact - lv_fact)


@function
def theta2temperature(theta: Field["float"], exn: Field["float"]) -> Field["float"]:
    """Convert potential temperature (theta) to temperature

    Args:
        t (Field[float]): temperature
        theta (Field[float]): potential temperature
        exner pressure (Field[float]): temperature

    Returns:
        Field[float]: temperature
    """

    return theta * exn


@function
def update_temperature(
    t: Field["float"],
    rc_in: Field["float"],
    rc_out: Field["float"],
    ri_in: Field["float"],
    ri_out: Field["float"],
    lv: Field["float"],
    ls: Field["float"],
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

    from __externals__ import CPD

    t = (
        t[0, 0, 0]
        + (
            (rc_out[0, 0, 0] - rc_in[0, 0, 0]) * lv[0, 0, 0]
            + (ri_out[0, 0, 0] - ri_in[0, 0, 0]) * ls[0, 0, 0]
        )
        / CPD
    )

    return t
