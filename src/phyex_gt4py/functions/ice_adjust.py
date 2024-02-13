# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function


@function
def vaporisation_latent_heat(
    t: Field["float"],
) -> Field["float"]:
    """Computes latent heat of vaporisation

    Args:
        t (Field[float]): field of temperature

    Returns:
        Field[float]: point wise vaporisation latent heat
    """

    from __externals__ import lvtt, cpv, Cl, tt

    return lvtt + (cpv - Cl) * (t - tt)


@function
def sublimation_latent_heat(
    t: Field["float"],
) -> Field["float"]:
    """Computes latent heat of sublimation

    Args:
        t (Field[float]): field of temperature

    Returns:
        Field[float]: point wise sublimation latent heat
    """

    from __externals__ import lstt, cpv, Ci, tt

    return lstt + (cpv - Ci) * (t - tt)


@function
def cph(
    rv: Field["float"],
    rc: Field["float"],
    ri: Field["float"],
    rr: Field["float"],
    rs: Field["float"],
    rg: Field["float"],
) -> Field["float"]:
    """Compute specific heat at constant pressure for a 
    moist parcel given mixing ratios

    Returns:
        Field[float]: specific heat of parcel
    """

    from __externals__ import cpd, cpv, Cl, Ci

    return cpd + cpv * rv + Cl * (rc + rr) + Ci * (ri + rs + rg)
