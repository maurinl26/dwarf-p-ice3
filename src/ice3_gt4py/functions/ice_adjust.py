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

    from __externals__ import LVTT, CPV, CL, TT

    return LVTT + (CPV - CL) * (t - TT)


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

    from __externals__ import LSTT, CPV, CI, TT

    return LSTT + (CPV - CI) * (t - TT)


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

    from __externals__ import CPD, CPV, CL, CI

    return CPD + CPV * rv + CL * (rc + rr) + CI * (ri + rs + rg)
