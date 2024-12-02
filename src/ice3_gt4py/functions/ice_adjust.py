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

    from __externals__ import CL, CPV, LVTT, TT

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

    from __externals__ import CI, CPV, LSTT, TT

    return LSTT + (CPV - CI) * (t - TT)


@function
def constant_pressure_heat_capacity(
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

    from __externals__ import CI, CL, CPD, CPV

    return CPD + CPV * rv + CL * (rc + rr) + CI * (ri + rs + rg)
