# -*- coding: utf-8 -*-
from __future__ import annotations

import dace
from ice3_gt4py.utils.typingx import dtype_float

@dace.program
def vaporisation_latent_heat(
    t: dtype_float,
    lv: dtype_float,
    ext: dace.compiletime
):
    """Computes latent heat of vaporisation

    Args:
        t (Field[float]): field of temperature

    Returns:
        Field[float]: point wise vaporisation latent heat
    """

    lv = ext.LVTT + (ext.CPV - ext.CL) * (t - ext.TT)

@dace.program
def sublimation_latent_heat(
    t: dtype_float,
    ls: dtype_float,
    ext: dace.compiletime
):
    """Computes latent heat of sublimation

    Args:
        t (Field[float]): field of temperature

    Returns:
        Field[float]: point wise sublimation latent heat
    """

    ls = ext.LSTT + (ext.CPV - ext.CI) * (t - ext.TT)


@dace.program
def constant_pressure_heat_capacity(
    rv: dtype_float,
    rc: dtype_float,
    ri: dtype_float,
    rr: dtype_float,
    rs: dtype_float,
    rg: dtype_float,
    cph: dtype_float,
    ext: dace.compiletime
):
    """Compute specific heat at constant pressure for a
    moist parcel given mixing ratios

    Returns:
        Field[float]: specific heat of parcel
    """
    cph = ext.CPD + ext.CPV * rv + ext.CL * (rc + rr) + ext.CI * (ri + rs + rg)

