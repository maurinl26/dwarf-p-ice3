# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function


@function
def vaporisation_latent_heat(
    t: Field,
):

    from __externals__ import lvtt, cpv, Cl, tt

    return lvtt + (cpv - Cl) * (t - tt)


@function
def sublimation_latent_heat(
    t: Field,
):

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
):

    from __externals__ import cpd, cpv, Cl, Ci

    return cpd + cpv * rv + Cl * (rc + rr) + Ci * (ri + rs + rg)
