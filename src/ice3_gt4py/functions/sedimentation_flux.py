# -*- coding: utf-8 -*-
from gt4py.cartesian.gtscript import Field, function, log


# FWSED1
# P1 in Bouteloup paper
@function
def weighted_sedimentation_flux_1(
    wsedw: Field["float"],
    dz: Field["float"],
    rhodref: Field["float"],
    rx_t: Field["float"],
    dt: "float",
):
    return min(rhodref * dz * rx_t / dt, wsedw * rhodref * rx_t)


# FWSED2
# P2 in Bouteloup paper
@function
def weighted_sedimentation_flux_2(
    wsedw: Field["float"], wsedsup: Field["float"], dz: Field["float"], dt: "float"
):
    return max(0, 1 - dz / (dt * wsedw)) * wsedsup[0, 0, 1]


@function
def other_species(
    fsed: "float", exsed: "float", content_in: Field["float"], rhodref: Field["float"]
):
    from __externals__ import CEXVT

    return fsed * content_in * (exsed - 1) * rhodref ** (exsed - CEXVT - 1)


@function
def pristine_ice(content: Field["float"], rhodref: Field["float"]):
    from __externals__ import CEXVT, FSEDI, EXCSEDI, I_RTMIN

    return (
        FSEDI
        * rhodref ** (-CEXVT)
        * max(5e-8, -1.5319e5 - 2.1454e5 * log(rhodref * content)) ** EXCSEDI
        if content > max(I_RTMIN, 1e-7)
        else 0
    )
