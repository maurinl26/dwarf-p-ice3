# -*- coding: utf-8 -*-
from gt4py.cartesian.gtscript import Field, function


# FWSED1
# P1 in Bouteloup paper
@function
def weighted_sedimentation_flux_1(
    wsedw: Field["float"],
    dz: Field["float"],
    rhodref: Field["float"],
    content_in: Field["float"],
    dt: "float",
):
    return min(rhodref * dz * content_in / dt, wsedw * rhodref, content_in)


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
    from __externals__ import cexvt

    wsedw = fsed * content_in * (exsed - 1) * rhodref ** (exsed - cexvt - 1)

    return wsedw
