# -*- coding: utf-8 -*-
from gt4py.cartesian.gtscript import Field, function, log


@function
def ray(sea: Field["int"]):

    from __externals__ import GAC, GC, GAC2, GC2

    return max(1, 0.5 * ((1 - sea) * GAC / GC + sea * GAC2 / GC2))


@function
def lbc(sea: Field["int"]):
    from __externals__ import LBC

    return max(min(LBC[0], LBC[1]), sea * LBC[0] + (1 - sea * LBC[1]))


@function
def fsedc(sea: Field["int"]):
    from __externals__ import FSEDC

    return max(min(FSEDC[0], FSEDC[1]), sea * FSEDC[0] + (1 - sea) * FSEDC[1])


@function
def conc3d(town: Field["float"], sea: Field["int"]):
    from __externals__ import CONC_LAND, CONC_SEA, CONC_URBAN

    return (1 - town) * (sea * CONC_SEA + (1 - sea) * CONC_LAND) + town * CONC_URBAN
