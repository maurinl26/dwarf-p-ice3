# -*- coding: utf-8 -*-
from gt4py.cartesian.gtscript import Field, function, min, max


@function
def ray(sea: Field["float"]) -> "float":
    """_summary_

    Args:
        sea (Field[float]): _description_

    Returns:
        float: _description_
    """

    from __externals__ import GAC, GC, GAC2, GC2

    return max(1, 0.5 * ((1 - sea) * GAC / GC + sea * GAC2 / GC2))


@function
def lbc(sea: Field["float"]) -> "float":
    """_summary_

    Args:
        sea (Field[float]): _description_

    Returns:
        float: _description_
    """
    from __externals__ import LBC_1, LBC_2

    return max(min(LBC_1, LBC_2), sea * LBC_1 + (1 - sea * LBC_2))


@function
def fsedc(sea: Field["float"]) -> "float":
    """_summary_

    Args:
        sea (Field[float]): _description_

    Returns:
        float: _description_
    """
    from __externals__ import FSEDC_1, FSEDC_2

    return max(min(FSEDC_1, FSEDC_2), sea * FSEDC_1 + (1 - sea) * FSEDC_2)


@function
def conc3d(town: Field["float"], sea: Field["float"]) -> "float":
    """_summary_

    Args:
        town (Field[float]): _description_
        sea (Field[float]): _description_

    Returns:
        float: _description_
    """
    from __externals__ import CONC_LAND, CONC_SEA, CONC_URBAN

    return (1 - town) * (sea * CONC_SEA + (1 - sea) * CONC_LAND) + town * CONC_URBAN
