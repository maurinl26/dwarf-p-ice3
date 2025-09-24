# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    IJ,
    Field,
    function,
)


@function
def upper_air_flux(
    wsed: Field["float"],
    max_tstep: Field[IJ, "float"],
    TSTEP: "float",
):
    """_summary_

    Args:
        wsed (Field[float]): _description_
        max_tstep (Field[IJ, float]): _description_
        TSTEP (float): _description_

    Returns:
        _type_: _description_
    """
    return wsed * (max_tstep / TSTEP)


@function
def mixing_ratio_update(
    max_tstep: Field[IJ, "float"],
    oorhodz: Field["float"],
    wsed: Field["float"],
    rs: Field["float"],
    r_t: Field["float"],
    TSTEP: "float",
) -> Field["float"]:
    """Update mixing ratio

    Args:
        max_tstep (Field[IJ, float]): maximum time step to use
        oorhodz (Field[float]): 1 / (rho * dz)
        wsed (Field[float]): sedimentation flux
        rs (Field[float]): tendency for mixing ratio
        r_t (Field[float]): mixing ratio at time t
        TSTEP (float): time step

    Returns:
        Field[float]: mixing ratio up to date
    """

    mrchange = max_tstep[0, 0] * oorhodz * (wsed[0, 0, 1] - wsed[0, 0, 0])
    r_t += mrchange + rs * max_tstep
    rs += mrchange / TSTEP

    return rs


@function
def maximum_time_step(
    rtmin: "float",
    rhodref: Field["float"],
    max_tstep: Field[IJ, "float"],
    r: Field["float"],
    dz: Field["float"],
    wsed: Field["float"],
    remaining_time: Field[IJ, "float"],
) -> Field["float"]:
    """_summary_

    Args:
        rtmin (float): _description_
        rhodref (Field[float]): _description_
        max_tstep (Field[IJ, float]): _description_
        r (Field[float]): _description_
        dz (Field[float]): _description_
        wsed (Field[float]): _description_
        remaining_time (Field[IJ, float]): _description_

    Returns:
        _type_: _description_
    """
    from __externals__ import SPLIT_MAXCFL

    tstep = max_tstep
    if r > rtmin and wsed > 1e-20 and remaining_time > 0:
        tstep[0, 0] = min(
            max_tstep,
            SPLIT_MAXCFL * rhodref[0, 0, 0] * r[0, 0, 0] * dz[0, 0, 0] / wsed[0, 0, 0],
        )

    return tstep


@function
def instant_precipitation(
    wsed: Field["float"], max_tstep: Field["float"], TSTEP: "float"
) -> Field["float"]:
    """_summary_

    Returns:
        _type_: _description_
    """
    from __externals__ import RHOLW

    return wsed[0, 0, 0] / RHOLW * (max_tstep / TSTEP)
