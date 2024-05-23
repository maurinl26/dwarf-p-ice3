# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, log, computation, interval, PARALLEL
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90.func.h", from_line=816, to_line=830
)
@stencil_collection("fog_deposition")
def fog_deposition(
    rcs: Field["float"],
    rc_t: Field["float"],
    rhodref: Field["float"],
    dzz: Field["float"],
    inprc: Field[IJ, "float"],
):
    """Compute fog deposition on vegetation.
    Not activated in AROME.

    Args:
        rcs (Field[float]): source of cloud droplets
        rc_t (Field[float]): cloud droplets m.r.
        rhodref (Field[float]): dry density of air
        dzz (Field[float]): vertical spacing of cells
        inprc (Field[IJ, float]): deposition on vegetation
    """

    from __externals__ import VDEPOSC, RHOLW

    # Note : activated if LDEPOSC is True in rain_ice.F90
    with computation(PARALLEL), interval(0, 1):
        rcs -= VDEPOSC * rc_t / dzz
        inprc += VDEPOSC * rc_t * rhodref / RHOLW
