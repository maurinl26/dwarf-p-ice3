# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, log, computation, interval, BACKWARD, PARALLEL
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=792, to_line=801
)
@stencil_collection("rain_fraction_sedimentation")
def rain_fraction_sedimentation(
    wr_r: Field["float"],
    wr_s: Field["float"],
    wr_g: Field["float"],
    rrs: Field["float"],
    rss: Field["float"],
    rgs: Field["float"],
):
    """Computes vertical rain fraction

    Args:
        wr_r (Field[float]): initial value for rain m.r.
        wr_s (Field[float]): initial value for snow m.r.
        wr_g (Field[float]): initial value for graupel m.r.
        rrs (Field[float]): tendency (source) for rain
        rss (Field[float]): tendency (source) for snow
        rgs (Field[float]): tendency (source) for graupel
    """

    from __externals__ import TSTEP

    with computation(PARALLEL), interval(0, 1):
        wr_r = rrs * TSTEP
        wr_s = rss * TSTEP
        wr_g = rgs * TSTEP


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=792, to_line=801
)
@stencil_collection("ice4_rainfr_vert")
def ice4_rainfr_vert(
    prfr: Field["float"], rr: Field["float"], rs: Field["float"], rg: Field["float"]
):
    from __externals__ import S_RTMIN, R_RTMIN, G_RTMIN

    with computation(BACKWARD), interval(0, -1):
        if rr > R_RTMIN or rs > S_RTMIN or rg > G_RTMIN:

            prfr[0, 0, 0] = max(prfr[0, 0, 0], prfr[0, 0, 1])
            if prfr == 0:
                prfr = 1
        else:
            prfr = 0
