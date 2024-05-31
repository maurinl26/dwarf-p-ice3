# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    Field,
    min,
    max,
    computation,
    interval,
    PARALLEL,
)
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_correct_negativities.F90")
@stencil_collection("ice4_correct_negativities")
def ice4_correct_negativities(
    th_t: Field["float"],
    rv_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    lv_fact: Field["float"],
    ls_fact: Field["float"],
):
    """Negativity corrector with conservation

    Args:
        rv_t (Field[float]): vapour m.r.
        rc_t (Field[float]): cloud droplets m.r.
        rr_t (Field[float]): rain m.r.
        ri_t (Field[float]): ice m.r.
        rs_t (Field[float]): snow m.r.
        rg_t (Field[float]): graupel m.r.
        lv_fact (Field[float]): latent heat of vaporisation (over cph)
        ls_fact (Field[float]): latent heat of sublimation (over cph)
    """

    from __externals__ import (
        S_RTMIN,
        G_RTMIN,
    )

    with computation(PARALLEL), interval(...):

        # 1. negative
        w = rc_t - max(rc_t, 0)
        rv_t += w
        th_t -= w * lv_fact
        rc_t -= w

        w = rr_t - max(rr_t, 0)
        rv_t += w
        th_t -= w * lv_fact
        rr_t -= w

        w = ri_t - max(ri_t, 0)
        rv_t += w
        th_t -= w * ls_fact
        ri_t -= w

        w = rs_t - max(rs_t, 0)
        rv_t += w
        th_t -= w * ls_fact
        rs_t -= w

        w = rg_t - max(rg_t, 0)
        rv_t += w
        th_t -= w * ls_fact
        rg_t -= w

        # 2. negative vapor mr
        w = min(max(rs_t, 0), max(S_RTMIN - rv_t, 0))  # rs -> rv
        rv_t += w
        rs_t -= w
        th_t -= w * ls_fact

        w = min(max(rg_t, 0), max(G_RTMIN - rv_t, 0))  # rg -> rv
        rv_t += w
        rg_t -= w
        th_t -= w * ls_fact

        # Translation note : l162 to l167 removed because hail is not taken into account
