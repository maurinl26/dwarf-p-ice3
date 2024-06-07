# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, log, computation, interval, PARALLEL
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=693, to_line=728
)
@stencil_collection("rain_ice_total_tendencies")
def rain_ice_total_tendencies(
    wr_th: Field["float"],
    wr_v: Field["float"],
    wr_c: Field["float"],
    wr_r: Field["float"],
    wr_i: Field["float"],
    wr_s: Field["float"],
    wr_g: Field["float"],
    ls_fact: Field["float"],
    lv_fact: Field["float"],
    exnref: Field["float"],
    ths: Field["float"],
    rvs: Field["float"],
    rcs: Field["float"],
    rrs: Field["float"],
    ris: Field["float"],
    rss: Field["float"],
    rgs: Field["float"],
    rvheni: Field["float"],
    rv_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
):
    """Update tendencies

    Args:
        wr_th (Field[float]): potential temperature initial value
        wr_v (Field[float]): vapour initial value
        wr_c (Field[float]): cloud droplets initial value
        wr_r (Field[float]): rain initial value
        wr_i (Field[float]): ice initial value
        wr_s (Field[float]): snow initial value
        wr_g (Field[float]): graupel initial value
        ls_fact (Field[float]): sublimation latent heat over heat capacity
        lv_fact (Field[float]): vapourisation latent heat over heat capacity
        exnref (Field[float]): reference exner pressure
        ths (Field[float]): source (tendency) of potential temperature
        rvs (Field[float]): source (tendency) of vapour
        rcs (Field[float]): source (tendency) of cloud droplets
        rrs (Field[float]): source (tendency) of rain
        ris (Field[float]): source (tendency) of ice
        rss (Field[float]): source (tendency) of snow
        rgs (Field[float]): source (tendency) of graupel
        rvheni (Field[float]): _description_
        rv_t (Field[float]): vapour m.r. at t
        rc_t (Field[float]): droplets m.r. at t
        rr_t (Field[float]): rain m.r. at t
        ri_t (Field[float]): ice m.r. at t
        rs_t (Field[float]): snow m.r. at t
        rg_t (Field[float]): graupel m.r. at t
    """

    from __externals__ import INV_TSTEP

    with computation(PARALLEL), interval(...):

        # Translation note ls, lv replaced by ls_fact, lv_fact

        # Hydrometeor tendency
        wr_v = (wr_v - rv_t) * INV_TSTEP
        wr_c = (wr_c - rc_t) * INV_TSTEP
        wr_r = (wr_r - rr_t) * INV_TSTEP
        wr_i = (wr_i - ri_t) * INV_TSTEP
        wr_s = (wr_s - rs_t) * INV_TSTEP
        wr_g = (wr_g - rg_t) * INV_TSTEP

        # Theta tendency
        wr_th = (wr_c + wr_r) * lv_fact + (wr_i + wr_s + wr_g) * ls_fact

        # Tendencies to sources, taking nucleation into account (rv_heni)
        ths += wr_th + rvheni * ls_fact
        rvs += wr_v - rvheni
        rcs += wr_c
        rrs += wr_r
        ris += wr_i + rvheni
        rss += wr_s
        rgs += wr_g
