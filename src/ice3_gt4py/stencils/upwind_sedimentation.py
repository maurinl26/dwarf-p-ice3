# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    IJ,
    Field,
    computation,
    PARALLEL,
    interval,
)
from ifs_physics_common.framework.stencil import stencil_collection

from ice3_gt4py.functions.sea_town_masks import conc3d, fsedc, lbc, ray
from ice3_gt4py.functions.upwind_sedimentation import (
    instant_precipitation,
    maximum_time_step,
    mixing_ratio_update,
    upper_air_flux,
)


@stencil_collection("upwind_sedimentation")
def upwind_sedimentation(
    rhodref: Field["float"],
    dzz: Field["float"],
    pabst: Field["float"],
    tht: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    rcs: Field["float"],
    rrs: Field["float"],
    ris: Field["float"],
    rss: Field["float"],
    rgs: Field["float"],
    inst_rr: Field[IJ, "float"],
    inst_rc: Field[IJ, "float"],
    inst_ri: Field[IJ, "float"],
    inst_rs: Field[IJ, "float"],
    inst_rg: Field[IJ, "float"],
    fpr_c: Field["float"],
    fpr_r: Field["float"],
    fpr_s: Field["float"],
    fpr_i: Field["float"],
    fpr_g: Field["float"],
    sea: Field["int"],
    town: Field["float"],
):

    from __externals__ import (
        C_RTMIN,
        T00,
        CC,
        CEXVT,
        CPD,
        G_RTMIN,
        I_RTMIN,
        LBEXC,
        P00,
        R_RTMIN,
        RD,
        S_RTMIN,
        TSTEP,
    )

    with computation(PARALLEL), interval(...):
        dt__rho_dz = TSTEP / (rhodref * dzz)
        oorhodz = 1 / (rhodref * dzz)

    # TODO
    # remaining time to be initialized
    # 2. Compute the fluxes
    # l219 to l262
    with computation(PARALLEL), interval(...):
        rcs -= rc_t / TSTEP
        ris -= ri_t / TSTEP
        rrs -= rr_t / TSTEP
        rss -= rs_t / TSTEP
        rgs -= rg_t / TSTEP

        wsed_c = 0
        wsed_r = 0
        wsed_i = 0
        wsed_s = 0
        wsed_g = 0

        remaining_time = TSTEP

    # in internal_sedim_split
    with computation(PARALLEL), interval(...):
        _ray = ray(sea)
        _lbc = lbc(sea)
        _fsedc = fsedc(sea)
        _conc3d = conc3d(town, sea)

    ## 2.1 For cloud droplets

    # TODO : share function with statistical sedimentation
    with computation(PARALLEL), interval(...):
        wlbdc = (_lbc * _conc3d / (rhodref * rc_t)) ** LBEXC
        _ray /= wlbdc
        t = tht * (pabst / P00) ** (RD / CPD)
        wlbda = 6.6e-8 * (P00 / pabst) * (t / T00)
        cc = CC * (1 + 1.26 * wlbda / _ray)
        wsed = rhodref ** (-CEXVT + 1) * wlbdc * cc * _fsedc

    # Translation note : l723 in mode_ice4_sedimentation_split.F90
    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            C_RTMIN, rhodref, max_tstep, rc_t, dzz, wsed_c, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_rc[0, 0] += instant_precipitation(wsed_c, max_tstep, TSTEP)

    # Translation note : l738 in mode_ice4_sedimentation_split.F90
    with computation(PARALLEL), interval(...):
        rcs = mixing_ratio_update(max_tstep, oorhodz, wsed_s, rcs, rc_t, TSTEP)
        fpr_c += upper_air_flux(wsed_s, max_tstep, TSTEP)

    ## 2.2 for ice
    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            I_RTMIN, rhodref, max_tstep, ri_t, dzz, wsed_i, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_ri[0, 0] += instant_precipitation(wsed_i, max_tstep, TSTEP)

    with computation(PARALLEL), interval(...):
        rcs = mixing_ratio_update(max_tstep, oorhodz, wsed_i, ris, ri_t, TSTEP)
        fpr_i += upper_air_flux(wsed_i, max_tstep, TSTEP)

    ## 2.3 for rain
    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            R_RTMIN, rhodref, max_tstep, rr_t, dzz, wsed_r, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_rr[0, 0] += instant_precipitation(wsed, max_tstep, TSTEP)

    with computation(PARALLEL), interval(...):
        rrs = mixing_ratio_update(max_tstep, oorhodz, wsed, rrs, rr_t, TSTEP)
        fpr_r += upper_air_flux(wsed, max_tstep, TSTEP)

    ## 2.4. for snow
    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            S_RTMIN, rhodref, max_tstep, rs_t, dzz, wsed, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_rs[0, 0] += instant_precipitation(wsed_s, max_tstep, TSTEP)

    with computation(PARALLEL), interval(...):
        rcs = mixing_ratio_update(max_tstep, oorhodz, wsed_s, rss, rs_t, TSTEP)
        fpr_s += upper_air_flux(wsed_s, max_tstep, TSTEP)

    # 2.5. for graupel
    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            G_RTMIN, rhodref, max_tstep, rg_t, dzz, wsed_g, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_rg[0, 0] += instant_precipitation(wsed_g, max_tstep, TSTEP)

    with computation(PARALLEL), interval(...):
        rcs = mixing_ratio_update(max_tstep, oorhodz, wsed_g, rgs, rg_t, TSTEP)
        fpr_g += upper_air_flux(wsed_g, max_tstep, TSTEP)
