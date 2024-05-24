# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    IJ,
    Field,
    function,
    computation,
    PARALLEL,
    interval,
)
from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection("upwind_sedimentation")
def upwind_sedimentation(
    dt: "float",
    rhodref: Field["float"],
    oorhodz: Field["float"],  # 1 / (rho * dz)
    dzz: Field["float"],
    pabst: Field["float"],
    tht: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    ray_t: Field["float"],  # Optional PRAY
    lbc_t: Field["float"],  # Optional LBC
    fsedc: Field["float"],  # Optional PFSEDC
    conc3d_t: Field["float"],  # Optional PCONC3D
    rcs_tnd: Field["float"],
    rrs_tnd: Field["float"],
    ris_tnd: Field["float"],
    rss_tnd: Field["float"],
    rgs_tnd: Field["float"],
    dt__rho_dz: Field["float"],
    inst_rr: Field[IJ, "float"],
    inst_rc: Field[IJ, "float"],
    inst_ri: Field[IJ, "float"],
    inst_rs: Field[IJ, "float"],
    inst_rg: Field[IJ, "float"],
    wsed_tmp: Field["float"],  # sedimentation fluxes
    t_tmp: Field["float"],  # temperature - temporary field
    remaining_time: Field[IJ, "float"],  # remaining time until time step end
    max_tstep: Field[IJ, "float"],
    fpr_c: Field["float"],  # upper-air precip fluxes for cloud droplets
    fpr_r: Field["float"],
    fpr_s: Field["float"],
    fpr_i: Field["float"],
    fpr_g: Field["float"],
):
    from __externals__ import C_RTMIN  # MIN CONTENT FOR CLOUD DROPLETS
    from __externals__ import T00  # 293.15
    from __externals__ import XCC  # FROM ICED
    from __externals__ import (
        CEXVT,
        CPD,
        G_RTMIN,
        I_RTMIN,
        LBEXC,
        P00,
        R_RTMIN,
        RD,
        S_RTMIN,
    )

    # TODO
    # remaining time to be initialized
    # 2. Compute the fluxes
    # l219 to l262
    rcs_tnd -= rc_t / dt
    ris_tnd -= ri_t / dt
    rrs_tnd -= rr_t / dt
    rss_tnd -= rs_t / dt
    rgs_tnd -= rg_t / dt

    # in internal_sedim_split

    ## 2.1 For cloud droplets
    # TODO : encapsulation in do while
    # TODO: extend by functions to other species
    # TODO add #else l590 to l629 for  #ifdef REPRO48
    with computation(PARALLEL), interval(...):
        wlbdc = (lbc_t * conc3d_t / (rhodref * rc_t)) ** lbexc
        ray_tmp = ray_t / wlbdc

        # TODO : replace with exner
        t_tmp = tht * (pabst / P00) ** (RD / CPD)
        wlbda = 6.6e-8 * (P00 / pabst) * (t_tmp / T00)
        cc_tmp = XCC * (1 + 1.26 * wlbda / ray_tmp)
        wsed_tmp = rhodref ** (-CEXVT + 1) * wlbdc * cc_tmp * fsedc

    # Translation note : l723 in mode_ice4_sedimentation_split.F90
    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            C_RTMIN, rhodref, max_tstep, rc_t, dzz, wsed_tmp, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_rc[0, 0] += instant_precipitation(wsed_tmp, max_tstep, dt)

    # Translation note : l738 in mode_ice4_sedimentation_split.F90
    with computation(PARALLEL), interval(...):
        rcs_tnd = mixing_ratio_update(max_tstep, oorhodz, wsed_tmp, rcs_tnd, rc_t, dt)
        fpr_c += upper_air_flux(wsed_tmp, max_tstep, dt)

    ## 2.2 for ice
    with computation(PARALLEL), interval(...):
        wsed_tmp = 0

    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            I_RTMIN, rhodref, max_tstep, ri_t, dzz, wsed_tmp, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_rc[0, 0] += instant_precipitation(wsed_tmp, max_tstep, dt)

    with computation(PARALLEL), interval(...):
        rcs_tnd = mixing_ratio_update(max_tstep, oorhodz, wsed_tmp, ris_tnd, ri_t, dt)
        fpr_c += upper_air_flux(wsed_tmp, max_tstep, dt)

    ## 2.3 for rain
    with computation(PARALLEL), interval(...):
        wsed_tmp = 0

    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            R_RTMIN, rhodref, max_tstep, rr_t, dzz, wsed_tmp, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_rr[0, 0] += instant_precipitation(wsed_tmp, max_tstep, dt)

    with computation(PARALLEL), interval(...):
        rcs_tnd = mixing_ratio_update(max_tstep, oorhodz, wsed_tmp, ris_tnd, ri_t, dt)
        fpr_c += upper_air_flux(wsed_tmp, max_tstep, dt)

    ## 2.4. for snow
    with computation(PARALLEL), interval(...):
        wsed_tmp = 0

    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            S_RTMIN, rhodref, max_tstep, rs_t, dz, wsed_tmp, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_rs[0, 0] += instant_precipitation(wsed_tmp, max_tstep, dt)

    with computation(PARALLEL), interval(...):
        rcs_tnd = mixing_ratio_update(max_tstep, oorhodz, wsed_tmp, rss_tnd, rs_t, dt)
        fpr_c += upper_air_flux(wsed_tmp, max_tstep, dt)

    # 2.5. for graupel
    with computation(PARALLEL), interval(...):
        wsed_tmp = 0

    with computation(PARALLEL), interval(0, 1):
        max_tstep = maximum_time_step(
            G_RTMIN, rhodref, max_tstep, rg_t, dz, wsed_tmp, remaining_time
        )
        remaining_time[0, 0] -= max_tstep[0, 0]
        inst_rg[0, 0] += instant_precipitation(wsed_tmp, max_tstep, dt)

    with computation(PARALLEL), interval(...):
        rcs_tnd = mixing_ratio_update(max_tstep, oorhodz, wsed_tmp, rgs_tnd, rg_t, dt)
        fpr_c += upper_air_flux(wsed_tmp, max_tstep, dt)


@function
def upper_air_flux(
    wsed_tmp: Field["float"],
    max_tstep: Field[IJ, "float"],
    dt: "float",
):
    return wsed_tmp * (max_tstep / dt)


@function
def mixing_ratio_update(
    max_tstep: Field[IJ, "float"],
    oorhodz: Field["float"],
    wsed: Field["float"],
    rs_tnd: Field["float"],
    r_t: Field["float"],
    dt: "float",
) -> Field["float"]:
    """Update mixing ratio

    Args:
        max_tstep (Field[IJ, float]): maximum time step to use
        oorhodz (Field[float]): 1 / (rho * dz)
        wsed (Field[float]): sedimentation flux
        rs_tnd (Field[float]): tendency for mixing ratio
        r_t (Field[float]): mixing ratio at time t
        dt (float): time step

    Returns:
        Field[float]: mixing ratio up to date
    """

    mrchange_tmp = max_tstep[0, 0] * oorhodz * (wsed[0, 0, 1] - wsed[0, 0, 0])
    r_t += mrchange_tmp + rs_tnd * max_tstep
    rs_tnd += mrchange_tmp / dt

    return rs_tnd


@function
def maximum_time_step(
    rtmin: "float",
    rhodref: Field["float"],
    max_tstep: Field[IJ, "float"],
    r: Field["float"],
    dz: Field["float"],
    wsed_tmp: Field["float"],
    remaining_time: Field["float"],
):
    from __externals__ import SPLIT_MAXCFL

    tstep = max_tstep
    if r > rtmin and wsed_tmp > 1e-20 and remaining_time > 0:
        tstep[0, 0] = min(
            max_tstep,
            SPLIT_MAXCFL
            * rhodref[0, 0, 0]
            * r[0, 0, 0]
            * dz[0, 0, 0]
            / wsed_tmp[0, 0, 0],
        )

    return tstep


@function
def instant_precipitation(
    wsed_tmp: Field["float"], max_tstep: Field["float"], dt: "float"
) -> Field["float"]:
    from __externals__ import RHOLW

    return wsed_tmp[0, 0, 0] / RHOLW * (max_tstep / dt)
