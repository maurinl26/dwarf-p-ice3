# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    Field,
    exp,
    log,
    computation,
    PARALLEL,
    interval,
    __externals__,
)
from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection("ice4_warm")
def ice4_slow(
    ldcompute: Field["bool"],  # boolean field for microphysics computation
    rhodref: Field["float"],
    lv_fact: Field["float"],
    t: Field["float"],  # temperature
    pres: Field["float"],
    tht: Field["float"],
    lbdar: Field["float"],  # slope parameter for the rain drop distribution
    lbdar_rf: Field["float"],  # slope parameter for the rain fraction part
    ka: Field["float"],  # thermal conductivity of the air
    dv: Field["float"],  # diffusivity of water vapour
    cj: Field["float"],  # function to compute the ventilation coefficient
    hlc_hcf: Field["float"],  # High Cloud Fraction in grid
    hlc_lcf: Field["float"],  # Low Cloud Fraction in grid
    hlc_hrc: Field["float"],  # LWC that is high in grid
    hlc_lrc: Field["float"],  # LWC that is low in grid
    cf: Field["float"],  # cloud fraction
    rf: Field["float"],  # rain fraction
    rv_t: Field["float"],  # water vapour mixing ratio at t
    rc_t: Field["float"],  # cloud water mixing ratio at t
    rr_t: Field["float"],  # rain water mixing ratio at t
    rcautr: Field["float"],  # autoconversion of rc for rr production
    rcaccr: Field["float"],  # accretion of r_c for r_r production
    rrevap: Field["float"],  # evaporation of rr
    ldsoft: "bool",
):
    from __externals__ import (
        ALPW,
        BETAW,
        C_RTMIN,
        CEXVT,
        CL,
        CPD,
        CPV,
        CRIAUTC,
        EPSILO,
        EX0EVAR,
        EX1EVAR,
        EXCACCR,
        FCACCR,
        GAMW,
        LVTT,
        O0EVAR,
        O1EVAR,
        R_RTMIN,
        RV,
        SUBG_RC_RR_ACCR,
        SUBG_RR_EVAP,
        TIMAUTC,
        TT,
    )

    # 4.2 compute the autoconversion of r_c for r_r : RCAUTR
    with computation(PARALLEL), interval(...):
        if hlc_hrc > C_RTMIN and hlc_hcf > 0 and ldcompute:
            rcautr = (
                TIMAUTC * max(hlc_hrc - hlc_hcf * CRIAUTC / rhodref, 0)
                if not ldsoft
                else rcautr
            )
        else:
            rcautr = 0

    # 4.3 compute the accretion of r_c for r_r : RCACCR
    # TODO : code hsubg_rcaccr to enumeration
    with computation(PARALLEL), interval(...):
        # Translation note : HSUBG_RC_RR_ACCR=='NONE'
        if SUBG_RC_RR_ACCR == 0:
            if rc_t > C_RTMIN and rr_t > R_RTMIN and ldcompute:
                rcaccr = (
                    FCACCR * rc_t * lbdar * EXCACCR * rhodref ** (-CEXVT)
                    if not ldsoft
                    else rcaccr
                )
            else:
                rcaccr = 0

        # Translation note : second option from l121 to l155 ommitted
        # elif csubg_rc_rr_accr == 1:

    # 4.4 computes the evaporation of r_r :  RREVAV
    with computation(PARALLEL), interval(...):
        # NONE in Fortran code
        if SUBG_RR_EVAP == 0:
            if rr_t > R_RTMIN and rc_t <= C_RTMIN and ldcompute:
                if not ldsoft:
                    rr_evav = exp(ALPW - BETAW / t - GAMW * log(t))
                    usw = 1 - rv_t * (pres - rr_evav)
                    rr_evav = (LVTT + (CPV - CL) * (t - TT)) ** 2 / (
                        ka * RV * t**2
                    ) + (RV * t) / (dv * rr_evav)
                    rr_evav = (max(0, usw / (rhodref * rr_evav))) * (
                        O0EVAR * lbdar**EX0EVAR + O1EVAR * cj * EX1EVAR
                    )

    # TODO : translate second option from line 178 to 227
    # HSUBG_RR_EVAP=='CLFR' .OR. HSUBG_RR_EVAP=='PRFR'
    with computation(PARALLEL), interval(...):
        if SUBG_RR_EVAP == 1 or SUBG_RR_EVAP == 2:
            # HSUBG_RR_EVAP=='CLFR'
            if SUBG_RR_EVAP == 1:
                zw4 = 1  # precipitation fraction
                zw3 = lbdar

            # HSUBG_RR_EVAP=='PRFR'
            elif SUBG_RR_EVAP == 2:
                zw4 = rf  # precipitation fraction
                zw3 = lbdar_rf

            if rr_t > R_RTMIN and zw4 > cf and ldcompute:
                if not ldsoft:
                    # outside the cloud (environment) the use of T^u (unsaturated) instead of T
                    # ! Bechtold et al. 1993

                    # ! T_l
                    thlt_tmp = tht - LVTT * tht / CPD / t * rc_t

                    # T^u = T_l = theta_l * (T/theta)
                    zw2 = thlt_tmp * t / tht

                    # saturation over water
                    rr_evav = exp(ALPW - BETAW / zw2 - GAMW * log(zw2))

                    # s, undersaturation over water (with new theta^u)
                    usw = 1 - rv_t * (pres - rr_evav) / (EPSILO * rr_evav)

                    rr_evav = (LVTT + (CPV - CL) * (zw2 - TT)) ** 2 / (
                        ka * RV * zw2**2
                    ) + RV * zw2 / (dv * rr_evav)
                    rr_evav = (
                        max(0, usw)
                        / (rhodref * rr_evav)
                        * (O0EVAR * zw3**EX0EVAR + O1EVAR * cj * zw3**EX1EVAR)
                    )
                    rr_evav = rr_evav * (zw4 - cf)

            else:
                rr_evav = 0
