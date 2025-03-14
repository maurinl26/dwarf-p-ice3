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
def ice4_warm(
    ldcompute: Field["bool"],  # boolean field for microphysics computation
    rhodref: Field["float"],
    t: Field["float"],  # temperature
    pres: Field["float"],
    tht: Field["float"],
    lbdar: Field["float"],  # slope parameter for the rain drop distribution
    lbdar_rf: Field["float"],  # slope parameter for the rain fraction part
    ka: Field["float"],  # thermal conductivity of the air
    dv: Field["float"],  # diffusivity of water vapour
    cj: Field["float"],  # function to compute the ventilation coefficient
    hlc_hcf: Field["float"],  # High Cloud Fraction in grid
    hlc_hrc: Field["float"],  # LWC that is high in grid
    cf: Field["float"],  # cloud fraction
    rf: Field["float"],  # rain fraction
    rvt: Field["float"],  # water vapour mixing ratio at t
    rct: Field["float"],  # cloud water mixing ratio at t
    rrt: Field["float"],  # rain water mixing ratio at t
    rcautr: Field["float"],  # autoconversion of rc for rr production
    rcaccr: Field["float"],  # accretion of r_c for r_r production
    rrevav: Field["float"],  # evaporation of rr
    ldsoft: "bool",
):
    """Computes slow processes.

    Args:
        ldcompute (Field[bool]): _description_
        lvfact (Field[float]): _description_
        t (Field[float]): _description_
        tht (Field[float]): _description_
        lbdar (Field[float]): _description_
    """
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
        SUBG_RR_EVAP,
        TIMAUTC,
        TT,
    )

    # 4.2 compute the autoconversion of r_c for r_r : RCAUTR
    with computation(PARALLEL), interval(...):
        if hlc_hrc > C_RTMIN and hlc_hcf > 0.0 and ldcompute:
            if not ldsoft:
                rcautr = (
                TIMAUTC * max(0.0, hlc_hrc - hlc_hcf * CRIAUTC / rhodref)
            )
        else:
            rcautr = 0

    # 4.3 compute the accretion of r_c for r_r : RCACCR
    with computation(PARALLEL), interval(...):
        # Translation note : HSUBG_RC_RR_ACCR=='NONE'
            if rct > C_RTMIN and rrt > R_RTMIN and ldcompute:
                if not ldsoft:
                    rcaccr = (
                    FCACCR * rct 
                    * lbdar ** EXCACCR 
                    * rhodref ** (-CEXVT)
                    )
            else:
                rcaccr = 0

        # Translation note : second option from l121 to l155 ommitted
        # elif csubg_rc_rr_accr == 1:

    # 4.4 computes the evaporation of r_r :  RREVAV
    with computation(PARALLEL), interval(...):
        # NONE in Fortran code
        if SUBG_RR_EVAP == 0:
            if rrt > R_RTMIN and rct <= C_RTMIN and ldcompute:
                if not ldsoft:
                    rrevav = exp(ALPW - BETAW / t - GAMW * log(t))
                    usw = 1 - rvt * (pres - rrevav)
                    rrevav = (LVTT + (CPV - CL) * (t - TT)) ** 2 / (
                        ka * RV * t**2
                    ) + (RV * t) / (dv * rrevav)
                    rrevav = (max(0, usw / (rhodref * rrevav))) * (
                        O0EVAR * lbdar**EX0EVAR + O1EVAR * cj * EX1EVAR
                    )

        if SUBG_RR_EVAP == 1 or SUBG_RR_EVAP == 2:
            # HSUBG_RR_EVAP=='CLFR'
            if SUBG_RR_EVAP == 1:
                zw4 = 1  # precipitation fraction
                zw3 = lbdar

            # HSUBG_RR_EVAP=='PRFR'
            elif SUBG_RR_EVAP == 2:
                zw4 = rf  # precipitation fraction
                zw3 = lbdar_rf

            if rrt > R_RTMIN and zw4 > cf and ldcompute:
                if not ldsoft:
                    # outside the cloud (environment) the use of T^u (unsaturated) instead of T
                    # ! Bechtold et al. 1993

                    # ! T_l
                    thlt_tmp = tht - LVTT * tht / CPD / t * rct

                    # T^u = T_l = theta_l * (T/theta)
                    zw2 = thlt_tmp * t / tht

                    # saturation over water
                    rrevav = exp(ALPW - BETAW / zw2 - GAMW * log(zw2))

                    # s, undersaturation over water (with new theta^u)
                    usw = 1 - rvt * (pres - rrevav) / (EPSILO * rrevav)

                    rrevav = (LVTT + (CPV - CL) * (zw2 - TT)) ** 2 / (
                        ka * RV * zw2**2
                    ) + RV * zw2 / (dv * rrevav)
                    rrevav = (
                        max(0, usw)
                        / (rhodref * rrevav)
                        * (O0EVAR * zw3**EX0EVAR + O1EVAR * cj * zw3**EX1EVAR)
                    )
                    rrevav = rrevav * (zw4 - cf)

            else:
                rrevav = 0
