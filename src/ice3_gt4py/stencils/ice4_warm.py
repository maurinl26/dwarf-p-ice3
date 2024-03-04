# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function, exp, log

from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection("ice4_warm")
def ice4_slow(
    ldcompute: Field["bool"],  # boolean field for microphysics computation
    rhodref: Field["float"],
    lv_fact: Field["float"],
    t: Field["float"],  # temperature
    pres: Field["float"],
    tht: Field["float"],
    lbda_r: Field["float"],  # slope parameter for the rain drop distribution
    lbda_r_rf: Field["float"],  # slope parameter for the rain fraction part
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
    rc_autr: Field["float"],  # autoconversion of rc for rr production
    rc_accr: Field["float"],  # accretion of r_c for r_r production
    rr_evap: Field["float"],  # evaporation of rr
    ldsoft: int,
):

    from __externals__ import (
        subg_rc_rr_accr,
        subg_rr_evap,
        c_rtmin,
        r_rtmin,
        timautc,
        criautc,
        cexvt,
        fcaccr,
        excaccr,
        alpw,
        betaw,
        gamw,
        Rv,
        Cl,
        lvtt,
        tt,
        cpv,
        o0evar,
        ex0evar,
        o1evar,
        ex1evar,
        cpd,
        epsilo,
    )

    # 4.2 compute the autoconversion of r_c for r_r : RCAUTR
    with computation(PARALLEL), interval(...):

        # Translation note : ldsoft omitted
        # TODO: see if ldsoft is used
        if hlc_hrc > c_rtmin and hlc_hcf > 0 and ldcompute:
            rc_autr = timautc * max(hlc_hrc - hlc_hcf * criautc / rhodref, 0)
        else:
            rc_autr = 0

    # 4.3 compute the accretion of r_c for r_r : RCACCR
    # TODO : code hsubg_rc_accr to enumeration
    with computation(PARALLEL), interval(...):

        # Translation note : HSUBG_RC_RR_ACCR=='NONE'
        if subg_rc_rr_accr == 0:
            if rc_t > c_rtmin and rr_t > r_rtmin and ldcompute:

                # Translation note :
                rc_accr = fcaccr * rc_t * lbda_r * excaccr * rhodref ** (-cexvt)
            else:
                rc_accr = 0

        # TODO : translate second option from l121 to l155
        # elif csubg_rc_rr_accr == 1:

    # 4.4 computes the evaporation of r_r :  RREVAV
    with computation(PARALLEL), interval(...):

        # NONE in Fortran code
        if subg_rr_evap == 0:
            if rr_t > r_rtmin and rc_t <= c_rtmin and ldcompute:

                rr_evav = exp(alpw - betaw / t - gamw * log(t))
                usw = 1 - rv_t * (pres - rr_evav)
                rr_evav = (lvtt + (cpv - Cl) * (t - tt)) ** 2 / (ka * Rv * t**2) + (
                    Rv * t
                ) / (dv * rr_evav)
                rr_evav = (max(0, usw / (rhodref * rr_evav))) * (
                    o0evar * lbda_r**ex0evar + o1evar * cj * ex1evar
                )

    # TODO : translate second option from line 178 to 227
    # HSUBG_RR_EVAP=='CLFR' .OR. HSUBG_RR_EVAP=='PRFR'
    with computation(PARALLEL), interval(...):

        # HSUBG_RR_EVAP=='CLFR'
        if subg_rr_evap == 1:
            zw4 = 1  # precipitation fraction
            zw3 = lbda_r

        # HSUBG_RR_EVAP=='PRFR'
        elif subg_rr_evap == 2:
            zw4 = rf  # precipitation fraction
            zw3 = lbda_r_rf

        if rr_t > r_rtmin and zw4 > cf and ldcompute:
            if ldsoft == 0:

                # outside the cloud (environment) the use of T^u (unsaturated) instead of T
                # ! Bechtold et al. 1993

                # ! T_l
                thlt_tmp = tht - lvtt * tht / cpd / t * rc_t

                # T^u = T_l = theta_l * (T/theta)
                zw2 = thlt_tmp * t / tht

                # saturation over water
                rr_evav = exp(alpw - betaw / zw2 - gamw * log(zw2))

                # s, undersaturation over water (with new theta^u)
                usw = 1 - rv_t * (pres - rr_evav) / (epsilo * rr_evav)

                rr_evav = (lvtt + (cpv - Cl) * (zw2 - tt)) ** 2 / (
                    ka * Rv * zw2**2
                ) + Rv * zw2 / (dv * rr_evav)
                rr_evav = (
                    max(0, usw)
                    / (rhodref * rr_evav)
                    * (o0evar * zw3**ex0evar + o1evar * cj * zw3**ex1evar)
                )
                rr_evav = rr_evav * (zw4 - cf)

        else:
            rr_evav = 0
