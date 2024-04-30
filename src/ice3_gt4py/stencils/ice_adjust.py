# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    PARALLEL,
    Field,
    GlobalTable,
    atan,
    computation,
    exp,
    floor,
    interval,
    log,
    sqrt,
    IJ,
)
from ifs_physics_common.framework.stencil import stencil_collection
import numpy as np

from ice3_gt4py.functions.compute_ice_frac import compute_frac_ice
from ice3_gt4py.functions.ice_adjust import (
    sublimation_latent_heat,
    vaporisation_latent_heat,
)
from ice3_gt4py.functions.temperature import update_temperature
from ice3_gt4py.functions.tiwmx import e_sat_i, e_sat_w

# TODO: remove POUT (not used in aro_adjust)
# TODO: add SSIO, SSIU, IFR, SRCS
@stencil_collection("ice_adjust")
def ice_adjust(
    sigqsat: Field["float"],
    exnref: Field["float"],
    exn: Field["float"],
    rhodref: Field["float"],
    pabs: Field["float"],
    sigs: Field["float"],
    cf_mf: Field["float"],
    rc_mf: Field["float"],
    ri_mf: Field["float"],
    th: Field["float"],
    rv: Field["float"],
    rc: Field["float"],
    ri: Field["float"],
    rr: Field["float"],
    rs: Field["float"],
    rg: Field["float"],
    ths: Field["float"],
    rvs: Field["float"],
    rcs: Field["float"],
    ris: Field["float"],
    cldfr: Field["float"],
    ifr: Field["float"],
    hlc_hrc: Field["float"],
    hlc_hcf: Field["float"],
    hli_hri: Field["float"],
    hli_hcf: Field["float"],
    sigrc: Field["float"],
    rv_tmp: Field["float"],
    ri_tmp: Field["float"],
    rc_tmp: Field["float"],
    t_tmp: Field["float"],
    cph: Field["float"],
    lv: Field["float"],
    ls: Field["float"],
    criaut: Field["float"],
    rt: Field["float"],
    pv: Field["float"],
    piv: Field["float"],
    qsl: Field["float"],
    qsi: Field["float"],
    frac_tmp: Field["float"],
    cond_tmp: Field["float"],
    a: Field["float"],
    sbar: Field["float"],
    sigma: Field["float"],
    q1: Field["float"],
    inq1: Field[np.int64],
    src_1d: GlobalTable[("float", (34))],
    dt: "float",
):
    """Microphysical adjustments for specific contents due to condensation.

    Args:
        sigqsat (Field[float]): _description_
        exnref (Field[float]): reference exner pressure
        exn (Field[float]): true exner pressure
        rhodref (Field[float]): reference density
        pabs (Field[float]): absolute pressure at time t
        sigs (Field[float]): _description_
        cf_mf (Field[float]): convective mass flux cloud fraction             (from shallow convection)
        rc_mf (Field[float]): convective mass flux liquid mixing ratio  (from shallow convection)
        ri_mf (Field[float]): convective mass flux ice mixing ratio     (from shallow convection)
        th (Field[float]): potential temperature
        rv (Field[float]): water vapour m.r. to adjust
        rc (Field[float]): cloud water m.r. to adjust
        ri (Field[float]): cloud ice m.r. to adjust
        rr (Field[float]): rain water m.r. to adjust
        rs (Field[float]): snow m.r. to adjust
        rg (Field[float]): graupel m.r. to adjust
        ths (Field[float]): potential temperature source
        rvs (Field[float]): water vapour source
        rcs (Field[float]): cloud droplets source
        ris (Field[float]): ice source
        cldfr (Field[float]): cloud fraction
        ifr (Field[float]): ratio cloud ice moist part to dry part
        hlc_hrc (Field[float]): _description_
        hlc_hcf (Field[float]): _description_
        hli_hri (Field[float]): _description_
        hli_hcf (Field[float]): _description_
        sigrc (Field[float]): _description_
        rv_tmp (Field[float]): _description_
        ri_tmp (Field[float]): _description_
        rc_tmp (Field[float]): _description_
        t_tmp (Field[float]): _description_
        cph (Field[float]): _description_
        frac_tmp (Field[float]): _description_
        sigma (Field[float]): _description_
        q1 (Field[float]): _description_
        dt (float): _description_
    """

    from __externals__ import (
        ACRIAUTI,
        BCRIAUTI,
        CI,
        CL,
        CPD,
        CPV,
        CRIAUTC,
        CRIAUTI,
        NRR,
        RD,
        RV,
        SUBG_COND,
        SUBG_MF_PDF,
        TT,
    )

    # 2.3 Compute the variation of mixing ratio
    with computation(PARALLEL), interval(...):
        t_tmp = th * exn
        lv = vaporisation_latent_heat(t_tmp)
        ls = sublimation_latent_heat(t_tmp)

        # Rem
        rv_tmp = rv
        ri_tmp = ri
        rc_tmp = rc

    # Translation note : in Fortran, ITERMAX = 1, DO JITER =1,ITERMAX
    # Translation note : version without iteration is kept (1 iteration)
    #                   IF jiter = 1; CALL ITERATION()
    # jiter > 0

    # numer of moist variables fixed to 6 (without hail)
    # 2.4 specific heat for moist air at t+1
    with computation(PARALLEL), interval(...):
        if NRR == 6:
            cph = CPD + CPV * rv_tmp + CL * (rc_tmp + rr) + CI * (ri_tmp + rs + rg)
        if NRR == 5:
            cph = CPD + CPV * rv_tmp + CL * (rc_tmp + rr) + CI * (ri_tmp + rs)
        if NRR == 4:
            cph = CPD + CPV * rv_tmp + CL * (rc_tmp + rr)
        if NRR == 2:
            cph = CPD + CPV * rv_tmp + CL * rc_tmp + CI * ri_tmp

    # 3. subgrid condensation scheme
    # Translation : only the case with subg_cond = True retained
    with computation(PARALLEL), interval(...):
        cldfr = 0
        sigrc = 0
        rv_tmp = 0
        rc_tmp = 0
        ri_tmp = 0

        # local fields
        # Translation note : 506 -> 514 kept (ocnd2 == False) # Arome default setting
        # Translation note : 515 -> 575 skipped (ocnd2 == True)
        prifact = 1  # ocnd2 == False for AROME
        ifr = 10
        frac_tmp = 0  # l340 in Condensation .f90

        # Translation note : 493 -> 503 : hlx_hcf and hlx_hrx are assumed to be present
        hlc_hcf = 0
        hlc_hrc = 0
        hli_hcf = 0
        hli_hri = 0

        # Translation note : 252 -> 263 if(present(PLV)) skipped (ls/lv are assumed to be present present)
        # Translation note : 264 -> 274 if(present(PCPH)) skipped (files are assumed to be present)

        # store total water mixing ratio (244 -> 248)
        rt = rv_tmp + rc_tmp + ri_tmp * prifact

        # Translation note : 276 -> 310 (not osigmas) skipped (osigmas = True) for Arome default version
        # Translation note : 316 -> 331 (ocnd2 == True) skipped

        #
        pv = min(
            e_sat_w(t_tmp),
            0.99 * pabs,
        )
        piv = min(
            e_sat_i(t_tmp),
            0.99 * pabs,
        )

        if rc_tmp > ri_tmp:
            if ri_tmp > 1e-20:
                frac_tmp = ri_tmp / (rc_tmp + ri_tmp)

        frac_tmp = compute_frac_ice(t_tmp)

        qsl = RD / RV * pv / (pabs - pv)
        qsi = RD / RV * piv / (pabs - piv)

        # # dtype_interpolate bewteen liquid and solid as a function of temperature
        qsl = (1 - frac_tmp) * qsl + frac_tmp * qsi
        lvs = (1 - frac_tmp) * lv + frac_tmp * ls

        # # coefficients a et b
        ah = lvs * qsl / (RV * t_tmp**2) * (1 + RV * qsl / RD)
        a = 1 / (1 + lvs / cph * ah)
        # # b = ah * a
        sbar = a * (rt - qsl + ah * lvs * (rc_tmp + ri_tmp * prifact) / CPD)

        sigma = sqrt((2 * sigs) ** 2 + (sigqsat * qsl * a) ** 2)
        sigma = max(1e-10, sigma)

        # Translation notes : 469 -> 504 (hcondens = "CB02")
        # 9.2.3 Fractional cloudiness and cloud condensate
        q1 = sbar / sigma
        if q1 > 0:
            if q1 <= 2:
                cond_tmp = min(
                    exp(-1) + 0.66 * q1 + 0.086 * q1**2, 2
                )  # we use the MIN function for continuity
        elif q1 > 2:
            cond_tmp = q1
        else:
            cond_tmp = exp(1.2 * q1 - 1)

        cond_tmp *= sigma

        # cloud fraction
        cldfr = max(0, min(1, 0.5 + 0.36 * atan(1.55 * q1))) if cond_tmp > 1e-12 else 0

        cond_tmp = 0 if cldfr == 0 else cond_tmp

        inq1 = floor(
            min(10, max(-22, min(-100, 2 * floor(q1))))
        )  # inner min/max prevents sigfpe when 2*zq1 does not fit dtype_into an "int"

        inc = 2 * q1 - inq1

        sigrc = min(1, (1 - inc) * src_1d.A[inq1] + inc * src_1d.A[inq1 + 1])

        # # Translation notes : 506 -> 514 (not ocnd2)
        rc_tmp = (1 - frac_tmp) * cond_tmp  # liquid condensate
        ri_tmp = frac_tmp * cond_tmp  # solid condensate
        t_tmp = update_temperature(t_tmp, rc_tmp, rc_tmp, ri_tmp, ri_tmp, lv, ls)
        rv_tmp = rt - rc_tmp - ri_tmp * prifact

        # Transaltion notes : 566 -> 578 HLAMBDA3 = CB
        sigrc *= min(3, max(1, 1 - q1))

    # Translation note : end jiter

    ##### 5.     COMPUTE THE SOURCES AND STORES THE CLOUD FRACTION #####
    with computation(PARALLEL), interval(...):
        # 5.0 compute the variation of mixing ratio
        w1 = (rc_tmp - rc) / dt
        w2 = (ri_tmp - ri) / dt

        # 5.1 compute the sources
        w1 = max(w1, -rcs) if w1 > 0 else min(w1, rvs)
        rvs -= w1
        rcs += w1
        ths += w1 * lv / (cph * exnref)

        w2 = max(w2, -ris) if w2 > 0 else min(w2, rvs)
        rvs -= w2
        rcs += w2
        ths += w2 * ls / (cph * exnref)

        # 5.2  compute the cloud fraction cldfr
        if SUBG_COND == 0:
            cldfr = 1 if rcs + ris > 1e-12 / dt else 0

        # Translation note : LSUBG_COND = TRUE for Arome
        else:
            w1 = rc_mf / dt
            w2 = ri_mf / dt

            if w1 + w2 > rvs:
                w1 *= rvs / (w1 + w2)
                w2 = rvs - w1

            cldfr = min(1, cldfr + cf_mf)
            rcs += w1
            ris += w2
            rvs -= w1 + w2
            ths += (w1 * lv + w2 * ls) / cph / exnref

    # Droplets subgrid autoconversion
    with computation(PARALLEL), interval(...):
        criaut = CRIAUTC / rhodref

        if SUBG_MF_PDF == 0:
            if w1 * dt > cf_mf * criaut:
                hlc_hrc += w1 * dt
                hlc_hcf = min(1, hlc_hcf + cf_mf)

        elif SUBG_MF_PDF == 1:
            if w1 * dt > cf_mf * criaut:
                hcf = 1 - 0.5 * (criaut * cf_mf) / max(1e-20, w1 * dt)
                hr = w1 * dt - (criaut * cf_mf) ** 3 / (3 * max(1e-20, w1 * dt) ** 2)

            elif 2 * w1 * dt <= cf_mf * criaut:
                hcf = 0
                hr = 0

            else:
                hcf = (2 * w1 * dt - criaut * cf_mf) ** 2 / (
                    2.0 * max(1.0e-20, w1 * dt) ** 2
                )
                hr = (
                    4.0 * (w1 * dt) ** 3
                    - 3.0 * w1 * dt * (criaut * cf_mf) ** 2
                    + (criaut * cf_mf) ** 3
                ) / (3 * max(1.0e-20, w1 * dt) ** 2)

            hcf *= cf_mf
            hlc_hcf = min(1, hlc_hcf + hcf)
            hlc_hrc += hr

    # Ice subgrid autoconversion
    with computation(PARALLEL), interval(...):
        criaut = min(
            CRIAUTI,
            10 ** (ACRIAUTI * (t_tmp - TT) + BCRIAUTI),
        )

        if SUBG_MF_PDF == 0:
            if w2 * dt > cf_mf * criaut:
                hli_hri += w2 * dt
                hli_hcf = min(1, hli_hcf + cf_mf)

        elif SUBG_MF_PDF == 1:
            if w2 * dt > cf_mf * criaut:
                hli_hcf = 1 - 0.5 * ((criaut * cf_mf) / (w2 * dt)) ** 2
                hli_hri = w2 * dt - (criaut * cf_mf) ** 3 / (3 * (w2 * dt) ** 2)

        elif 2 * w2 * dt <= cf_mf * criaut:
            hli_hcf = 0
            hli_hri = 0

        else:
            hli_hcf = (2 * w2 * dt - criaut * cf_mf) ** 2 / (2.0 * (w2 * dt) ** 2)
            hli_hri = (
                4.0 * (w2 * dt) ** 3
                - 3.0 * w2 * dt * (criaut * cf_mf) ** 2
                + (criaut * cf_mf) ** 3
            ) / (3 * (w2 * dt) ** 2)

        hli_hcf *= cf_mf
        hli_hcf = min(1, hli_hcf + hli_hcf)
        hli_hri += hli_hri

    # Translation note : 402 -> 427 (removed pout_x not present )
