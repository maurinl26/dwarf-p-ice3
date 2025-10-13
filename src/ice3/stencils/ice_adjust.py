# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    __INLINED,
    PARALLEL,
    atan,
    computation,
    exp,
    floor,
    interval,
    sqrt,
    Field,
    GlobalTable,
)
from ifs_physics_common.framework.stencil import stencil_collection

from ice3.functions.tiwmx import e_sat_i, e_sat_w
from ice3.functions.ice_adjust import vaporisation_latent_heat, sublimation_latent_heat


# PHYEX/src/common/micro/ice_adjust.F90
@stencil_collection("ice_adjust")
def ice_adjust(
        sigqsat: Field["float"],
        pabs: Field["float"],
        sigs: Field["float"],
        th: Field["float"],
        exn: Field["float"],
        exn_ref: Field["float"],
        rho_dry_ref: Field["float"],
        t: Field["float"],
        rv: Field["float"],
        ri: Field["float"],
        rc: Field["float"],
        rr: Field["float"],
        rs: Field["float"],
        rg: Field["float"],
        cf_mf: Field["float"],
        rc_mf: Field["float"],
        ri_mf: Field["float"],
        rv_out: Field["float"],
        rc_out: Field["float"],
        ri_out: Field["float"],
        hli_hri: Field["float"],
        hli_hcf: Field["float"],
        hlc_hrc: Field["float"],
        hlc_hcf: Field["float"],
        ths: Field["float"],
        rvs: Field["float"],
        rcs: Field["float"],
        ris: Field["float"],
        cldfr: Field["float"],
        cph: Field["float"],
        lv: Field["float"],
        ls: Field["float"],
        q1: Field["float"],
        sigma_rc: Field["float"],
        pv: Field["float"],
        piv: Field["float"],
        src_1d: GlobalTable[("float", (34))],
        dt: "float"
):
    """Microphysical adjustments for specific contents due to condensation."""

    from __externals__ import (
        RD, RV, FRAC_ICE_ADJUST, TMAXMIX, TMINMIX,
        OCND2, LSIGMAS, LSTATNW, CONDENS,
        NRR, CPV, CPD, CL, CI,
        LSUBG_COND,
        SUBG_MF_PDF,
        CRIAUTC,
        CRIAUTI,
        ACRIAUTI,
        BCRIAUTI,
        TT
    )

    # 2.3 Compute the variation of mixing ratio
    with computation(PARALLEL), interval(...):
        t = th * exn
        lv = vaporisation_latent_heat(t)
        ls = sublimation_latent_heat(t)

    # Translation note : in Fortran, ITERMAX = 1, DO JITER =1,ITERMAX
    # Translation note : version without iteration is kept (1 iteration)
    #                   IF jiter = 1; CALL ITERATION()
    # jiter > 0

    # numer of moist variables fixed to 6 (without hail)

    # Translation note :
    # 2.4 specific heat for moist air at t+1
    with computation(PARALLEL), interval(...):
        # Translation note : case(7) removed because hail is not taken into account
        # Translation note : l453 to l456 removed
        if __INLINED(NRR == 6):
            cph = CPD + CPV * rv + CL * (rc + rr) + CI * (ri + rs + rg)
        if __INLINED(NRR == 5):
            cph = CPD + CPV * rv + CL * (rc + rr) + CI * (ri + rs)
        if __INLINED(NRR == 4):
            cph = CPD + CPV * rv + CL * (rc + rr)
        if __INLINED(NRR == 2):
            cph = CPD + CPV * rv + CL * rc + CI * ri

    # initialize values
    with computation(PARALLEL), interval(...):
        cldfr = 0.0
        rv_out = 0.0
        rc_out = 0.0
        ri_out = 0.0

    # 3. subgrid condensation scheme
    # Translation note : only the case with LSUBG_COND = True retained (l475 in ice_adjust.F90)
    # sigqsat and sigs must be provided by the user
    with computation(PARALLEL), interval(...):
        # local
        # Translation note : 506 -> 514 kept (ocnd2 == False) # Arome default setting
        # Translation note : 515 -> 575 skipped (ocnd2 == True)
        prifact = 1  # ocnd2 == False for AROME
        frac_tmp = 0  # l340 in Condensation .f90

        # Translation note : 252 -> 263 if(present(PLV)) skipped (ls/lv are assumed to be present)
        # Translation note : 264 -> 274 if(present(PCPH)) skipped (files are assumed to be present)

        # store total water mixing ratio (244 -> 248)
        rt = rv + rc + ri * prifact

        # Translation note : 276 -> 310 (not osigmas) skipped : (osigmas = True) for Arome default version
        # Translation note : 316 -> 331 (ocnd2 == True) skipped : ocnd2 = False for Arome

        # l334 to l337
        if __INLINED(not OCND2):
            pv = min(
                e_sat_w(t),
                0.99 * pabs,
            )
            piv = min(
                e_sat_i(t),
                0.99 * pabs,
            )

        # TODO : l341 -> l350
        # Translation note : OUSERI = TRUE, OCND2 = False
        if __INLINED(not OCND2):
            frac_tmp = (
                rc / (rc + ri)
                if rc + ri > 1e-20 else 0
            )

            # Compute frac ice inlined
            # Default Mode (S)
            if __INLINED(FRAC_ICE_ADJUST == 3):
                frac_tmp = max(0, min(1, frac_tmp))

            # AROME mode
            if __INLINED(FRAC_ICE_ADJUST == 0):
                frac_tmp = max(0, min(1, ((TMAXMIX - t) / (TMAXMIX - TMINMIX))))

        # Supersaturation coefficients
        qsl = RD / RV * pv / (pabs - pv)
        qsi = RD / RV * piv / (pabs - piv)

        # interpolate between liquid and solid as a function of temperature
        qsl = (1 - frac_tmp) * qsl + frac_tmp * qsi
        lvs = (1 - frac_tmp) * lv + frac_tmp * ls

        # coefficients a et b
        ah = lvs * qsl / (RV * t ** 2) * (1 + RV * qsl / RD)
        a = 1 / (1 + lvs / cph * ah)
        b = ah * a
        sbar = a * (rt - qsl + ah * lvs * (rc + ri * prifact) / cph)

        # Translation note : l369 - l390 kept
        # Translation note : l391 - l406 skipped (OSIGMAS = False)
        # Translation note : LSTATNW = False
        # Translation note : l381 retained for sigmas formulation
        # Translation note : OSIGMAS = TRUE
        # Translation npte : LHGT_QS = False (and ZDZFACT unused)
        if __INLINED(LSIGMAS and not LSTATNW):
            sigma = (
                sqrt((2 * sigs) ** 2 + (sigqsat * qsl * a) ** 2)
                if sigqsat != 0
                else 2 * sigs
            )

        # Translation note : l407 - l411
        sigma = max(1e-10, sigma)
        q1 = sbar / sigma

        # Translation notes : l413 to l468 skipped (HCONDENS=="GAUS")
        # TODO : add hcondens == "GAUS" option
        # Translation notes : l469 to l504 kept (HCONDENS = "CB02")
        # 9.2.3 Fractional cloudiness and cloud condensate
        # HCONDENS = 0 is CB02 option
        if __INLINED(CONDENS == 0):
            # Translation note : l470 to l479
            if q1 > 0.0:
                cond_tmp = (
                    min(exp(-1.0) + 0.66 * q1 + 0.086 * q1 ** 2, 2.0) if q1 <= 2.0 else q1
                )  # we use the MIN function for continuity
            else:
                cond_tmp = exp(1.2 * q1 - 1.0)
            cond_tmp *= sigma

            # Translation note : l482 to l489
            # cloud fraction
            cldfr = (
                max(0.0, min(1.0, 0.5 + 0.36 * atan(1.55 * q1))) if cond_tmp >= 1e-12 else 0
            )

            # Translation note : l487 to l489
            cond_tmp = 0 if cldfr == 0 else cond_tmp

            # Translation note : l496 to l503 removed (because initialized further in cloud_fraction diagnostics)

            # Translation notes : 506 -> 514 (not ocnd2)
            # Translation notes : l515 to l565 (removed)
            if __INLINED(not OCND2):
                rc_out = (1 - frac_tmp) * cond_tmp  # liquid condensate
                ri_out = frac_tmp * cond_tmp  # solid condensate
                t += ((rc_out - rc) * lv + (ri_out - ri) * ls) / cph
                rv_out = rt - rc_out - ri_out * prifact

            # Translation note : sigrc computation out of scope
            # Translation note : l491 to l494 skept
            # sigrc computation in sigrc_computation stencil

        # Translation note : end jiter

    # Compute sigma_rc using global table
    with computation(PARALLEL), interval(...):
        inq1 = floor(min(100., max(-100., 2 * q1[0, 0, 0])))
        inq2 = min(max(-22, inq1), 10)
        # inner min/max prevents sigfpe when 2*zq1 does not fit dtype_into an "int"
        inc = 2 * q1  # - inq2
        sigma_rc = min(1, (1 - inc) * src_1d.A[inq2 + 22] + inc * src_1d.A[inq2 + 23])

    # Cloud fraction 1
    with computation(PARALLEL), interval(...):
        # 5.0 compute the variation of mixing ratio
        w1 = (rc_out - rc) / dt
        w2 = (ri_out - ri) / dt

        # 5.1 compute the sources
        w1 = max(w1, -rcs) if w1 < 0.0 else min(w1, rvs)
        rvs -= w1
        rcs += w1
        ths += w1 * lv / (cph * exn_ref)

        w2 = max(w2, -ris) if w2 < 0.0 else min(w2, rvs)
        rvs -= w2
        ris += w2
        ths += w2 * ls / (cph * exn_ref)

    # Cloud fraction 2
    with computation(PARALLEL), interval(...):

        if __INLINED(not LSUBG_COND):
            cldfr = 1.0 if ((rcs + ris) * dt > 1e-12) else 0.0
        # Translation note : OCOMPUTE_SRC is taken False
        # Translation note : l320 to l322 removed

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
            rvs -= (w1 + w2)
            ths += (w1 * lv + w2 * ls) / (cph * exn_ref)

            # Droplets subgrid autoconversion
            # LLHLC_H is True (AROME like) phlc_hrc and phlc_hcf are present
            # LLHLI_H is True (AROME like) phli_hri and phli_hcf are present
            criaut = CRIAUTC / rho_dry_ref

            # ice_adjust.F90 IF LLNONE; IF CSUBG_MF_PDF is None
            if __INLINED(SUBG_MF_PDF == 0):
                if w1 * dt > cf_mf * criaut:
                    hlc_hrc += w1 * dt
                    hlc_hcf = min(1.0, hlc_hcf + cf_mf)

            # Translation note : if LLTRIANGLE in .F90
            if __INLINED(SUBG_MF_PDF == 1):
                if w1 * dt > cf_mf * criaut:
                    hcf = 1.0 - 0.5 * (criaut * cf_mf / max(1e-20, w1 * dt)) ** 2
                    hr = w1 * dt - (criaut * cf_mf) ** 3 / (
                            3 * max(1e-20, w1 * dt) ** 2
                    )

                elif 2.0 * w1 * dt <= cf_mf * criaut:
                    hcf = 0.0
                    hr = 0.0

                else:
                    hcf = (2.0 * w1 * dt - criaut * cf_mf) ** 2 / (
                            2.0 * max(1.0e-20, w1 * dt) ** 2
                    )
                    hr = (
                                 4.0 * (w1 * dt) ** 3
                                 - 3.0 * w1 * dt * (criaut * cf_mf) ** 2
                                 + (criaut * cf_mf ** 3)
                         ) / (3 * max(1.0e-20, w1 * dt) ** 2)

                hcf *= cf_mf
                hlc_hcf = min(1.0, hlc_hcf + hcf)
                hlc_hrc += hr

            # Ice subgrid autoconversion
            criaut = min(
                CRIAUTI,
                10 ** (ACRIAUTI * (t - TT) + BCRIAUTI),
            )

            # LLNONE in ice_adjust.F90
            if __INLINED(SUBG_MF_PDF == 0):
                if w2 * dt > cf_mf * criaut:
                    hli_hri += w2 * dt
                    hli_hcf = min(1.0, hli_hcf + cf_mf)

            # LLTRIANGLE in ice_adjust.F90
            if __INLINED(SUBG_MF_PDF == 1):
                if w2 * dt > cf_mf * criaut:
                    hcf = 1.0 - 0.5 * ((criaut * cf_mf) / (w2 * dt)) ** 2
                    hri = w2 * dt - (criaut * cf_mf) ** 3 / (3 * (w2 * dt) ** 2)

                elif 2 * w2 * dt <= cf_mf * criaut:
                    hcf = 0.0
                    hri = 0.0

                else:
                    hcf = (2.0 * w2 * dt - criaut * cf_mf) ** 2 / (
                            2.0 * (w2 * dt) ** 2
                    )
                    hri = (
                                  4.0 * (w2 * dt) ** 3
                                  - 3.0 * w2 * dt * (criaut * cf_mf) ** 2
                                  + (criaut * cf_mf) ** 3
                          ) / (3.0 * (w2 * dt) ** 2)

                hcf *= cf_mf
                hli_hcf = min(1.0, hli_hcf + hcf)
                hli_hri += hri

