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
    IJ
)
from ifs_physics_common.framework.stencil import stencil_collection
from ice3_gt4py.functions.tiwmx import e_sat_i, e_sat_w
from ifs_physics_common.utils.f2py import ported_method


@ported_method(
    from_file="PHYEX/src/common/micro/condensation.F90",
)
@stencil_collection("condensation")
def condensation(
    sigqsat: Field[IJ, "float"],
    pabs: Field["float"],
    sigs: Field["float"],
    t: Field["float"],
    rv_in: Field["float"],
    ri_in: Field["float"],
    rc_in: Field["float"],
    rv_out: Field["float"],
    rc_out: Field["float"],
    ri_out: Field["float"],
    cldfr: Field["float"],
    cph: Field["float"],
    lv: Field["float"],
    ls: Field["float"],
    q1: Field["float"],
    # Temporaries
    piv: Field["float"],
    pv: Field["float"],
    qsl: Field["float"],
    qsi: Field["float"], 
    frac_tmp: Field["float"],
    sigma: Field["float"],
    cond_tmp: Field["float"],
    a: Field["float"],
    b: Field["float"], 
    sbar: Field["float"]
):
    """Microphysical adjustments for specific contents due to condensation."""

    from __externals__ import (
        RD, RV, FRAC_ICE_ADJUST, TMAXMIX, TMINMIX, 
        OCND2, LSIGMAS, LSTATNW, CONDENS
    )
    
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
        rt = rv_in + rc_in + ri_in * prifact

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
            frac_tmp =(
                rc_in / (rc_in + ri_in) 
                if rc_in + ri_in > 1e-20 else 0
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
        ah = lvs * qsl / (RV * t**2) * (1 + RV * qsl / RD)
        a = 1 / (1 + lvs / cph * ah)
        b = ah * a
        sbar = a * (rt - qsl + ah * lvs * (rc_in + ri_in * prifact) / cph)

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
                min(exp(-1.0) + 0.66 * q1 + 0.086 * q1**2, 2.0) if q1 <= 2.0 else q1
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
                t += ((rc_out - rc_in) * lv + (ri_out - ri_in) * ls) / cph
                rv_out = rt - rc_out - ri_out * prifact

            # Translation note : sigrc computation out of scope
            # Translation note : l491 to l494 skept
            # sigrc computation in sigrc_computation stencil

        # Translation note : end jiter


@ported_method(
    from_file="./PHYEX/src/common/micro/condensation.F90",
    from_line=186,
    to_line=189
)
@stencil_collection("sigrc_diagnostic")
def sigrc_computation(
    q1: Field["float"], 
    sigrc: Field["float"], 
    inq1: Field["int"],
    src_1d: GlobalTable["float", (34)]
):

    from __externals__ import LAMBDA3

    with computation(PARALLEL), interval(...):

        inq1 = floor(min(100., max(-100., 2 * q1)))
        inq2 = min(max(-22, inq1), 10)
        # inner min/max prevents sigfpe when 2*zq1 does not fit dtype_into an "int"
        inc = 2 * q1 - inq2
        sigrc = min(1, (1 - inc) * src_1d.A[inq2 + 22] + inc * src_1d.A[inq2 + 23])

        # Transaltion notes : 566 -> 578 HLAMBDA3 = CB
        if __INLINED(LAMBDA3 == 0):
            sigrc *= min(3, max(1, 1 - q1))

