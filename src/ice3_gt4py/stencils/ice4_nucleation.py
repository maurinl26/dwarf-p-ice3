# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, log
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(from_file="PHYEX/src/common/micro/ice4_nucleation.func.h")
@stencil_collection("ice4_nucleation")
def ice4_fast_rg(
    ldcompute: Field["bool"],
    tht: Field["float"],
    pabs_t: Field["float"],
    rhodref: Field["float"],
    exn: Field["float"],
    ls_fact: Field["float"],
    t: Field["float"],
    rv_t: Field["float"],
    ci_t: Field["float"],
    rv_heni_mr: Field["float"],
    usw: Field["float"],  # PBUF(USW) in Fortran
    w2: Field["float"],
    w1: Field["float"],
    ssi: Field["float"],
):
    """Compute nucleation

    Args:
        ldcompute (Field[bool]): _description_
        tht (Field[float]): potential temperature at t
        pabs_t (Field[float]): absolute pressure at t
        rhodref (Field[float]): reference density
        exn (Field[float]): exner pressure at t
        ls_fact (Field[float]): latent heat of sublimation
        t (Field[float]): temperature
        rv_t (Field[float]): vapour mixing ratio at t
        ci_t (Field[float]): ice content at t
        rv_heni_mr (Field[float]): mixing ratio change of vapour
    """

    from __externals__ import (
        ALPHA1,
        ALPHA2,
        ALPI,
        ALPW,
        BETA1,
        BETA2,
        BETAI,
        BETAW,
        EPSILO,
        GAMI,
        GAMW,
        LFEEDBACKT,
        MNU0,
        NU10,
        NU20,
        TT,
        V_RTMIN,
    )

    # l72
    with computation(PARALLEL), interval(...):
        if t < TT and rv_t > V_RTMIN and ldcompute:
            usw = 0
            w2 = 0

            w2 = log(t)
            usw = exp(ALPW - BETAW / t - GAMW * w2)
            w2 = exp(ALPI - BETAI / t - GAMI * w2)

    # l83
    with computation(PARALLEL), interval(...):
        if t < TT and rv_t > V_RTMIN and ldcompute:
            ssi = 0
            w2 = min(pabs_t / 2, w2)
            ssi = rv_t * (pabs_t - w2) / (EPSILO * w2) - 1
            # supersaturation over ice

            usw = min(pabs_t / 2, usw)
            usw = (usw / w2) * ((pabs_t - w2) / (pabs_t - usw))
            # supersaturation of saturated water vapor over ice

            ssi = min(ssi, usw)  # limitation of ssi according to ssw = 0

    # l96
    with computation(PARALLEL), interval(...):
        w2 = 0
        if t < TT and rv_t > V_RTMIN and ldcompute:
            if t < TT - 5 and ssi > 0:
                w2 = NU20 * exp(ALPHA2 * ssi - BETA2)
            elif t < TT - 2 and t > TT - 5 and ssi > 0:
                w2 = max(
                    NU20 * exp(-BETA2),
                    NU10 * exp(-BETA1 * (t - TT)) * (ssi / usw) ** ALPHA1,
                )

    # l107
    with computation(PARALLEL), interval(...):
        w2 = w2 - ci_t
        w2 = min(w2, 5e4)

    # l114
    with computation(PARALLEL), interval(...):
        rv_heni_mr = 0
        if t < TT and rv_t > V_RTMIN and ldcompute:
            rv_heni_mr = max(w2, 0) * MNU0 / rhodref
            rv_heni_mr = min(rv_t, rv_heni_mr)

    # l122
    with computation(PARALLEL), interval(...):
        if LFEEDBACKT:
            w1 = 0
            if t < TT and rv_t > V_RTMIN and ldcompute:
                w1 = min(rv_heni_mr, max(0, (TT / exn - tht)) / ls_fact) / max(
                    rv_heni_mr, 1e-20
                )

            rv_heni_mr *= w1
            w2 *= w1

    # l134
    with computation(PARALLEL), interval(...):
        if t < TT and rv_t > V_RTMIN and ldcompute:
            ci_t = max(w2 + ci_t, ci_t)


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_tendencies.F90",
    from_line=152,
    to_line=157,
)
@stencil_collection("nucleation_post_processing")
def ice4_nucleation_post_processing(
    t: Field["float"],
    exn: Field["float"],
    ls_fact: Field["float"],
    lv_fact: Field["float"],
    theta_t: Field["float"],
    rv_t: Field["float"],
    ri_t: Field["float"],
    rvheni_mr: Field["float"],
):
    """adjust mixing ratio with nucleation increments

    Args:
        t (Field[float]): temperature
        exn (Field[float]): exner pressure
        ls_fact (Field[float]): sublimation latent heat over heat capacity
        theta_t (Field[float]): potential temperature
        rv_t (Field[float]): vapour m.r.
        ri_t (Field[float]): ice m.r.
        rvheni_mr (Field[float]): vapour m.r. increment due to HENI (heteroegenous nucleation over ice)
    """

    with computation(PARALLEL), interval(...):
        theta_t += rvheni_mr * (ls_fact - lv_fact)
        t = theta_t / exn
        rv_t -= rvheni_mr
        ri_t += rvheni_mr
