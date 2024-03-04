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
        tt,
        v_rtmin,
        alpi,
        betai,
        gami,
        alpw,
        betaw,
        gamw,
        epsilo,
        nu20,
        alpha2,
        beta2,
        nu10,
        beta1,
        alpha1,
        mnu0,
        lfeedbackt,
    )

    # l72
    with computation(PARALLEL), interval(...):

        if t < tt and rv_t > v_rtmin and ldcompute:
            usw = 0
            w2 = 0

            w2 = log(t)
            usw = exp(alpw - betaw / t - gamw * w2)
            w2 = exp(alpi - betai / t - gami * w2)

    # l83
    with computation(PARALLEL), interval(...):

        if t < tt and rv_t > v_rtmin and ldcompute:

            ssi = 0
            w2 = min(pabs_t / 2, w2)
            ssi = rv_t * (pabs_t - w2) / (epsilo * w2) - 1
            # supersaturation over ice

            usw = min(pabs_t / 2, usw)
            usw = (usw / w2) * ((pabs_t - w2) / (pabs_t - usw))
            # supersaturation of saturated water vapor over ice

            ssi = min(ssi, usw)  # limitation of ssi according to ssw = 0

    # l96
    with computation(PARALLEL), interval(...):

        w2 = 0
        if t < tt and rv_t > v_rtmin and ldcompute:
            if t < tt - 5 and ssi > 0:
                w2 = nu20 * exp(alpha2 * ssi - beta2)
            elif t < tt - 2 and t > tt - 5 and ssi > 0:
                w2 = max(
                    nu20 * exp(-beta2),
                    nu10 * exp(-beta1 * (t - tt)) * (ssi / usw) ** alpha1,
                )

    # l107
    with computation(PARALLEL), interval(...):
        w2 = w2 - ci_t
        w2 = min(w2, 5e4)

    # l114
    with computation(PARALLEL), interval(...):
        rv_heni_mr = 0
        if t < tt and rv_t > v_rtmin and ldcompute:
            rv_heni_mr = max(w2, 0) * mnu0 / rhodref
            rv_heni_mr = min(rv_t, rv_heni_mr)

    # l122
    with computation(PARALLEL), interval(...):
        if lfeedbackt:
            w1 = 0
            if t < tt and rv_t > v_rtmin and ldcompute:
                w1 = min(rv_heni_mr, max(0, (tt / exn - tht)) / ls_fact) / max(
                    rv_heni_mr, 1e-20
                )

            rv_heni_mr *= w1
            w2 *= w1

    # l134
    with computation(PARALLEL), interval(...):
        if t < tt and rv_t > v_rtmin and ldcompute:
            ci_t = max(w2 + ci_t, ci_t)
