# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, log, computation, interval, PARALLEL
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(from_file="PHYEX/src/common/micro/ice4_nucleation.func.h")
@stencil_collection("ice4_nucleation")
def ice4_nucleation(
    ldcompute: Field["bool"],
    tht: Field["float"],
    pabst: Field["float"],
    rhodref: Field["float"],
    exn: Field["float"],
    lsfact: Field["float"],
    t: Field["float"],
    rvt: Field["float"],
    cit: Field["float"],
    rvheni_mr: Field["float"],
    ssi: Field["float"],
):
    """Compute nucleation

    Args:
        ldcompute (Field[bool]): compuation mask for microphysical sources
        tht (Field[float]): potential temperature at t
        pabst (Field[float]): absolute pressure at t
        rhodref (Field[float]): reference density
        exn (Field[float]): exner pressure at t
        lsfact (Field[float]): latent heat of sublimation
        t (Field[float]): temperature
        rvt (Field[float]): vapour mixing ratio at t
        cit (Field[float]): ice content at t
        rvheni_mr (Field[float]): mixing ratio change of vapour
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
    
    with computation(PARALLEL), interval(...):
        usw = 0.0
        zw = 0.0

    # l72
    with computation(PARALLEL), interval(...):
        if t < TT and rvt > V_RTMIN and ldcompute:
            zw = log(t)
            usw = exp(ALPW - BETAW / t - GAMW * zw)
            zw = exp(ALPI - BETAI / t - GAMI * zw)

    with computation(PARALLEL), interval(...):
        ssi = 0.0

    # l83
    with computation(PARALLEL), interval(...):
        if t < TT and rvt > V_RTMIN and ldcompute:
            zw = min(pabst / 2, zw)
            ssi = rvt * (pabst - zw) / (EPSILO * zw) - 1
            # supersaturation over ice

            usw = min(pabst / 2, usw)
            usw = (usw / zw) * ((pabst - zw) / (pabst - usw))
            # supersaturation of saturated water vapor over ice

            ssi = min(ssi, usw)  # limitation of ssi according to ssw = 0

    # l96
    with computation(PARALLEL), interval(...):
        zw = 0.0
        if t < TT and rvt > V_RTMIN and ldcompute:
            if t < TT - 5 and ssi > 0:
                zw = NU20 * exp(ALPHA2 * ssi - BETA2)
            elif t < TT - 2.0 and t > TT - 5.0 and ssi > 0.0:
                zw = max(
                    NU20 * exp(-BETA2),
                    NU10 * exp(-BETA1 * (t - TT)) * (ssi / usw) ** ALPHA1,
                )

    # l107
    with computation(PARALLEL), interval(...):
        zw = zw - cit
        zw = min(zw, 5e4)

    # l114
    with computation(PARALLEL), interval(...):
        rvheni_mr = 0
        if t < TT and rvt > V_RTMIN and ldcompute:
            rvheni_mr = max(zw, 0.0) * MNU0 / rhodref
            rvheni_mr = min(rvt, rvheni_mr)

    # l122
    with computation(PARALLEL), interval(...):
        
        if LFEEDBACKT:
            w1 = 0
            if t < TT and rvt > V_RTMIN and ldcompute:
                w1 = min(rvheni_mr, 
                         max(0.0, (TT / exn - tht)) / lsfact) / max(
                    rvheni_mr, 1e-20
                )

            rvheni_mr *= w1
            zw *= w1

    # l134
    with computation(PARALLEL), interval(...):
        if t < TT and rvt > V_RTMIN and ldcompute:
            cit = max(zw + cit, cit)


