# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (PARALLEL, Field, computation, exp,
                                      interval, log)


# "PHYEX/src/common/micro/ice4_nucleation.func.h"
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
    """Compute heterogeneous ice nucleation through water vapor deposition.
    
    This stencil computes ice crystal nucleation by heterogeneous nucleation on ice
    nuclei (HENI process). The nucleation rate depends on temperature and supersaturation
    over ice (ssi). The scheme uses empirical parameterizations that distinguish between
    different temperature regimes:
    - T < -5°C: Uses NU20 parameterization with supersaturation dependency
    - -5°C ≤ T < -2°C: Transition regime using max of two parameterizations
    - T ≥ -2°C: No nucleation
    
    The supersaturation over ice is computed from the water vapor mixing ratio and
    limited by the supersaturation of water-saturated air over ice to ensure physical
    consistency. An optional temperature feedback (LFEEDBACKT) limits nucleation to
    prevent temperature from exceeding the freezing point.

    Args:
        ldcompute (Field[bool]): Computation mask for microphysical sources - true where nucleation is computed
        tht (Field[float]): Potential temperature at time t (K)
        pabst (Field[float]): Absolute pressure at time t (Pa)
        rhodref (Field[float]): Reference air density (kg/m³)
        exn (Field[float]): Exner function (dimensionless pressure) at time t
        lsfact (Field[float]): Latent heat factor for sublimation (K·kg/kg), used for temperature feedback
        t (Field[float]): Temperature (K)
        rvt (Field[float]): Water vapor mixing ratio at time t (kg/kg)
        cit (Field[float]): Input/Output - Ice crystal number concentration at time t (1/kg), updated by nucleation
        rvheni_mr (Field[float]): Output - Water vapor mixing ratio change due to heterogeneous nucleation (kg/kg)
        ssi (Field[float]): Output - Supersaturation over ice (dimensionless), computed from vapor and temperature
    """

    from __externals__ import (ALPHA1, ALPHA2, ALPI, ALPW, BETA1, BETA2, BETAI,
                               BETAW, EPSILO, GAMI, GAMW, LFEEDBACKT, MNU0,
                               NU10, NU20, TT, V_RTMIN)
    
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
            usw = (usw / zw) * ((pabst - zw) / (pabst - usw)) - 1.0
            # supersaturation of saturated water vapor over ice

            ssi = min(ssi, usw)  # limitation of ssi according to ssw = 0

    # l96
    with computation(PARALLEL), interval(...):
        zw = 0.0

    with computation(PARALLEL), interval(...):
        if t < TT and rvt > V_RTMIN and ldcompute:
            if t < TT - 5 and ssi > 0:
                zw = NU20 * exp(ALPHA2 * ssi - BETA2)
            elif t <= TT - 2.0 and t >= TT - 5.0 and ssi > 0.0:
                zw = max(
                    NU20 * exp(-BETA2),
                    NU10 * exp(-BETA1 * (t - TT)) * (ssi / usw) ** ALPHA1,
                )

    # l107
    with computation(PARALLEL), interval(...):
        if t < TT and rvt > V_RTMIN and ldcompute:
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
            w1 = 0.0
            if t < TT and rvt > V_RTMIN and ldcompute:
                w1 = min(rvheni_mr, 
                         max(0.0, (TT / exn - tht)) / lsfact) / max(
                    rvheni_mr, 1e-20
                )
        else:
            w1 = 1.0
            
        if LFEEDBACKT:
            rvheni_mr = rvheni_mr * w1
            zw = zw * w1

    # l134
    with computation(PARALLEL), interval(...):
        if t < TT and rvt > V_RTMIN and ldcompute:
            cit = max(zw + cit, cit)
