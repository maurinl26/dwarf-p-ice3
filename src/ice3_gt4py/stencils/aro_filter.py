# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, computation, PARALLEL, interval
from ifs_physics_common.framework.stencil import stencil_collection

from ice3_gt4py.functions.ice_adjust import (
    cph,
    sublimation_latent_heat,
    vaporisation_latent_heat,
)


@stencil_collection("aro_filter")
def aro_filter(
    exnref: Field["float"],
    tht: Field["float"],
    ths: Field["float"],
    rcs: Field["float"],
    rrs: Field["float"],
    ris: Field["float"],
    rvs: Field["float"],
    rgs: Field["float"],
    rss: Field["float"],
    dt: "float",
):
    """Negativity filter for sources

    Args:
        exnref (Field[float]): reference exner pressure
        tht (Field[float]): potential temperature at time t
        ths (Field[float]): potential temperature source
        rcs (Field[float]): cloud droplets source
        rrs (Field[float]): rain source
        ris (Field[float]): ice source
        rvs (Field[float]): water vapour source
        rgs (Field[float]): graupel source
        rss (Field[float]): snow source
        dt (float): time step un seconds
    """

    # 3.1. Remove negative values
    with computation(PARALLEL), interval(...):
        rrs[0, 0, 0] = max(0, rrs[0, 0, 0])
        rss[0, 0, 0] = max(0, rss[0, 0, 0])
        rgs[0, 0, 0] = max(0, rgs[0, 0, 0])

    # 3.2. Adjustment for solid and liquid cloud
    with computation(PARALLEL), interval(...):
        t = tht[0, 0, 0] * exnref[0, 0, 0]
        ls = sublimation_latent_heat(t)
        lv = vaporisation_latent_heat(t)
        cph_ = cph(rvs, rcs, ris, rrs, rss, rgs)

    with computation(PARALLEL), interval(...):
        if ris[0, 0, 0] > 0:
            rvs[0, 0, 0] = rvs[0, 0, 0] + ris[0, 0, 0]
            ths[0, 0, 0] = (
                ths[0, 0, 0]
                - ris[0, 0, 0] * ls[0, 0, 0] / cph[0, 0, 0] / exnref[0, 0, 0]
            )
            ris[0, 0, 0] = 0

    with computation(PARALLEL), interval(...):
        if rcs[0, 0, 0] < 0:
            rvs[0, 0, 0] = rvs[0, 0, 0] + rcs[0, 0, 0]
            ths[0, 0, 0] = (
                ths[0, 0, 0]
                - rcs[0, 0, 0] * lv[0, 0, 0] / cph_[0, 0, 0] / exnref[0, 0, 0]
            )
            rcs[0, 0, 0] = 0

    # cloud droplets
    with computation(PARALLEL), interval(...):
        cor = (
            min(-rvs[0, 0, 0], rcs[0, 0, 0])
            if rvs[0, 0, 0] < 0 and rcs[0, 0, 0] > 0
            else 0
        )
        rvs[0, 0, 0] = rvs[0, 0, 0] + cor[0, 0, 0]
        ths[0, 0, 0] = (
            ths[0, 0, 0] - cor[0, 0, 0] * lv[0, 0, 0] / cph_[0, 0, 0] / exnref[0, 0, 0]
        )
        rcs[0, 0, 0] = rcs[0, 0, 0] - cor[0, 0, 0]

    # ice
    with computation(PARALLEL), interval(...):
        cor = (
            min(-rvs[0, 0, 0], ris[0, 0, 0])
            if rvs[0, 0, 0] < 0 and ris[0, 0, 0] > 0
            else 0
        )
        rvs[0, 0, 0] = rvs[0, 0, 0] + cor[0, 0, 0]
        ths[0, 0, 0] = (
            ths[0, 0, 0] - cor[0, 0, 0] * lv[0, 0, 0] / cph_[0, 0, 0] / exnref[0, 0, 0]
        )
        ris[0, 0, 0] = ris[0, 0, 0] - cor[0, 0, 0]

    # 9. Transform sources to tendencies (*= 2 dt)
    with computation(PARALLEL), interval(...):
        rvs[0, 0, 0] = rvs[0, 0, 0] * 2 * dt
        rcs[0, 0, 0] = rcs[0, 0, 0] * 2 * dt
        rrs[0, 0, 0] = rrs[0, 0, 0] * 2 * dt
        ris[0, 0, 0] = ris[0, 0, 0] * 2 * dt
        rss[0, 0, 0] = rss[0, 0, 0] * 2 * dt
        rgs[0, 0, 0] = rgs[0, 0, 0] * 2 * dt

    # (Call ice_adjust - saturation adjustment - handled by AroAdjust ImplicitTendencyComponent + ice_adjust stencil)
