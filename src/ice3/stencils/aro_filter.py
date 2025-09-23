# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, computation, PARALLEL, interval
from ifs_physics_common.framework.stencil import stencil_collection

from ice3_gt4py.functions.ice_adjust import (
    constant_pressure_heat_capacity,
    sublimation_latent_heat,
    vaporisation_latent_heat,
)


@stencil_collection("aro_filter")
def aro_filter(
    exnref: Field["float"],
    cph: Field["float"],
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
        rrs  = max(0, rrs )
        rss  = max(0, rss )
        rgs  = max(0, rgs )

    # 3.2. Adjustment for solid and liquid cloud
    with computation(PARALLEL), interval(...):
        t = tht  * exnref 
        ls = sublimation_latent_heat(t)
        lv = vaporisation_latent_heat(t)
        cph = constant_pressure_heat_capacity(rvs, rcs, ris, rrs, rss, rgs)

    with computation(PARALLEL), interval(...):
        if ris  > 0:
            rvs  = rvs  + ris 
            ths  = (
                ths 
                - ris  * ls  / cph  / exnref 
            )
            ris  = 0

    with computation(PARALLEL), interval(...):
        if rcs  < 0:
            rvs  = rvs  + rcs 
            ths  = (
                ths 
                - rcs  * lv  / cph  / exnref 
            )
            rcs  = 0

    # cloud droplets
    with computation(PARALLEL), interval(...):
        cor = (
            min(-rvs , rcs )
            if rvs  < 0 and rcs  > 0
            else 0
        )
        rvs  = rvs  + cor 
        ths  = (
            ths  - cor  * lv  / cph  / exnref 
        )
        rcs  = rcs  - cor 

    # ice
    with computation(PARALLEL), interval(...):
        cor = (
            min(-rvs , ris )
            if rvs  < 0 and ris  > 0
            else 0
        )
        rvs  = rvs  + cor 
        ths  = (
            ths  - cor  * lv  / cph  / exnref 
        )
        ris  = ris  - cor 

    # 9. Transform sources to tendencies (*= 2 dt)
    with computation(PARALLEL), interval(...):
        rvs  = rvs  * 2 * dt
        rcs  = rcs  * 2 * dt
        rrs  = rrs  * 2 * dt
        ris  = ris  * 2 * dt
        rss  = rss  * 2 * dt
        rgs  = rgs  * 2 * dt

    # (Call ice_adjust - saturation adjustment - handled by AroAdjust ImplicitTendencyComponent + ice_adjust stencil)
