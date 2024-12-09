# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import __INLINED, PARALLEL, computation, interval, Field
from ifs_physics_common.framework.stencil import stencil_collection

from ice3_gt4py.functions.ice_adjust import (
    sublimation_latent_heat,
    vaporisation_latent_heat,
)


@stencil_collection("thermodynamic_fields")
def thermodynamic_fields(
    th: Field["float"],
    exn: Field["float"],
    rv: Field["float"],
    rc: Field["float"],
    rr: Field["float"],
    ri: Field["float"],
    rs: Field["float"],
    rg: Field["float"],
    lv: Field["float"],
    ls: Field["float"],
    cph: Field["float"],
    t: Field["float"],
):
    from __externals__ import NRR, CPV, CPD, CL, CI

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


# 5.
@stencil_collection("ice_adjust_sources")
def compute_sources(
    rvs: Field["float"],
    ris: Field["float"],
    ths: Field["float"],
    rc_in: Field["float"],
    rv_in: Field["float"],
    ri_in: Field["float"],
    rc_out: Field["float"],
    rv_out: Field["float"],
    ri_out: Field["float"],
    cph: Field["float"],
    ls: Field["float"],
    lv: Field["float"],
    exnref: Field["float"],
    dt: "float",
):

    # 5.     COMPUTE THE SOURCES AND STORES THE CLOUD FRACTION #####
    with computation(PARALLEL), interval(...):

        # 5.0 compute the variation of mixing ratio
        w1 = (rc_out - rc_in) / dt
        w2 = (ri_out - ri_in) / dt

        # 5.1 compute the sources
        w1 = max(w1, -rcs) if w1 < 0 else min(w1, rvs)
        rvs -= w1
        rcs += w1
        ths += w1 * lv / (cph * exnref)

        w2 = max(w2, -ris) if w2 < 0 else min(w2, rvs)
        rvs -= w2
        rcs += w2
        ths += w2 * ls / (cph * exnref)
