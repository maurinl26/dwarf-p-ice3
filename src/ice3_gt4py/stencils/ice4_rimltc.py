# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field

from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection("ice4_rimltc")
def ice4_rimltc(
    ldcompute: Field["bool"],
    t: Field["float"],
    exn: Field["float"],
    lv_fact: Field["float"],
    ls_fact: Field["float"],
    tht: Field["float"],  # theta at time t
    ri_t: Field["float"],  # rain water mixing ratio at t
    rimltc_mr: Field["float"],
):
    """Compute cloud ice melting process RIMLTC

    Args:
        ldcompute (Field[bool]): switch to activate microphysical sources computation on column
        t (Field[float]): temperature
        exn (Field[float]): exner pressure
        lv_fact (Field[float]): vaporisation latent heat
        ls_fact (Field[float]): sublimation latent heat
        tht (Field[float]): potential temperature at t
        ri_t (Field[float]): cloud ice mixing ratio at t
        rimltc_mr (Field[float]): mixing ratio change due to cloud ice melting
    """

    from __externals__ import (
        TT,
        LFEEDBACKT,
    )

    with computation(PARALLEL), interval(...):

        # 7.1 cloud ice melting
        if ri_t > 0 and t > TT and ldcompute:
            rimltc_mr = ri_t

            # limitation due to zero crossing of temperature
            if LFEEDBACKT:
                rimltc_mr = min(
                    rimltc_mr, max(0, (tht - TT / exn) / (ls_fact - lv_fact))
                )

        else:
            rimltc_mr = 0
