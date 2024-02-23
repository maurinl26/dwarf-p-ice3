# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function

from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection("ice4_rrhong")
def ice4_rrhong(
    ldcompute: Field["float"],
    t: Field["float"],
    exn: Field["float"],
    lv_fact: Field["float"],
    ls_fact: Field["float"],
    tht: Field["float"],  # theta at time t
    rrhong_mr: Field["float"],
    rr_t: Field["float"],  # rain water mixing ratio at t
):
    """Compute the spontaneous frezzing source RRHONG

    Args:
        ldcompute (Field[float]): switch to activate microphysical processes on column
        t (Field[float]): temperature at t
        exn (Field[float]): exner pressure
        lv_fact (Field[float]): vaporisation latent heat
        ls_fact (Field[float]): sublimation latent heat
        tht (Field[float]): potential temperature
        rr_t (Field[float]): rain mixing ratio at t
        rrhong_mr (Field[float]): mixing ratio for spontaneous freezing source
    """

    from __externals__ import r_rtmin, tt, lfeedbackt

    # 3.3 compute the spontaneous frezzing source: RRHONG
    with computation(PARALLEL), interval(...):

        if t < tt - 35 and rr_t > r_rtmin and ldcompute == 1:
            rrhong_mr = rr_t

            # limitation for -35 degrees crossing
            if lfeedbackt == 1:
                rrhong_mr = min(
                    rrhong_mr, max(0, ((tt - 35) / exn - tht) / (ls_fact - lv_fact))
                )

        else:
            rrhong_mr = 0
