# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function

from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection("ice4_rrhong")
def ice4_rrhong(
    t: Field["float"],
    exn: Field["float"],
    lv_fact: Field["float"],
    ls_fact: Field["float"],
    tht: Field["float"],  # theta at time t
    rrhong_mr_out: Field["float"],
    rr_in: Field["float"],  # rain water mixing ratio at t
    ld_compute: Field["float"],  # mask of computation
):

    from __externals__ import r_rtmin, tt, lfeedbackt

    with computation(PARALLEL), interval(...):

        if t < tt - 35 and rr_in > r_rtmin and ldcompute == 1:
            rrhong_mr_out = rr_in
            if lfeedbackt == 1:
                rrhong_mr_out = min(
                    rrhong_mr_out, max(0, ((tt - 35) / exn - tht) / (ls_fact - lv_fact))
                )

        else:
            rrhong_mr_out = 0
