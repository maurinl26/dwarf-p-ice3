# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field

from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90")
@stencil_collection("step_limiter")
def step_limiter(
    exn: Field["float"],
    theta_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    theta_a_tnd: Field["float"],
    rc_a_tnd: Field["float"],
    rr_a_tnd: Field["float"],
    ri_a_tnd: Field["float"],
    rs_a_tnd: Field["float"],
    rg_a_tnd: Field["float"],
    theta_b: Field["float"],
    rc_b: Field["float"],
    rr_b: Field["float"],
    ri_b: Field["float"],
    rs_b: Field["float"],
    rg_b: Field["float"],
    delta_t_micro: Field["float"],
    t_micro: Field["float"],
    delta_t_soft: Field["float"],
    t_soft: Field["float"],
    ldcompute: Field["bool"],
    ldmicro: Field["bool"],
    ci_t: Field["float"],
):
    from __externals__ import TSTEP, LFEEDBACKT, TT, C_RTMIN, MNH_TINY, TSTEP_TS

    # 4.6 Time integration
    with computation(PARALLEL), interval(...):
        delta_t_micro = TSTEP - t_micro if ldcompute else 0

    # Adjustment of tendencies when temperature reaches 0
    with computation(PARALLEL), interval(...):

        theta_tt = TT / exn
        if (theta_t - theta_tt) * (theta_t + theta_b - theta_tt) < 0:
            delta_t_micro = 0

        if abs(theta_a_tnd > 1e-20):
            delta_t_tmp = (theta_tt - theta_b - theta_t) / theta_a_tnd
            if delta_t_tmp > 0:
                delta_t_micro = min(delta_t_micro, delta_t_tmp)

    # Tendencies adjustment if a speci disappears
    # TODO: change mnh_tiny with epsilon machine
    with computation(PARALLEL), interval(...):
        if rc_a_tnd < -1e20 and rc_t > C_RTMIN:
            delta_t_micro = min(delta_t_micro, -(rc_b + rc_t) / rc_a_tnd)
            delta_t_micro = max(delta_t_micro, MNH_TINY)

    # TODO: repeat for each specy (r, i, s, g)
    # We stop when the end of the timestep is reached
    with computation(PARALLEL), interval(...):
        llcompute = False if t_micro + delta_t_micro > TSTEP else llcompute

    # TODO : mnh lastcall
    # if TSTEP_TS != 0:
    #     with computation(PARALLEL), interval(...):

    # 4.7 New values of variables for next iteration
    with computation(PARALLEL), interval(...):
        theta_t += theta_a_tnd * delta_t_micro + theta_b

    with computation(PARALLEL), interval(...):
        if ri_t > 0 and ldmicro:
            t_micro += delta_t_micro
            ci_t = 0
