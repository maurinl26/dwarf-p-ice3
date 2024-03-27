# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field

from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method

from ice3_gt4py.functions.sign import sign
from ice3_gt4py.functions.stepping import mixing_ratio_step_limiter


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
    time_threshold_tmp: Field["float"],
    r_b_max: Field["float"],
):
    from __externals__ import (
        TSTEP,
        LFEEDBACKT,
        TT,
        C_RTMIN,
        R_RTMIN,
        I_RTMIN,
        S_RTMIN,
        G_RTMIN,
        MNH_TINY,
        TSTEP_TS,
        XMRSTEP,
    )

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
    # (c)
    with computation(PARALLEL), interval(...):
        delta_t_micro = mixing_ratio_step_limiter(
            rc_a_tnd, rc_b, rc_t, delta_t_micro, C_RTMIN, MNH_TINY
        )
    # (r)
    with computation(PARALLEL), interval(...):
        delta_t_micro = mixing_ratio_step_limiter(
            rr_a_tnd, rr_b, rr_t, delta_t_micro, R_RTMIN, MNH_TINY
        )
    # (i)
    with computation(PARALLEL), interval(...):
        delta_t_micro = mixing_ratio_step_limiter(
            ri_a_tnd, ri_b, ri_t, delta_t_micro, I_RTMIN, MNH_TINY
        )
    # (s)
    with computation(PARALLEL), interval(...):
        delta_t_micro = mixing_ratio_step_limiter(
            rs_a_tnd, rs_b, rs_t, delta_t_micro, S_RTMIN, MNH_TINY
        )
    # (g)
    with computation(PARALLEL), interval(...):
        delta_t_micro = mixing_ratio_step_limiter(
            rg_a_tnd, rg_b, rg_t, delta_t_micro, G_RTMIN, MNH_TINY
        )

    # We stop when the end of the timestep is reached
    with computation(PARALLEL), interval(...):
        llcompute = False if t_micro + delta_t_micro > TSTEP else llcompute

    # TODO : TSTEP_TS out of the loop
    with computation(PARALLEL), interval(...):
        if TSTEP_TS != 0:
            if t_micro + delta_t_micro > t_soft + delta_t_soft:
                delta_t_micro = t_soft + delta_t_soft - t_micro
                ldcompute = False


############################ MRSTEP != 0 ################################
@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=346,
    to_line=388,
)
@stencil_collection("mixing_ratio_limiter")
def state_update(
    rc_0r_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    rc_b: Field["float"],
    rr_b: Field["float"],
    ri_b: Field["float"],
    rs_b: Field["float"],
    rg_b: Field["float"],
    rc_tnd_a: Field["float"],
    rr_tnd_a: Field["float"],
    ri_tnd_a: Field["float"],
    rs_tnd_a: Field["float"],
    rg_tnd_a: Field["float"],
    delta_t_micro: Field["float"],
    ldmicro: Field["float"],
    time_threshold_tmp: Field["float"],
):

    from __externals__ import MRSTEP

    # TODO:iteration of species
    # l356
    with computation(PARALLEL), interval(...):
        # TODO: add condition on LL_ANY_ITER
        time_threshold_tmp = (
            (sign(1, rc_tnd_a) * MRSTEP + rc_0r_t - rc_t - rc_b)
            if abs(rc_a_tnd) > 1e-20
            else -1
        )

    # l363
    with computation(PARALLEL), interval(...):
        if (
            time_threshold_tmp >= 0
            and time_threshold_tmp < delta_t_micro
            and (rc_t > C_RTMIN or rc_a_tnd > 0)
        ):
            delta_t_micro = min(delta_t_micro, time_threshold_tmp)
            ldcompute = False

    # l370
    # Translation note : l370 to l378 in mode_ice4_stepping. F90 contracted in a single stencil
    with computation(PARALLEL), interval(...):
        r_b_max = abs(rc_b)  # (c)
        r_b_max = max(r_b_max, abs(rr_b))  # (r)
        r_b_max = max(r_b_max, abs(ri_b))  # (i)
        r_b_max = max(r_b_max, abs(rs_b))  # (s)
        r_b_max = max(r_b_max, abs(rg_b))  # (g)

    # TODO : insert LLCPZ0RT
    # l381
    with computation(PARALLEL), interval(...):
        if r_b_max > XMRSTEP:
            delta_t_micro = 0
            ldcompute = False


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=391,
    to_line=404,
)
@stencil_collection("state_update")
def state_update(
    theta_t: Field["float"],
    theta_b: Field["float"],
    theta_tnd_a: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    rc_b: Field["float"],
    rr_b: Field["float"],
    ri_b: Field["float"],
    rs_b: Field["float"],
    rg_b: Field["float"],
    rc_tnd_a: Field["float"],
    rr_tnd_a: Field["float"],
    ri_tnd_a: Field["float"],
    rs_tnd_a: Field["float"],
    rg_tnd_a: Field["float"],
    delta_t_micro: Field["float"],
    ldmicro: Field["float"],
    ci_t: Field["float"],
):
    """Update values of guess of potential temperature and mixing ratios after each step

    Args:
        theta_t (Field[float]): _description_
        theta_b (Field[float]): _description_
        theta_tnd_a (Field[float]): _description_
        rc_t (Field[float]): _description_
        rr_t (Field[float]): _description_
        ri_t (Field[float]): _description_
        rs_t (Field[float]): _description_
        rg_t (Field[float]): _description_
        rc_b (Field[float]): _description_
        rr_b (Field[float]): _description_
        ri_b (Field[float]): _description_
        rs_b (Field[float]): _description_
        rg_b (Field[float]): _description_
        rc_tnd_a (Field[float]): _description_
        rr_tnd_a (Field[float]): _description_
        ri_tnd_a (Field[float]): _description_
        rs_tnd_a (Field[float]): _description_
        rg_tnd_a (Field[float]): _description_
        delta_t_micro (Field[float]): _description_
        ldmicro (Field[float]): _description_
        ci_t (Field[float]): _description_
    """

    # 4.7 New values of variables for next iteration
    with computation(PARALLEL), interval(...):
        theta_t += theta_tnd_a * delta_t_micro + theta_b
        rc_t += rc_tnd_a * delta_t_micro + rc_b
        rr_t += rr_tnd_a * delta_t_micro + rr_b
        ri_t += ri_tnd_a * delta_t_micro + ri_b
        rs_t += rs_tnd_a * delta_t_micro + rs_b
        rg_t += rg_tnd_a * delta_t_micro + rg_b

    with computation(PARALLEL), interval(...):
        if ri_t <= 0 and ldmicro:
            t_micro += delta_t_micro
            ci_t = 0

    # 4.8 Mixing ratio change due to each process
    # Translation note : l409 to 431 have been omitted since no budget calculations


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=440,
    to_line=452,
)
@stencil_collection("external_tendencies_update")
def external_tendencies_update(
    theta_t: Field["float"],
    theta_tnd_ext: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    rc_tnd_ext: Field["float"],
    rr_tnd_ext: Field["float"],
    ri_tnd_ext: Field["float"],
    rs_tnd_ext: Field["float"],
    rg_tnd_ext: Field["float"],
    ldmicro: Field["bool"],
):
    from __externals__ import TSTEP

    with computation(PARALLEL), interval(...):
        if ldmicro:
            theta_t -= theta_tnd_ext * TSTEP
            rc_t -= rc_tnd_ext * TSTEP
            rr_t -= rr_tnd_ext * TSTEP
            ri_t -= ri_tnd_ext * TSTEP
            rs_t -= rs_tnd_ext * TSTEP
            rg_t -= rg_tnd_ext * TSTEP
