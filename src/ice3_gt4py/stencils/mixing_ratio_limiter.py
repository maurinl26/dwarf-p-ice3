# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    Field,
    __externals__,
    computation,
    interval,
    PARALLEL,
)
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method

from ice3_gt4py.functions.sign import sign


############################ MRSTEP != 0 ################################
@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=346,
    to_line=388,
)
@stencil_collection("mixing_ratio_step_limiter")
def mixing_ratio_step_limiter(
    rc_0r_t: Field["float"],
    rr_0r_t: Field["float"],
    ri_0r_t: Field["float"],
    rs_0r_t: Field["float"],
    rg_0r_t: Field["float"],
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
    ldcompute: Field["bool"],
):
    """_summary_

    Args:
        rc_0r_t (Field[float]): _description_
        rr_0r_t (Field[float]): _description_
        ri_0r_t (Field[float]): _description_
        rs_0r_t (Field[float]): _description_
        rg_0r_t (Field[float]): _description_
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
        time_threshold_tmp (Field[float]): _description_
    """
    from __externals__ import C_RTMIN, G_RTMIN, I_RTMIN, MRSTEP, R_RTMIN, S_RTMIN

    ############## (c) ###########
    # l356
    with computation(PARALLEL), interval(...):
        # TODO: add condition on LL_ANY_ITER
        time_threshold_tmp = (
            (sign(1, rc_tnd_a) * MRSTEP + rc_0r_t - rc_t - rc_b)
            if abs(rc_tnd_a) > 1e-20
            else -1
        )

    # l363
    with computation(PARALLEL), interval(...):
        if (
            time_threshold_tmp >= 0
            and time_threshold_tmp < delta_t_micro
            and (rc_t > C_RTMIN or rc_tnd_a > 0)
        ):
            delta_t_micro = min(delta_t_micro, time_threshold_tmp)
            ldcompute = False

            # Translation note : ldcompute is LLCOMPUTE in mode_ice4_stepping.F90

    # l370
    # Translation note : l370 to l378 in mode_ice4_stepping. F90 contracted in a single stencil
    with computation(PARALLEL), interval(...):
        r_b_max = abs(rr_b)

    ################ (r) #############
    with computation(PARALLEL), interval(...):
        time_threshold_tmp = (
            (sign(1, rr_tnd_a) * MRSTEP + rr_0r_t - rr_t - rr_b)
            if abs(rr_tnd_a) > 1e-20
            else -1
        )

    with computation(PARALLEL), interval(...):
        if (
            time_threshold_tmp >= 0
            and time_threshold_tmp < delta_t_micro
            and (rr_t > R_RTMIN or rr_tnd_a > 0)
        ):
            delta_t_micro = min(delta_t_micro, time_threshold_tmp)
            ldcompute = False

    with computation(PARALLEL), interval(...):
        r_b_max = max(r_b_max, abs(rr_b))

    ################ (i) #############
    with computation(PARALLEL), interval(...):
        time_threshold_tmp = (
            (sign(1, ri_tnd_a) * MRSTEP + ri_0r_t - ri_t - ri_b)
            if abs(ri_tnd_a) > 1e-20
            else -1
        )

    with computation(PARALLEL), interval(...):
        if (
            time_threshold_tmp >= 0
            and time_threshold_tmp < delta_t_micro
            and (rc_t > I_RTMIN or ri_tnd_a > 0)
        ):
            delta_t_micro = min(delta_t_micro, time_threshold_tmp)
            ldcompute = False

    with computation(PARALLEL), interval(...):
        r_b_max = max(r_b_max, abs(ri_b))

    ################ (s) #############
    with computation(PARALLEL), interval(...):
        time_threshold_tmp = (
            (sign(1, rs_tnd_a) * MRSTEP + rs_0r_t - rs_t - rs_b)
            if abs(rs_tnd_a) > 1e-20
            else -1
        )

    with computation(PARALLEL), interval(...):
        if (
            time_threshold_tmp >= 0
            and time_threshold_tmp < delta_t_micro
            and (rs_t > S_RTMIN or rs_tnd_a > 0)
        ):
            delta_t_micro = min(delta_t_micro, time_threshold_tmp)
            ldcompute = False

    with computation(PARALLEL), interval(...):
        r_b_max = max(r_b_max, abs(rs_b))

    ################ (g) #############
    with computation(PARALLEL), interval(...):
        time_threshold_tmp = (
            (sign(1, rg_tnd_a) * MRSTEP + rg_0r_t - rg_t - rg_b)
            if abs(rg_tnd_a) > 1e-20
            else -1
        )

    with computation(PARALLEL), interval(...):
        if (
            time_threshold_tmp >= 0
            and time_threshold_tmp < delta_t_micro
            and (rg_t > G_RTMIN or rg_tnd_a > 0)
        ):
            delta_t_micro = min(delta_t_micro, time_threshold_tmp)
            ldcompute = False

    with computation(PARALLEL), interval(...):
        r_b_max = max(r_b_max, abs(rg_b))  # (g)

    # Limiter on max mixing ratio
    with computation(PARALLEL), interval(...):
        if r_b_max > MRSTEP:
            delta_t_micro = 0
            ldcompute = False
