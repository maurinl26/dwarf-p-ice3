# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, interval, computation, PARALLEL
from ifs_physics_common.framework.stencil import stencil_collection

from ifs_physics_common.utils.f2py import ported_method

from ice3_gt4py.functions.ice_adjust import (
    constant_pressure_heat_capacity,
    sublimation_latent_heat,
    vaporisation_latent_heat,
)
from ice3_gt4py.functions.temperature import theta2temperature
from ice3_gt4py.functions.sign import sign
from ice3_gt4py.functions.stepping import mixing_ratio_step_limiter



@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=215,
    to_line=221,
)
@stencil_collection("ice4_stepping_tmicro_init")
def ice4_stepping_tmicro_init(t_micro: Field["float"], ldmicro: Field["bool"]):
    """Initialise t_soft with value of t_micro after each loop
    on LSOFT condition.

    Args:
        t_micro (Field[float]): time for microphsyics loops
        ldmicro (Field[bool]): microphsyics activation mask
    """

    from __externals__ import TSTEP

    # 4.4 Temporal loop
    with computation(PARALLEL), interval(...):
        t_micro = 0 if ldmicro else TSTEP


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=225,
    to_line=228,
)
@stencil_collection("ice4_stepping_tsoft_init")
def ice4_stepping_init_tsoft(t_micro: Field["float"], t_soft: Field["float"]):
    """Initialise t_soft with value of t_micro after each loop
    on LSOFT condition.

    Args:
        t_micro (Field[float]): time for microphsyics loops
        t_soft (Field[float]): time for lsoft blocks loops
    """

    with computation(PARALLEL), interval(...):
        t_soft = t_micro


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=244,
    to_line=254,
)
@stencil_collection("ice4_stepping_heat")
def ice4_stepping_heat(
    rv_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    exn: Field["float"],
    th_t: Field["float"],
    ls_fact: Field["float"],
    lv_fact: Field["float"],
    t: Field["float"],
):
    """Compute and convert heat variables before computations

    Args:
        rv_t (Field[float]): vapour mixing ratio
        rc_t (Field[float]): cloud droplet mixing ratio
        rr_t (Field[float]): rain m.r.
        ri_t (Field[float]): ice m.r.
        rs_t (Field[float]): snow m.r.
        rg_t (Field[float]): graupel m.r.
        exn (Field[float]): exner pressure
        th_t (Field[float]): potential temperature
        ls_fact (Field[float]): sublimation latent heat over heat capacity
        lv_fact (Field[float]): vapourisation latent heat over heat capacity
        t (Field[float]): temperature
    """
    with computation(PARALLEL), interval(...):
        specific_heat = constant_pressure_heat_capacity(rv_t, rc_t, ri_t, rr_t, rs_t, rg_t)
        t = theta2temperature(th_t, exn)
        ls_fact = sublimation_latent_heat(t) / specific_heat
        lv_fact = vaporisation_latent_heat(t) / specific_heat


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=230,
    to_line=237,
)
@stencil_collection("ice4_stepping_ldcompute_init")
def ice4_stepping_ldcompute_init(ldcompute: Field["bool"], t_micro: Field["float"]):
    """Initialize ldcompute mask

    Args:
        ldcompute (Field[bool]): mask to compute microphysical species
        t_micro (Field[float]): microphysical time-step
    """

    from __externals__ import TSTEP

    with computation(PARALLEL), interval(...):
        ldcompute = True if t_micro < TSTEP else False
        

############################ MRSTEP != 0 ################################
@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=346,
    to_line=388,
)
@stencil_collection("ice4_mixing_ratio_step_limiter")
def ice4_mixing_ratio_step_limiter(
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
    """Step limiter for processes, based on tendencies thresholds.

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


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=290,
    to_line=332,
)
@stencil_collection("ice4_step_limiter")
def ice4_step_limiter(
    exn: Field["float"],
    theta_t: Field["float"],
    theta_a_tnd: Field["float"],
    theta_b: Field["float"],
    theta_ext_tnd: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    rc_a_tnd: Field["float"],
    rr_a_tnd: Field["float"],
    ri_a_tnd: Field["float"],
    rs_a_tnd: Field["float"],
    rg_a_tnd: Field["float"],
    rc_ext_tnd: Field["float"],
    rr_ext_tnd: Field["float"],
    ri_ext_tnd: Field["float"],
    rs_ext_tnd: Field["float"],
    rg_ext_tnd: Field["float"],
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
):
    from __externals__ import (
        C_RTMIN,
        G_RTMIN,
        I_RTMIN,
        MNH_TINY,
        R_RTMIN,
        S_RTMIN,
        TSTEP,
        TSTEP_TS,
        TT,
    )

    # Adding externals tendencies
    with computation(PARALLEL), interval(...):
        theta_a_tnd += theta_ext_tnd
        rc_a_tnd += rc_ext_tnd
        rr_a_tnd += rr_ext_tnd
        ri_a_tnd += ri_ext_tnd
        rs_a_tnd += rs_ext_tnd
        rg_a_tnd += rg_ext_tnd

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
    with computation(PARALLEL), interval(...):
        # (c)
        delta_t_micro = mixing_ratio_step_limiter(
            rc_a_tnd, rc_b, rc_t, delta_t_micro, C_RTMIN, MNH_TINY
        )
        # (r)
        delta_t_micro = mixing_ratio_step_limiter(
        rr_a_tnd, rr_b, rr_t, delta_t_micro, R_RTMIN, MNH_TINY
    )

        # (i)
        delta_t_micro = mixing_ratio_step_limiter(
            ri_a_tnd, ri_b, ri_t, delta_t_micro, I_RTMIN, MNH_TINY
        )

        # (s)
        delta_t_micro = mixing_ratio_step_limiter(
            rs_a_tnd, rs_b, rs_t, delta_t_micro, S_RTMIN, MNH_TINY
        )

        # (g)
        delta_t_micro = mixing_ratio_step_limiter(
            rg_a_tnd, rg_b, rg_t, delta_t_micro, G_RTMIN, MNH_TINY
        )

    # We stop when the end of the timestep is reached
    with computation(PARALLEL), interval(...):
        ldcompute = False if t_micro + delta_t_micro > TSTEP else ldcompute

    # TODO : TSTEP_TS out of the loop
    with computation(PARALLEL), interval(...):
        if TSTEP_TS != 0:
            if t_micro + delta_t_micro > t_soft + delta_t_soft:
                delta_t_micro = t_soft + delta_t_soft - t_micro
                ldcompute = False


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=391,
    to_line=404,
)
@stencil_collection("state_update")
def state_update(
    th_t: Field["float"],
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
    ldmicro: Field["bool"],
    ci_t: Field["float"],
    t_micro: Field["float"],
):
    """Update values of guess of potential temperature and mixing ratios after each step

    Args:
        th_t (Field[float]): potential temperature at t
        theta_b (Field[float]): increase of potential temperature
        theta_tnd_a (Field[float]): potential temperature source (tendency update)
        rc_t (Field[float]): cloud droplets m.r. at t
        rr_t (Field[float]): rain m.r. at t
        ri_t (Field[float]): ice m.r. at t
        rs_t (Field[float]): snow m.r. at t
        rg_t (Field[float]): graupel m.r. at t
        rc_b (Field[float]): increase of cloud droplets m.r.
        rr_b (Field[float]): increase of rain m.r.
        ri_b (Field[float]): increase of ice m.r.
        rs_b (Field[float]): increase of snow m.r.
        rg_b (Field[float]): increase of graupel m.r.
        rc_tnd_a (Field[float]): cloud droplets source (tendency update)
        rr_tnd_a (Field[float]): rain source (tendency update)
        ri_tnd_a (Field[float]): ice (tendency update)
        rs_tnd_a (Field[float]): snow (tendency update)
        rg_tnd_a (Field[float]): graupel (tendency update)
        delta_t_micro (Field[float]): timestep for explicit microphysics
        ldmicro (Field[float]): boolean mask for microphsyics computations
        ci_t (Field[float]): concentration of ice at t
    """

    # 4.7 New values of variables for next iteration
    with computation(PARALLEL), interval(...):
        th_t += theta_tnd_a * delta_t_micro + theta_b
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
    th_t: Field["float"],
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
    dt: "float"
):

    with computation(PARALLEL), interval(...):
        if ldmicro:
            th_t -= theta_tnd_ext * dt
            rc_t -= rc_tnd_ext * dt
            rr_t -= rr_tnd_ext * dt
            ri_t -= ri_tnd_ext * dt
            rs_t -= rs_tnd_ext * dt
            rg_t -= rg_tnd_ext * dt

