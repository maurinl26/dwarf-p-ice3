# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field

from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method

from ice3_gt4py.functions.ice_adjust import (
    cph,
    sublimation_latent_heat,
    vaporisation_latent_heat,
)
from ice3_gt4py.functions.temperature import theta2temperature


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=230,
    to_line=237,
)
@stencil_collection("ice4_stepping_ldcompute")
def ice4_stepping_ldcompute(
    sub_time: Field["float"], ldcompute: Field["bool"], tstep: float
):
    """Compute ldcompute mask

    Args:
        sub_time (Field[bool]): time in sub_step (from 0 to tstep_ts)
        ldcompute (Field[bool]): mask of computations
        tstep: time step
    """

    with computation(PARALLEL), interval(...):

        if sub_time < tstep:
            ldcompute = True
        else:
            ldcompute = False


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
    """"""

    from __externals__ import cpd, cpv, Cl, Ci, tt

    with computation(PARALLEL), interval(...):

        specific_heat = cph(rv_t, rc_t, ri_t, rr_t, rs_t, rg_t)
        t = theta2temperature(t, th_t, exn)
        ls_fact = sublimation_latent_heat(t) / specific_heat
        lv_fact = vaporisation_latent_heat(t) / specific_heat


# 4.6 Time integration
@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=244,
    to_line=254,
)
@stencil_collection("ice4_stepping_time_integration")
def ice4_stepping_time_integration(
    ldcompute: Field["bool"],
    time: Field["float"],
    max_time: Field["float"],
    exn: Field["float"],
    theta: Field["float"],
    theta_a: Field["float"],
    theta_b: Field["float"],  # theta stored in ZB
    tstep: "float",
):
    """_summary_

    Args:
        ldcompute (Field[bool]): mask of computation
        time (Field[float]): current time (between 0 and tstep)
        max_time (Field[float]): max remaining step
        tstep (float): time step for dycore
    """
    from __externals__ import tstep_ts, tt

    # l290
    # if we can, we shall use these tenedencies until the end of the time step
    with computation(PARALLEL), interval(...):
        if ldcompute:
            max_time = tstep - time
        else:
            max_time = 0

    # l297
    # TODO : insert LFEEDBACKT
    with computation(PARALLEL), interval(...):
        th_tt = tt / exn
        if (theta - th_tt) * (theta + theta_b - th_tt) < 0:
            max_time = 0

        if abs(theta_a > 1e-20):
            time_threshold = (th_tt - theta_b - theta) / theta_a
            if time_threshold > 0:
                max_time = min(max_time, time_threshold)

    # We stop when the end ot the timestep is reached
    with computation(PARALLEL), interval(...):
        if time + max_time > tstep:
            ldcompute = False
