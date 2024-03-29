# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field
from ifs_physics_common.framework.stencil import (
    stencil_collection,
    interval,
    computation,
    PARALLEL,
    __externals__,
)
from ifs_physics_common.utils.f2py import ported_method

from ice3_gt4py.functions.ice_adjust import (
    cph,
    sublimation_latent_heat,
    vaporisation_latent_heat,
)
from ice3_gt4py.functions.temperature import theta2temperature


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=215,
    to_line=221,
)
@stencil_collection("ice4_stepping_tmicro_init")
def ice4_stepping_init_tmicro(t_micro: Field["float"], ldmicro: Field["bool"]):
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

    from __externals__ import TSTEP_TS

    with computation(PARALLEL), interval(...):
        t_soft = t_micro


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_stepping.F90",
    from_line=230,
    to_line=237,
)
@stencil_collection("ice4_stepping_ldcompute")
def ice4_stepping_ldcompute(
    sub_time: Field["float"],
    ldcompute: Field["bool"],
):
    """Compute ldcompute mask

    Args:
        sub_time (Field[bool]): time in sub_step (from 0 to tstep_ts)
        ldcompute (Field[bool]): mask of computations
        tstep: time step
    """
    from __externals__ import TSTEP

    with computation(PARALLEL), interval(...):
        ldcompute = True if sub_time < TSTEP else False


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
        specific_heat = cph(rv_t, rc_t, ri_t, rr_t, rs_t, rg_t)
        t = theta2temperature(t, th_t, exn)
        ls_fact = sublimation_latent_heat(t) / specific_heat
        lv_fact = vaporisation_latent_heat(t) / specific_heat
