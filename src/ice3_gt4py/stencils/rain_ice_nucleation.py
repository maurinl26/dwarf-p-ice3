# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    Field,
    computation,
    PARALLEL,
    interval,
    __externals__,
)
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=452, to_line=463
)
@stencil_collection("ice4_nucleation_pre_processing")
def ice4_nucleation_pre_processing(
    ldmicro: Field["bool"],
    lw3d: Field["bool"],
    ci_t: Field["float"],
    w3d: Field["float"],
    ls_fact: Field["float"],
    exn: Field["float"],
):

    with computation(PARALLEL), interval(...):
        lw3d = not ldmicro
        if lw3d:
            w3d = ls_fact / exn
            ci_t = 0


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=473, to_line=477
)
@stencil_collection("ice4_nucleation_post_processing")
def ice4_nucleation_pre_processing(
    rvs: Field["float"],
    rvheni: Field["float"],
):
    """rvheni limiter (heterogeneous nucleation of ice)

    Args:
        rvs (Field[float]): source of vapour
        rvheni (Field[float]): vapour mr change due to heni
    """

    from __externals__ import TSTEP

    with computation(PARALLEL), interval(...):
        rvheni = min(rvs, rvheni / TSTEP)
