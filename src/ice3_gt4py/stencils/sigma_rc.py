# -*- coding: utf-8 -*-
from gt4py.cartesian.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    interval,
)
from ifs_physics_common.framework.stencil import stencil_collection


# TODO : shift stencil compilation
@stencil_collection("sigrc")
def sigrc_computation(
    q1: Field["float"], sigrc: Field["float"], src_1d: GlobalTable["float", (34)]
):

    from __externals__ import LAMBDA3

    with computation(PARALLEL), interval(...):

        inq1 = floor(
            min(10, max(-22, min(-100, 2 * floor(q1))))
        )  # inner min/max prevents sigfpe when 2*zq1 does not fit dtype_into an "int"
        inc = 2 * q1 - inq1
        sigrc = min(1, (1 - inc) * src_1d.A[inq1] + inc * src_1d.A[inq1 + 1])

        # Transaltion notes : 566 -> 578 HLAMBDA3 = CB
        if __INLINED(LAMBDA3 == 0):
            sigrc *= min(3, max(1, 1 - q1))
