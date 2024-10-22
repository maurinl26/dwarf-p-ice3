# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    Field,
    exp,
    log,
    computation,
    PARALLEL,
    interval,
    __externals__,
)
from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection("multiply_ab2c")
def multiply_ab2c(
    a: Field["float"],
    b: Field["float"],
    c: Field["float"]
):
    """Multiplies a and b to give c."""

    # 4.2 compute the autoconversion of r_c for r_r : RCAUTR
    with computation(PARALLEL), interval(...):
        c = a * b

