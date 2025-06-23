# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    Field,
    computation,
    PARALLEL,
    interval,
)
from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection("multiply_ab2c")
def multiply_ab2c(
    a: Field["float"],
    b: Field["float"],
    c: Field["float"]
):
    """Multiplies a and b to give c."""

    with computation(PARALLEL), interval(...):
        c = a * b
        
@stencil_collection("double_a")
def double_a(
    a: Field["float"],
    c: Field["float"]
):
    """Returns the double of a

    Args:
        a (Field["float"]): input field
        c (Field["float"]): double of input field
    """
    with computation(PARALLEL), interval(...):
        c = 2.0 * a
        
@stencil_collection("multioutput")
def multioutput(
    a: Field["float"],
    b: Field["float"],
    c: Field["float"]
):
    with computation(PARALLEL), interval(...):
        b = 2.0 * a
        c = 3.0 * a

