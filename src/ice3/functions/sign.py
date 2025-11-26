# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import function

@function
def sign(a: "float", b: "float") -> "float":
    if a >= 0.0:
        sign_b = 1 * abs(a)
    elif a < 0.0:
        sign_b = -1 * abs(a)


    return sign_b