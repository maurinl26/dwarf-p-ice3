# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import function

@function
def sign(x: float):
    if x > 0:
        sign_x = 1
    elif x == 0:
        sign_x = 0
    elif x < 0:
        sign_x = -1

    return sign_x