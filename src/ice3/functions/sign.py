# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import function

@function
def sign(x: float):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    elif x < 0:
        return -1