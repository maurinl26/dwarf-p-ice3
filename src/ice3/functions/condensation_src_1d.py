# -*- coding: utf-8 -*-
from gt4py.cartesian.gtscript import function, GlobalTable


# TODO: change as a global table
@function
def src_1d(inq: int, src: GlobalTable[float, 34]) -> float:
    """Retrieve value for src_1d table.

    Args:
        inq (int): increment

    Returns:
        float: src_1d value on increment index
    """

    # TODO: specify src as a global table
    return src.A[inq]
