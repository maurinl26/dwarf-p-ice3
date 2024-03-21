# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple

from gt4py.cartesian.gtscript import Field, GlobalTable, function, log, floor
from ifs_physics_common.utils.f2py import ported_method


@ported_method(
    from_file="PHYEX/src/common/micro/interp_micro.func.h", from_line=5, to_line=124
)
@function
def index_interp_micro_1d(
    zw: Field["float"],
) -> Field["float"]:
    """Compute index in logspace for table

    Args:
        zw (Field[float]): point (x) to compute log index

    Returns:
        Field[float]: floating index in lookup table (index + offset)
    """

    from __externals__ import rimintp1, rimintp2, ngaminc

    # Real index for interpolation
    return max(1, min(ngaminc) - 1e-5, rimintp1 * log(zw) + rimintp2)


def interp_micro_1d(
    index: Field["float"], lookup_table: GlobalTable[float, (80)]
) -> Field["float"]:
    """Perform 1d interpolation on global table with index

    Args:
        index (Field[float]): index for interpolation table (integer index + offset)
        lookup_table (GlobalTable[float, (80)]): lookup_table for value retrieval

    Returns:
        Field[float]: interpolated value
    """

    lut_index = floor(index)
    floating_index = index - lut_index
    return (
        floating_index * lookup_table.at[lut_index + 1]
        + (1 - floating_index) * lookup_table.at[lut_index]
    )
