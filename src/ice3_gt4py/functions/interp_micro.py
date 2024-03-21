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


@ported_method(
    from_file="PHYEX/src/common/micro/interp_micro.func.h", from_line=126, to_line=269
)
@function
def index_interp_micro_2d(
    lambda_r: Field["float"], lambda_s: Field["float"]
) -> Field["float"]:
    """Compute index in logspace for table

    Args:
        zw (Field[float]): point (x) to compute log index

    Returns:
        Field[float]: floating index in lookup table (index + offset)
    """

    from __externals__ import (
        accintp1s,
        accintp2s,
        accintp1r,
        accintp2r,
        nacclbdas,
        nacclbdar,
    )

    # Real index for interpolation
    index_r = max(
        1 + 1e-5, min(nacclbdar) - 1e-5, accintp1r * log(lambda_r) + accintp2r
    )
    index_s = max(
        1 + 1e-5, min(nacclbdas) - 1e-5, accintp1s * log(lambda_s) + accintp2s
    )

    return index_r, index_s


def interp_micro_2d(
    index_r: Field["float"],
    index_s: Field[float],
    lookup_table: GlobalTable[float, (40, 40)],
) -> Field["float"]:
    """Perform 1d interpolation on global table with index

    (40, 40) = (NLBDAR, NBLDAS)

    Args:
        index (Field[float]): index for interpolation table (integer index + offset)
        lookup_table (GlobalTable[float, (80)]): lookup_table for value retrieval

    Returns:
        Field[float]: interpolated value
    """

    lut_index_r, lut_index_s = floor(index_r), floor(index_s)

    floating_index_r, floating_index_s = index_r - lut_index_r, index_s - lut_index_s
    return floating_index_s * (
        floating_index_r * lookup_table.at[lut_index_s + 1, lut_index_r + 1]
        + (1 - floating_index_r) * lookup_table.at[lut_index_s + 1, lut_index_r]
    ) + (1 - floating_index_s) * (
        floating_index_r * lookup_table.at[lut_index_s, lut_index_r + 1]
        + (1 - floating_index_r) * lookup_table.at[lut_index_s, lut_index_r]
    )
