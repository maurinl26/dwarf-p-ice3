# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

from gt4py.cartesian.gtscript import Field, GlobalTable, floor, function, log, max, min
from ifs_physics_common.utils.f2py import ported_method


@ported_method(
    from_file="PHYEX/src/common/micro/interp_micro.func.h", from_line=5, to_line=124
)
@function
def index_interp_micro_1d(
    zw: Field["float"],
) -> Field["int"]:
    """Compute index in logspace for table

    Args:
        zw (Field[float]): point (x) to compute log index

    Returns:
        Field[float]: floating index in lookup table (index + offset)
    """

    from __externals__ import NGAMINC, RIMINTP1, RIMINTP2

    # Real index for interpolation
    return max(1, min(NGAMINC - 1e-5, RIMINTP1 * log(zw) + RIMINTP2))


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
        floating_index * lookup_table.A[lut_index + 1]
        + (1 - floating_index) * lookup_table.A[lut_index]
    )


######################### Index 2D ###############################
@ported_method(
    from_file="PHYEX/src/common/micro/interp_micro.func.h", from_line=126, to_line=269
)
@function
def index_micro2d_acc_r(lambda_r: Field["float"]) -> Field["float"]:
    """Compute index in logspace for table

    Args:
        zw (Field[float]): point (x) to compute log index

    Returns:
        Field[float]: floating index in lookup table (index + offset)
    """

    from __externals__ import (
        ACCINTP1R,
        ACCINTP2R,
        NACCLBDAR,
    )

    # Real index for interpolation
    return max(1 + 1e-5, min(NACCLBDAR - 1e-5, ACCINTP1R * log(lambda_r) + ACCINTP2R))


@ported_method(
    from_file="PHYEX/src/common/micro/interp_micro.func.h", from_line=126, to_line=269
)
@function
def index_micro2d_acc_s(lambda_s: Field["float"]) -> Field["float"]:
    """Compute index in logspace for table

    Args:
        zw (Field[float]): point (x) to compute log index

    Returns:
        Field[float]: floating index in lookup table (index + offset)
    """

    from __externals__ import (
        ACCINTP1S,
        ACCINTP2S,
        NACCLBDAS,
    )

    return max(1 + 1e-5, min(NACCLBDAS - 1e-5, ACCINTP1S * log(lambda_s) + ACCINTP2S))


################ DRY COLLECTION #####################
# (s) -> (g)
@function
def index_micro2d_dry_g(lambda_g: Field["float"]) -> Field["float"]:
    """Compute index in logspace for table

    Args:
        zw (Field[float]): point (x) to compute log index

    Returns:
        Field[float]: floating index in lookup table (index + offset)
    """

    from __externals__ import (
        DRYINTP1G,
        DRYINTP2G,
        NDRYLBDAG,
    )

    # Real index for interpolation
    return max(1 + 1e-5, min(NDRYLBDAG - 1e-5, DRYINTP1G * log(lambda_g) + DRYINTP2G))


@function
def index_micro2d_dry_s(lambda_s: Field["float"]) -> Field["float"]:
    """Compute index in logspace for table

    Args:
        zw (Field[float]): point (x) to compute log index

    Returns:
        Field[float]: floating index in lookup table (index + offset)
    """

    from __externals__ import (
        DRYINTP1S,
        DRYINTP2S,
        NDRYLBDAS,
    )

    return max(1 + 1e-5, min(NDRYLBDAS - 1e-5, DRYINTP1S * log(lambda_s) + DRYINTP2S))


# (r) -> (g)
@function
def index_micro2d_dry_r(lambda_r: Field["float"]) -> Field["float"]:
    """Compute index in logspace for table

    Args:
        zw (Field[float]): point (x) to compute log index

    Returns:
        Field[float]: floating index in lookup table (index + offset)
    """

    from __externals__ import (
        DRYINTP1R,
        DRYINTP2R,
        NDRYLBDAR,
    )

    # Real index for interpolation
    return max(1 + 1e-5, min(NDRYLBDAR - 1e-5, DRYINTP1R * log(lambda_r) + DRYINTP2R))


### Look up + interpolation


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
        floating_index_r * lookup_table.A[lut_index_s + 1, lut_index_r + 1]
        + (1 - floating_index_r) * lookup_table.A[lut_index_s + 1, lut_index_r]
    ) + (1 - floating_index_s) * (
        floating_index_r * lookup_table.A[lut_index_s, lut_index_r + 1]
        + (1 - floating_index_r) * lookup_table.A[lut_index_s, lut_index_r]
    )
