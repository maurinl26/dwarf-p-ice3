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

    index = max(1, min(NGAMINC - 1e-5, RIMINTP1 * log(zw) + RIMINTP2))
    # Real index for interpolation
    return floor(index), index - floor(index)


######################### Index 2D ###############################
@ported_method(
    from_file="PHYEX/src/common/micro/interp_micro.func.h", from_line=126, to_line=269
)
@function
def index_micro2d_acc_r(lambda_r: Field["float"]) -> Tuple["int", "float"]:
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
    index = max(1 + 1e-5, min(NACCLBDAR - 1e-5, ACCINTP1R * log(lambda_r) + ACCINTP2R))
    return floor(index), index - floor(index)


@ported_method(
    from_file="PHYEX/src/common/micro/interp_micro.func.h", from_line=126, to_line=269
)
@function
def index_micro2d_acc_s(lambda_s: Field["float"]) -> Tuple["int", "float"]:
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

    index = max(1 + 1e-5, min(NACCLBDAS - 1e-5, ACCINTP1S * log(lambda_s) + ACCINTP2S))
    return floor(index), index - floor(index)


################ DRY COLLECTION #####################
# (s) -> (g)
@function
def index_micro2d_dry_g(lambda_g: Field["float"]) -> Tuple["int", "float"]:
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
    index = max(1 + 1e-5, min(NDRYLBDAG - 1e-5, DRYINTP1G * log(lambda_g) + DRYINTP2G))
    return floor(index), index - floor(index)


@function
def index_micro2d_dry_s(lambda_s: Field["float"]) -> Tuple["int", "float"]:
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

    index = max(1 + 1e-5, min(NDRYLBDAS - 1e-5, DRYINTP1S * log(lambda_s) + DRYINTP2S))
    return floor(index), index - floor(index)


# (r) -> (g)
@function
def index_micro2d_dry_r(lambda_r: Field["float"]) -> Tuple["int", "float"]:
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
    index = max(1 + 1e-5, min(NDRYLBDAR - 1e-5, DRYINTP1R * log(lambda_r) + DRYINTP2R))
    return floor(index), index - floor(index)
