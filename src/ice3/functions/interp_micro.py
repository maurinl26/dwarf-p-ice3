# -*- coding: utf-8 -*-
"""
Microphysical lookup table interpolation functions.

This module provides GT4Py implementations of index calculation functions
for bilinear interpolation in pre-computed microphysical kernel lookup tables.
These tables store collection efficiencies and other microphysical parameters
as functions of particle size distribution slope parameters (lambda).

The functions compute both integer floor indices and fractional offsets
for bilinear interpolation in 1D and 2D tables.
"""
from __future__ import annotations

from typing import Tuple

from gt4py.cartesian.gtscript import Field, floor, function, log, max, min


@function
def index_interp_micro_1d(
    zw: "float",
) -> Tuple["float", "int"]:
    """
    Compute interpolation index in logarithmic space for 1D lookup tables.
    
    This function calculates the position in a 1D lookup table where
    values are distributed logarithmically. It returns both the floor
    index and the fractional offset for linear interpolation.
    
    Parameters
    ----------
    zw : float
        Input value for which to compute the table index.
        Typically a microphysical parameter like incomplete gamma function argument.
        
    Returns
    -------
    Tuple[float, int]
        A tuple containing:
        - floor(index): Integer floor index in the lookup table
        - offset: Fractional offset from the floor index (0 to 1)
        
    Notes
    -----
    The index is computed as:
    index = RIMINTP1 * ln(zw) + RIMINTP2
    
    where RIMINTP1 and RIMINTP2 are external calibration constants
    that map the logarithmic space to table indices.
    
    The index is clamped to [1, NGAMINC-1e-5] to stay within table bounds.
    
    Source Reference
    ----------------
    Translated from PHYEX/src/common/micro/interp_micro.func.h,
    lines 5-124
    """
    from __externals__ import NGAMINC, RIMINTP1, RIMINTP2

    index = max(1, min(NGAMINC - 1e-5, RIMINTP1 * log(zw) + RIMINTP2))
    # Real index for interpolation
    return floor(index), index - floor(index)


# 2D Interpolation Functions for Accretion Processes

@function
def index_micro2d_acc_r(lambda_r: "float") -> Tuple["int", "float"]:
    """
    Compute rain slope parameter index for accretion lookup tables.
    
    This function calculates the table index for the rain (r) dimension
    in 2D accretion kernel lookup tables. The rain slope parameter lambda_r
    characterizes the rain drop size distribution.
    
    Parameters
    ----------
    lambda_r : float
        Slope parameter of the rain particle size distribution (m⁻¹).
        From the exponential distribution N(D) = N₀ exp(-λᵣD).
        
    Returns
    -------
    Tuple[int, float]
        A tuple containing:
        - floor(index): Integer floor index in the rain dimension
        - offset: Fractional offset from the floor index (0 to 1)
        
    Notes
    -----
    The index is computed as:
    index = ACCINTP1R * ln(λᵣ) + ACCINTP2R
    
    The index is clamped to [1+1e-5, NACCLBDAR-1e-5] to ensure valid
    table lookup bounds.
    
    Used for rain-snow and rain-graupel accretion processes.
    
    Source Reference
    ----------------
    Translated from PHYEX/src/common/micro/interp_micro.func.h,
    lines 126-269
    """
    from __externals__ import (
        ACCINTP1R,
        ACCINTP2R,
        NACCLBDAR,
    )

    # Real index for interpolation
    index = max(1 + 1e-5, min(NACCLBDAR - 1e-5, ACCINTP1R * log(lambda_r) + ACCINTP2R))
    return floor(index), index - floor(index)


@function
def index_micro2d_acc_s(lambda_s: "float") -> Tuple["int", "float"]:
    """
    Compute snow slope parameter index for accretion lookup tables.
    
    This function calculates the table index for the snow (s) dimension
    in 2D accretion kernel lookup tables. The snow slope parameter lambda_s
    characterizes the snow particle size distribution.
    
    Parameters
    ----------
    lambda_s : float
        Slope parameter of the snow particle size distribution (m⁻¹).
        From the exponential distribution N(D) = N₀ exp(-λₛD).
        
    Returns
    -------
    Tuple[int, float]
        A tuple containing:
        - floor(index): Integer floor index in the snow dimension
        - offset: Fractional offset from the floor index (0 to 1)
        
    Notes
    -----
    The index is computed as:
    index = ACCINTP1S * ln(λₛ) + ACCINTP2S
    
    The index is clamped to [1+1e-5, NACCLBDAS-1e-5] to ensure valid
    table lookup bounds.
    
    Used for rain-snow accretion processes.
    
    Source Reference
    ----------------
    Translated from PHYEX/src/common/micro/interp_micro.func.h,
    lines 126-269
    """
    from __externals__ import (
        ACCINTP1S,
        ACCINTP2S,
        NACCLBDAS,
    )

    index = max(1 + 1e-5, min(NACCLBDAS - 1e-5, ACCINTP1S * log(lambda_s) + ACCINTP2S))
    return floor(index), index - floor(index)


# 2D Interpolation Functions for Dry Growth Processes

@function
def index_micro2d_dry_g(lambda_g: "float") -> Tuple["int", "float"]:
    """
    Compute graupel slope parameter index for dry growth lookup tables.
    
    This function calculates the table index for the graupel (g) dimension
    in 2D dry growth kernel lookup tables. The graupel slope parameter lambda_g
    characterizes the graupel particle size distribution.
    
    Parameters
    ----------
    lambda_g : float
        Slope parameter of the graupel particle size distribution (m⁻¹).
        From the exponential distribution N(D) = N₀ exp(-λ_gD).
        
    Returns
    -------
    Tuple[int, float]
        A tuple containing:
        - floor(index): Integer floor index in the graupel dimension
        - offset: Fractional offset from the floor index (0 to 1)
        
    Notes
    -----
    The index is computed as:
    index = DRYINTP1G * ln(λ_g) + DRYINTP2G
    
    The index is clamped to [1+1e-5, NDRYLBDAG-1e-5] to ensure valid
    table lookup bounds.
    
    Used for both snow-graupel and rain-graupel dry growth collection.
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
def index_micro2d_dry_s(lambda_s: "float") -> Tuple["int", "float"]:
    """
    Compute snow slope parameter index for dry growth lookup tables.
    
    This function calculates the table index for the snow (s) dimension
    in 2D dry growth kernel lookup tables used in snow-graupel interactions.
    
    Parameters
    ----------
    lambda_s : float
        Slope parameter of the snow particle size distribution (m⁻¹).
        From the exponential distribution N(D) = N₀ exp(-λₛD).
        
    Returns
    -------
    Tuple[int, float]
        A tuple containing:
        - floor(index): Integer floor index in the snow dimension
        - offset: Fractional offset from the floor index (0 to 1)
        
    Notes
    -----
    The index is computed as:
    index = DRYINTP1S * ln(λₛ) + DRYINTP2S
    
    The index is clamped to [1+1e-5, NDRYLBDAS-1e-5] to ensure valid
    table lookup bounds.
    
    Used for snow-graupel dry growth collection processes.
    """
    from __externals__ import (
        DRYINTP1S,
        DRYINTP2S,
        NDRYLBDAS,
    )

    index = max(1 + 1e-5, min(NDRYLBDAS - 1e-5, DRYINTP1S * log(lambda_s) + DRYINTP2S))
    return floor(index), index - floor(index)


@function
def index_micro2d_dry_r(lambda_r: "float") -> Tuple["int", "float"]:
    """
    Compute rain slope parameter index for dry growth lookup tables.
    
    This function calculates the table index for the rain (r) dimension
    in 2D dry growth kernel lookup tables used in rain-graupel interactions.
    
    Parameters
    ----------
    lambda_r : float
        Slope parameter of the rain particle size distribution (m⁻¹).
        From the exponential distribution N(D) = N₀ exp(-λᵣD).
        
    Returns
    -------
    Tuple[int, float]
        A tuple containing:
        - floor(index): Integer floor index in the rain dimension
        - offset: Fractional offset from the floor index (0 to 1)
        
    Notes
    -----
    The index is computed as:
    index = DRYINTP1R * ln(λᵣ) + DRYINTP2R
    
    The index is clamped to [1+1e-5, NDRYLBDAR-1e-5] to ensure valid
    table lookup bounds.
    
    Used for rain-graupel dry growth collection processes when rain
    droplets are collected by graupel in dry growth mode (temperature
    below freezing but insufficient liquid water for wet growth).
    """
    from __externals__ import (
        DRYINTP1R,
        DRYINTP2R,
        NDRYLBDAR,
    )

    # Real index for interpolation
    index = max(1 + 1e-5, min(NDRYLBDAR - 1e-5, DRYINTP1R * log(lambda_r) + DRYINTP2R))
    return floor(index), index - floor(index)
