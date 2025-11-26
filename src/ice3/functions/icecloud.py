# -*- coding: utf-8 -*-
"""
Subgrid-scale ice supersaturation calculations.

This module provides GT4Py implementations for computing subgrid-scale
fractions of supersaturation with respect to ice in clouds. This accounts
for spatial variability of relative humidity within a grid cell that is
not resolved by the model's horizontal and vertical resolution.
"""
from __future__ import annotations

from gt4py.cartesian.gtscript import IJ, Field, function, sqrt

from ice3.functions.tiwmx import esati, esatw


@function
def icecloud(
    p: Field[IJ, float],
    z: Field[IJ, float],
    dz: Field[IJ, float],
    t: Field[IJ, float],
    r: Field[IJ, float],
    pblh: float,
    wcld: Field[IJ, float],
    w2d: float,
    sifrc: Field[IJ, float],
    ssio: Field[IJ, float],
    ssiu: Field[IJ, float],
    w2d_out: Field[IJ, float],
    rsi: Field[IJ, float],
):
    """
    Calculate subgrid-scale supersaturation fraction with respect to ice.
    
    This function computes the fraction of a grid cell that is
    supersaturated with respect to ice, accounting for subgrid-scale
    variability in relative humidity. It assumes a linear distribution
    of relative humidity within the grid cell, with variability that
    depends on model level thickness and boundary layer characteristics.
    
    Parameters
    ----------
    p : Field[IJ, float]
        Pressure at model level (Pa).
    z : Field[IJ, float]
        Model level height above ground (m).
    dz : Field[IJ, float]
        Model level thickness (m).
    t : Field[IJ, float]
        Temperature (K).
    r : Field[IJ, float]
        Water vapor mixing ratio (kg/kg).
    pblh : float
        Planetary boundary layer height (m).
        Negative values indicate unknown boundary layer height.
    wcld : Field[IJ, float]
        Water and mixed-phase cloud cover fraction (0-1).
        Negative values indicate unknown cloud cover.
    w2d : float
        Ratio between ice crystal concentration in dry vs wet regions.
        Used to maintain consistency between grid-box mean and subgrid parts.
    sifrc : Field[IJ, float]
        Output: Subgrid-scale fraction with supersaturation w.r.t. ice (0-1).
    ssio : Field[IJ, float]
        Output: Super-saturation w.r.t. ice in the supersaturated fraction.
    ssiu : Field[IJ, float]
        Output: Sub-saturation w.r.t. ice in the subsaturated fraction.
    w2d_out : Field[IJ, float]
        Output: Consistency factor between grid-box mean and subgrid parts.
    rsi : Field[IJ, float]
        Output: Saturation mixing ratio over ice (kg/kg).
        
    Returns
    -------
    Tuple[Field[IJ, float], ...]
        Returns (sifrc, ssio, ssiu, w2d_out, rsi)
        
    Notes
    -----
    Algorithm Overview:
    1. Compute relative humidity w.r.t. water (rhw) and ice (rhi)
    2. Estimate subgrid-scale RH variability from:
       - Horizontal variability (sigmax, sigmay)
       - Vertical variability (sigmaz or drhdz)
       - Boundary layer vs free troposphere differences
    3. Determine the fraction of the grid cell that is supersaturated
    4. Compute mean supersaturation in saturated and unsaturated parts
    
    Assumptions:
    - Linear distribution of relative humidity
    - RH variability scales with grid spacing and layer thickness
    - Enhanced variability in the boundary layer
    - Reduced variability aloft where vertical RH gradients dominate
    
    External Parameters:
    - TSTEP: Model time step (s)
    - LVTT: Latent heat of vaporization at triple point (J/kg)
    - GRAVITY0: Gravitational acceleration (m/s²)
    - RD: Gas constant for dry air (J/(kg·K))
    - CPD: Specific heat of dry air at constant pressure (J/(kg·K))
    - EPSILO: Ratio of molecular weights (Mw/Md ≈ 0.622)
    
    Physical Constants:
    - sigmax = sigmay = 3×10⁻⁴: Assumed horizontal RH variability
    - sigmaz = 1×10⁻²: Assumed vertical RH variability
    - xdist = ydist = 2500 m: Assumed grid spacing
    
    Special Cases:
    - T > 273.1 K: No ice supersaturation computed (all liquid)
    - r ≤ 0: No moisture, no supersaturation
    - e_sat_i ≥ 0.5·p: Pressure too low for valid ice saturation
    
    References
    ----------
    This implements a subgrid-scale cloud scheme similar to those
    used in climate models to represent unresolved cloud variability.
    The method allows for partial cloud cover and accounts for
    supersaturation needed for ice crystal nucleation and growth.
    """
    from __externals__ import TSTEP, LVTT, GRAVITY0, RD, CPD, EPSILO

    # Assumed RH variability in horizontal and vertical directions
    sigmax = 3e-4  # assumed rh variation in x axis direction
    sigmay = sigmax  # assumed rh variation in y axis direction
    sigmaz = 1e-2  # assumed rh variation in vertical

    xdist = 2500  # gridsize in x axis (m)
    ydist = xdist  # gridsize in y axis (m)

    # Ensure non-negative mixing ratio
    zr = max(0, r[0, 0, 0] * TSTEP)
    sifrc = 0
    
    # Compute vapor pressure from mixing ratio
    a = zr[0, 0, 0] * p[0, 0, 0] / (EPSILO + zr)

    # Compute relative humidity w.r.t. water and ice
    rhw = a / esatw(t[0, 0, 0])
    rhi = a / esati(t[0, 0, 0])
    
    # Ratio of saturation vapor pressures (water/ice)
    i2w = esatw(t[0, 0, 0]) / esati(t[0, 0, 0])

    # Initialize supersaturation values
    ssiu = min(i2w, rhi)
    ssio = ssiu[0, 0, 0]
    w2d = 1

    # Check if ice supersaturation is physically possible
    if t[0, 0, 0] > 273.1 or r <= 0 or esati(t[0, 0, 0]) >= p[0, 0, 0] * 0.5:
        # Conditions not suitable for ice supersaturation
        ssiu -= 1
        ssio = ssiu
        if wcld >= 0:
            sifrc = wcld[0, 0, 0]

    # Compute vertical RH gradient
    rhin = max(0.05, min(1, rhw))
    drhdz = (
        rhin * GRAVITY0 / (t[0, 0, 0] * RD) * (EPSILO * LVTT / (CPD * t[0, 0, 0]) - 1)
    )

    # Determine if we're in boundary layer or free troposphere
    zz = 0
    if pblh < 0:
        # Boundary layer height unknown - use height-based estimate
        zz = min(1, max(0, z[0, 0, 0] * 0.001))
    elif z[0, 0, 0] > 35 and z[0, 0, 0] > pblh:
        # Above boundary layer
        zz = 1

    # Compute total RH variability from horizontal and vertical components
    rhdist = sqrt(
        xdist * sigmax**2
        + ydist * sigmay**2
        + (1 - zz) * (dz[0, 0, 0] * drhdz) ** 2
        + zz * dz[0, 0, 0] * sigmaz**2
    )

    # Normalize RH variation in free troposphere
    if zz > 0.1:
        rhdist /= 1 + rhdist

    # Compute RH threshold for cloud formation
    rhlim = max(0.5, min(0.99, 1 - 0.5 * rhdist))
    
    # Estimate cloud cover if not provided
    if wcld < 0:
        rhdif = 1 - sqrt(max(0, (1 - rhw) / (1 - rhlim)))
        wcld = min(1, max(rhdif, 0))
    else:
        wcld = wcld[0, 0, 0]

    sifrc = wcld

    # Compute ice supersaturation threshold
    rhlimice = 1 + i2w * (rhlim - 1)
    
    if rhlim < 0.999:
        rhliminv = 1 / (1 - rhlimice)
        rhdif = (rhi - rhlimice) * rhliminv

        if wcld == 0:
            # No pre-existing cloud - compute supersaturated fraction
            sifrc = min(1, 0.5 * max(0, rhdif))
        else:
            # Adjust for existing cloud cover
            sifrc = min(1, a * 0.5 / (1 - rhlim))
            sifrc = min(1, wcld + sifrc)

    # Compute mean supersaturation in each fraction
    if sifrc > 0.01:
        ssiu = min(1, a * 0.5 / (1 - rhlim))
        ssio = (rhi - (1 - sifrc) * ssiu) / sifrc
    else:
        sifrc = 0
        a = min(1, a * 0.5 / (1 - rhlim))
        ssiu = max(0, sifrc + rhlimice * (1 - sifrc) + 2 * a)

    # Convert to supersaturation (Si - 1)
    ssiu -= 1
    ssio -= 1

    # Compute consistency factor if needed
    if w2d > 1:
        w2d_out = 1 / (1 - (1 + w2d) * sifrc)

    return sifrc, ssio, ssiu, w2d_out, rsi
