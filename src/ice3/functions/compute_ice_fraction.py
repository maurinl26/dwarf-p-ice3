# -*- coding: utf-8 -*-
"""
Ice fraction computation for mixed-phase cloud microphysics.

This module provides GT4Py implementations for calculating the fraction
of condensate that exists as ice versus liquid water in mixed-phase clouds.
The ice fraction depends on temperature and is controlled by different
parameterization modes.
"""
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function, __INLINED


@function
def compute_frac_ice(
    frac_ice: Field["float"],
    t: Field["float"],
) -> Field["float"]:
    """
    Compute ice fraction in mixed-phase clouds based on temperature.
    
    This function calculates the fraction of total condensate (ice + liquid)
    that exists as ice, using temperature-dependent parameterizations.
    The computation mode is selected via the FRAC_ICE_ADJUST external parameter.
    
    Parameters
    ----------
    frac_ice : Field[float]
        Ice fraction field to be computed (dimensionless, 0 to 1).
        Input value may be used in mode 3 (previous value).
    t : Field[float]
        Temperature field in Kelvin (K).
        
    Returns
    -------
    Field[float]
        Ice fraction field with values between 0 (all liquid) and 1 (all ice).
        
    Notes
    -----
    External Parameters:
    - FRAC_ICE_ADJUST: Mode selector (0, 1, 2, or 3)
    - TMAXMIX: Maximum temperature for mixed-phase (K), typically 273.15 K
    - TMINMIX: Minimum temperature for mixed-phase (K), typically 233.15 K
    - TT: Triple point temperature (K), 273.15 K
    
    Computation Modes
    -----------------
    Mode 0 (T): Linear interpolation between TMINMIX and TMAXMIX
        frac_ice = (TMAXMIX - T) / (TMAXMIX - TMINMIX)
        This provides a smooth transition from all liquid above TMAXMIX
        to all ice below TMINMIX.
        
    Mode 1 (O): Old formula using triple point
        frac_ice = (TT - T) / 40
        Linear decrease over 40 K range below triple point.
        
    Mode 2 (N): No ice
        frac_ice = 0
        All condensate remains liquid regardless of temperature.
        
    Mode 3 (S): Same as previous value
        frac_ice = max(0, min(1, frac_ice))
        Maintains the input frac_ice value, only clamping to [0,1].
        
    All modes clamp the result to [0, 1] to ensure physical values.
    
    Examples
    --------
    Mode 0 with TMAXMIX=273.15, TMINMIX=233.15:
    - T = 273.15 K → frac_ice = 0 (all liquid)
    - T = 253.15 K → frac_ice = 0.5 (50% ice)
    - T = 233.15 K → frac_ice = 1.0 (all ice)
    """
    from __externals__ import FRAC_ICE_ADJUST, TMAXMIX, TMINMIX, TT

    frac_ice = 0

    # Mode 0: using temperature with linear interpolation
    # FracIceAdjust.T.value
    if __INLINED(FRAC_ICE_ADJUST == 0):
        frac_ice = max(0, min(1, ((TMAXMIX - t) / (TMAXMIX - TMINMIX))))

    # Mode 1: using temperature with old formula
    # FracIceAdjust.O.value
    elif __INLINED(FRAC_ICE_ADJUST == 1):
        frac_ice = max(0, min(1, ((TT - t) / 40)))

    # Mode 2: no ice
    # FracIceAdjust.N.value
    elif __INLINED(FRAC_ICE_ADJUST == 2):
        frac_ice = 0

    # Mode 3: same as previous
    # FracIceAdjust.S.value
    elif __INLINED(FRAC_ICE_ADJUST == 3):
        frac_ice = max(0, min(1, frac_ice))

    return frac_ice
