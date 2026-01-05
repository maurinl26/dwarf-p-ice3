# -*- coding: utf-8 -*-
"""
Saturation vapor pressure functions over water and ice.

This module provides GT4Py implementations of saturation vapor pressure
calculations using empirical formulas for both liquid water and ice surfaces.
These are fundamental functions for cloud microphysics calculations.
"""
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, function, log

@function
def e_sat_w(t: Field["float"]) -> Field["float"]:
    """
    Calculate saturation vapor pressure over liquid water.
    
    This function computes the saturation vapor pressure over a plane
    liquid water surface using an empirical exponential formula with
    temperature-dependent coefficients.
    
    Parameters
    ----------
    t : Field[float]
        Temperature field in Kelvin (K).
        
    Returns
    -------
    Field[float]
        Saturation vapor pressure over liquid water in Pascals (Pa).
        
    Notes
    -----
    The formula uses external constants:
    - ALPW, BETAW, GAMW: Empirical coefficients for the saturation
      vapor pressure formula over liquid water.
      
    The saturation vapor pressure is computed as:
    e_sat_w = exp(ALPW - BETAW/T - GAMW*ln(T))
    
    This formulation is commonly used in atmospheric models and follows
    standard meteorological conventions.
    """
    from __externals__ import ALPW, BETAW, GAMW

    return exp(ALPW - BETAW / t - GAMW * log(t))


@function
def e_sat_i(t: Field["float"]):
    """
    Calculate saturation vapor pressure over ice.
    
    This function computes the saturation vapor pressure over a plane
    ice surface using an empirical exponential formula with
    temperature-dependent coefficients.
    
    Parameters
    ----------
    t : Field[float]
        Temperature field in Kelvin (K).
        
    Returns
    -------
    Field[float]
        Saturation vapor pressure over ice in Pascals (Pa).
        
    Notes
    -----
    The formula uses external constants:
    - ALPI, BETAI, GAMI: Empirical coefficients for the saturation
      vapor pressure formula over ice.
      
    The saturation vapor pressure is computed as:
    e_sat_i = exp(ALPI - BETAI/T - GAMI*ln(T))
    
    This formulation is commonly used in atmospheric models for
    temperatures below freezing where ice is the stable phase.
    The difference between e_sat_w and e_sat_i is important for
    determining supersaturation conditions in mixed-phase clouds.
    """
    from __externals__ import ALPI, BETAI, GAMI

    return exp(ALPI - BETAI / t - GAMI * log(t))

