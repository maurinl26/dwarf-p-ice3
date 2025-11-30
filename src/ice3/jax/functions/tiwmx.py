# -*- coding: utf-8 -*-
"""
JAX implementation of saturation vapor pressure functions over water and ice.

This module provides JAX-compatible implementations of saturation vapor pressure
calculations using empirical formulas for both liquid water and ice surfaces.
These are fundamental functions for cloud microphysics calculations.

Translated from GT4Py implementation in src/ice3/functions/tiwmx.py
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from typing import Dict, Any


def e_sat_w(t: Array, constants: Dict[str, Any]) -> Array:
    """
    Calculate saturation vapor pressure over liquid water.
    
    This function computes the saturation vapor pressure over a plane
    liquid water surface using an empirical exponential formula with
    temperature-dependent coefficients.
    
    Parameters
    ----------
    t : Array
        Temperature field in Kelvin (K).
    constants : Dict[str, Any]
        Dictionary containing empirical constants:
        - ALPW: Alpha coefficient for water saturation formula
        - BETAW: Beta coefficient for water saturation formula
        - GAMW: Gamma coefficient for water saturation formula
        
    Returns
    -------
    Array
        Saturation vapor pressure over liquid water in Pascals (Pa).
        
    Notes
    -----
    The formula uses empirical constants:
    - ALPW, BETAW, GAMW: Empirical coefficients for the saturation
      vapor pressure formula over liquid water.
      
    The saturation vapor pressure is computed as:
    e_sat_w = exp(ALPW - BETAW/T - GAMW*ln(T))
    
    This formulation is commonly used in atmospheric models and follows
    standard meteorological conventions.
    """
    ALPW = constants["ALPW"]
    BETAW = constants["BETAW"]
    GAMW = constants["GAMW"]
    
    return jnp.exp(ALPW - BETAW / t - GAMW * jnp.log(t))


def e_sat_i(t: Array, constants: Dict[str, Any]) -> Array:
    """
    Calculate saturation vapor pressure over ice.
    
    This function computes the saturation vapor pressure over a plane
    ice surface using an empirical exponential formula with
    temperature-dependent coefficients.
    
    Parameters
    ----------
    t : Array
        Temperature field in Kelvin (K).
    constants : Dict[str, Any]
        Dictionary containing empirical constants:
        - ALPI: Alpha coefficient for ice saturation formula
        - BETAI: Beta coefficient for ice saturation formula
        - GAMI: Gamma coefficient for ice saturation formula
        
    Returns
    -------
    Array
        Saturation vapor pressure over ice in Pascals (Pa).
        
    Notes
    -----
    The formula uses empirical constants:
    - ALPI, BETAI, GAMI: Empirical coefficients for the saturation
      vapor pressure formula over ice.
      
    The saturation vapor pressure is computed as:
    e_sat_i = exp(ALPI - BETAI/T - GAMI*ln(T))
    
    This formulation is commonly used in atmospheric models for
    temperatures below freezing where ice is the stable phase.
    The difference between e_sat_w and e_sat_i is important for
    determining supersaturation conditions in mixed-phase clouds.
    """
    ALPI = constants["ALPI"]
    BETAI = constants["BETAI"]
    GAMI = constants["GAMI"]
    
    return jnp.exp(ALPI - BETAI / t - GAMI * jnp.log(t))
