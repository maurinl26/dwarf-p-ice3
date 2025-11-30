# -*- coding: utf-8 -*-
"""
JAX implementation of thermodynamic functions for ice adjustment.

This module provides JAX-compatible implementations of fundamental thermodynamic
functions used in saturation adjustment schemes for mixed-phase clouds.
These include latent heat calculations and specific heat capacity computations.

Translated from GT4Py implementation in src/ice3/functions/ice_adjust.py
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from typing import Dict, Any


def vaporisation_latent_heat(
    t: Array,
    constants: Dict[str, Any]
) -> Array:
    """
    Compute latent heat of vaporization as a function of temperature.
    
    This function calculates the temperature-dependent latent heat
    released when water vapor condenses to liquid water (or absorbed
    during evaporation). The latent heat varies with temperature due
    to differences in specific heat capacities between phases.
    
    Parameters
    ----------
    t : Array
        Temperature field in Kelvin (K).
    constants : Dict[str, Any]
        Dictionary containing physical constants:
        - LVTT: Latent heat of vaporization at triple point (J/kg)
        - CPV: Specific heat of water vapor at constant pressure (J/(kg·K))
        - CL: Specific heat of liquid water (J/(kg·K))
        - TT: Triple point temperature (K), 273.15 K
        
    Returns
    -------
    Array
        Latent heat of vaporization in J/kg.
        
    Notes
    -----
    The formula is:
    L_v(T) = L_v(T_t) + (c_pv - c_l) * (T - T_t)
    
    where:
    - L_v(T_t) = LVTT: Latent heat at triple point (J/kg)
    - c_pv = CPV: Specific heat of water vapor at constant pressure (J/(kg·K))
    - c_l = CL: Specific heat of liquid water (J/(kg·K))
    - T_t = TT: Triple point temperature (K), 273.15 K
    
    Typical values:
    - LVTT ≈ 2.5×10⁶ J/kg at 273.15 K
    - CPV ≈ 1850 J/(kg·K)
    - CL ≈ 4218 J/(kg·K)
    """
    LVTT = constants["LVTT"]
    CPV = constants["CPV"]
    CL = constants["CL"]
    TT = constants["TT"]
    
    return LVTT + (CPV - CL) * (t - TT)


def sublimation_latent_heat(
    t: Array,
    constants: Dict[str, Any]
) -> Array:
    """
    Compute latent heat of sublimation as a function of temperature.
    
    This function calculates the temperature-dependent latent heat
    released when water vapor deposits to ice (or absorbed during
    sublimation). The latent heat varies with temperature due to
    differences in specific heat capacities between phases.
    
    Parameters
    ----------
    t : Array
        Temperature field in Kelvin (K).
    constants : Dict[str, Any]
        Dictionary containing physical constants:
        - LSTT: Latent heat of sublimation at triple point (J/kg)
        - CPV: Specific heat of water vapor at constant pressure (J/(kg·K))
        - CI: Specific heat of ice (J/(kg·K))
        - TT: Triple point temperature (K), 273.15 K
        
    Returns
    -------
    Array
        Latent heat of sublimation in J/kg.
        
    Notes
    -----
    The formula is:
    L_s(T) = L_s(T_t) + (c_pv - c_i) * (T - T_t)
    
    where:
    - L_s(T_t) = LSTT: Latent heat at triple point (J/kg)
    - c_pv = CPV: Specific heat of water vapor at constant pressure (J/(kg·K))
    - c_i = CI: Specific heat of ice (J/(kg·K))
    - T_t = TT: Triple point temperature (K), 273.15 K
    
    The latent heat of sublimation equals the sum of latent heats
    of fusion and vaporization: L_s = L_f + L_v
    
    Typical values:
    - LSTT ≈ 2.83×10⁶ J/kg at 273.15 K
    - CPV ≈ 1850 J/(kg·K)
    - CI ≈ 2106 J/(kg·K)
    """
    LSTT = constants["LSTT"]
    CPV = constants["CPV"]
    CI = constants["CI"]
    TT = constants["TT"]
    
    return LSTT + (CPV - CI) * (t - TT)


def constant_pressure_heat_capacity(
    rv: Array,
    rc: Array,
    ri: Array,
    rr: Array,
    rs: Array,
    rg: Array,
    constants: Dict[str, Any]
) -> Array:
    """
    Compute specific heat at constant pressure for a moist air parcel.
    
    This function calculates the effective heat capacity of a moist
    air parcel containing various hydrometeor species. The total heat
    capacity is the mass-weighted average of the heat capacities of
    all components (dry air, water vapor, and condensed phases).
    
    Parameters
    ----------
    rv : Array
        Water vapor mixing ratio (kg/kg).
    rc : Array
        Cloud liquid water mixing ratio (kg/kg).
    ri : Array
        Cloud ice mixing ratio (kg/kg).
    rr : Array
        Rain water mixing ratio (kg/kg).
    rs : Array
        Snow mixing ratio (kg/kg).
    rg : Array
        Graupel mixing ratio (kg/kg).
    constants : Dict[str, Any]
        Dictionary containing physical constants:
        - CPD: Specific heat of dry air at constant pressure (J/(kg·K))
        - CPV: Specific heat of water vapor at constant pressure (J/(kg·K))
        - CL: Specific heat of liquid water (J/(kg·K))
        - CI: Specific heat of ice (J/(kg·K))
        
    Returns
    -------
    Array
        Specific heat capacity at constant pressure in J/(kg·K).
        
    Notes
    -----
    The formula is:
    c_p = c_pd + c_pv·r_v + c_l·(r_c + r_r) + c_i·(r_i + r_s + r_g)
    
    where:
    - c_pd = CPD: Specific heat of dry air at constant pressure (J/(kg·K))
    - c_pv = CPV: Specific heat of water vapor at constant pressure (J/(kg·K))
    - c_l = CL: Specific heat of liquid water (J/(kg·K))
    - c_i = CI: Specific heat of ice (J/(kg·K))
    
    Typical values:
    - CPD ≈ 1004 J/(kg·K)
    - CPV ≈ 1850 J/(kg·K)
    - CL ≈ 4218 J/(kg·K)
    - CI ≈ 2106 J/(kg·K)
    """
    CPD = constants["CPD"]
    CPV = constants["CPV"]
    CL = constants["CL"]
    CI = constants["CI"]
    
    return CPD + CPV * rv + CL * (rc + rr) + CI * (ri + rs + rg)
