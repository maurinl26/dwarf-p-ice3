# -*- coding: utf-8 -*-
"""
Thermodynamic functions for ice adjustment calculations.

This module provides GT4Py implementations of fundamental thermodynamic
functions used in saturation adjustment schemes for mixed-phase clouds.
These include latent heat calculations and specific heat capacity
computations for moist air parcels.
"""
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function


@function
def vaporisation_latent_heat(
    t: Field["float"],
) -> Field["float"]:
    """
    Compute latent heat of vaporization as a function of temperature.
    
    This function calculates the temperature-dependent latent heat
    released when water vapor condenses to liquid water (or absorbed
    during evaporation). The latent heat varies with temperature due
    to differences in specific heat capacities between phases.
    
    Parameters
    ----------
    t : Field[float]
        Temperature field in Kelvin (K).
        
    Returns
    -------
    Field[float]
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
    
    The temperature dependence accounts for the energy difference in
    heating/cooling the phases between the reference temperature and
    the current temperature.
    
    Typical values:
    - LVTT ≈ 2.5×10⁶ J/kg at 273.15 K
    - CPV ≈ 1850 J/(kg·K)
    - CL ≈ 4218 J/(kg·K)
    """
    from __externals__ import CL, CPV, LVTT, TT

    return LVTT + (CPV - CL) * (t - TT)


@function
def sublimation_latent_heat(
    t: Field["float"],
) -> Field["float"]:
    """
    Compute latent heat of sublimation as a function of temperature.
    
    This function calculates the temperature-dependent latent heat
    released when water vapor deposits to ice (or absorbed during
    sublimation). The latent heat varies with temperature due to
    differences in specific heat capacities between phases.
    
    Parameters
    ----------
    t : Field[float]
        Temperature field in Kelvin (K).
        
    Returns
    -------
    Field[float]
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
    from __externals__ import CI, CPV, LSTT, TT

    return LSTT + (CPV - CI) * (t - TT)


@function
def constant_pressure_heat_capacity(
    rv: Field["float"],
    rc: Field["float"],
    ri: Field["float"],
    rr: Field["float"],
    rs: Field["float"],
    rg: Field["float"],
) -> Field["float"]:
    """
    Compute specific heat at constant pressure for a moist air parcel.
    
    This function calculates the effective heat capacity of a moist
    air parcel containing various hydrometeor species. The total heat
    capacity is the mass-weighted average of the heat capacities of
    all components (dry air, water vapor, and condensed phases).
    
    Parameters
    ----------
    rv : Field[float]
        Water vapor mixing ratio (kg/kg).
    rc : Field[float]
        Cloud liquid water mixing ratio (kg/kg).
    ri : Field[float]
        Cloud ice mixing ratio (kg/kg).
    rr : Field[float]
        Rain water mixing ratio (kg/kg).
    rs : Field[float]
        Snow mixing ratio (kg/kg).
    rg : Field[float]
        Graupel mixing ratio (kg/kg).
        
    Returns
    -------
    Field[float]
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
    
    This function is essential for temperature adjustments during
    phase changes, as it determines how much the temperature changes
    for a given amount of latent heat release or absorption.
    """
    from __externals__ import CI, CL, CPD, CPV

    return CPD + CPV * rv + CL * (rc + rr) + CI * (ri + rs + rg)
