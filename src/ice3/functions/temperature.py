# -*- coding: utf-8 -*-
"""
Temperature update functions for microphysics calculations.

This module provides GT4Py implementations for updating temperature
fields based on phase changes (condensation, deposition, freezing, melting)
and for converting between temperature and potential temperature.
"""
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function


@function
def update_potential_temperature(
    theta: Field["float"],
    transfo_mixing_ratio: Field["float"],
    ls_fact: Field["float"],
    lv_fact: Field["float"],
):
    """
    Update potential temperature due to phase transformation.
    
    This function adjusts potential temperature when water substance
    changes phase (e.g., liquid to ice). The transformation is defined
    as going from liquid water to ice, releasing latent heat of fusion.
    
    Parameters
    ----------
    theta : Field[float]
        Potential temperature to update (K).
    transfo_mixing_ratio : Field[float]
        Mass of water undergoing phase transformation per unit mass of dry air (kg/kg).
        Positive values indicate freezing (liquid → ice).
    ls_fact : Field[float]
        Latent heat of sublimation divided by heat capacity, L_s/c_p (K).
    lv_fact : Field[float]
        Latent heat of vaporization divided by heat capacity, L_v/c_p (K).
        
    Returns
    -------
    Field[float]
        Updated potential temperature (K).
        
    Notes
    -----
    The latent heat of fusion (freezing/melting) is:
    L_f = L_s - L_v
    
    Therefore, the temperature change is:
    Δθ = (L_f/c_p) * Δr = (L_s/c_p - L_v/c_p) * Δr
    
    Where Δr is the mixing ratio change (transfo_mixing_ratio).
    
    For freezing (liquid → ice):
    - transfo_mixing_ratio > 0
    - Heat is released: θ increases
    
    For melting (ice → liquid):
    - transfo_mixing_ratio < 0
    - Heat is absorbed: θ decreases
    
    This function is used in saturation adjustment schemes to maintain
    thermodynamic consistency when partitioning condensate between
    liquid and ice phases.
    """
    return theta + transfo_mixing_ratio * (ls_fact - lv_fact)


@function
def theta2temperature(theta: Field["float"], exn: Field["float"]) -> Field["float"]:
    """
    Convert potential temperature to actual temperature.
    
    This function performs the standard conversion from potential
    temperature (θ) to absolute temperature (T) using the Exner
    function, which accounts for the effects of pressure.
    
    Parameters
    ----------
    theta : Field[float]
        Potential temperature (K).
    exn : Field[float]
        Exner function (dimensionless), π = (p/p₀)^(R_d/c_p).
        
    Returns
    -------
    Field[float]
        Absolute temperature (K).
        
    Notes
    -----
    The relation between temperature and potential temperature is:
    T = θ · π
    
    where π = (p/p₀)^(R_d/c_p) is the Exner function with:
    - p: pressure at the level
    - p₀: reference pressure (typically 1000 hPa)
    - R_d: gas constant for dry air
    - c_p: specific heat at constant pressure
    
    The Exner function represents the factor by which an air parcel's
    temperature would change if brought adiabatically from pressure p₀
    to pressure p.
    
    Examples
    --------
    At sea level (p ≈ 1000 hPa), π ≈ 1.0, so T ≈ θ
    At 500 hPa, π ≈ 0.83, so a parcel with θ = 300 K has T ≈ 249 K
    """
    return theta * exn


@function
def update_temperature(
    t: Field["float"],
    rc_in: Field["float"],
    rc_out: Field["float"],
    ri_in: Field["float"],
    ri_out: Field["float"],
    lv: Field["float"],
    ls: Field["float"],
) -> Field["float"]:
    """
    Update temperature based on changes in ice and liquid mixing ratios.
    
    This function computes the temperature change resulting from phase
    transformations (condensation/evaporation, deposition/sublimation,
    freezing/melting) by accounting for the latent heat exchanges.
    
    Parameters
    ----------
    t : Field[float]
        Current temperature to update (K).
    rc_in : Field[float]
        Cloud liquid water mixing ratio before the transformation (kg/kg).
    rc_out : Field[float]
        Cloud liquid water mixing ratio after the transformation (kg/kg).
    ri_in : Field[float]
        Ice mixing ratio before the transformation (kg/kg).
    ri_out : Field[float]
        Ice mixing ratio after the transformation (kg/kg).
    lv : Field[float]
        Latent heat of vaporization at current temperature (J/kg).
    ls : Field[float]
        Latent heat of sublimation at current temperature (J/kg).
        
    Returns
    -------
    Field[float]
        Updated temperature (K).
        
    Notes
    -----
    The temperature change is computed from the energy balance:
    ΔT = [Δr_c · L_v + Δr_i · L_s] / c_pd
    
    where:
    - Δr_c = rc_out - rc_in: change in liquid water mixing ratio
    - Δr_i = ri_out - ri_in: change in ice mixing ratio
    - L_v: latent heat of vaporization (vapor ↔ liquid)
    - L_s: latent heat of sublimation (vapor ↔ ice)
    - c_pd = CPD: specific heat of dry air at constant pressure
    
    Physical Process Examples:
    - Condensation (Δr_c > 0): releases L_v, temperature increases
    - Evaporation (Δr_c < 0): absorbs L_v, temperature decreases
    - Deposition (Δr_i > 0): releases L_s, temperature increases
    - Sublimation (Δr_i < 0): absorbs L_s, temperature decreases
    - Freezing: Δr_c < 0 and Δr_i > 0, net effect = (L_s - L_v) = L_f
    
    This function is fundamental to saturation adjustment schemes,
    ensuring energy conservation during phase changes.
    """
    from __externals__ import CPD

    t = (
        t[0, 0, 0]
        + (
            (rc_out[0, 0, 0] - rc_in[0, 0, 0]) * lv[0, 0, 0]
            + (ri_out[0, 0, 0] - ri_in[0, 0, 0]) * ls[0, 0, 0]
        )
        / CPD
    )

    return t
