"""
ICE4 Fast RS (Snow/Aggregate) Processes - JAX Implementation

This module implements the fast growth processes for snow/aggregates in the ICE4
microphysics scheme, translated to JAX from GT4Py stencils.

Processes implemented:
- Freezing rate computation for snow processes
- Conversion-melting of aggregates

Reference:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_rs.F90
"""
import jax
import jax.numpy as jnp
from typing import Tuple


def compute_freezing_rate(
    prhodref: jnp.ndarray,
    ppres: jnp.ndarray,
    pdv: jnp.ndarray,
    pka: jnp.ndarray,
    pcj: jnp.ndarray,
    plbdas: jnp.ndarray,
    pt: jnp.ndarray,
    prvt: jnp.ndarray,
    prst: jnp.ndarray,
    priaggs: jnp.ndarray,
    ldcompute: jnp.ndarray,
    ldsoft: bool,
    levlimit: bool,
    constants: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute maximum freezing rate for snow processes.
    
    This rate limits the riming and accretion processes on snow aggregates
    based on thermal balance and vapor deposition constraints.
    
    Args:
        prhodref: Air density (kg/m³)
        ppres: Pressure (Pa)
        pdv: Vapor diffusivity (m²/s)
        pka: Thermal conductivity (J/m/s/K)
        pcj: Ventilation coefficient
        plbdas: Snow slope parameter (m⁻¹)
        pt: Temperature (K)
        prvt: Water vapor mixing ratio (kg/kg)
        prst: Snow mixing ratio (kg/kg)
        priaggs: Aggregation rate of ice
        ldcompute: Computation mask
        ldsoft: Soft budget mode flag
        levlimit: Vapor pressure limitation flag
        constants: Dictionary of physical constants
        
    Returns:
        Tuple of (ZFREEZ_RATE, FREEZ1_TEND, FREEZ2_TEND)
    """
    # Extract constants
    S_RTMIN = constants["S_RTMIN"]
    EPSILO = constants["EPSILO"]
    ALPI = constants["ALPI"]
    BETAI = constants["BETAI"]
    GAMI = constants["GAMI"]
    TT = constants["TT"]
    LVTT = constants["LVTT"]
    CPV = constants["CPV"]
    CL = constants["CL"]
    CI = constants["CI"]
    LMTT = constants["LMTT"]
    ESTT = constants["ESTT"]
    RV = constants["RV"]
    O0DEPS = constants["O0DEPS"]
    O1DEPS = constants["O1DEPS"]
    EX0DEPS = constants["EX0DEPS"]
    EX1DEPS = constants["EX1DEPS"]
    
    # Initialize outputs
    zfreez_rate = jnp.zeros_like(prhodref)
    freez1_tend = jnp.zeros_like(prhodref)
    freez2_tend = jnp.zeros_like(prhodref)
    
    # Compute mask
    mask = (prst > S_RTMIN) & ldcompute
    
    if not ldsoft:
        # Compute vapor pressure
        prs_ev = prvt * ppres / (EPSILO + prvt)
        
        # Apply saturation limit if requested
        if levlimit:
            prs_ev_sat = jnp.exp(ALPI - BETAI / pt - GAMI * jnp.log(pt))
            prs_ev = jnp.minimum(prs_ev, prs_ev_sat)
        
        # Compute first freezing term (vapor deposition)
        zzw_temp = pka * (TT - pt) + (
            pdv * (LVTT + (CPV - CL) * (pt - TT)) * (ESTT - prs_ev) / (RV * pt)
        )
        
        freez1_tend = (
            zzw_temp
            * (O0DEPS * jnp.power(plbdas, EX0DEPS) + O1DEPS * pcj * jnp.power(plbdas, EX1DEPS))
            / (prhodref * (LMTT - CL * (TT - pt)))
        )
        
        # Compute second freezing term (heat capacity factor)
        freez2_tend = (prhodref * (LMTT + (CI - CL) * (TT - pt))) / (
            prhodref * (LMTT - CL * (TT - pt))
        )
        
        # Compute total freezing rate
        zfreez_rate = jnp.maximum(
            0.0,
            jnp.maximum(0.0, freez1_tend + freez2_tend * priaggs) - priaggs
        )
        
        # Apply mask
        freez1_tend = jnp.where(mask, freez1_tend, 0.0)
        freez2_tend = jnp.where(mask, freez2_tend, 0.0)
        zfreez_rate = jnp.where(mask, zfreez_rate, 0.0)
    
    return zfreez_rate, freez1_tend, freez2_tend


def conversion_melting_snow(
    prhodref: jnp.ndarray,
    ppres: jnp.ndarray,
    pdv: jnp.ndarray,
    pka: jnp.ndarray,
    pcj: jnp.ndarray,
    plbdas: jnp.ndarray,
    pt: jnp.ndarray,
    prvt: jnp.ndarray,
    prst: jnp.ndarray,
    rcrims_tend: jnp.ndarray,
    rraccs_tend: jnp.ndarray,
    ldcompute: jnp.ndarray,
    ldsoft: bool,
    levlimit: bool,
    constants: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute conversion-melting of snow aggregates above 0°C.
    
    Args:
        prhodref: Air density (kg/m³)
        ppres: Pressure (Pa)
        pdv: Vapor diffusivity (m²/s)
        pka: Thermal conductivity (J/m/s/K)
        pcj: Ventilation coefficient
        plbdas: Snow slope parameter (m⁻¹)
        pt: Temperature (K)
        prvt: Water vapor mixing ratio (kg/kg)
        prst: Snow mixing ratio (kg/kg)
        rcrims_tend: Cloud riming tendency
        rraccs_tend: Rain accretion tendency
        ldcompute: Computation mask
        ldsoft: Soft budget mode flag
        levlimit: Vapor pressure limitation flag
        constants: Dictionary of physical constants
        
    Returns:
        Tuple of (PRSMLTG, PRCMLTSR)
    """
    # Extract constants
    S_RTMIN = constants["S_RTMIN"]
    EPSILO = constants["EPSILO"]
    ALPW = constants["ALPW"]
    BETAW = constants["BETAW"]
    GAMW = constants["GAMW"]
    TT = constants["TT"]
    LVTT = constants["LVTT"]
    CPV = constants["CPV"]
    CL = constants["CL"]
    LMTT = constants["LMTT"]
    ESTT = constants["ESTT"]
    RV = constants["RV"]
    O0DEPS = constants["O0DEPS"]
    O1DEPS = constants["O1DEPS"]
    EX0DEPS = constants["EX0DEPS"]
    EX1DEPS = constants["EX1DEPS"]
    FSCVMG = constants["FSCVMG"]
    
    # Initialize outputs
    prsmltg = jnp.zeros_like(prhodref)
    prcmltsr = jnp.zeros_like(prhodref)
    
    # Compute mask
    mask = (prst > S_RTMIN) & (pt > TT) & ldcompute
    
    if not ldsoft:
        # Compute vapor pressure
        prs_ev = prvt * ppres / (EPSILO + prvt)
        
        # Apply saturation limit if requested
        if levlimit:
            prs_ev_sat = jnp.exp(ALPW - BETAW / pt - GAMW * jnp.log(pt))
            prs_ev = jnp.minimum(prs_ev, prs_ev_sat)
        
        # Compute melting term
        zzw_temp = pka * (TT - pt) + (
            pdv * (LVTT + (CPV - CL) * (pt - TT)) * (ESTT - prs_ev) / (RV * pt)
        )
        
        # Compute RSMLT
        prsmltg = FSCVMG * jnp.maximum(
            0.0,
            (
                -zzw_temp
                * (O0DEPS * jnp.power(plbdas, EX0DEPS) + O1DEPS * pcj * jnp.power(plbdas, EX1DEPS))
                - (rcrims_tend + rraccs_tend) * (prhodref * CL * (TT - pt))
            )
            / (prhodref * LMTT)
        )
        
        # Collection rate (both species liquid, no heat exchange)
        prcmltsr = rcrims_tend
        
        # Apply mask
        prsmltg = jnp.where(mask, prsmltg, 0.0)
        prcmltsr = jnp.where(mask, prcmltsr, 0.0)
    
    return prsmltg, prcmltsr
