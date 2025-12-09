"""
ICE4 Fast RG (Graupel) Processes - JAX Implementation

This module implements the fast growth processes for graupel in the ICE4
microphysics scheme, translated to JAX from GT4Py stencils.

Processes implemented:
- Rain contact freezing (RICFRRG, RRCFRIG, PRICFRR)
- Cloud and pristine ice collection on graupel (wet and dry)
- Graupel melting

Reference:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_rg.F90
"""
import jax
import jax.numpy as jnp
from typing import Tuple


def rain_contact_freezing(
    prhodref: jnp.ndarray,
    plbdar: jnp.ndarray,
    pt: jnp.ndarray,
    prit: jnp.ndarray,
    prrt: jnp.ndarray,
    pcit: jnp.ndarray,
    ldcompute: jnp.ndarray,
    ldsoft: bool,
    lcrflimit: bool,
    constants: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute rain contact freezing processes.
    
    Args:
        prhodref: Air density (kg/m³)
        plbdar: Rain slope parameter (m⁻¹)
        pt: Temperature (K)
        prit: Pristine ice mixing ratio (kg/kg)
        prrt: Rain mixing ratio (kg/kg)
        pcit: Pristine ice concentration (#/m³)
        ldcompute: Computation mask
        ldsoft: Soft budget mode flag
        lcrflimit: Contact freezing rate limitation flag
        constants: Dictionary of physical constants
        
    Returns:
        Tuple of (RICFRRG, RRCFRIG, PRICFRR) tendencies
    """
    # Extract constants
    I_RTMIN = constants["I_RTMIN"]
    R_RTMIN = constants["R_RTMIN"]
    ICFRR = constants["ICFRR"]
    EXICFRR = constants["EXICFRR"]
    CEXVT = constants["CEXVT"]
    RCFRI = constants["RCFRI"]
    EXRCFRI = constants["EXRCFRI"]
    TT = constants["TT"]
    CI = constants["CI"]
    CL = constants["CL"]
    LVTT = constants["LVTT"]
    
    # Initialize outputs
    pricfrrg = jnp.zeros_like(prhodref)
    prrcfrig = jnp.zeros_like(prhodref)
    pricfrr = jnp.zeros_like(prhodref)
    
    # Compute mask
    mask = (prit > I_RTMIN) & (prrt > R_RTMIN) & ldcompute
    
    if not ldsoft:
        # RICFRRG - pristine ice collection by rain → graupel
        pricfrrg = jnp.where(
            mask,
            ICFRR * prit * jnp.power(plbdar, EXICFRR) * jnp.power(prhodref, -CEXVT),
            0.0
        )
        
        # RRCFRIG - rain freezing by contact with pristine ice
        prrcfrig = jnp.where(
            mask,
            RCFRI * pcit * jnp.power(plbdar, EXRCFRI) * jnp.power(prhodref, -CEXVT - 1.0),
            0.0
        )
        
        if lcrflimit:
            # Heat balance limitation
            zzw_denominator = jnp.maximum(1.0e-20, LVTT * prrcfrig)
            zzw_ratio = (pricfrrg * CI + prrcfrig * CL) * (TT - pt) / zzw_denominator
            zzw_prop = jnp.clip(zzw_ratio, 0.0, 1.0)
            
            prrcfrig = jnp.where(mask, zzw_prop * prrcfrig, 0.0)
            pricfrr = jnp.where(mask, (1.0 - zzw_prop) * pricfrrg, 0.0)
            pricfrrg = jnp.where(mask, zzw_prop * pricfrrg, 0.0)
    
    return pricfrrg, prrcfrig, pricfrr


def cloud_pristine_collection_graupel(
    prhodref: jnp.ndarray,
    plbdag: jnp.ndarray,
    pt: jnp.ndarray,
    prct: jnp.ndarray,
    prit: jnp.ndarray,
    prgt: jnp.ndarray,
    ldcompute: jnp.ndarray,
    ldsoft: bool,
    constants: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute wet and dry collection of cloud droplets and pristine ice on graupel.
    
    Args:
        prhodref: Air density (kg/m³)
        plbdag: Graupel slope parameter (m⁻¹)
        pt: Temperature (K)
        prct: Cloud water mixing ratio (kg/kg)
        prit: Pristine ice mixing ratio (kg/kg)
        prgt: Graupel mixing ratio (kg/kg)
        ldcompute: Computation mask
        ldsoft: Soft budget mode flag
        constants: Dictionary of physical constants
        
    Returns:
        Tuple of (RCDRYG_TEND, RIDRYG_TEND, RIWETG_TEND)
    """
    # Extract constants
    C_RTMIN = constants["C_RTMIN"]
    I_RTMIN = constants["I_RTMIN"]
    G_RTMIN = constants["G_RTMIN"]
    TT = constants["TT"]
    FCDRYG = constants["FCDRYG"]
    FIDRYG = constants["FIDRYG"]
    COLIG = constants["COLIG"]
    COLEXIG = constants["COLEXIG"]
    CXG = constants["CXG"]
    DG = constants["DG"]
    CEXVT = constants["CEXVT"]
    
    # Initialize outputs
    rcdryg_tend = jnp.zeros_like(prhodref)
    ridryg_tend = jnp.zeros_like(prhodref)
    riwetg_tend = jnp.zeros_like(prhodref)
    
    if not ldsoft:
        # Cloud droplet collection
        mask_cloud = (prgt > G_RTMIN) & (prct > C_RTMIN) & ldcompute
        base_cloud = jnp.power(plbdag, CXG - DG - 2.0) * jnp.power(prhodref, -CEXVT)
        rcdryg_tend = jnp.where(mask_cloud, FCDRYG * prct * base_cloud, 0.0)
        
        # Pristine ice collection
        mask_ice = (prgt > G_RTMIN) & (prit > I_RTMIN) & ldcompute
        base_ice = jnp.power(plbdag, CXG - DG - 2.0) * jnp.power(prhodref, -CEXVT)
        exp_term = jnp.exp(COLEXIG * (pt - TT))
        
        ridryg_tend = jnp.where(
            mask_ice,
            FIDRYG * exp_term * prit * base_ice,
            0.0
        )
        
        riwetg_tend = jnp.where(
            mask_ice,
            ridryg_tend / (COLIG * exp_term),
            0.0
        )
    
    return rcdryg_tend, ridryg_tend, riwetg_tend


def graupel_melting(
    prhodref: jnp.ndarray,
    ppres: jnp.ndarray,
    pdv: jnp.ndarray,
    pka: jnp.ndarray,
    pcj: jnp.ndarray,
    plbdag: jnp.ndarray,
    pt: jnp.ndarray,
    prvt: jnp.ndarray,
    prgt: jnp.ndarray,
    rcdryg_tend: jnp.ndarray,
    rrdryg_tend: jnp.ndarray,
    ldcompute: jnp.ndarray,
    ldsoft: bool,
    levlimit: bool,
    constants: dict,
) -> jnp.ndarray:
    """
    Compute melting of graupel above 0°C.
    
    Args:
        prhodref: Air density (kg/m³)
        ppres: Pressure (Pa)
        pdv: Vapor diffusivity (m²/s)
        pka: Thermal conductivity (J/m/s/K)
        pcj: Ventilation coefficient
        plbdag: Graupel slope parameter (m⁻¹)
        pt: Temperature (K)
        prvt: Water vapor mixing ratio (kg/kg)
        prgt: Graupel mixing ratio (kg/kg)
        rcdryg_tend: Cloud collection tendency
        rrdryg_tend: Rain collection tendency
        ldcompute: Computation mask
        ldsoft: Soft budget mode flag
        levlimit: Vapor pressure limitation flag
        constants: Dictionary of physical constants
        
    Returns:
        PRGMLTR: Graupel melting rate
    """
    # Extract constants
    G_RTMIN = constants["G_RTMIN"]
    TT = constants["TT"]
    EPSILO = constants["EPSILO"]
    ALPW = constants["ALPW"]
    BETAW = constants["BETAW"]
    GAMW = constants["GAMW"]
    LVTT = constants["LVTT"]
    CPV = constants["CPV"]
    CL = constants["CL"]
    ESTT = constants["ESTT"]
    RV = constants["RV"]
    LMTT = constants["LMTT"]
    O0DEPG = constants["O0DEPG"]
    O1DEPG = constants["O1DEPG"]
    EX0DEPG = constants["EX0DEPG"]
    EX1DEPG = constants["EX1DEPG"]
    
    # Initialize output
    prgmltr = jnp.zeros_like(prhodref)
    
    # Compute mask
    mask = (prgt > G_RTMIN) & (pt > TT) & ldcompute
    
    if not ldsoft:
        # Compute vapor pressure
        prs_ev = prvt * ppres / (EPSILO + prvt)
        
        # Apply saturation limit if requested
        if levlimit:
            prs_ev_sat = jnp.exp(ALPW - BETAW / pt - GAMW * jnp.log(pt))
            prs_ev = jnp.minimum(prs_ev, prs_ev_sat)
        
        # Compute melting rate
        zzw_temp = pka * (TT - pt) + pdv * (
            LVTT + (CPV - CL) * (pt - TT)
        ) * (ESTT - prs_ev) / (RV * pt)
        
        prgmltr = jnp.maximum(
            0.0,
            (
                -zzw_temp
                * (O0DEPG * jnp.power(plbdag, EX0DEPG) + O1DEPG * pcj * jnp.power(plbdag, EX1DEPG))
                - (rcdryg_tend + rrdryg_tend) * (prhodref * CL * (TT - pt))
            )
            / (prhodref * LMTT)
        )
        
        prgmltr = jnp.where(mask, prgmltr, 0.0)
    
    return prgmltr
