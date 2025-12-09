"""
ICE4 Warm Rain Processes - JAX Implementation

This module implements warm rain microphysical processes: autoconversion,
accretion, and evaporation.

Reference:
    PHYEX/src/common/micro/mode_ice4_warm.F90
"""
import jax
import jax.numpy as jnp
from typing import Tuple


def ice4_warm(
    rhodref: jnp.ndarray,
    t: jnp.ndarray,
    pres: jnp.ndarray,
    tht: jnp.ndarray,
    lbdar: jnp.ndarray,
    lbdar_rf: jnp.ndarray,
    ka: jnp.ndarray,
    dv: jnp.ndarray,
    cj: jnp.ndarray,
    hlc_hcf: jnp.ndarray,
    hlc_hrc: jnp.ndarray,
    cf: jnp.ndarray,
    rf: jnp.ndarray,
    rvt: jnp.ndarray,
    rct: jnp.ndarray,
    rrt: jnp.ndarray,
    ldcompute: jnp.ndarray,
    ldsoft: bool,
    subg_rr_evap: int,
    constants: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute warm rain microphysical processes.
    
    Processes:
    1. Autoconversion: Cloud droplets → rain drops
    2. Accretion: Cloud droplets collected by rain
    3. Evaporation: Rain drops → vapor
    
    Args:
        rhodref: Reference air density (kg/m³)
        t: Temperature (K)
        pres: Pressure (Pa)
        tht: Potential temperature (K)
        lbdar: Rain slope parameter (m⁻¹)
        lbdar_rf: Rain slope parameter for rain fraction (m⁻¹)
        ka: Thermal conductivity of air (W/(m·K))
        dv: Water vapor diffusivity (m²/s)
        cj: Ventilation coefficient
        hlc_hcf: High cloud fraction from subgrid scheme (0-1)
        hlc_hrc: High cloud liquid water content (kg/kg)
        cf: Total cloud fraction (0-1)
        rf: Rain/precipitation fraction (0-1)
        rvt: Water vapor mixing ratio (kg/kg)
        rct: Cloud droplet mixing ratio (kg/kg)
        rrt: Rain mixing ratio (kg/kg)
        ldcompute: Computation mask
        ldsoft: Soft threshold mode flag
        subg_rr_evap: Evaporation scheme (0=NONE, 1=CLFR, 2=PRFR)
        constants: Dictionary of physical constants
        
    Returns:
        Tuple of (RCAUTR, RCACCR, RREVAV) tendencies
    """
    # Extract constants
    C_RTMIN = constants["C_RTMIN"]
    R_RTMIN = constants["R_RTMIN"]
    CRIAUTC = constants["CRIAUTC"]
    TIMAUTC = constants["TIMAUTC"]
    FCACCR = constants["FCACCR"]
    EXCACCR = constants["EXCACCR"]
    CEXVT = constants["CEXVT"]
    ALPW = constants["ALPW"]
    BETAW = constants["BETAW"]
    GAMW = constants["GAMW"]
    EPSILO = constants["EPSILO"]
    LVTT = constants["LVTT"]
    CPV = constants["CPV"]
    CL = constants["CL"]
    CPD = constants["CPD"]
    TT = constants["TT"]
    RV = constants["RV"]
    O0EVAR = constants["O0EVAR"]
    O1EVAR = constants["O1EVAR"]
    EX0EVAR = constants["EX0EVAR"]
    EX1EVAR = constants["EX1EVAR"]
    
    # Initialize outputs
    rcautr = jnp.zeros_like(rhodref)
    rcaccr = jnp.zeros_like(rhodref)
    rrevav = jnp.zeros_like(rhodref)
    
    if not ldsoft:
        # ====================
        # 1. Autoconversion
        # ====================
        mask_auto = ldcompute & (hlc_hrc > C_RTMIN) & (hlc_hcf > 0.0)
        rcautr = jnp.where(
            mask_auto,
            TIMAUTC * jnp.maximum(0.0, hlc_hrc - hlc_hcf * CRIAUTC / rhodref),
            0.0
        )
        
        # ====================
        # 2. Accretion
        # ====================
        mask_accr = ldcompute & (rct > C_RTMIN) & (rrt > R_RTMIN)
        rcaccr = jnp.where(
            mask_accr,
            FCACCR * rct * jnp.power(lbdar, EXCACCR) * jnp.power(rhodref, -CEXVT),
            0.0
        )
        
        # ====================
        # 3. Evaporation
        # ====================
        if subg_rr_evap == 0:  # NONE - grid-mean evaporation
            mask_evap = ldcompute & (rrt > R_RTMIN) & (rct <= C_RTMIN)
            
            # Saturation vapor pressure over water
            esat_w = jnp.exp(ALPW - BETAW / t - GAMW * jnp.log(t))
            
            # Undersaturation
            usw = 1.0 - rvt * (pres - esat_w) / (EPSILO * esat_w)
            
            # Thermodynamic coefficient
            av = (
                jnp.square(LVTT + (CPV - CL) * (t - TT)) / (ka * RV * jnp.square(t))
                + (RV * t) / (dv * esat_w)
            )
            
            # Evaporation rate
            rrevav = jnp.where(
                mask_evap,
                (jnp.maximum(0.0, usw) / (rhodref * av)) * (
                    O0EVAR * jnp.power(lbdar, EX0EVAR) + 
                    O1EVAR * cj * jnp.power(lbdar, EX1EVAR)
                ),
                0.0
            )
            
        elif subg_rr_evap == 1:  # CLFR - cloud fraction method
            zw4 = 1.0  # precipitation fraction
            zw3 = lbdar
            
            mask_evap = ldcompute & (rrt > R_RTMIN) & (zw4 > cf)
            
            # Liquid water potential temperature
            thlt_tmp = tht - LVTT * tht / CPD / t * rct
            
            # Unsaturated temperature
            zw2 = thlt_tmp * t / tht
            
            # Saturation over water (with unsaturated temp)
            esat_w = jnp.exp(ALPW - BETAW / zw2 - GAMW * jnp.log(zw2))
            
            # Undersaturation
            usw = 1.0 - rvt * (pres - esat_w) / (EPSILO * esat_w)
            
            # Thermodynamic coefficient
            av = (
                jnp.square(LVTT + (CPV - CL) * (zw2 - TT)) / (ka * RV * jnp.square(zw2))
                + RV * zw2 / (dv * esat_w)
            )
            
            # Evaporation rate in clear sky fraction
            rrevav = jnp.where(
                mask_evap,
                (jnp.maximum(0.0, usw) / (rhodref * av)) * (
                    O0EVAR * jnp.power(zw3, EX0EVAR) + 
                    O1EVAR * cj * jnp.power(zw3, EX1EVAR)
                ) * (zw4 - cf),
                0.0
            )
            
        elif subg_rr_evap == 2:  # PRFR - precipitation fraction method
            zw4 = rf  # precipitation fraction
            zw3 = lbdar_rf
            
            mask_evap = ldcompute & (rrt > R_RTMIN) & (zw4 > cf)
            
            # Liquid water potential temperature
            thlt_tmp = tht - LVTT * tht / CPD / t * rct
            
            # Unsaturated temperature
            zw2 = thlt_tmp * t / tht
            
            # Saturation over water (with unsaturated temp)
            esat_w = jnp.exp(ALPW - BETAW / zw2 - GAMW * jnp.log(zw2))
            
            # Undersaturation
            usw = 1.0 - rvt * (pres - esat_w) / (EPSILO * esat_w)
            
            # Thermodynamic coefficient
            av = (
                jnp.square(LVTT + (CPV - CL) * (zw2 - TT)) / (ka * RV * jnp.square(zw2))
                + RV * zw2 / (dv * esat_w)
            )
            
            # Evaporation rate in rain shaft outside cloud
            rrevav = jnp.where(
                mask_evap,
                (jnp.maximum(0.0, usw) / (rhodref * av)) * (
                    O0EVAR * jnp.power(zw3, EX0EVAR) + 
                    O1EVAR * cj * jnp.power(zw3, EX1EVAR)
                ) * (zw4 - cf),
                0.0
            )
    
    return rcautr, rcaccr, rrevav
