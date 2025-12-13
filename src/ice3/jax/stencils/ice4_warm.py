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
    
    # ====================
    # 1. Autoconversion (Only if not ldsoft)
    # ====================
    mask_auto = ldcompute & (hlc_hrc > C_RTMIN) & (hlc_hcf > 0.0) & (~ldsoft)
    rcautr = jnp.where(
        mask_auto,
        TIMAUTC * jnp.maximum(0.0, hlc_hrc - hlc_hcf * CRIAUTC / rhodref),
        0.0
    )
    
    # ====================
    # 2. Accretion (Only if not ldsoft)
    # ====================
    mask_accr = ldcompute & (rct > C_RTMIN) & (rrt > R_RTMIN) & (~ldsoft)
    rcaccr = jnp.where(
        mask_accr,
        FCACCR * rct * jnp.power(lbdar, EXCACCR) * jnp.power(rhodref, -CEXVT),
        0.0
    )
    
    # ====================
    # 3. Evaporation (Only if not ldsoft)
    # ====================
    # Saturation vapor pressure over water
    esat_w_std = jnp.exp(ALPW - BETAW / t - GAMW * jnp.log(t))
    
    # Thermodynamic coefficient for std
    av_std = (
        jnp.square(LVTT + (CPV - CL) * (t - TT)) / (ka * RV * jnp.square(t))
        + (RV * t) / (dv * esat_w_std)
    )

    # NONE - grid-mean evaporation
    mask_evap_none = ldcompute & (rrt > R_RTMIN) & (rct <= C_RTMIN) & (~ldsoft)
    usw_none = 1.0 - rvt * (pres - esat_w_std) / (EPSILO * esat_w_std)
    rrevav_none = (jnp.maximum(0.0, usw_none) / (rhodref * av_std)) * (
        O0EVAR * jnp.power(lbdar, EX0EVAR) + 
        O1EVAR * cj * jnp.power(lbdar, EX1EVAR)
    )

    # Liquid water potential temperature
    thlt_tmp = tht - LVTT * tht / CPD / t * rct
    # Unsaturated temperature (common for CLFR and PRFR)
    zw2 = thlt_tmp * t / tht
    esat_w_zw2 = jnp.exp(ALPW - BETAW / zw2 - GAMW * jnp.log(zw2))
    usw_zw2 = 1.0 - rvt * (pres - esat_w_zw2) / (EPSILO * esat_w_zw2)
    av_zw2 = (
        jnp.square(LVTT + (CPV - CL) * (zw2 - TT)) / (ka * RV * jnp.square(zw2))
        + RV * zw2 / (dv * esat_w_zw2)
    )

    # CLFR - cloud fraction method
    zw4_clfr = 1.0
    mask_evap_clfr = ldcompute & (rrt > R_RTMIN) & (zw4_clfr > cf) & (~ldsoft)
    rrevav_clfr = (jnp.maximum(0.0, usw_zw2) / (rhodref * av_zw2)) * (
        O0EVAR * jnp.power(lbdar, EX0EVAR) + 
        O1EVAR * cj * jnp.power(lbdar, EX1EVAR)
    ) * (zw4_clfr - cf)

    # PRFR - precipitation fraction method
    zw4_prfr = rf
    mask_evap_prfr = ldcompute & (rrt > R_RTMIN) & (zw4_prfr > cf) & (~ldsoft)
    rrevav_prfr = (jnp.maximum(0.0, usw_zw2) / (rhodref * av_zw2)) * (
        O0EVAR * jnp.power(lbdar_rf, EX0EVAR) + 
        O1EVAR * cj * jnp.power(lbdar_rf, EX1EVAR)
    ) * (zw4_prfr - cf)

    # Final evaporation tendency based on subg_rr_evap
    rrevav = jnp.where(
        subg_rr_evap == 0,
        jnp.where(mask_evap_none, rrevav_none, 0.0),
        jnp.where(
            subg_rr_evap == 1,
            jnp.where(mask_evap_clfr, rrevav_clfr, 0.0),
            jnp.where(mask_evap_prfr, rrevav_prfr, 0.0)
        )
    )
    
    return rcautr, rcaccr, rrevav

