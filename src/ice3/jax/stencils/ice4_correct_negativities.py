"""
ICE4 Negativity Correction - JAX Implementation

This module corrects negative mixing ratios while conserving mass and energy.

Reference:
    PHYEX/src/common/micro/mode_ice4_correct_negativities.F90
"""
import jax
import jax.numpy as jnp
from typing import Tuple


def ice4_correct_negativities(
    th_t: jnp.ndarray,
    rv_t: jnp.ndarray,
    rc_t: jnp.ndarray,
    rr_t: jnp.ndarray,
    ri_t: jnp.ndarray,
    rs_t: jnp.ndarray,
    rg_t: jnp.ndarray,
    lv_fact: jnp.ndarray,
    ls_fact: jnp.ndarray,
    constants: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Correct negative mixing ratios while conserving mass and energy.
    
    This function ensures all hydrometeor mixing ratios remain non-negative
    by converting negative values to vapor and adjusting temperature accordingly.
    
    Args:
        th_t: Potential temperature (K)
        rv_t: Water vapor mixing ratio (kg/kg)
        rc_t: Cloud water mixing ratio (kg/kg)
        rr_t: Rain mixing ratio (kg/kg)
        ri_t: Pristine ice mixing ratio (kg/kg)
        rs_t: Snow mixing ratio (kg/kg)
        rg_t: Graupel mixing ratio (kg/kg)
        lv_fact: Latent heat of vaporization factor (Lv/Cp) (K)
        ls_fact: Latent heat of sublimation factor (Ls/Cp) (K)
        constants: Dictionary of physical constants
        
    Returns:
        Tuple of corrected (th_t, rv_t, rc_t, rr_t, ri_t, rs_t, rg_t)
        
    Notes:
        The correction proceeds in two steps:
        1. Convert any negative hydrometeor values to vapor
           - For liquid species (rc, rr): evaporation
           - For ice species (ri, rs, rg): sublimation
        2. If vapor becomes negative, sublimate snow and graupel
        
        Energy is conserved through temperature adjustment using
        latent heat factors.
    """
    # Extract constants
    S_RTMIN = constants["S_RTMIN"]
    G_RTMIN = constants["G_RTMIN"]
    
    # Step 1: Correct negative hydrometeors
    
    # Cloud water
    w = rc_t - jnp.maximum(rc_t, 0.0)
    rv_t = rv_t + w
    th_t = th_t - w * lv_fact
    rc_t = rc_t - w
    
    # Rain
    w = rr_t - jnp.maximum(rr_t, 0.0)
    rv_t = rv_t + w
    th_t = th_t - w * lv_fact
    rr_t = rr_t - w
    
    # Pristine ice
    w = ri_t - jnp.maximum(ri_t, 0.0)
    rv_t = rv_t + w
    th_t = th_t - w * ls_fact
    ri_t = ri_t - w
    
    # Snow
    w = rs_t - jnp.maximum(rs_t, 0.0)
    rv_t = rv_t + w
    th_t = th_t - w * ls_fact
    rs_t = rs_t - w
    
    # Graupel
    w = rg_t - jnp.maximum(rg_t, 0.0)
    rv_t = rv_t + w
    th_t = th_t - w * ls_fact
    rg_t = rg_t - w
    
    # Step 2: Correct negative vapor by sublimating snow and graupel
    
    # Snow -> vapor
    w = jnp.minimum(
        jnp.maximum(rs_t, 0.0),
        jnp.maximum(S_RTMIN - rv_t, 0.0)
    )
    rv_t = rv_t + w
    rs_t = rs_t - w
    th_t = th_t - w * ls_fact
    
    # Graupel -> vapor
    w = jnp.minimum(
        jnp.maximum(rg_t, 0.0),
        jnp.maximum(G_RTMIN - rv_t, 0.0)
    )
    rv_t = rv_t + w
    rg_t = rg_t - w
    th_t = th_t - w * ls_fact
    
    return th_t, rv_t, rc_t, rr_t, ri_t, rs_t, rg_t
