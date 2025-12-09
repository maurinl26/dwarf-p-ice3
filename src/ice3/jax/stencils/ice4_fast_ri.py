"""
Fast ice crystal processes - Bergeron-Findeisen effect - JAX Implementation

This module implements the Bergeron-Findeisen effect in JAX, where cloud droplets
evaporate to provide water vapor that deposits onto ice crystals.

Reference:
    PHYEX/src/common/micro/mode_ice4_fast_ri.F90
"""
import jax
import jax.numpy as jnp


def ice4_fast_ri(
    rhodref: jnp.ndarray,
    ai: jnp.ndarray,
    cj: jnp.ndarray,
    cit: jnp.ndarray,
    ssi: jnp.ndarray,
    rct: jnp.ndarray,
    rit: jnp.ndarray,
    ldcompute: jnp.ndarray,
    ldsoft: bool,
    constants: dict,
) -> jnp.ndarray:
    """
    Compute the Bergeron-Findeisen effect (RCBERI tendency).
    
    This function calculates the rate at which cloud droplets evaporate
    to deposit as ice on existing ice crystals in supersaturated conditions
    with respect to ice.
    
    Args:
        rhodref: Reference air density (kg/m³)
        ai: Thermodynamical function for vapor diffusion (m²·s·kg⁻¹)
        cj: Ventilation coefficient function (dimensionless)
        cit: Ice crystal number concentration (m⁻³)
        ssi: Supersaturation with respect to ice (dimensionless)
        rct: Cloud liquid water mixing ratio (kg/kg)
        rit: Pristine ice crystal mixing ratio (kg/kg)
        ldcompute: Mask indicating which grid points require computation
        ldsoft: If True, skip computation and return zeros
        constants: Dictionary of physical constants
        
    Returns:
        rc_beri_tnd: Tendency for cloud water due to Bergeron-Findeisen effect (kg/kg/s)
        
    Notes:
        The Bergeron-Findeisen effect occurs when:
        - Air is supersaturated with respect to ice (ssi > 0)
        - Both liquid droplets and ice crystals coexist
        - Cloud droplets evaporate and water vapor deposits onto ice crystals
    """
    # Extract constants
    C_RTMIN = constants["C_RTMIN"]
    I_RTMIN = constants["I_RTMIN"]
    LBI = constants["LBI"]
    LBEXI = constants["LBEXI"]
    O0DEPI = constants["O0DEPI"]
    O2DEPI = constants["O2DEPI"]
    DI = constants["DI"]
    
    # Initialize output
    rc_beri_tnd = jnp.zeros_like(rhodref)
    
    if not ldsoft:
        # Compute mask for Bergeron-Findeisen conditions
        mask = (
            (ssi > 0) & 
            (rct > C_RTMIN) & 
            (rit > I_RTMIN) & 
            (cit > 1e-20) & 
            ldcompute
        )
        
        # Compute ice crystal slope parameter lambda_i
        lambda_i = jnp.minimum(
            1e8, 
            LBI * jnp.power(rhodref * rit / cit, LBEXI)
        )
        
        # Compute deposition rate with ventilation correction
        rc_beri_tnd = (
            (ssi / (rhodref * ai))
            * cit
            * (
                O0DEPI / lambda_i + 
                O2DEPI * jnp.square(cj) / jnp.power(lambda_i, DI + 2.0)
            )
        )
        
        # Apply mask
        rc_beri_tnd = jnp.where(mask, rc_beri_tnd, 0.0)
    
    return rc_beri_tnd
