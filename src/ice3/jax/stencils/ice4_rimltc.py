"""
ICE4 RIMLTC - JAX Implementation

This module computes melting of cloud ice crystals above freezing temperature.

Reference:
    PHYEX/src/common/micro/mode_ice4_rimltc.F90
"""
import jax
import jax.numpy as jnp


def ice4_rimltc(
    t: jnp.ndarray,
    exn: jnp.ndarray,
    lvfact: jnp.ndarray,
    lsfact: jnp.ndarray,
    tht: jnp.ndarray,
    rit: jnp.ndarray,
    ldcompute: jnp.ndarray,
    lfeedbackt: bool,
    constants: dict,
) -> jnp.ndarray:
    """
    Compute melting of cloud ice crystals (RIMLTC process).
    
    This function computes instantaneous melting of ice crystals when
    temperature rises above 0°C. The phase transition converts ice
    crystals to cloud droplets.
    
    Args:
        t: Temperature (K)
        exn: Exner function (dimensionless)
        lvfact: Vaporization latent heat factor Lv/(cph×π) (K)
        lsfact: Sublimation latent heat factor Ls/(cph×π) (K)
        tht: Potential temperature θ (K)
        rit: Ice crystal mixing ratio (kg/kg)
        ldcompute: Computation mask
        lfeedbackt: Temperature feedback flag
        constants: Dictionary of physical constants
        
    Returns:
        rimltc_mr: Ice melting rate ri → rc (kg/kg/s)
        
    Notes:
        When T > 0°C, ice crystals melt to cloud droplets.
        With temperature feedback, melting is limited to prevent
        temperature from dropping below freezing due to latent
        heat consumption.
    """
    # Extract constants
    TT = constants["TT"]
    
    # Initialize output
    rimltc_mr = jnp.zeros_like(t)
    
    # Compute mask for melting conditions
    mask = (rit > 0.0) & (t > TT) & ldcompute
    
    # All ice melts above freezing
    rimltc_mr = jnp.where(mask, rit, 0.0)
    
    # Temperature feedback limitation
    if lfeedbackt:
        # Limit melting to prevent T from dropping below freezing
        # Maximum ice that can melt: (θ - TT/π) / (Ls_fact - Lv_fact)
        max_melt = jnp.maximum(0.0, (tht - TT / exn) / (lsfact - lvfact))
        rimltc_mr = jnp.where(mask, jnp.minimum(rimltc_mr, max_melt), 0.0)
    
    return rimltc_mr
