"""
ICE4 RRHONG - JAX Implementation

This module computes spontaneous (homogeneous) freezing of supercooled rain drops.

Reference:
    PHYEX/src/common/micro/mode_ice4_rrhong.F90
"""
import jax
import jax.numpy as jnp


def ice4_rrhong(
    t: jnp.ndarray,
    exn: jnp.ndarray,
    lvfact: jnp.ndarray,
    lsfact: jnp.ndarray,
    tht: jnp.ndarray,
    rrt: jnp.ndarray,
    ldcompute: jnp.ndarray,
    lfeedbackt: bool,
    constants: dict,
) -> jnp.ndarray:
    """
    Compute spontaneous freezing of supercooled rain (RRHONG process).
    
    This function computes instantaneous freezing of liquid rain drops
    at very cold temperatures (T < -35°C). The frozen rain becomes graupel.
    
    Args:
        t: Temperature (K)
        exn: Exner function (dimensionless)
        lvfact: Vaporization latent heat factor Lv/(cph×π) (K)
        lsfact: Sublimation latent heat factor Ls/(cph×π) (K)
        tht: Potential temperature θ (K)
        rrt: Rain mixing ratio (kg/kg)
        ldcompute: Computation mask
        lfeedbackt: Temperature feedback flag
        constants: Dictionary of physical constants
        
    Returns:
        rrhong_mr: Rain homogeneous freezing rate rr → rg (kg/kg/s)
        
    Notes:
        When T < -35°C, supercooled rain freezes spontaneously to graupel.
        With temperature feedback, freezing is limited to prevent
        temperature from rising above -35°C due to latent heat release.
    """
    # Extract constants
    TT = constants["TT"]
    R_RTMIN = constants["R_RTMIN"]
    
    # Initialize output
    rrhong_mr = jnp.zeros_like(t)
    
    # Compute mask for freezing conditions (T < -35°C)
    mask = (t < TT - 35.0) & (rrt > R_RTMIN) & ldcompute
    
    # All rain freezes below -35°C
    rrhong_mr = jnp.where(mask, rrt, 0.0)
    
    # Temperature feedback limitation
    if lfeedbackt:
        # Limit freezing to prevent T from rising above -35°C
        # Maximum rain that can freeze: ((TT-35)/π - θ) / (Ls_fact - Lv_fact)
        max_freeze = jnp.maximum(0.0, ((TT - 35.0) / exn - tht) / (lsfact - lvfact))
        rrhong_mr = jnp.where(mask, jnp.minimum(rrhong_mr, max_freeze), 0.0)
    
    return rrhong_mr
