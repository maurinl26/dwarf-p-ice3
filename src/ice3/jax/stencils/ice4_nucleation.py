"""
ICE4 Nucleation - JAX Implementation

This module computes heterogeneous ice nucleation through water vapor deposition.

Reference:
    PHYEX/src/common/micro/ice4_nucleation.func.h
"""
import jax
import jax.numpy as jnp
from typing import Tuple


def ice4_nucleation(
    tht: jnp.ndarray,
    pabst: jnp.ndarray,
    rhodref: jnp.ndarray,
    exn: jnp.ndarray,
    lsfact: jnp.ndarray,
    t: jnp.ndarray,
    rvt: jnp.ndarray,
    cit: jnp.ndarray,
    ldcompute: jnp.ndarray,
    lfeedbackt: bool,
    constants: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute heterogeneous ice nucleation (HENI process).
    
    The nucleation rate depends on temperature and supersaturation over ice.
    Three temperature regimes:
    - T < -5°C: NU20 parameterization with supersaturation dependency
    - -5°C ≤ T < -2°C: Transition using max of two parameterizations  
    - T ≥ -2°C: No nucleation
    
    Args:
        tht: Potential temperature (K)
        pabst: Absolute pressure (Pa)
        rhodref: Reference air density (kg/m³)
        exn: Exner function (dimensionless)
        lsfact: Latent heat factor for sublimation (K·kg/kg)
        t: Temperature (K)
        rvt: Water vapor mixing ratio (kg/kg)
        cit: Ice crystal number concentration (1/kg)
        ldcompute: Computation mask
        lfeedbackt: Temperature feedback flag
        constants: Dictionary of physical constants
        
    Returns:
        Tuple of (cit_new, rvheni_mr, ssi)
    """
    # Extract constants
    V_RTMIN = constants["V_RTMIN"]
    TT = constants["TT"]
    ALPW = constants["ALPW"]
    BETAW = constants["BETAW"]
    GAMW = constants["GAMW"]
    ALPI = constants["ALPI"]
    BETAI = constants["BETAI"]
    GAMI = constants["GAMI"]
    EPSILO = constants["EPSILO"]
    NU10 = constants["NU10"]
    NU20 = constants["NU20"]
    ALPHA1 = constants["ALPHA1"]
    ALPHA2 = constants["ALPHA2"]
    BETA1 = constants["BETA1"]
    BETA2 = constants["BETA2"]
    MNU0 = constants["MNU0"]
    
    # Initialize outputs
    rvheni_mr = jnp.zeros_like(rhodref)
    ssi = jnp.zeros_like(rhodref)
    cit_new = cit.copy()
    
    # Compute mask for nucleation conditions
    mask = (t < TT) & (rvt > V_RTMIN) & ldcompute
    
    # Compute saturation vapor pressures
    log_t = jnp.log(t)
    
    # Saturation over water
    usw = jnp.exp(ALPW - BETAW / t - GAMW * log_t)
    
    # Saturation over ice
    zw = jnp.exp(ALPI - BETAI / t - GAMI * log_t)
    
    # Limit saturation pressures
    zw = jnp.minimum(pabst / 2.0, zw)
    
    # Supersaturation over ice
    ssi = rvt * (pabst - zw) / (EPSILO * zw) - 1.0
    
    # Supersaturation of water-saturated air over ice
    usw = jnp.minimum(pabst / 2.0, usw)
    usw_over_ice = (usw / zw) * ((pabst - zw) / (pabst - usw)) - 1.0
    
    # Limit ssi to water saturation
    ssi = jnp.minimum(ssi, usw_over_ice)
    
    # Apply mask to ssi
    ssi = jnp.where(mask, ssi, 0.0)
    
    # Compute nucleation rate based on temperature regime
    zw_nucl = jnp.zeros_like(rhodref)
    
    # Regime 1: T < -5°C
    mask_cold = mask & (t < TT - 5.0) & (ssi > 0.0)
    zw_nucl = jnp.where(
        mask_cold,
        NU20 * jnp.exp(ALPHA2 * ssi - BETA2),
        zw_nucl
    )
    
    # Regime 2: -5°C ≤ T < -2°C
    mask_transition = mask & (t >= TT - 5.0) & (t <= TT - 2.0) & (ssi > 0.0)
    nucl_option1 = NU20 * jnp.exp(-BETA2)
    nucl_option2 = NU10 * jnp.exp(-BETA1 * (t - TT)) * jnp.power(ssi / usw_over_ice, ALPHA1)
    
    zw_nucl = jnp.where(
        mask_transition,
        jnp.maximum(nucl_option1, nucl_option2),
        zw_nucl
    )
    
    # Net nucleation (zw_nucl - existing concentration)
    zw_nucl = jnp.where(mask, zw_nucl - cit, 0.0)
    zw_nucl = jnp.where(mask, jnp.minimum(zw_nucl, 5e4), 0.0)
    
    # Vapor mixing ratio consumed by nucleation
    rvheni_mr = jnp.where(
        mask,
        jnp.maximum(zw_nucl, 0.0) * MNU0 / rhodref,
        0.0
    )
    rvheni_mr = jnp.where(mask, jnp.minimum(rvt, rvheni_mr), 0.0)
    
    # Temperature feedback limitation
    if lfeedbackt:
        # Limit to prevent T > TT
        w1 = jnp.where(
            mask,
            jnp.minimum(
                rvheni_mr,
                jnp.maximum(0.0, (TT / exn - tht)) / lsfact
            ) / jnp.maximum(rvheni_mr, 1e-20),
            1.0
        )
        
        rvheni_mr = rvheni_mr * w1
        zw_nucl = zw_nucl * w1
    
    # Update ice crystal concentration
    cit_new = jnp.where(
        mask,
        jnp.maximum(zw_nucl + cit, cit),
        cit
    )
    
    return cit_new, rvheni_mr, ssi
