"""Saturation mixing ratio computations.

Translation of convect_satmixratio.h to JAX.
Computes vapor saturation mixing ratio over liquid water and related thermodynamic quantities.
"""

import jax.numpy as jnp
from jax import Array

from ice3.phyex_common.constants import Constants


def convect_satmixratio(
    ppres: Array,
    pt: Array,
    cst: Constants,
) -> tuple[Array, Array, Array, Array]:
    """Compute vapor saturation mixing ratio over liquid water.
    
    This function determines the saturation mixing ratio and returns values
    for latent heats (L_v, L_s) and specific heat (C_ph).
    
    Args:
        ppres: Pressure (Pa)
        pt: Temperature (K)
        cst: Physical constants
        
    Returns:
        Tuple of (pew, plv, pls, pcph):
            pew: Vapor saturation mixing ratio (kg/kg)
            plv: Latent heat of vaporization L_v (J/kg)
            pls: Latent heat of sublimation L_s (J/kg)
            pcph: Specific heat C_ph (J/kg/K)
            
    Reference:
        Book1,2 of PHYEX documentation (routine CONVECT_SATMIXRATIO)
        Original: P. BECHTOLD, Laboratoire d'Aerologie, 07/11/95
    """
    # Compute PEPS = XRD / XRV
    peps = cst.XRD / cst.XRV
    
    # Temperature bounds to prevent overflow
    zt = jnp.minimum(400.0, jnp.maximum(pt, 10.0))
    
    # Compute saturation vapor pressure and mixing ratio
    # Using Clausius-Clapeyron-like equation
    pew = jnp.exp(cst.XALPW - cst.XBETAW / zt - cst.XGAMW * jnp.log(zt))
    pew = peps * pew / (ppres - pew)
    
    # Compute latent heat of vaporization L_v
    plv = cst.XLVTT + (cst.XCPV - cst.XCL) * (zt - cst.XTT)
    
    # Compute latent heat of sublimation L_s
    pls = cst.XLSTT + (cst.XCPV - cst.XCI) * (zt - cst.XTT)
    
    # Compute specific heat C_ph
    pcph = cst.XCPD + cst.XCPV * pew
    
    return pew, plv, pls, pcph
