"""Mixing function for convection entrainment/detrainment.

Translation of convect_mixing_funct.F90 to JAX.
Determines entrainment and detrainment rates by evaluating the area under a distribution function.
"""

import jax.numpy as jnp
from jax import Array


def convect_mixing_funct(
    pmixc: Array,  # (nit,) - critical mixed fraction
    kmf: int = 1,  # switch for distribution function (1=gaussian, 2=triangular)
) -> tuple[Array, Array]:
    """Determine normalized entrainment and detrainment rates.
    
    The purpose of this routine is to determine the entrainment and
    detrainment rate by evaluating the area under the distribution function.
    The integration interval is limited by the critical mixed fraction PMIXC.
    
    Args:
        pmixc: Critical mixed fraction (dimensionless), shape (nit,)
        kmf: Switch for distribution function (1=gaussian, 2=triangular), default=1
        
    Returns:
        Tuple of (per, pdr):
            per: Normalized entrainment rate (dimensionless), shape (nit,)
            pdr: Normalized detrainment rate (dimensionless), shape (nit,)
            
    Reference:
        Book2 of PHYEX documentation (routine MIXING_FUNCT)
        Abramowitz and Stegun (1968), handbook of math. functions
        Original: P. BECHTOLD, Laboratoire d'Aerologie, 07/11/95
    """
    # Constants for Gaussian distribution
    ZSIGMA = 0.166666667  # standard deviation
    ZFE = 4.931813949  # integral normalization
    ZSQRTP = 2.506628  # sqrt(2*pi)
    ZP = 0.33267
    ZA1 = 0.4361836
    ZA2 = -0.1201676
    ZA3 = 0.9372980
    ZT1 = 0.500498
    ZE45 = 0.01111
    
    # Use Gaussian function (KMF=1)
    if kmf == 1:
        # Transform critical mixing fraction
        zx = 6.0 * pmixc - 3.0
        
        # Compute work variables for error function approximation
        zw1 = 1.0 / (1.0 + ZP * jnp.abs(zx))
        zy = jnp.exp(-0.5 * zx * zx)
        zw2 = ZA1 * zw1 + ZA2 * zw1 ** 2 + ZA3 * zw1 ** 3
        zw11 = ZA1 * ZT1 + ZA2 * ZT1 ** 2 + ZA3 * ZT1 ** 3
        
        # Compute entrainment and detrainment for positive zx
        per_pos = (ZSIGMA * (0.5 * (ZSQRTP - ZE45 * zw11 - zy * zw2) + 
                   ZSIGMA * (ZE45 - zy)) - 
                   0.5 * ZE45 * pmixc ** 2)
        
        pdr_pos = (ZSIGMA * (0.5 * (zy * zw2 - ZE45 * zw11) + 
                   ZSIGMA * (ZE45 - zy)) - 
                   ZE45 * (0.5 + 0.5 * pmixc ** 2 - pmixc))
        
        # Compute entrainment and detrainment for negative zx
        per_neg = (ZSIGMA * (0.5 * (zy * zw2 - ZE45 * zw11) + 
                   ZSIGMA * (ZE45 - zy)) - 
                   0.5 * ZE45 * pmixc ** 2)
        
        pdr_neg = (ZSIGMA * (0.5 * (ZSQRTP - ZE45 * zw11 - zy * zw2) + 
                   ZSIGMA * (ZE45 - zy)) - 
                   ZE45 * (0.5 + 0.5 * pmixc ** 2 - pmixc))
        
        # Select based on sign of zx
        per = jnp.where(zx >= 0.0, per_pos, per_neg)
        pdr = jnp.where(zx >= 0.0, pdr_pos, pdr_neg)
        
        # Apply normalization factor
        per = per * ZFE
        pdr = pdr * ZFE
        
    else:
        # Triangular distribution (KMF=2) - not yet implemented
        # Return zeros for now
        per = jnp.zeros_like(pmixc)
        pdr = jnp.zeros_like(pmixc)
    
    return per, pdr
