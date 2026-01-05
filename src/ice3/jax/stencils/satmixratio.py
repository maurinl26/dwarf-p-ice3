"""
Saturation mixing ratio calculations.

Translated from: fortran/conv/internals/convect_satmixratio.F90
"""

import jax.numpy as jnp
from jax import Array
from typing import Tuple

from .constants import PHYS_CONSTANTS


def convect_satmixratio(
    pres: Array,
    temp: Array,
) -> Tuple[Array, Array, Array, Array]:
    """
    Compute vapor saturation mixing ratio over liquid water.
    
    This function determines saturation mixing ratio and returns values
    for latent heats and specific heat.
    
    Parameters
    ----------
    pres : Array
        Pressure (Pa), shape (klon,) or (...,)
    temp : Array  
        Temperature (K), shape (klon,) or (...,)
        
    Returns
    -------
    ew : Array
        Saturation mixing ratio (kg/kg)
    lv : Array
        Latent heat of vaporization L_v (J/kg)
    ls : Array
        Latent heat of sublimation L_s (J/kg)
    cph : Array
        Specific heat C_ph (J/kg/K)
        
    Notes
    -----
    Uses Bolton (1980) formula for saturation vapor pressure.
    Temperature is bounded to [10, 400] K to prevent overflow.
    """
    
    # Constants
    eps = PHYS_CONSTANTS.eps
    alpw = PHYS_CONSTANTS.alpw
    betaw = PHYS_CONSTANTS.betaw
    gamw = PHYS_CONSTANTS.gamw
    xlvtt = PHYS_CONSTANTS.xlvtt
    xlstt = PHYS_CONSTANTS.xlstt
    xcpv = PHYS_CONSTANTS.cpv
    xcl = PHYS_CONSTANTS.cl
    xci = PHYS_CONSTANTS.ci
    xcpd = PHYS_CONSTANTS.cpd
    xtt = PHYS_CONSTANTS.tt
    
    # Bound temperature to prevent overflow
    t = jnp.clip(temp, 10.0, 400.0)
    
    # Compute saturation vapor pressure using Bolton (1980) formula
    # es = exp(alpw - betaw/T - gamw * log(T))
    es = jnp.exp(alpw - betaw / t - gamw * jnp.log(t))
    
    # Convert to mixing ratio: ew = eps * es / (p - es)
    ew = eps * es / (pres - es)
    
    # Compute latent heat of vaporization (temperature dependent)
    # L_v = L_v(T_t) + (C_pv - C_l) * (T - T_t)
    lv = xlvtt + (xcpv - xcl) * (t - xtt)
    
    # Compute latent heat of sublimation (temperature dependent)
    # L_s = L_s(T_t) + (C_pv - C_i) * (T - T_t)
    ls = xlstt + (xcpv - xci) * (t - xtt)
    
    # Compute specific heat (moisture dependent)
    # C_ph = C_pd + C_pv * e_w
    cph = xcpd + xcpv * ew
    
    return ew, lv, ls, cph
