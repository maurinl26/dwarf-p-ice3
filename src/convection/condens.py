"""Condensation routine for convection.

Translation of convect_condens.F90 to JAX.
Computes temperature, cloud water, and ice content from enthalpy and total water.
"""

import jax.numpy as jnp
from jax import Array

from ice3.phyex_common.constants import Constants


def convect_condens(
    cst: Constants,
    pice: float,
    peps0: float,
    ppres: Array,  # (nit,) - pressure
    pthl: Array,  # (nit,) - enthalpy (J/kg)
    prw: Array,  # (nit,) - total water mixing ratio
    prco: Array,  # (nit,) - cloud water estimate
    prio: Array,  # (nit,) - cloud ice estimate
    pz: Array,  # (nit,) - level height
    xtfrz1: float,  # begin of freezing interval
    xtfrz2: float,  # end of freezing interval
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Compute temperature and cloud condensate from enthalpy and total water.
    
    Condensate is extracted iteratively using saturation mixing ratios.
    
    Args:
        cst: Physical constants
        pice: Flag for ice (1 = yes, 0 = no ice)
        peps0: R_d / R_v
        ppres: Pressure (Pa), shape (nit,)
        pthl: Enthalpy (J/kg), shape (nit,)
        prw: Total water mixing ratio (kg/kg), shape (nit,)
        prco: Cloud water estimate (kg/kg), shape (nit,)
        prio: Cloud ice estimate (kg/kg), shape (nit,)
        pz: Level height (m), shape (nit,)
        xtfrz1: Begin of freezing interval (K)
        xtfrz2: End of freezing interval (K)
        
    Returns:
        Tuple of (pt, pew, prc, pri, plv, pls, pcph):
            pt: Temperature (K), shape (nit,)
            pew: Water saturation mixing ratio (kg/kg), shape (nit,)
            prc: Cloud water mixing ratio (kg/kg), shape (nit,)
            pri: Cloud ice mixing ratio (kg/kg), shape (nit,)
            plv: Latent heat of vaporization L_v (J/kg), shape (nit,)
            pls: Latent heat of sublimation L_s (J/kg), shape (nit,)
            pcph: Specific heat C_ph (J/kg/K), shape (nit,)
            
    Reference:
        Book1,2 of PHYEX documentation (routine CONVECT_CONDENS)
        Original: P. BECHTOLD, Laboratoire d'Aerologie, 07/11/95
    """
    # Initialize specific heat and work arrays
    pcph = cst.XCPD + cst.XCPV * prw
    zwork1 = (1.0 + prw) * cst.XG * pz
    
    # Make first temperature estimate based on lower level r_c and r_i
    pt = jnp.maximum(180.0, jnp.minimum(330.0,
        (pthl + prco * cst.XLVTT + prio * cst.XLSTT - zwork1) / pcph))
    
    # Iteration loop to refine temperature and condensate
    for jiter in range(6):
        # Compute saturation mixing ratios for water and ice
        pew = jnp.exp(cst.XALPW - cst.XBETAW / pt - cst.XGAMW * jnp.log(pt))
        pew = peps0 * pew / (ppres - pew)
        
        zei = jnp.exp(cst.XALPI - cst.XBETAI / pt - cst.XGAMI * jnp.log(pt))
        zei = peps0 * zei / (ppres - zei)
        
        # Compute freezing fraction (linear transition between xtfrz1 and xtfrz2)
        zwork2 = jnp.maximum(0.0, jnp.minimum(1.0,
            (xtfrz1 - pt) / (xtfrz1 - xtfrz2))) * pice
        
        # Compute mixed saturation mixing ratio
        zwork3 = (1.0 - zwork2) * pew + zwork2 * zei
        
        # Compute cloud water and ice
        prc = jnp.maximum(0.0, (1.0 - zwork2) * (prw - zwork3))
        pri = jnp.maximum(0.0, zwork2 * (prw - zwork3))
        
        # Compute latent heats
        plv = cst.XLVTT + (cst.XCPV - cst.XCL) * (pt - cst.XTT)
        pls = cst.XLSTT + (cst.XCPV - cst.XCI) * (pt - cst.XTT)
        
        # Update temperature estimate
        zt = (pthl + prc * plv + pri * pls - zwork1) / pcph
        # Force convergence with relaxation factor 0.4
        pt = jnp.maximum(175.0, jnp.minimum(330.0, pt + (zt - pt) * 0.4))
    
    return pt, pew, prc, pri, plv, pls, pcph
