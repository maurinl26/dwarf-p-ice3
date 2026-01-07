"""
Bougeault-Lacarrere (1989) mixing length computation.

This module implements the BL89 mixing length scheme, which computes
the turbulent mixing length based on vertical displacement of an air parcel
having an initial kinetic energy equal to its TKE.

Physical Principle
-----------------
The mixing length is determined by the maximum vertical distance an air parcel
can travel before being stopped by buoyancy forces. The parcel starts with
kinetic energy equal to the local TKE and loses energy through buoyancy work.

The energy balance equation for a displaced parcel is:

    e(z) = e₀ - ∫[z₀ to z] (g/θ_v) Δθ_v dz'

where:
- e₀: Initial TKE at starting level
- Δθ_v: Virtual potential temperature difference from environment
- The integral represents buoyancy work

The mixing length at level k is computed as:

    L_m(k) = [L_up(k) · L_down(k) / (L_up(k) + L_down(k))]^α

where:
- L_up: Maximum upward displacement distance
- L_down: Maximum downward displacement distance
- α = 2/3 (default exponent, Bougeault-Lacarrere 1989)

The RM17 variant includes a shear term in the energy budget:

    e(z) = e₀ - ∫[z₀ to z] [(g/θ_v) Δθ_v - C_s S √e₀] dz'

where C_s = 0.5 (Rodier et al. 2017) and S is the wind shear.

Key References
-------------
- Bougeault, P., and P. Lacarrere, 1989: Parameterization of orography-induced
  turbulence in a mesobeta-scale model. Mon. Wea. Rev., 117, 1872-1890.
  https://doi.org/10.1175/1520-0493(1989)117<1872:POOITI>2.0.CO;2
  (Original formulation, Equations 6-9)

- Rodier, Q., H. Masson, E. Couvreux, and A. Paci, 2017: Evaluation of a
  buoyancy and shear based mixing length for a turbulence scheme.
  Bound.-Layer Meteor., 165, 401-419.
  https://doi.org/10.1007/s10546-017-0272-3
  (Enhanced version with shear term, Equation 12)

Source Code
----------
Translated from PHYEX-IAL_CY50T1/turb/mode_bl89.F90
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Tuple


def compute_bl89_mixing_length(
    zz: Array,
    dzz: Array,
    thvref: Array,
    thlm: Array,
    rm: Array,
    tke: Array,
    shear: Array,
    constants: dict,
    phys_constants: dict,
    xlini: float = 10.0,
    lbl89top: bool = False,
    ocean: bool = False,
) -> Array:
    """
    Compute Bougeault-Lacarrere (1989) mixing length.

    The mixing length is based on the vertical displacement of an air parcel
    having an initial internal energy equal to its TKE and stopped by buoyancy forces.

    Algorithm
    ---------
    For each level k:
    1. Compute L_down: integrate downward from k until energy is depleted
    2. Compute L_up: integrate upward from k until energy is depleted
    3. Combine using: L_m = L_down · (2 / (1 + (L_down/L_up)^α))^(1/α)
       where α = xbl89exp (default 2/3)

    The energy equation at each integration step:
        de/dz = -(g/θ_v)(θ_v - θ_v,ref) + C_s S √e
    where the second term (shear) is the RM17 enhancement.

    Parameters
    ----------
    zz : Array
        Physical height (m), shape (nz,)
    dzz : Array
        Vertical grid spacing (m), shape (nz,)
    thvref : Array
        Virtual potential temperature reference (K), shape (nz,)
    thlm : Array
        Liquid potential temperature (K), shape (nz,)
    rm : Array
        Mixing ratios (kg/kg), shape (nz, krr)
        rm[..., 0] is water vapor mixing ratio
    tke : Array
        Turbulent kinetic energy (m²/s²), shape (nz,)
    shear : Array
        Wind shear (s⁻¹), shape (nz,)
    constants : dict
        Turbulence constants (CSTURB)
    phys_constants : dict
        Physical constants (CST) with keys 'g', 'rd', 'rv'
    xlini : float, optional
        Minimum mixing length (m). Default: 10.0
    lbl89top : bool, optional
        Apply BL89 top constraint. Default: False
    ocean : bool, optional
        Ocean model mode. Default: False

    Returns
    -------
    Array
        Mixing length (m), shape (nz,)

    References
    ----------
    Bougeault, P., and P. Lacarrere, 1989: Parameterization of orography-induced
    turbulence in a mesobeta-scale model. Mon. Wea. Rev., 117, 1872-1890.

    Rodier, Q., H. Masson, E. Couvreux, and A. Paci, 2017: Evaluation of a
    buoyancy and shear based mixing length for a turbulence scheme.
    Bound.-Layer Meteor., 165, 401-419.

    Fortran Source
    --------------
    PHYEX-IAL_CY50T1/turb/mode_bl89.F90
    Subroutine: BL89 (main mixing length computation)
    """
    nz = zz.shape[0]
    krr = rm.shape[1] if rm.ndim > 1 else 0

    # Extract constants
    g = phys_constants['g']
    rv = phys_constants['rv']
    rd = phys_constants['rd']
    zrvord = rv / rd
    xrm17 = constants.get('xrm17', 0.5)
    xlinf = constants.get('xlinf', 1.0e-10)
    xbl89exp = constants.get('xbl89exp', 2.0/3.0)
    xusrbl89 = constants.get('xusrbl89', 1.5)

    # Compute g/thvref or alpha for ocean
    if ocean:
        alphaoc = phys_constants.get('alphaoc', 1.9e-4)
        g_o_thvref = jnp.full_like(thvref, g * alphaoc)
    else:
        g_o_thvref = g / thvref

    # Compute sqrt(TKE)
    sqrt_tke = jnp.sqrt(tke)

    # Compute virtual potential temperature on model grid
    if krr > 0:
        # Sum all mixing ratios
        rm_sum = jnp.sum(rm, axis=-1)
        zvpt = thlm * (1.0 + zrvord * rm[..., 0]) / (1.0 + rm_sum)
    else:
        zvpt = thlm

    # Compute ZVPT differences and half-level values
    # Use second-order centered differences
    zdeltvpt = jnp.zeros_like(zvpt)
    zhlvpt = jnp.zeros_like(zvpt)

    # Interior points
    zdeltvpt = zdeltvpt.at[1:-1].set(zvpt[1:-1] - zvpt[:-2])
    zhlvpt = zhlvpt.at[1:-1].set(0.5 * (zvpt[1:-1] + zvpt[:-2]))

    # Boundaries
    zdeltvpt = zdeltvpt.at[-1].set(zvpt[-1] - zvpt[-2])
    zdeltvpt = zdeltvpt.at[0].set(0.0)
    zhlvpt = zhlvpt.at[-1].set(0.5 * (zvpt[-1] + zvpt[-2]))
    zhlvpt = zhlvpt.at[0].set(zvpt[0])

    # Prevent division by zero
    zdeltvpt = jnp.where(jnp.abs(zdeltvpt) < xlinf, xlinf, zdeltvpt)

    # Initialize mixing length arrays
    lm_down = jnp.zeros_like(zvpt)
    lm_up = jnp.zeros_like(zvpt)

    # Compute downward and upward mixing lengths
    # This is a simplified vectorized version - the full BL89 requires iterative computation
    # For a functional version, we approximate using energy balance

    # Downward mixing length (from level k down to surface)
    for k in range(1, nz-1):
        # Initial energy at level k
        inte = tke[k]

        # Integrate downward
        lwork = 0.0
        for kk in range(k, 0, -1):
            if inte > 0:
                # Potential energy change
                zpote = (-g_o_thvref[k] * (zhlvpt[kk] - zvpt[k]) +
                        xrm17 * shear[kk] * sqrt_tke[k]) * dzz[kk]

                if inte > zpote:
                    # Full level can be reached
                    lwork += dzz[kk]
                    inte -= zpote
                else:
                    # Partial level
                    # Solve quadratic equation for exact height
                    a = g_o_thvref[k] * zdeltvpt[kk] / dzz[kk]
                    b = (g_o_thvref[k] * (zvpt[kk] - zvpt[k]) -
                         xrm17 * shear[kk] * sqrt_tke[k])
                    c = -inte

                    if jnp.abs(a) > xlinf:
                        disc = b**2 - 4*a*c
                        if disc > 0:
                            lwork += (-b + jnp.sqrt(disc)) / (2*a)
                    else:
                        lwork += -c / b if jnp.abs(b) > xlinf else dzz[kk]
                    break

        # Limit to distance from surface
        lm_down = lm_down.at[k].set(jnp.minimum(lwork, zz[k] - zz[0]))

    # Upward mixing length (from level k up to top)
    for k in range(1, nz-1):
        # Initial energy at level k
        inte = tke[k]

        # Integrate upward
        lwork = 0.0
        for kk in range(k+1, nz):
            if inte > 0:
                # Potential energy change
                zpote = (g_o_thvref[k] * (zhlvpt[kk] - zvpt[k]) +
                        xrm17 * shear[kk] * sqrt_tke[k]) * dzz[kk]

                if inte > zpote:
                    # Full level can be reached
                    lwork += dzz[kk]
                    inte -= zpote
                else:
                    # Partial level
                    a = g_o_thvref[k] * zdeltvpt[kk] / dzz[kk]
                    b = (-g_o_thvref[k] * (zvpt[kk-1] - zvpt[k]) -
                         xrm17 * shear[kk] * sqrt_tke[k])
                    c = -inte

                    if jnp.abs(a) > xlinf:
                        disc = b**2 - 4*a*c
                        if disc > 0:
                            lwork += (-b + jnp.sqrt(disc)) / (2*a)
                    else:
                        lwork += -c / b if jnp.abs(b) > xlinf else dzz[kk]
                    break

        lm_up = lm_up.at[k].set(lwork)

    # Apply BL89 top constraint if requested
    if lbl89top:
        for k in range(nz-2, 0, -1):
            lm_up = lm_up.at[k].set(jnp.maximum(lm_up[k], lm_up[k+1] - dzz[k+1]))

    # Final mixing length (geometric mean with exponent)
    lm_down_safe = jnp.maximum(lm_down, 1.0e-10)
    lm_up_safe = jnp.maximum(lm_up, 1.0e-10)

    # BL89 formula: L = Ldown * (2 / (1 + (Ldown/Lup)^exp))^(1/exp)
    ratio = lm_down_safe / lm_up_safe
    lm = lm_down_safe * (2.0 / (1.0 + ratio**xbl89exp))**xusrbl89

    # Apply minimum mixing length
    lm = jnp.maximum(lm, xlini)

    # Set boundaries
    lm = lm.at[0].set(lm[1])
    lm = lm.at[-2:].set(lm[-3])

    return lm


def compute_bl89_mixing_length_vectorized(
    zz: Array,
    dzz: Array,
    thvref: Array,
    thlm: Array,
    rm: Array,
    tke: Array,
    shear: Array,
    constants: dict,
    phys_constants: dict,
    xlini: float = 10.0,
    lbl89top: bool = False,
    ocean: bool = False,
) -> Array:
    """
    Vectorized version of BL89 mixing length (simplified).

    This is a simplified vectorized approximation of the full BL89 algorithm.
    For production use, the full iterative version should be used.

    Parameters are the same as compute_bl89_mixing_length.

    Returns
    -------
    Array
        Mixing length (m), shape (nz,)

    Fortran Source
    --------------
    PHYEX-IAL_CY50T1/turb/mode_bl89.F90
    Simplified JAX-compatible version using gradient-based approximation
    """
    nz = zz.shape[0]
    krr = rm.shape[1] if rm.ndim > 1 else 0

    # Extract constants
    g = phys_constants['g']
    rv = phys_constants['rv']
    rd = phys_constants['rd']
    zrvord = rv / rd
    xrm17 = constants.get('xrm17', 0.5)
    xlinf = constants.get('xlinf', 1.0e-10)
    xbl89exp = constants.get('xbl89exp', 2.0/3.0)
    xusrbl89 = constants.get('xusrbl89', 1.5)

    # Compute g/thvref
    if ocean:
        alphaoc = phys_constants.get('alphaoc', 1.9e-4)
        g_o_thvref = jnp.full_like(thvref, g * alphaoc)
    else:
        g_o_thvref = g / thvref

    sqrt_tke = jnp.sqrt(tke)

    # Virtual potential temperature
    if krr > 0:
        rm_sum = jnp.sum(rm, axis=-1)
        zvpt = thlm * (1.0 + zrvord * rm[..., 0]) / (1.0 + rm_sum)
    else:
        zvpt = thlm

    # Simplified estimate using characteristic length scale
    # L_down ~ sqrt(2 * TKE / (g/theta_v * dtheta_v/dz))
    # where dtheta_v/dz is the Brunt-Väisälä frequency squared

    dzvpt_dz = jnp.gradient(zvpt) / jnp.gradient(zz)
    n2 = g_o_thvref * dzvpt_dz  # Brunt-Väisälä frequency squared

    # Avoid division by small or negative N²
    n2_safe = jnp.where(n2 > 1e-6, n2, 1e-6)

    # Characteristic displacement scale
    lm_down = jnp.sqrt(2.0 * tke / n2_safe)
    lm_up = lm_down  # Simplified: assume symmetric

    # Limit to distance from surface/top
    lm_down = jnp.minimum(lm_down, zz - zz[0])
    lm_up = jnp.minimum(lm_up, zz[-1] - zz)

    # BL89 formula
    lm_down_safe = jnp.maximum(lm_down, 1.0e-10)
    lm_up_safe = jnp.maximum(lm_up, 1.0e-10)
    ratio = lm_down_safe / lm_up_safe
    lm = lm_down_safe * (2.0 / (1.0 + ratio**xbl89exp))**xusrbl89

    # Apply minimum
    lm = jnp.maximum(lm, xlini)

    return lm
