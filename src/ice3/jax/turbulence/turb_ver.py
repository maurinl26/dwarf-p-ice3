"""
Vertical turbulent fluxes and diffusion.

This module computes vertical turbulent fluxes of momentum, heat, and moisture
using a 1.5-order closure with implicit time integration.

Theoretical Framework
--------------------

1. **Turbulent Flux Parameterization** (Redelsperger & Sommeria 1981, Eq. 2.5-2.7):

   The vertical turbulent fluxes are parameterized using K-theory:

   u'w' = -K_m ∂u/∂z    (momentum flux)
   v'w' = -K_m ∂v/∂z    (momentum flux)
   w'θ' = -K_h ∂θ/∂z    (heat flux)
   w'r' = -K_h ∂r/∂z    (moisture flux)

2. **Eddy Diffusivity** (Cuxart et al. 2000, Eq. 2):

   K_m = C_k L_m √e

   where:
   - C_k ≈ 0.143 is the momentum flux constant (XCMFS)
   - L_m is the mixing length (Bougeault-Lacarrere 1989)
   - e is the TKE

3. **Turbulent Prandtl Number** (Redelsperger & Sommeria 1981):

   Pr_t = K_m / K_h

   The Prandtl number depends on atmospheric stability (Richardson number):
   - Neutral conditions (Ri ≈ 0): Pr_t ≈ 1/3
   - Stable conditions (Ri > 0): Pr_t increases (reduced heat mixing)
   - Unstable conditions (Ri < 0): Pr_t decreases (enhanced convection)

4. **Implicit Time Integration**:

   The vertical diffusion equation:

   ∂φ/∂t = ∂/∂z(K ∂φ/∂z)

   is discretized implicitly as:

   φ^(n+1) - φ^n = Δt ∂/∂z(K ∂φ^(n+1)/∂z)

   This leads to a tridiagonal system solved using the Thomas algorithm.
   The implicit scheme ensures stability for large time steps (CFL-free).

5. **Boundary Conditions**:

   - Lower boundary: prescribed surface flux (e.g., from surface layer scheme)
   - Upper boundary: typically zero flux (free atmosphere)

Key References
-------------
- Redelsperger, J.-L., and G. Sommeria, 1981: Méthode de représentation de
  la turbulence d'echelle inférieure à la maille pour un modèle
  tri-dimensionnel de convection nuageuse. Boundary-Layer Meteor., 21, 509-530.
  https://doi.org/10.1007/BF02033592
  (K-theory formulation, Section 2.2)

- Cuxart, J., P. Bougeault, and J.-L. Redelsperger, 2000: A turbulence scheme
  allowing for mesoscale and large-eddy simulations. Q. J. R. Meteorol. Soc.,
  126, 1-30. https://doi.org/10.1002/qj.49712656202
  (Complete implementation, Section 2)

- Thomas, L. H., 1949: Elliptic problems in linear differential equations over
  a network. Watson Sci. Comput. Lab Report, Columbia University, New York.
  (Tridiagonal matrix algorithm)

Source Code
----------
Translated from PHYEX-IAL_CY50T1/turb/mode_turb_ver.F90
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Tuple, Dict


def compute_prandtl_number(
    tke: Array,
    lm: Array,
    shear: Array,
    buoy_gradient: Array,
    constants: dict,
) -> Tuple[Array, Array]:
    """
    Compute turbulent Prandtl number.

    The Prandtl number relates momentum and heat diffusivities:
    Pr_t = K_m / K_h

    Parameters
    ----------
    tke : Array
        Turbulent kinetic energy (m²/s²), shape (nz,)
    lm : Array
        Mixing length (m), shape (nz,)
    shear : Array
        Wind shear squared (s⁻²), shape (nz,)
    buoy_gradient : Array
        Buoyancy gradient g/θ * dθ/dz (s⁻²), shape (nz,)
    constants : dict
        Turbulence constants

    Returns
    -------
    prt : Array
        Turbulent Prandtl number (dimensionless), shape (nz,)
    km : Array
        Momentum diffusivity (m²/s), shape (nz,)

    Fortran Source
    --------------
    PHYEX-IAL_CY50T1/turb/mode_prandtl.F90
    Subroutine: PRANDTL (compute turbulent Prandtl/Schmidt numbers)
    """
    # Extract constants
    xcmfs = constants.get('xcmfs', 0.143)
    xcmfb = constants.get('xcmfb', 0.19)
    xcshf = constants.get('xcshf', 0.143)
    xcpr1 = constants.get('xcpr1', 0.19)

    sqrt_tke = jnp.sqrt(tke)

    # Momentum diffusivity: K_m = C_k * L_m * sqrt(TKE)
    km = xcmfs * lm * sqrt_tke

    # Richardson number
    ri = buoy_gradient / jnp.maximum(shear, 1.0e-10)
    ri = jnp.clip(ri, -10.0, 10.0)

    # Stability functions (simplified)
    # For neutral: Pr_t ~ 1/3
    # For stable (Ri > 0): Pr_t increases
    # For unstable (Ri < 0): Pr_t decreases

    # Simplified Prandtl number formulation
    prt = jnp.where(
        ri >= 0.0,
        # Stable: increase Prandtl number
        1.0 / 3.0 + ri / (1.0 + 3.0 * ri),
        # Unstable: decrease Prandtl number
        (1.0 / 3.0) * (1.0 - 10.0 * ri) / (1.0 - 5.0 * ri)
    )

    # Limit Prandtl number
    prt = jnp.clip(prt, 0.1, 10.0)

    return prt, km


def turb_ver_implicit(
    u: Array,
    v: Array,
    thl: Array,
    rt: Array,
    tke: Array,
    prt: Array,
    km: Array,
    dzz: Array,
    dt: float,
    surf_flux_u: float,
    surf_flux_v: float,
    surf_flux_th: float,
    surf_flux_rv: float,
    ximpl: float = 1.0,
) -> Tuple[Array, Array, Array, Array]:
    """
    Compute vertical turbulent tendencies using implicit scheme.

    Solves the vertical diffusion equation implicitly using a tridiagonal solver.

    Parameters
    ----------
    u, v : Array
        Horizontal wind components (m/s), shape (nz,)
    thl : Array
        Liquid potential temperature (K), shape (nz,)
    rt : Array
        Total water mixing ratio (kg/kg), shape (nz,)
    tke : Array
        Turbulent kinetic energy (m²/s²), shape (nz,)
    prt : Array
        Turbulent Prandtl number, shape (nz,)
    km : Array
        Momentum diffusivity (m²/s), shape (nz,)
    dzz : Array
        Vertical grid spacing (m), shape (nz,)
    dt : float
        Time step (s)
    surf_flux_u, surf_flux_v : float
        Surface momentum fluxes (m²/s²)
    surf_flux_th : float
        Surface heat flux (K·m/s)
    surf_flux_rv : float
        Surface moisture flux (kg/kg·m/s)
    ximpl : float, optional
        Degree of implicitness (0=explicit, 1=fully implicit). Default: 1.0

    Returns
    -------
    du_dt : Array
        U tendency (m/s²), shape (nz,)
    dv_dt : Array
        V tendency (m/s²), shape (nz,)
    dthl_dt : Array
        Theta_l tendency (K/s), shape (nz,)
    drt_dt : Array
        R_t tendency (kg/kg/s), shape (nz,)

    Fortran Source
    --------------
    PHYEX-IAL_CY50T1/turb/mode_turb_ver.F90
    Subroutine: TURB_VER_THERMO_FLUX (implicit vertical diffusion solver)
    Uses tridiagonal matrix algorithm (Thomas algorithm)
    """
    nz = u.shape[0]

    # Heat diffusivity: K_h = K_m / Pr_t
    kh = km / prt

    # Compute fluxes at half-levels (w-points)
    # Flux = -K * dvar/dz

    # For implicit scheme, we need to solve:
    # var(n+1) - var(n) = dt * d/dz(K * dvar(n+1)/dz)
    #
    # Discretized:
    # (var[k]^(n+1) - var[k]^n) / dt =
    #     (flux[k+1/2] - flux[k-1/2]) / dzz[k]
    #
    # where flux[k+1/2] = -K[k+1/2] * (var[k+1] - var[k]) / dzz[k+1/2]

    # Build tridiagonal system: A * var^(n+1) = B
    # where A is tridiagonal matrix

    def solve_implicit_diffusion(var, k_diff, surf_flux, top_flux=0.0):
        """Solve implicit vertical diffusion equation."""
        # Half-level diffusivities (average)
        k_half = jnp.zeros(nz + 1)
        k_half = k_half.at[1:-1].set(0.5 * (k_diff[:-1] + k_diff[1:]))
        k_half = k_half.at[0].set(k_diff[0])
        k_half = k_half.at[-1].set(k_diff[-1])

        # Half-level grid spacings
        dzz_half = jnp.zeros(nz + 1)
        dzz_half = dzz_half.at[1:-1].set(0.5 * (dzz[:-1] + dzz[1:]))
        dzz_half = dzz_half.at[0].set(dzz[0])
        dzz_half = dzz_half.at[-1].set(dzz[-1])

        # Tridiagonal coefficients
        # Lower diagonal
        a = jnp.zeros(nz)
        # Main diagonal
        b = jnp.ones(nz)
        # Upper diagonal
        c = jnp.zeros(nz)
        # Right-hand side
        rhs = var.copy()

        # Interior points
        for k in range(1, nz - 1):
            coef_lower = ximpl * dt * k_half[k] / (dzz[k] * dzz_half[k])
            coef_upper = ximpl * dt * k_half[k + 1] / (dzz[k] * dzz_half[k + 1])

            a = a.at[k].set(-coef_lower)
            b = b.at[k].set(1.0 + coef_lower + coef_upper)
            c = c.at[k].set(-coef_upper)

        # Bottom boundary (surface flux)
        coef_upper = ximpl * dt * k_half[1] / (dzz[0] * dzz_half[1])
        b = b.at[0].set(1.0 + coef_upper)
        c = c.at[0].set(-coef_upper)
        rhs = rhs.at[0].add(dt * surf_flux / dzz[0])

        # Top boundary (zero flux or specified)
        coef_lower = ximpl * dt * k_half[-2] / (dzz[-1] * dzz_half[-2])
        a = a.at[-1].set(-coef_lower)
        b = b.at[-1].set(1.0 + coef_lower)
        rhs = rhs.at[-1].add(dt * top_flux / dzz[-1])

        # Solve tridiagonal system
        var_new = tridiagonal_solve(a, b, c, rhs)

        # Compute tendency
        tendency = (var_new - var) / dt

        return tendency

    # Solve for each variable
    du_dt = solve_implicit_diffusion(u, km, surf_flux_u)
    dv_dt = solve_implicit_diffusion(v, km, surf_flux_v)
    dthl_dt = solve_implicit_diffusion(thl, kh, surf_flux_th)
    drt_dt = solve_implicit_diffusion(rt, kh, surf_flux_rv)

    return du_dt, dv_dt, dthl_dt, drt_dt


def tridiagonal_solve(a: Array, b: Array, c: Array, d: Array) -> Array:
    """
    Solve tridiagonal system: A * x = d

    where A has lower diagonal a, main diagonal b, upper diagonal c.

    Parameters
    ----------
    a : Array
        Lower diagonal (size n, a[0] not used)
    b : Array
        Main diagonal (size n)
    c : Array
        Upper diagonal (size n, c[-1] not used)
    d : Array
        Right-hand side (size n)

    Returns
    -------
    x : Array
        Solution (size n)

    Notes
    -----
    This implements the Thomas algorithm for tridiagonal matrices.

    Fortran Source
    --------------
    Standard tridiagonal solver used throughout PHYEX
    Thomas algorithm (Thomas, 1949)
    """
    n = b.shape[0]

    # Forward elimination
    c_star = jnp.zeros_like(c)
    d_star = jnp.zeros_like(d)

    c_star = c_star.at[0].set(c[0] / b[0])
    d_star = d_star.at[0].set(d[0] / b[0])

    for i in range(1, n):
        denom = b[i] - a[i] * c_star[i - 1]
        c_star = c_star.at[i].set(c[i] / denom)
        d_star = d_star.at[i].set((d[i] - a[i] * d_star[i - 1]) / denom)

    # Back substitution
    x = jnp.zeros_like(d)
    x = x.at[-1].set(d_star[-1])

    for i in range(n - 2, -1, -1):
        x = x.at[i].set(d_star[i] - c_star[i] * x[i + 1])

    return x
