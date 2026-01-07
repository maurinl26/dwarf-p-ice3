"""
TKE production and dissipation sources.

This module computes the sources and sinks for the TKE evolution equation
following the 1.5-order closure framework.

TKE Budget Equation
------------------
The complete TKE equation (Redelsperger & Sommeria 1981, Eq. 2.4):

    ∂e/∂t = P_s + P_b + T - ε

1. **Shear Production** (Eq. 2.5):

   P_s = K_m S²

   where S² = (∂u/∂z)² + (∂v/∂z)² is the wind shear squared.
   This term is always positive and represents conversion of mean kinetic
   energy to TKE through velocity shear.

2. **Buoyancy Production** (Eq. 2.7):

   P_b = (g/θ_v) K_h ∂θ_v/∂z

   - Positive (P_b > 0) in unstable conditions (∂θ_v/∂z < 0): convection
   - Negative (P_b < 0) in stable conditions (∂θ_v/∂z > 0): suppression
   This represents conversion between potential and kinetic energy.

3. **Turbulent Transport** (Cuxart et al. 2000, Eq. 4):

   T = ∂/∂z(K_e ∂e/∂z)

   where K_e = C_e L_m √e is the TKE diffusivity (C_e ≈ 0.4).
   This term redistributes TKE vertically without changing the total amount.

4. **Dissipation** (Cuxart et al. 2000, Eq. 3):

   ε = C_ε e^(3/2) / L_ε

   where:
   - C_ε = 0.845 (AROME, Schmidt-Schumann 1989)
   - C_ε = 0.34 (RM17, Rodier et al. 2017)
   - L_ε = min(L_m, κz) is the dissipative length

   This represents the energy cascade to small scales where it is
   dissipated as heat by molecular viscosity.

Key References
-------------
- Redelsperger, J.-L., and G. Sommeria, 1981: Méthode de représentation de
  la turbulence d'echelle inférieure à la maille pour un modèle
  tri-dimensionnel de convection nuageuse. Boundary-Layer Meteor., 21, 509-530.
  https://doi.org/10.1007/BF02033592
  (Original TKE budget formulation, Section 2.2)

- Cuxart, J., P. Bougeault, and J.-L. Redelsperger, 2000: A turbulence scheme
  allowing for mesoscale and large-eddy simulations. Q. J. R. Meteorol. Soc.,
  126, 1-30. https://doi.org/10.1002/qj.49712656202
  (Complete implementation details)

- Schmidt, H., and U. Schumann, 1989: Coherent structure of the convective
  boundary layer derived from large-eddy simulations. J. Fluid Mech., 200, 511-562.
  https://doi.org/10.1017/S0022112089000753
  (Dissipation constant C_ε = 0.845)

- Rodier, Q., H. Masson, E. Couvreux, and A. Paci, 2017: Evaluation of a
  buoyancy and shear based mixing length for a turbulence scheme.
  Bound.-Layer Meteor., 165, 401-419.
  https://doi.org/10.1007/s10546-017-0272-3
  (Alternative dissipation constant C_ε = 0.34)

Source Code
----------
Translated from PHYEX-IAL_CY50T1/turb/mode_tke_eps_sources.F90
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Tuple


def compute_tke_sources(
    tke: Array,
    u: Array,
    v: Array,
    thl: Array,
    rt: Array,
    thv: Array,
    km: Array,
    kh: Array,
    lm: Array,
    leps: Array,
    dzz: Array,
    constants: dict,
    phys_constants: dict,
) -> Tuple[Array, Array, Array, Array]:
    """
    Compute TKE production and dissipation terms.

    Parameters
    ----------
    tke : Array
        Turbulent kinetic energy (m²/s²), shape (nz,)
    u, v : Array
        Horizontal wind components (m/s), shape (nz,)
    thl : Array
        Liquid potential temperature (K), shape (nz,)
    rt : Array
        Total water mixing ratio (kg/kg), shape (nz,)
    thv : Array
        Virtual potential temperature (K), shape (nz,)
    km : Array
        Momentum diffusivity (m²/s), shape (nz,)
    kh : Array
        Heat diffusivity (m²/s), shape (nz,)
    lm : Array
        Mixing length (m), shape (nz,)
    leps : Array
        Dissipative length (m), shape (nz,)
    dzz : Array
        Vertical grid spacing (m), shape (nz,)
    constants : dict
        Turbulence constants
    phys_constants : dict
        Physical constants

    Returns
    -------
    prod_shear : Array
        Shear production term (m²/s³), shape (nz,)
    prod_buoy : Array
        Buoyancy production term (m²/s³), shape (nz,)
    transport : Array
        Transport/diffusion term (m²/s³), shape (nz,)
    dissipation : Array
        Dissipation term (m²/s³), shape (nz,)

    Notes
    -----
    The TKE equation:
    ∂e/∂t = P_s + P_b + D - ε

    where:
    - P_s = -u'w' ∂u/∂z - v'w' ∂v/∂z (shear production)
    - P_b = (g/θ_v) w'θ_v' (buoyancy production/destruction)
    - D = ∂/∂z(K_e ∂e/∂z) (turbulent transport)
    - ε = C_ε e^(3/2) / L_ε (dissipation)

    Fortran Source
    --------------
    PHYEX-IAL_CY50T1/turb/mode_tke_eps_sources.F90
    Subroutine: TKE_EPS_SOURCES (compute all TKE source terms)
    """
    nz = tke.shape[0]
    g = phys_constants['g']

    # Extract constants
    xced = constants.get('xced', 0.85)
    xcet = constants.get('xcet', 0.4)

    # Compute vertical gradients
    du_dz = jnp.gradient(u) / dzz
    dv_dz = jnp.gradient(v) / dzz
    dthv_dz = jnp.gradient(thv) / dzz

    # Shear production: P_s = K_m * S²
    # where S² = (∂u/∂z)² + (∂v/∂z)²
    shear_squared = du_dz**2 + dv_dz**2
    prod_shear = km * shear_squared

    # Buoyancy production: P_b = (g/θ_v) * K_h * ∂θ_v/∂z
    # Positive for unstable (dθ_v/dz < 0), negative for stable
    g_over_thv = g / thv
    prod_buoy = g_over_thv * kh * dthv_dz

    # Note: In stable conditions, buoyancy production is negative (suppression)

    # Dissipation: ε = C_ε * e^(3/2) / L_ε
    sqrt_tke = jnp.sqrt(tke)
    dissipation = xced * tke * sqrt_tke / leps

    # Transport term: ∂/∂z(K_e ∂e/∂z)
    # where K_e = C_e * L_m * sqrt(e)
    ke = xcet * lm * sqrt_tke

    # Compute flux: F = -K_e * ∂e/∂z
    dtke_dz = jnp.gradient(tke) / dzz
    flux = -ke * dtke_dz

    # Transport = -∂F/∂z
    transport = -jnp.gradient(flux) / dzz

    return prod_shear, prod_buoy, transport, dissipation


def compute_tke_tendency(
    tke: Array,
    prod_shear: Array,
    prod_buoy: Array,
    transport: Array,
    dissipation: Array,
    tke_min: float = 1.0e-6,
) -> Array:
    """
    Compute total TKE tendency.

    Parameters
    ----------
    tke : Array
        Current TKE (m²/s²), shape (nz,)
    prod_shear : Array
        Shear production (m²/s³), shape (nz,)
    prod_buoy : Array
        Buoyancy production (m²/s³), shape (nz,)
    transport : Array
        Transport term (m²/s³), shape (nz,)
    dissipation : Array
        Dissipation (m²/s³), shape (nz,)
    tke_min : float, optional
        Minimum TKE value (m²/s²). Default: 1e-6

    Returns
    -------
    dtke_dt : Array
        TKE tendency (m²/s³), shape (nz,)

    Fortran Source
    --------------
    PHYEX-IAL_CY50T1/turb/mode_tke_eps_sources.F90
    Combines all TKE sources to compute total tendency
    """
    # Total tendency
    dtke_dt = prod_shear + prod_buoy + transport - dissipation

    # Ensure TKE doesn't go below minimum
    # If TKE would become negative, limit the tendency
    dtke_dt = jnp.where(tke <= tke_min, jnp.maximum(dtke_dt, 0.0), dtke_dt)

    return dtke_dt


def compute_dissipative_length(
    lm: Array,
    zz: Array,
    constants: dict,
) -> Array:
    """
    Compute dissipative length scale.

    The dissipative length can differ from the mixing length,
    especially near the surface.

    Parameters
    ----------
    lm : Array
        Mixing length (m), shape (nz,)
    zz : Array
        Height above surface (m), shape (nz,)
    constants : dict
        Turbulence constants

    Returns
    -------
    leps : Array
        Dissipative length (m), shape (nz,)

    Notes
    -----
    Near the surface, the dissipative length is often taken as:
    L_ε = κz where κ is von Kármán constant (0.4)

    Aloft, L_ε ≈ L_m

    Fortran Source
    --------------
    PHYEX-IAL_CY50T1/turb/mode_tke_eps_sources.F90
    Computes dissipative length scale L_ε for dissipation term
    """
    karman = constants.get('karman', 0.4)

    # Near-surface dissipative length: κz
    l_surface = karman * zz

    # Use minimum of mixing length and surface-based length
    # This ensures proper dissipation near the surface
    leps = jnp.minimum(lm, l_surface)

    # But don't let it go to zero
    leps = jnp.maximum(leps, 0.1)

    return leps
