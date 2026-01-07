"""
Main turbulence scheme for AROME (1D vertical).

This module provides the main turbulence routine that coordinates
mixing length computation, vertical fluxes, and TKE evolution.

Theoretical Framework
--------------------
The scheme implements a 1.5-order closure based on:

1. **TKE Prognostic Equation** (Redelsperger & Sommeria 1981, Eq. 2.4):

   ∂e/∂t + u_j ∂e/∂x_j = P_s + P_b + T - ε

   where:
   - P_s = -u'_i u'_j ∂u_i/∂x_j : Shear production
   - P_b = (g/θ_v) w'θ'_v : Buoyancy production/destruction
   - T = -∂(u'_j e')/∂x_j : Turbulent transport
   - ε : Dissipation rate

2. **Turbulent Flux Parameterizations** (Redelsperger & Sommeria 1981, Eq. 2.5-2.7):

   u'_i u'_j = -K_m (∂u_i/∂x_j + ∂u_j/∂x_i)
   w'θ' = -K_h ∂θ/∂z
   w'r' = -K_h ∂r/∂z

3. **Eddy Diffusivity** (Cuxart et al. 2000, Eq. 2):

   K_m = C_k L_m √e
   K_h = K_m / Pr_t

   where:
   - C_k ≈ 0.143 (momentum flux constant)
   - L_m: Mixing length (Bougeault-Lacarrere 1989)
   - Pr_t: Turbulent Prandtl number (stability-dependent)

4. **Dissipation Rate** (Cuxart et al. 2000, Eq. 3):

   ε = C_ε e^(3/2) / L_ε

   where:
   - C_ε = 0.845 (AROME) or 0.34 (RM17)
   - L_ε: Dissipative length scale

5. **Implicit Time Integration**:
   The vertical diffusion is solved implicitly using a tridiagonal solver
   to ensure numerical stability for large time steps.

Key References
-------------
- Cuxart, J., P. Bougeault, and J.-L. Redelsperger, 2000: A turbulence scheme
  allowing for mesoscale and large-eddy simulations. Q. J. R. Meteorol. Soc.,
  126, 1-30. https://doi.org/10.1002/qj.49712656202
  (Main reference for the complete scheme)

- Redelsperger, J.-L., and G. Sommeria, 1981: Méthode de représentation de
  la turbulence d'echelle inférieure à la maille pour un modèle
  tri-dimensionnel de convection nuageuse. Boundary-Layer Meteor., 21, 509-530.
  https://doi.org/10.1007/BF02033592
  (Foundation of the closure constants)

- Bougeault, P., and P. Lacarrere, 1989: Parameterization of orography-induced
  turbulence in a mesobeta-scale model. Mon. Wea. Rev., 117, 1872-1890.
  https://doi.org/10.1175/1520-0493(1989)117<1872:POOITI>2.0.CO;2
  (Mixing length formulation)

Source Code
----------
Translated from PHYEX-IAL_CY50T1/turb/turb.F90
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Dict, Tuple, Optional

from .constants import TurbulenceConstants, TURB_CONSTANTS
from .bl89 import compute_bl89_mixing_length_vectorized
from .turb_ver import compute_prandtl_number, turb_ver_implicit
from .tke_eps import (
    compute_tke_sources,
    compute_tke_tendency,
    compute_dissipative_length,
)


def turb_scheme(
    # Vertical grid
    zz: Array,
    dzz: Array,
    # Thermodynamic fields
    theta: Array,
    thl: Array,
    rt: Array,
    rv: Array,
    rc: Array,
    ri: Array,
    # Dynamic fields
    u: Array,
    v: Array,
    w: Array,
    tke: Array,
    # Reference state
    thvref: Array,
    pabst: Array,
    exn: Array,
    # Surface fluxes
    surf_flux_u: float,
    surf_flux_v: float,
    surf_flux_th: float,
    surf_flux_rv: float,
    # Time step
    dt: float,
    # Configuration
    turb_constants: Optional[TurbulenceConstants] = None,
    phys_constants: Optional[Dict] = None,
    ximpl: float = 1.0,
    xlini: float = 10.0,
    tke_min: float = 1.0e-6,
) -> Tuple[Array, Array, Array, Array, Array, Dict]:
    """
    Main 1D vertical turbulence scheme (AROME configuration).

    Computes turbulent mixing using:
    1. Bougeault-Lacarrere (1989) mixing length
    2. 1.5-order closure with prognostic TKE
    3. Implicit vertical diffusion

    Parameters
    ----------
    zz : Array
        Physical height (m), shape (nz,)
    dzz : Array
        Vertical grid spacing (m), shape (nz,)
    theta : Array
        Potential temperature (K), shape (nz,)
    thl : Array
        Liquid potential temperature (K), shape (nz,)
    rt : Array
        Total water mixing ratio (kg/kg), shape (nz,)
    rv : Array
        Water vapor mixing ratio (kg/kg), shape (nz,)
    rc : Array
        Cloud liquid water mixing ratio (kg/kg), shape (nz,)
    ri : Array
        Cloud ice mixing ratio (kg/kg), shape (nz,)
    u, v, w : Array
        Wind components (m/s), shape (nz,)
    tke : Array
        Turbulent kinetic energy (m²/s²), shape (nz,)
    thvref : Array
        Reference virtual potential temperature (K), shape (nz,)
    pabst : Array
        Absolute pressure (Pa), shape (nz,)
    exn : Array
        Exner function, shape (nz,)
    surf_flux_u, surf_flux_v : float
        Surface momentum fluxes (m²/s²)
    surf_flux_th : float
        Surface heat flux (K·m/s)
    surf_flux_rv : float
        Surface moisture flux (kg/kg·m/s)
    dt : float
        Time step (s)
    turb_constants : TurbulenceConstants, optional
        Turbulence constants. Default: TURB_CONSTANTS
    phys_constants : dict, optional
        Physical constants. Default: standard values
    ximpl : float, optional
        Implicit weight (0=explicit, 1=fully implicit). Default: 1.0
    xlini : float, optional
        Minimum mixing length (m). Default: 10.0
    tke_min : float, optional
        Minimum TKE (m²/s²). Default: 1e-6

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
    dtke_dt : Array
        TKE tendency (m²/s³), shape (nz,)
    diagnostics : dict
        Diagnostic fields (mixing length, Prandtl number, productions, etc.)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from ice3.jax.turbulence import turb_scheme
    >>>
    >>> # Setup vertical grid
    >>> nz = 50
    >>> zz = jnp.linspace(0, 5000, nz)  # 0-5000m
    >>> dzz = jnp.diff(zz, prepend=0.0)
    >>>
    >>> # Initialize fields
    >>> theta = jnp.linspace(285, 295, nz)  # K
    >>> thl = theta.copy()
    >>> rt = jnp.full(nz, 0.01)  # kg/kg
    >>> u = jnp.linspace(0, 10, nz)  # m/s
    >>> v = jnp.zeros(nz)
    >>> tke = jnp.full(nz, 0.1)  # m²/s²
    >>>
    >>> # Run turbulence scheme
    >>> du_dt, dv_dt, dthl_dt, drt_dt, dtke_dt, diag = turb_scheme(
    ...     zz, dzz, theta, thl, rt, rv, rc, ri,
    ...     u, v, w, tke, thvref, pabst, exn,
    ...     surf_flux_u=0.1, surf_flux_v=0.0,
    ...     surf_flux_th=0.01, surf_flux_rv=1e-5,
    ...     dt=60.0
    ... )
    >>>
    >>> # Access diagnostics
    >>> print(f"Max mixing length: {diag['lm'].max():.1f} m")
    >>> print(f"Max shear production: {diag['prod_shear'].max():.2e} m²/s³")

    Fortran Source
    --------------
    PHYEX-IAL_CY50T1/turb/turb.F90
    Subroutine: TURB (main 1D turbulence scheme)
    Called subroutines:
    - BL89: mixing length computation
    - TURB_VER: vertical turbulent fluxes
    - TKE_EPS: TKE sources and dissipation
    """
    nz = zz.shape[0]

    # Set default constants
    if turb_constants is None:
        turb_constants = TURB_CONSTANTS

    if phys_constants is None:
        phys_constants = {
            'g': 9.80665,
            'rd': 287.06,
            'rv': 461.51,
            'karman': 0.4,
            'alphaoc': 1.9e-4,
        }

    # Convert constants to dict for JAX compatibility
    const_dict = turb_constants.to_dict()

    # ========================================================================
    # 1. Compute virtual potential temperature
    # ========================================================================
    zrvord = phys_constants['rv'] / phys_constants['rd']

    # Combine all condensed water
    rm = jnp.stack([rv, rc, ri], axis=-1)

    # θ_v = θ_l * (1 + 0.61*r_v) / (1 + r_t)
    thv = thl * (1.0 + zrvord * rv) / (1.0 + rt)

    # ========================================================================
    # 2. Compute shear
    # ========================================================================
    du_dz = jnp.gradient(u) / dzz
    dv_dz = jnp.gradient(v) / dzz
    shear = jnp.sqrt(du_dz**2 + dv_dz**2)

    # ========================================================================
    # 3. Compute mixing length (BL89)
    # ========================================================================
    lm = compute_bl89_mixing_length_vectorized(
        zz=zz,
        dzz=dzz,
        thvref=thvref,
        thlm=thl,
        rm=rm,
        tke=tke,
        shear=shear,
        constants=const_dict,
        phys_constants=phys_constants,
        xlini=xlini,
        lbl89top=False,
        ocean=False,
    )

    # ========================================================================
    # 4. Compute dissipative length
    # ========================================================================
    leps = compute_dissipative_length(
        lm=lm,
        zz=zz,
        constants=const_dict,
    )

    # ========================================================================
    # 5. Compute buoyancy gradient
    # ========================================================================
    g = phys_constants['g']
    dthv_dz = jnp.gradient(thv) / dzz
    buoy_gradient = (g / thv) * dthv_dz
    shear_squared = shear**2

    # ========================================================================
    # 6. Compute Prandtl number and diffusivities
    # ========================================================================
    prt, km = compute_prandtl_number(
        tke=tke,
        lm=lm,
        shear=shear_squared,
        buoy_gradient=buoy_gradient,
        constants=const_dict,
    )
    kh = km / prt

    # ========================================================================
    # 7. Compute vertical turbulent fluxes (implicit)
    # ========================================================================
    du_dt, dv_dt, dthl_dt, drt_dt = turb_ver_implicit(
        u=u,
        v=v,
        thl=thl,
        rt=rt,
        tke=tke,
        prt=prt,
        km=km,
        dzz=dzz,
        dt=dt,
        surf_flux_u=surf_flux_u,
        surf_flux_v=surf_flux_v,
        surf_flux_th=surf_flux_th,
        surf_flux_rv=surf_flux_rv,
        ximpl=ximpl,
    )

    # ========================================================================
    # 8. Compute TKE sources
    # ========================================================================
    prod_shear, prod_buoy, transport, dissipation = compute_tke_sources(
        tke=tke,
        u=u,
        v=v,
        thl=thl,
        rt=rt,
        thv=thv,
        km=km,
        kh=kh,
        lm=lm,
        leps=leps,
        dzz=dzz,
        constants=const_dict,
        phys_constants=phys_constants,
    )

    # ========================================================================
    # 9. Compute TKE tendency
    # ========================================================================
    dtke_dt = compute_tke_tendency(
        tke=tke,
        prod_shear=prod_shear,
        prod_buoy=prod_buoy,
        transport=transport,
        dissipation=dissipation,
        tke_min=tke_min,
    )

    # ========================================================================
    # 10. Prepare diagnostics
    # ========================================================================
    diagnostics = {
        'lm': lm,
        'leps': leps,
        'km': km,
        'kh': kh,
        'prt': prt,
        'shear': shear,
        'buoy_gradient': buoy_gradient,
        'prod_shear': prod_shear,
        'prod_buoy': prod_buoy,
        'transport': transport,
        'dissipation': dissipation,
        'thv': thv,
    }

    return du_dt, dv_dt, dthl_dt, drt_dt, dtke_dt, diagnostics


def turb_scheme_jit(
    *args,
    turb_constants: Optional[TurbulenceConstants] = None,
    phys_constants: Optional[Dict] = None,
    **kwargs
):
    """
    JIT-compiled version of turb_scheme.

    Parameters are the same as turb_scheme.

    Fortran Source
    --------------
    PHYEX-IAL_CY50T1/turb/turb.F90
    JAX JIT-compiled wrapper for performance optimization
    """
    @jax.jit
    def _turb_jit(*args, **kwargs):
        return turb_scheme(
            *args,
            turb_constants=turb_constants,
            phys_constants=phys_constants,
            **kwargs
        )

    return _turb_jit(*args, **kwargs)
