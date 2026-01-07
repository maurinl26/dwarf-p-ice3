"""
JAX Implementation of PHYEX Turbulence Scheme.

This package provides a JAX translation of the PHYEX turbulence scheme,
focusing on the AROME operational configuration (1D vertical turbulence).

Scientific Background
--------------------
This implementation is based on a 1.5-order turbulence closure scheme with
prognostic Turbulent Kinetic Energy (TKE) and diagnostic mixing length.

The fundamental TKE equation is:

    ∂e/∂t = P_s + P_b + T - ε

where:
- e: Turbulent kinetic energy (m²/s²)
- P_s: Shear production = K_m (∂u/∂z)²
- P_b: Buoyancy production = (g/θ_v) K_h (∂θ_v/∂z)
- T: Turbulent transport = ∂/∂z(K_e ∂e/∂z)
- ε: Dissipation = C_ε e^(3/2) / L_ε

The turbulent diffusivities are parameterized as:
- K_m = C_k L_m √e  (momentum)
- K_h = K_m / Pr_t  (heat/moisture)

Main Components
--------------
- constants: Turbulence scheme constants (CSTURB)
- bl89: Bougeault-Lacarrere (1989) mixing length
- prandtl: Turbulent Prandtl/Schmidt numbers
- turb_ver: Vertical turbulent fluxes
- tke_eps: TKE production and dissipation
- turb: Main turbulence routine

Key References
-------------
- Cuxart, J., P. Bougeault, and J.-L. Redelsperger, 2000: A turbulence scheme
  allowing for mesoscale and large-eddy simulations. Q. J. R. Meteorol. Soc.,
  126, 1-30. https://doi.org/10.1002/qj.49712656202

- Bougeault, P., and P. Lacarrere, 1989: Parameterization of orography-induced
  turbulence in a mesobeta-scale model. Mon. Wea. Rev., 117, 1872-1890.
  https://doi.org/10.1175/1520-0493(1989)117<1872:POOITI>2.0.CO;2

- Redelsperger, J.-L., and G. Sommeria, 1981: Méthode de représentation de
  la turbulence d'echelle inférieure à la maille pour un modèle
  tri-dimensionnel de convection nuageuse. Boundary-Layer Meteor., 21, 509-530.
  https://doi.org/10.1007/BF02033592

- Rodier, Q., H. Masson, E. Couvreux, and A. Paci, 2017: Evaluation of a
  buoyancy and shear based mixing length for a turbulence scheme.
  Bound.-Layer Meteor., 165, 401-419.
  https://doi.org/10.1007/s10546-017-0272-3

Source Code
----------
Translated from PHYEX-IAL_CY50T1/turb/
"""

from .constants import TurbulenceConstants, TURB_CONSTANTS, get_turb_constants
from .bl89 import compute_bl89_mixing_length, compute_bl89_mixing_length_vectorized
from .turb import turb_scheme, turb_scheme_jit
from .turb_ver import compute_prandtl_number, turb_ver_implicit
from .tke_eps import compute_tke_sources, compute_tke_tendency, compute_dissipative_length

__all__ = [
    # Constants
    'TurbulenceConstants',
    'TURB_CONSTANTS',
    'get_turb_constants',
    # Mixing length
    'compute_bl89_mixing_length',
    'compute_bl89_mixing_length_vectorized',
    # Main scheme
    'turb_scheme',
    'turb_scheme_jit',
    # Vertical fluxes
    'compute_prandtl_number',
    'turb_ver_implicit',
    # TKE evolution
    'compute_tke_sources',
    'compute_tke_tendency',
    'compute_dissipative_length',
]
