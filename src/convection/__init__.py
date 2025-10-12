"""JAX implementation of shallow convection parameterization.

This module contains the translation of PHYEX shallow convection routines to JAX.
It follows the same pattern as the ice_adjust module, with functional JAX implementations
of the Fortran convection schemes.
"""

from .convpar_shal import CONVPAR_SHAL, init_convpar_shal
from .satmixratio import convect_satmixratio
from .trigger_shal import convect_trigger_shal
from .condens import convect_condens
from .mixing_funct import convect_mixing_funct
from .updraft_shal import convect_updraft_shal

__all__ = [
    "CONVPAR_SHAL",
    "init_convpar_shal",
    "convect_satmixratio",
    "convect_trigger_shal",
    "convect_condens",
    "convect_mixing_funct",
    "convect_updraft_shal",
]
