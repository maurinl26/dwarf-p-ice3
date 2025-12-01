"""
Test de reproductibilité du stencil sigrc_computation (DaCe) par rapport au Fortran.

Ce module valide que l'implémentation Python DaCe du calcul de sigma_rc 
(écart-type sous-maille de l'eau nuageuse) produit des résultats numériquement 
identiques à l'implémentation Fortran de référence.

Le calcul de SIGRC est utilisé dans le schéma de condensation sous-maille 
Chaboureau-Bechtold (CB) pour représenter la variabilité sous-maille de l'eau nuageuse.

Référence:
    mode_sigrc_computation.F90
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ice3.stencils_dace.sigrc_computation import sigrc_computation, SRC_1D
from ice3.utils.compile_fortran import compile_fortran_stencil


def test_sigrc_computation(dtypes, backend, externals, domain, origin):
    ...


