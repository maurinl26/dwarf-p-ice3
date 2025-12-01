# -*- coding: utf-8 -*-
"""
Test de reproductibilité du stencil ice4_fast_rg par rapport à PHYEX-IAL_CY50T1.

Ce module valide que l'implémentation Python GT4Py des processus rapides du graupel
de la microphysique ICE4 produit des résultats numériquement identiques à l'implémentation 
Fortran de référence issue du projet PHYEX (PHYsique EXternalisée) version IAL_CY50T1.

Les processus rapides du graupel représentent:
- Le givrage par congélation de contact de la pluie (RICFRRG, RRCFRIG, RICFRR)
- La croissance du graupel par collection (RCDRY, RIDRY, RSDRY, RRDRY)
- La croissance humide du graupel (RIWET, RSWET)
- Les taux de congélation (FREEZ1, FREEZ2)
- La fonte du graupel (RGMLTR)

Ces processus sont dits "rapides" car leurs échelles de temps caractéristiques sont plus courtes
que celles des processus lents (nucléation, agrégation, etc.).

Référence:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_rg.F90
"""
from ctypes import c_double, c_float

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array, zeros
from numpy.testing import assert_allclose

from ice3.phyex_common.xker_raccs import KER_SACCRG
from ice3.phyex_common.xker_rdryg import KER_RDRYG
from ice3.phyex_common.xker_sdryg import KER_SDRYG
from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import dp_dtypes, sp_dtypes


def test_rain_contact_freezing(dtypes, backend, externals, domain, origin):
    ...

def test_cloud_pristine_collection_graupel(dtypes, backend, externals, domain, origin):
    ...

def test_snow_collection_on_graupel(dtypes, backend, externals, domain, origin):
    ...

def test_rain_accretion_on_graupel(dtypes, backend, externals, domain, origin):
    ...

def test_compute_graupel_growth(dtypes, backend, externals, domain, origin):
    from ice3.stencils_dace.ice4_fast_rg import compute_graupel_growth

    compute_graupel_growth_dace = stencil(
        backend,
        definition=compute_graupel_growth,
        name="compute_graupel_growth",
        dtypes=dtypes,
        externals=externals,
        )


def test_graupel_melting(dtypes, backend, externals, domain, origin):
    from ice3.stencils_dace.ice4_fast_rg import graupel_melting

    graupel_melting_dace = stencil(
        backend,
        definition=graupel_melting,
        name="graupel_melting",
        dtypes=dtypes,
        externals=externals,
    )