"""
Test de reproductibilité des processus microphysiques ICE4.

Ce module valide que les implémentations Python GT4Py des processus microphysiques
ICE4 produisent des résultats numériquement identiques aux implémentations Fortran
de référence issues du projet PHYEX (PHYsique EXternalisée) version IAL_CY50T1.

Les processus testés incluent:
- ICE4_RIMLTC: Fonte du givre (riming melting)

Référence:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_*.F90
"""
from ctypes import c_float, c_double

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array, zeros
from numpy.testing import assert_allclose

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import dp_dtypes, sp_dtypes



@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("debug", marks=pytest.mark.debug),
        pytest.param("numpy", marks=pytest.mark.numpy),
        pytest.param("gt:cpu_ifirst", marks=pytest.mark.cpu),
        pytest.param("gt:gpu", marks=pytest.mark.gpu),
    ],
)
def test_ice4_rimltc(dtypes, backend, externals, packed_dims, domain, origin):
    """
    Test de reproductibilité du stencil ice4_rimltc (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre
    l'implémentation Python/GT4Py et l'implémentation Fortran de référence PHYEX-IAL_CY50T1
    du processus de fonte du givre (ICE4_RIMLTC - Riming Melting/Collection).
    
    La fonte du givre est le processus par lequel la glace collectée par givrage
    (riming) fond lorsque la température dépasse 0°C, transformant la glace en eau
    liquide et libérant de la chaleur latente. Ce processus affecte la température
    potentielle du système.
    
    Champs validés:
        - rimltc_mr: Taux de fonte du givre [kg/kg/s]
    
    Tolérance:
        rtol=1e-6
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        packed_dims: Dimensions pour l'interface Fortran (kproma, ksize)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_rimltc import ice4_rimltc

    ice4_rimltc_gt4py = stencil(
        backend,
        definition=ice4_rimltc,
        name="ice4_rimltc",
        dtypes=dtypes,
        externals=externals,
    )
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_rimltc.F90", "mode_ice4_rimltc", "ice4_rimltc"
    )

    # Initialize fields
    input_field_names = ["t", "exn", "lvfact", "lsfact", "tht", "rit"]
    fields = {
        name: np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in input_field_names
    }
    
    fields["rimltc_mr"] = np.array(
        np.zeros(domain),
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    
    ldcompute = np.array(np.random.rand(*domain) > 0.2, dtype=np.bool_, order="F")
    
    # Create GT4Py storages
    ldcompute_gt4py = from_array(ldcompute, dtype=dtypes["bool"], backend=backend)
    t_gt4py = from_array(fields["t"], dtype=dtypes["float"], backend=backend)
    exn_gt4py = from_array(fields["exn"], dtype=dtypes["float"], backend=backend)
    lvfact_gt4py = from_array(fields["lvfact"], dtype=dtypes["float"], backend=backend)
    lsfact_gt4py = from_array(fields["lsfact"], dtype=dtypes["float"], backend=backend)
    tht_gt4py = from_array(fields["tht"], dtype=dtypes["float"], backend=backend)
    rit_gt4py = from_array(fields["rit"], dtype=dtypes["float"], backend=backend)
    rimltc_mr_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    
    # Execute GT4Py stencil
    ice4_rimltc_gt4py(
        ldcompute=ldcompute_gt4py,
        t=t_gt4py,
        exn=exn_gt4py,
        lvfact=lvfact_gt4py,
        lsfact=lsfact_gt4py,
        tht=tht_gt4py,
        rit=rit_gt4py,
        rimltc_mr=rimltc_mr_gt4py,
        domain=domain,
        origin=origin,
    )
    
    # Execute Fortran stencil
    result = fortran_stencil(
        ldcompute=ldcompute.ravel(),
        pexn=fields["exn"].ravel(),
        plvfact=fields["lvfact"].ravel(),
        plsfact=fields["lsfact"].ravel(),
        pt=fields["t"].ravel(),
        ptht=fields["tht"].ravel(),
        prit=fields["rit"].ravel(),
        primltc_mr=fields["rimltc_mr"].ravel(),
        xtt=externals["TT"],
        lfeedbackt=externals["LFEEDBACKT"],
        **packed_dims,
    )
    rimltc_mr_out = result[0]
    
    # Compare results
    assert_allclose(rimltc_mr_out, rimltc_mr_gt4py.ravel(), rtol=1e-6, atol=1e-8)

