"""
Test de reproductibilité des stencils cloud_fraction par rapport à PHYEX-IAL_CY50T1.

Ce module valide que les implémentations Python GT4Py des schémas de fraction nuageuse
produisent des résultats numériquement identiques aux implémentations Fortran de référence
issues du projet PHYEX (PHYsique EXternalisée) version IAL_CY50T1.

Les stencils testés font partie du schéma d'ajustement microphysique ICE_ADJUST qui
calcule les transferts entre phases (vapeur, liquide, glace) et détermine la fraction
nuageuse en tenant compte de la variabilité sous-maille.

Référence:
    PHYEX-IAL_CY50T1/micro/ice_adjust.F90
"""
from ctypes import c_double, c_float

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
def test_thermodynamic_fields_repro(dtypes, externals, fortran_dims, backend, domain, origin):
    """
    Test de reproductibilité du calcul des champs thermodynamiques (PHYEX-IAL_CY50T1).
    
    Ce test valide le calcul de:
    - Température (T = theta * exner)
    - Chaleur latente de vaporisation (Lv)
    - Chaleur latente de sublimation (Ls)
    - Chaleur spécifique de l'air humide (Cph)
    
    Référence PHYEX:
        ice_adjust.F90, lignes 450-473
        
    Configuration testée:
        - Support 2 à 6 hydrométéores (NRR)
        - Formules de Clausius-Clapeyron pour Lv et Ls
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        externals: Paramètres externes (constantes physiques)
        fortran_dims: Dimensions pour l'interface Fortran
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.cloud_fraction import thermodynamic_fields

    thermo_stencil = stencil(
        backend,
        definition=thermodynamic_fields,
        name="thermo",
        externals=externals,
        dtypes=dtypes,
    )

    fortran_stencil = compile_fortran_stencil(
        "mode_cloud_fraction.F90", "mode_cloud_fraction", "thermodynamic_fields"
    )

    # Initialisation des champs d'entrée avec valeurs aléatoires
    FloatFieldsIJK_Names = [
        "th", "exn", "rv", "rc", "rr", "ri", "rs", "rg",
        "lv", "ls", "cph", "t",
    ]

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    # Conversion en storages GT4Py (inputs)
    th_gt4py = from_array(FloatFieldsIJK["th"], dtype=dtypes["float"], backend=backend)
    exn_gt4py = from_array(FloatFieldsIJK["exn"], dtype=dtypes["float"], backend=backend)
    rv_gt4py = from_array(FloatFieldsIJK["rv"], dtype=dtypes["float"], backend=backend)
    rc_gt4py = from_array(FloatFieldsIJK["rc"], dtype=dtypes["float"], backend=backend)
    rr_gt4py = from_array(FloatFieldsIJK["rr"], dtype=dtypes["float"], backend=backend)
    ri_gt4py = from_array(FloatFieldsIJK["ri"], dtype=dtypes["float"], backend=backend)
    rs_gt4py = from_array(FloatFieldsIJK["rs"], dtype=dtypes["float"], backend=backend)
    rg_gt4py = from_array(FloatFieldsIJK["rg"], dtype=dtypes["float"], backend=backend)

    # Outputs (initialisés à zéro)
    lv_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    ls_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    cph_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    t_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)

    # Exécution du stencil Python
    thermo_stencil(
        th=th_gt4py,
        exn=exn_gt4py,
        rv=rv_gt4py,
        rc=rc_gt4py,
        rr=rr_gt4py,
        ri=ri_gt4py,
        rs=rs_gt4py,
        rg=rg_gt4py,
        lv=lv_gt4py,
        ls=ls_gt4py,
        cph=cph_gt4py,
        t=t_gt4py,
        domain=domain,
        origin=origin,
    )

    # Exécution de la référence Fortran
    (
        zlv, zls, zcph, zt
    ) = fortran_stencil(
        nrr=6,
        prv=FloatFieldsIJK["rv"].reshape(domain[0]*domain[1], domain[2]),
        prc=FloatFieldsIJK["rc"].reshape(domain[0]*domain[1], domain[2]),
        pri=FloatFieldsIJK["ri"].reshape(domain[0]*domain[1], domain[2]),
        prr=FloatFieldsIJK["rr"].reshape(domain[0]*domain[1], domain[2]),
        prs=FloatFieldsIJK["rs"].reshape(domain[0]*domain[1], domain[2]),
        prg=FloatFieldsIJK["rg"].reshape(domain[0]*domain[1], domain[2]),
        pth=FloatFieldsIJK["th"].reshape(domain[0]*domain[1], domain[2]),
        pexn=FloatFieldsIJK["exn"].reshape(domain[0]*domain[1], domain[2]),
        cpd=externals["CPD"],
        **fortran_dims,
    )

    # ======================================================================
    # VALIDATION - Comparaison Python vs Fortran PHYEX
    # ======================================================================
    print("\n" + "="*75)
    print("TEST REPRODUCTIBILITÉ: thermodynamic_fields vs PHYEX-IAL_CY50T1")
    print("="*75)

    assert_allclose(
        zt, 
        t_gt4py.reshape(domain[0] * domain[1], domain[2]), 
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] Température (T = theta*exner)"
    )
    print("✓ T (température)")

    assert_allclose(
        zlv, 
        lv_gt4py.reshape(domain[0] * domain[1], domain[2]), 
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] Chaleur latente vaporisation (Lv)"
    )
    print("✓ Lv (chaleur latente vaporisation)")

    assert_allclose(
        zls, 
        ls_gt4py.reshape(domain[0] * domain[1], domain[2]), 
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] Chaleur latente sublimation (Ls)"
    )
    print("✓ Ls (chaleur latente sublimation)")

    assert_allclose(
        zcph, 
        cph_gt4py.reshape(domain[0] * domain[1], domain[2]), 
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] Chaleur spécifique air humide (Cph)"
    )
    print("✓ Cph (chaleur spécifique air humide)")

    print("\n" + "="*75)
    print("SUCCÈS: Champs thermodynamiques validés!")
    print("="*75 + "\n")


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
def test_cloud_fraction_1(externals, fortran_dims, dtypes, backend, domain, origin):
    """
    Test de reproductibilité du schéma de fraction nuageuse #1 (PHYEX-IAL_CY50T1).
    
    Ce test valide le calcul des sources microphysiques après la boucle
    de condensation, incluant:
    - Ajustement des rapports de mélange (rv, rc, ri)
    - Conservation de l'eau totale
    - Mise à jour de la température potentielle
    
    Référence PHYEX:
        ice_adjust.F90, lignes 278-312
    
    Configuration testée:
        - LSUBG_COND = True (schéma sous-maille activé)
        - Pas de temps dt = 50s
        
    Champs validés:
        - ths: Température potentielle [K]
        - rvs: Rapport mélange vapeur [kg/kg]
        - rcs: Rapport mélange liquide [kg/kg]
        - ris: Rapport mélange glace [kg/kg]
    
    Args:
        externals: Paramètres externes
        fortran_dims: Dimensions Fortran
        dtypes: Types de données
        backend: Backend GT4Py
        domain: Domaine de calcul
        origin: Origine GT4Py
    """
    externals.update({"LSUBG_COND": True})

    from ice3.stencils.cloud_fraction import cloud_fraction_1

    cloud_fraction_1_stencil = stencil(
        backend,
        definition=cloud_fraction_1,
        name="cloud_fraction_1",
        externals=externals,
        dtypes=dtypes,
    )

    dt = dtypes["float"](50.0)

    FloatFieldsIJK_Names = [
        "lv", "ls", "cph", "exnref", "rc", "ri",
        "ths", "rvs", "rcs", "ris", "rc_tmp", "ri_tmp",
    ]

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    # Création des storages GT4Py
    lv_gt4py = from_array(FloatFieldsIJK["lv"], backend=backend, dtype=dtypes["float"])
    ls_gt4py = from_array(FloatFieldsIJK["ls"], backend=backend, dtype=dtypes["float"])
    cph_gt4py = from_array(FloatFieldsIJK["cph"], backend=backend, dtype=dtypes["float"])
    exnref_gt4py = from_array(FloatFieldsIJK["exnref"], backend=backend, dtype=dtypes["float"])
    rc_gt4py = from_array(FloatFieldsIJK["rc"], backend=backend, dtype=dtypes["float"])
    ri_gt4py = from_array(FloatFieldsIJK["ri"], backend=backend, dtype=dtypes["float"])
    ths_gt4py = from_array(FloatFieldsIJK["ths"], backend=backend, dtype=dtypes["float"])
    rvs_gt4py = from_array(FloatFieldsIJK["rvs"], backend=backend, dtype=dtypes["float"])
    rcs_gt4py = from_array(FloatFieldsIJK["rcs"], backend=backend, dtype=dtypes["float"])
    ris_gt4py = from_array(FloatFieldsIJK["ris"], backend=backend, dtype=dtypes["float"])
    rc_tmp_gt4py = from_array(FloatFieldsIJK["rc_tmp"], backend=backend, dtype=dtypes["float"])
    ri_tmp_gt4py = from_array(FloatFieldsIJK["ri_tmp"], backend=backend, dtype=dtypes["float"])

    # Exécution stencil Python
    cloud_fraction_1_stencil(
        lv=lv_gt4py,
        ls=ls_gt4py,
        cph=cph_gt4py,
        exnref=exnref_gt4py,
        rc=rc_gt4py,
        ri=ri_gt4py,
        ths=ths_gt4py,
        rvs=rvs_gt4py,
        rcs=rcs_gt4py,
        ris=ris_gt4py,
        rc_tmp=rc_tmp_gt4py,
        ri_tmp=ri_tmp_gt4py,
        dt=dt,
        domain=domain,
        origin=origin,
    )

    # Exécution référence Fortran
    fortran_stencil = compile_fortran_stencil(
        "mode_cloud_fraction.F90", "mode_cloud_fraction", "cloud_fraction_1"
    )

    (pths_out, prvs_out, prcs_out, pris_out) = fortran_stencil(
        pdt=50.0, # timestep AROME
        prc_tmp=FloatFieldsIJK["rc_tmp"].reshape(domain[0]*domain[1], domain[2]),
        pri_tmp=FloatFieldsIJK["ri_tmp"].reshape(domain[0]*domain[1], domain[2]),
        pexnref=FloatFieldsIJK["exnref"].reshape(domain[0]*domain[1], domain[2]),
        pcph=FloatFieldsIJK["cph"].reshape(domain[0]*domain[1], domain[2]),
        plv=FloatFieldsIJK["lv"].reshape(domain[0]*domain[1], domain[2]),
        pls=FloatFieldsIJK["ls"].reshape(domain[0]*domain[1], domain[2]),
        prc=FloatFieldsIJK["rc"].reshape(domain[0]*domain[1], domain[2]),
        pri=FloatFieldsIJK["ri"].reshape(domain[0]*domain[1], domain[2]),
        prvs=FloatFieldsIJK["rvs"].reshape(domain[0]*domain[1], domain[2]),
        prcs=FloatFieldsIJK["rcs"].reshape(domain[0]*domain[1], domain[2]),
        pths=FloatFieldsIJK["ths"].reshape(domain[0]*domain[1], domain[2]),
        pris=FloatFieldsIJK["ris"].reshape(domain[0]*domain[1], domain[2]),
        **fortran_dims
    )

    # ======================================================================
    # VALIDATION
    # ======================================================================
    print("\n" + "="*75)
    print("TEST REPRODUCTIBILITÉ: cloud_fraction_1 vs PHYEX-IAL_CY50T1")
    print("="*75)

    # Adjust tolerances for double precision when Fortran uses single precision
    rtol = 1e-4 if dtypes["float"] == np.float64 else 1e-6
    atol = 1e-6 if dtypes["float"] == np.float64 else 1e-8
    
    assert_allclose(
        pths_out,
        ths_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=rtol,
        atol=atol,
        err_msg="[ÉCHEC] Température potentielle (ths)"
    )
    print("✓ ths (température potentielle)")

    assert_allclose(
        prvs_out,
        rvs_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=rtol,
        atol=atol,
        err_msg="[ÉCHEC] Rapport mélange vapeur (rvs)"
    )
    print("✓ rvs (rapport mélange vapeur)")

    assert_allclose(
        prcs_out,
        rcs_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=rtol,
        atol=atol,
        err_msg="[ÉCHEC] Rapport mélange liquide (rcs)"
    )
    print("✓ rcs (rapport mélange liquide)")

    assert_allclose(
        pris_out,
        ris_gt4py.reshape(domain[0] * domain[1], domain[2]),
        rtol=rtol,
        atol=atol,
        err_msg="[ÉCHEC] Rapport mélange glace (ris)"
    )
    print("✓ ris (rapport mélange glace)")

    print("\n" + "="*75)
    print("SUCCÈS: Fraction nuageuse #1 validée!")
    print("="*75 + "\n")
