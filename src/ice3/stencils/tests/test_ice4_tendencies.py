# -*- coding: utf-8 -*-
"""
Test de reproductibilité des stencils ice4_tendencies par rapport à PHYEX-IAL_CY50T1.

Ce module valide que l'implémentation Python GT4Py des fonctions de tendances
de la microphysique ICE4 produit des résultats numériquement identiques à l'implémentation 
Fortran de référence issue du projet PHYEX (PHYsique EXternalisée) version IAL_CY50T1.

Les fonctions de tendances représentent:
- Post-processing de la nucléation (ice4_nucleation_post_processing)
- Post-processing du givrage homogène (ice4_rrhong_post_processing)
- Post-processing de la fonte du givrage (ice4_rimltc_post_processing)
- Pré-processing fast_rg (ice4_fast_rg_pre_processing)
- Mise à jour des incréments (ice4_increment_update)
- Champs dérivés (ice4_derived_fields)
- Paramètres de pente (ice4_slope_parameters)
- Mise à jour des tendances totales (ice4_total_tendencies_update)

Référence:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_tendencies.F90
"""
from ctypes import c_double, c_float

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array, zeros
from numpy.testing import assert_allclose

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import dp_dtypes, sp_dtypes


@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("debug", marks=pytest.mark.debug),
        pytest.param("numpy", marks=pytest.mark.numpy),
        pytest.param("gt:cpu_ifirst", marks=pytest.mark.cpu),
        pytest.param("gt:gpu", marks=pytest.mark.gpu),
    ],
)
def test_ice4_nucleation_post_processing(
    externals, packed_dims, dtypes, backend, domain, origin
):
    """
    Test de reproductibilité du stencil ice4_nucleation_post_processing (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre:
    - L'implémentation Python/GT4Py du post-processing de la nucléation
    - L'implémentation Fortran de référence PHYEX-IAL_CY50T1
    
    Le post-processing de nucléation (RVHENI - heterogeneous nucleation of ice) ajuste
    les rapports de mélange et la température potentielle après la nucléation hétérogène
    de la glace pristine:
    
    Processus:
    - Ajustement de la température potentielle par chaleur latente de sublimation
    - Conversion de vapeur d'eau en glace pristine
    - Mise à jour de la température absolue
    
    Champs vérifiés:
        - tht: Température potentielle [K]
        - rvt: Rapport de mélange de vapeur d'eau [kg/kg]
        - rit: Rapport de mélange de glace pristine [kg/kg]
        - t: Température absolue [K]
    
    Tolérance:
        rtol=1e-6, atol=1e-8
    
    Args:
        externals: Paramètres externes (constantes physiques)
        packed_dims: Dimensions pour l'interface Fortran (kproma, ksize)
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_tendencies import ice4_nucleation_post_processing

    ice4_nucleation_post_processing_gt4py = stencil(
        backend,
        definition=ice4_nucleation_post_processing,
        name="ice4_nucleation_post_processing",
        dtypes=dtypes,
        externals=externals,
    )
    
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90",
        "mode_ice4_tendencies",
        "ice4_nucleation_post_processing",
    )
    
    # Initialize input fields with realistic values
    t = np.array(
        np.random.rand(*domain) * 70 + 233,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    exn = np.array(
        np.random.rand(*domain) * 0.3 + 0.7,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    lsfact = np.array(
        np.random.rand(*domain) * 1000 + 2500,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    tht = np.array(
        np.random.rand(*domain) * 50 + 250,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    rvt = np.array(
        np.random.rand(*domain) * 0.015,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    rit = np.array(
        np.random.rand(*domain) * 0.002,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    rvheni_mr = np.array(
        np.random.rand(*domain) * 1e-6,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    
    # Convert to GT4Py storages
    t_gt4py = from_array(t.copy(), dtype=dtypes["float"], backend=backend)
    exn_gt4py = from_array(exn, dtype=dtypes["float"], backend=backend)
    lsfact_gt4py = from_array(lsfact, dtype=dtypes["float"], backend=backend)
    tht_gt4py = from_array(tht.copy(), dtype=dtypes["float"], backend=backend)
    rvt_gt4py = from_array(rvt.copy(), dtype=dtypes["float"], backend=backend)
    rit_gt4py = from_array(rit.copy(), dtype=dtypes["float"], backend=backend)
    rvheni_mr_gt4py = from_array(rvheni_mr, dtype=dtypes["float"], backend=backend)
    
    # Execute GT4Py stencil
    ice4_nucleation_post_processing_gt4py(
        t=t_gt4py,
        exn=exn_gt4py,
        lsfact=lsfact_gt4py,
        tht=tht_gt4py,
        rvt=rvt_gt4py,
        rit=rit_gt4py,
        rvheni_mr=rvheni_mr_gt4py,
        domain=domain,
        origin=(0, 0, 0),
    )
    
    # Execute Fortran stencil
    tht_fortran, rvt_fortran, rit_fortran, t_fortran = fortran_stencil(
        plsfact=lsfact.ravel(),
        pexn=exn.ravel(),
        prit=rit.ravel(),
        prvt=rvt.ravel(),
        zt=t.ravel(),
        rvheni_mr=rvheni_mr.ravel(),
        tht=tht.ravel(),
        **packed_dims,
    )

    # Validate
    assert_allclose(tht_fortran, tht_gt4py.ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(rit_fortran, rit_gt4py.ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(t_fortran, t_gt4py.ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(rvt_fortran, rvt_gt4py.ravel(), rtol=1e-6, atol=1e-8)


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
def test_ice4_rrhong_post_processing(
    externals, packed_dims, dtypes, domain, origin, backend
):
    """
    Test de reproductibilité du stencil ice4_rrhong_post_processing (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre:
    - L'implémentation Python/GT4Py du post-processing du givrage homogène
    - L'implémentation Fortran de référence PHYEX-IAL_CY50T1
    
    Le post-processing du givrage homogène (RRHONG - homogeneous freezing of rain drops)
    ajuste les rapports de mélange et la température potentielle après la congélation
    homogène des gouttes de pluie en graupel:
    
    Processus:
    - Ajustement de la température potentielle par différence des chaleurs latentes
      (sublimation - vaporisation)
    - Conversion de pluie en graupel par congélation homogène
    - Mise à jour de la température absolue
    
    Champs vérifiés:
        - tht: Température potentielle [K]
        - t: Température absolue [K]
        - rrt: Rapport de mélange de pluie [kg/kg]
        - rgt: Rapport de mélange de graupel [kg/kg]
    
    Tolérance:
        rtol=1e-6, atol=1e-8
    
    Args:
        externals: Paramètres externes (constantes physiques)
        packed_dims: Dimensions pour l'interface Fortran (kproma, ksize)
        dtypes: Dictionnaire des types (simple/double précision)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
    """
    from ice3.stencils.ice4_tendencies import ice4_rrhong_post_processing

    ice4_rrhong_post_processing_gt4py = stencil(
        backend,
        name="ice4_rrhong_post_processing",
        definition=ice4_rrhong_post_processing,
        dtypes=dtypes,
        externals=externals,
    )

    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90",
        "mode_ice4_tendencies",
        "ice4_rrhong_post_processing",
    )

    # Initialize input fields
    t = np.array(
        np.random.rand(*domain) * 70 + 233,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    exn = np.array(
        np.random.rand(*domain) * 0.3 + 0.7,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    lsfact = np.array(
        np.random.rand(*domain) * 1000 + 2500,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    lvfact = np.array(
        np.random.rand(*domain) * 500 + 2000,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    tht = np.array(
        np.random.rand(*domain) * 50 + 250,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    rrt = np.array(
        np.random.rand(*domain) * 0.005,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    rgt = np.array(
        np.random.rand(*domain) * 0.006,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    rrhong_mr = np.array(
        np.random.rand(*domain) * 1e-6,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    
    # Convert to GT4Py storages
    t_gt4py = from_array(t.copy(), dtype=dtypes["float"], backend=backend)
    exn_gt4py = from_array(exn, dtype=dtypes["float"], backend=backend)
    lsfact_gt4py = from_array(lsfact, dtype=dtypes["float"], backend=backend)
    lvfact_gt4py = from_array(lvfact, dtype=dtypes["float"], backend=backend)
    tht_gt4py = from_array(tht.copy(), dtype=dtypes["float"], backend=backend)
    rrt_gt4py = from_array(rrt.copy(), dtype=dtypes["float"], backend=backend)
    rgt_gt4py = from_array(rgt.copy(), dtype=dtypes["float"], backend=backend)
    rrhong_mr_gt4py = from_array(rrhong_mr, dtype=dtypes["float"], backend=backend)
    
    # Execute GT4Py stencil
    ice4_rrhong_post_processing_gt4py(
        t=t_gt4py,
        exn=exn_gt4py,
        lsfact=lsfact_gt4py,
        lvfact=lvfact_gt4py,
        tht=tht_gt4py,
        rrt=rrt_gt4py,
        rgt=rgt_gt4py,
        rrhong_mr=rrhong_mr_gt4py,
        domain=domain,
        origin=(0, 0, 0),
    )
    
    # Execute Fortran stencil
    tht_fortran, t_fortran, rrt_fortran, rgt_fortran = fortran_stencil(
        plsfact=lsfact.ravel(),
        plvfact=lvfact.ravel(),
        pexn=exn.ravel(),
        prrhong_mr=rrhong_mr.ravel(),
        ptht=tht.ravel(),
        pt=t.ravel(),
        prrt=rrt.ravel(),
        prgt=rgt.ravel(),
        **packed_dims,
    )

    # Validate
    assert_allclose(tht_fortran, tht_gt4py.ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(t_fortran, t_gt4py.ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(rrt_fortran, rrt_gt4py.ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(rgt_fortran, rgt_gt4py.ravel(), rtol=1e-6, atol=1e-8)


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
def test_ice4_rimltc_post_processing(
    backend, externals, packed_dims, dtypes, domain, origin
):
    """
    Test de reproductibilité du stencil ice4_rimltc_post_processing (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre:
    - L'implémentation Python/GT4Py du post-processing de la fonte du givrage
    - L'implémentation Fortran de référence PHYEX-IAL_CY50T1
    
    Le post-processing de la fonte du givrage (RIMLTC - melting of ice crystals captured by cloud)
    ajuste les rapports de mélange et la température potentielle après la fonte des cristaux
    de glace capturés par les gouttelettes nuageuses:
    
    Processus:
    - Ajustement de la température potentielle (effet de refroidissement)
    - Conversion de glace pristine en eau nuageuse par fonte
    - Mise à jour de la température absolue
    
    Champs vérifiés:
        - tht: Température potentielle [K]
        - t: Température absolue [K]
        - rct: Rapport de mélange d'eau nuageuse [kg/kg]
        - rit: Rapport de mélange de glace pristine [kg/kg]
    
    Tolérance:
        rtol=1e-6, atol=1e-8
    
    Args:
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques)
        packed_dims: Dimensions pour l'interface Fortran (kproma, ksize)
        dtypes: Dictionnaire des types (simple/double précision)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_tendencies import ice4_rimltc_post_processing

    ice4_rimltc_post_processing_gt4py = stencil(
        backend,
        name="ice4_rimltc_post_processing",
        definition=ice4_rimltc_post_processing,
        dtypes=dtypes,
        externals=externals,
    )
    
    fortran_stencil = compile_fortran_stencil(
        fortran_script="mode_ice4_tendencies.F90",
        fortran_module="mode_ice4_tendencies",
        fortran_stencil="ice4_rimltc_post_processing",
    )

    # Initialize input fields
    t = np.array(
        np.random.rand(*domain) * 70 + 233,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    exn = np.array(
        np.random.rand(*domain) * 0.3 + 0.7,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    lsfact = np.array(
        np.random.rand(*domain) * 1000 + 2500,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    lvfact = np.array(
        np.random.rand(*domain) * 500 + 2000,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    rimltc_mr = np.array(
        np.random.rand(*domain) * 1e-6,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    tht = np.array(
        np.random.rand(*domain) * 50 + 250,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    rct = np.array(
        np.random.rand(*domain) * 0.003,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    rit = np.array(
        np.random.rand(*domain) * 0.002,
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    
    # Convert to GT4Py storages
    t_gt4py = from_array(t.copy(), dtype=dtypes["float"], backend=backend)
    exn_gt4py = from_array(exn, dtype=dtypes["float"], backend=backend)
    lsfact_gt4py = from_array(lsfact, dtype=dtypes["float"], backend=backend)
    lvfact_gt4py = from_array(lvfact, dtype=dtypes["float"], backend=backend)
    rimltc_mr_gt4py = from_array(rimltc_mr, dtype=dtypes["float"], backend=backend)
    tht_gt4py = from_array(tht.copy(), dtype=dtypes["float"], backend=backend)
    rct_gt4py = from_array(rct.copy(), dtype=dtypes["float"], backend=backend)
    rit_gt4py = from_array(rit.copy(), dtype=dtypes["float"], backend=backend)
    
    # Execute GT4Py stencil
    ice4_rimltc_post_processing_gt4py(
        t=t_gt4py,
        exn=exn_gt4py,
        lsfact=lsfact_gt4py,
        lvfact=lvfact_gt4py,
        rimltc_mr=rimltc_mr_gt4py,
        tht=tht_gt4py,
        rct=rct_gt4py,
        rit=rit_gt4py,
        domain=domain,
        origin=(0, 0, 0),
    )
    
    # Execute Fortran stencil
    tht_fortran, t_fortran, rct_fortran, rit_fortran = fortran_stencil(
        plsfact=lsfact.ravel(),
        plvfact=lvfact.ravel(),
        pexn=exn.ravel(),
        primltc_mr=rimltc_mr.ravel(),
        ptht=tht.ravel(),
        pt=t.ravel(),
        prit=rit.ravel(),
        prct=rct.ravel(),
        **packed_dims,
    )

    # Validate
    assert_allclose(tht_fortran, tht_gt4py.ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(t_fortran, t_gt4py.ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(rct_fortran, rct_gt4py.ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(rit_fortran, rit_gt4py.ravel(), rtol=1e-6, atol=1e-8)


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
def test_ice4_fast_rg_pre_processing(
    externals, packed_dims, dtypes, backend, domain, origin
):
    """
    Test de reproductibilité du stencil ice4_fast_rg_pre_processing (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre:
    - L'implémentation Python/GT4Py du pré-processing des processus rapides du graupel
    - L'implémentation Fortran de référence PHYEX-IAL_CY50T1
    
    Le pré-processing fast_rg calcule les sources instantanées totales de graupel (RGSI)
    avant l'appel aux processus rapides, en sommant les contributions de:
    
    Processus sources de graupel:
    - RVDEPG: Déposition de vapeur d'eau sur le graupel
    - RSMLTG: Fonte des agrégats de neige en graupel
    - RRACCSG: Accrétion de pluie sur gros agrégats → graupel
    - RSACCRG: Accrétion-conversion neige → graupel
    - RCRIMSG: Givrage sur gros agrégats → graupel
    - RSRIMCG: Conversion neige → graupel par givrage
    
    Ce kernel correspond aux lignes 386-390 de mode_ice4_tendencies.F90 dans PHYEX-IAL_CY50T1,
    juste avant l'appel à 'call ice4_fast_rg'.
    
    Champs vérifiés:
        - rgsi: Source de graupel instantanée [kg/kg/s]
        - rgsi_mr: Rapport de mélange source [kg/kg]
    
    Tolérance:
        rtol=1e-6, atol=1e-8
    
    Référence Fortran:
        PHYEX-IAL_CY50T1/common/micro/mode_ice4_tendencies.F90:386-390
    
    Args:
        externals: Paramètres externes (constantes physiques)
        packed_dims: Dimensions pour l'interface Fortran (kproma, ksize)
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_tendencies import ice4_fast_rg_pre_post_processing

    ice4_fast_rg_pre_processing_gt4py = stencil(
        backend,
        name="ice4_fast_rg_pre_post_processing",
        definition=ice4_fast_rg_pre_post_processing,
        dtypes=dtypes,
        externals=externals,
    )
    
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90",
        "mode_ice4_tendencies",
        "ice4_fast_rg_pre_processing",
    )

    # Initialize input fields
    field_names = [
        "rvdepg", "rsmltg", "rraccsg", "rsaccrg",
        "rcrimsg", "rsrimcg", "rrhong_mr", "rsrimcg_mr"
    ]
    
    fields = {
        name: np.array(
            np.random.rand(*domain) * 1e-6,
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in field_names
    }
    
    # Output fields
    rgsi = np.zeros(domain, dtype=(c_float if dtypes["float"] == np.float32 else c_double), order="F")
    rgsi_mr = np.zeros(domain, dtype=(c_float if dtypes["float"] == np.float32 else c_double), order="F")
    
    # Convert to GT4Py storages
    gt4py_fields = {
        name: from_array(arr, dtype=dtypes["float"], backend=backend)
        for name, arr in fields.items()
    }
    rgsi_gt4py = from_array(rgsi.copy(), dtype=dtypes["float"], backend=backend)
    rgsi_mr_gt4py = from_array(rgsi_mr.copy(), dtype=dtypes["float"], backend=backend)
    
    # Execute GT4Py stencil
    ice4_fast_rg_pre_processing_gt4py(
        rgsi=rgsi_gt4py,
        rgsi_mr=rgsi_mr_gt4py,
        rvdepg=gt4py_fields["rvdepg"],
        rsmltg=gt4py_fields["rsmltg"],
        rraccsg=gt4py_fields["rraccsg"],
        rsaccrg=gt4py_fields["rsaccrg"],
        rcrimsg=gt4py_fields["rcrimsg"],
        rsrimcg=gt4py_fields["rsrimcg"],
        rrhong_mr=gt4py_fields["rrhong_mr"],
        rsrimcg_mr=gt4py_fields["rsrimcg_mr"],
        domain=domain,
        origin=(0, 0, 0),
    )
    
    # Execute Fortran stencil
    zrgsi_fortran, zrgsi_mr_fortran = fortran_stencil(
        rvdepg=fields["rvdepg"].ravel(),
        rsmltg=fields["rsmltg"].ravel(),
        rraccsg=fields["rraccsg"].ravel(),
        rsaccrg=fields["rsaccrg"].ravel(),
        rcrimsg=fields["rcrimsg"].ravel(),
        rsrimcg=fields["rsrimcg"].ravel(),
        rrhong_mr=fields["rrhong_mr"].ravel(),
        rsrimcg_mr=fields["rsrimcg_mr"].ravel(),
        zgrsi=rgsi.ravel(),
        zrgsi_mr=rgsi_mr.ravel(),
        **packed_dims,
    )

    # Validate
    assert_allclose(zrgsi_fortran, rgsi_gt4py.ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(zrgsi_mr_fortran, rgsi_mr_gt4py.ravel(), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("debug", marks=pytest.mark.debug),
        pytest.param("numpy", marks=pytest.mark.numpy),
        pytest.param("gt:cpu_ifirst", marks=pytest.mark.cpu),
        pytest.param("gt:gpu", marks=pytest.mark.gpu),
    ],
)
def test_ice4_increment_update(
    externals, packed_dims, dtypes, backend, domain, origin
):
    """
    Test de reproductibilité du stencil ice4_increment_update (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre:
    - L'implémentation Python/GT4Py de la mise à jour des incréments
    - L'implémentation Fortran de référence PHYEX-IAL_CY50T1
    
    La mise à jour des incréments calcule les tendances totales (instantanées) dues aux
    processus microphysiques de changement de phase qui interviennent entre les étapes
    de splitting de la microphysique:
    
    Processus inclus:
    - RVHENI_MR: Nucléation hétérogène de glace (vapeur → glace)
    - RRHONG_MR: Congélation homogène (pluie → graupel)
    - RIMLTC_MR: Fonte du givrage (glace → eau nuageuse)
    - RSRIMCG_MR: Conversion neige → graupel
    
    Ajustements:
    - Température potentielle: chaleur latente de sublimation/vaporisation
    - Rapports de mélange: conservation de la masse totale
    
    Champs vérifiés:
        - theta_increment: Incrément de température potentielle [K]
        - rv_increment: Incrément vapeur d'eau [kg/kg]
        - rc_increment: Incrément eau nuageuse [kg/kg]
        - rr_increment: Incrément pluie [kg/kg]
        - ri_increment: Incrément glace pristine [kg/kg]
        - rs_increment: Incrément neige [kg/kg]
        - rg_increment: Incrément graupel [kg/kg]
    
    Tolérance:
        rtol=1e-6, atol=1e-8
    
    Args:
        externals: Paramètres externes (constantes physiques)
        packed_dims: Dimensions pour l'interface Fortran (kproma, ksize)
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_tendencies import ice4_increment_update

    ice4_increment_update_gt4py = stencil(
        backend,
        name="ice4_increment_update",
        definition=ice4_increment_update,
        dtypes=dtypes,
        externals=externals,
    )
    
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_tendencies.F90", "mode_ice4_tendencies", "ice4_increment_update"
    )
    
    # Initialize input fields
    field_names = [
        "lsfact", "lvfact", "theta_increment", "rv_increment", "rc_increment",
        "rr_increment", "ri_increment", "rs_increment", "rg_increment",
        "rvheni_mr", "rimltc_mr", "rrhong_mr", "rsrimcg_mr"
    ]
    
    fields = {}
    for name in field_names:
        if name in ["lsfact", "lvfact"]:
            fields[name] = np.array(
                np.random.rand(*domain) * 1000 + 2000,
                dtype=(c_float if dtypes["float"] == np.float32 else c_double),
                order="F",
            )
        else:
            fields[name] = np.array(
                np.random.rand(*domain) * 1e-6,
                dtype=(c_float if dtypes["float"] == np.float32 else c_double),
                order="F",
            )
    
    # Convert to GT4Py storages
    gt4py_fields = {}
    for name, arr in fields.items():
        if "increment" in name:
            gt4py_fields[name] = from_array(arr.copy(), dtype=dtypes["float"], backend=backend)
        else:
            gt4py_fields[name] = from_array(arr, dtype=dtypes["float"], backend=backend)
    
    # Execute GT4Py stencil
    ice4_increment_update_gt4py(
        lsfact=gt4py_fields["lsfact"],
        lvfact=gt4py_fields["lvfact"],
        theta_increment=gt4py_fields["theta_increment"],
        rv_increment=gt4py_fields["rv_increment"],
        rc_increment=gt4py_fields["rc_increment"],
        rr_increment=gt4py_fields["rr_increment"],
        ri_increment=gt4py_fields["ri_increment"],
        rs_increment=gt4py_fields["rs_increment"],
        rg_increment=gt4py_fields["rg_increment"],
        rvheni_mr=gt4py_fields["rvheni_mr"],
        rimltc_mr=gt4py_fields["rimltc_mr"],
        rrhong_mr=gt4py_fields["rrhong_mr"],
        rsrimcg_mr=gt4py_fields["rsrimcg_mr"],
        domain=domain,
        origin=(0, 0, 0),
    )
    
    # Execute Fortran stencil
    (
        pth_inst_fortran,
        prv_inst_fortran,
        prc_inst_fortran,
        prr_inst_fortran,
        pri_inst_fortran,
        prs_inst_fortran,
        prg_inst_fortran,
    ) = fortran_stencil(
        plsfact=fields["lsfact"].ravel(),
        plvfact=fields["lvfact"].ravel(),
        prvheni_mr=fields["rvheni_mr"].ravel(),
        primltc_mr=fields["rimltc_mr"].ravel(),
        prrhong_mr=fields["rrhong_mr"].ravel(),
        prsrimcg_mr=fields["rsrimcg_mr"].ravel(),
        pth_inst=fields["theta_increment"].ravel(),
        prv_inst=fields["rv_increment"].ravel(),
        prc_inst=fields["rc_increment"].ravel(),
        prr_inst=fields["rr_increment"].ravel(),
        pri_inst=fields["ri_increment"].ravel(),
        prs_inst=fields["rs_increment"].ravel(),
        prg_inst=fields["rg_increment"].ravel(),
        **packed_dims,
    )

    # Validate
    assert_allclose(pth_inst_fortran, gt4py_fields["theta_increment"].ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(prv_inst_fortran, gt4py_fields["rv_increment"].ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(prc_inst_fortran, gt4py_fields["rc_increment"].ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(pri_inst_fortran, gt4py_fields["ri_increment"].ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(prr_inst_fortran, gt4py_fields["rr_increment"].ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(prs_inst_fortran, gt4py_fields["rs_increment"].ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(prg_inst_fortran, gt4py_fields["rg_increment"].ravel(), rtol=1e-6, atol=1e-8)
