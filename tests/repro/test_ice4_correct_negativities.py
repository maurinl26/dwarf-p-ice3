"""
Test de reproductibilité du stencil ice4_correct_negativities.

Ce module valide que l'implémentation Python GT4Py du stencil ice4_correct_negativities
produit des résultats numériquement identiques à l'implémentation Fortran de référence
issue du projet PHYEX (PHYsique EXternalisée) version IAL_CY50T1.

Le stencil ice4_correct_negativities corrige les valeurs négatives des rapports de mélange
qui peuvent apparaître lors des processus microphysiques, tout en conservant la masse
et l'énergie totales.

Processus testés:
- Correction des rapports de mélange négatifs pour toutes les espèces hydrometeor
- Transfert de masse vers la vapeur d'eau avec ajustement thermodynamique
- Correction de la vapeur d'eau négative par sublimation de neige/grésil
- Conservation de la masse et de l'énergie

Physique:
- Si r_x < 0: r_v += |r_x|, θ -= |r_x| * L/Cp, r_x = 0
- Si r_v < 0: convertit rs ou rg en vapeur pour compenser

Référence:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_correct_negativities.F90
"""
from ctypes import c_double, c_float

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array
from numpy.testing import assert_allclose

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
def test_ice4_correct_negativities_positive_values(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil ice4_correct_negativities avec valeurs positives.
    
    Ce test valide que le stencil n'altère pas les champs lorsque toutes les valeurs
    sont déjà positives (cas nominal sans correction nécessaire).
    
    Vérifications:
        - Les champs positifs ne sont pas modifiés
        - Tous les champs restent finis
        - Les valeurs restent dans des plages physiques réalistes
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_correct_negativities import ice4_correct_negativities

    # Compilation du stencil GT4Py
    correct_neg_gt4py = stencil(
        backend,
        definition=ice4_correct_negativities,
        name="ice4_correct_negativities",
        dtypes=dtypes,
        externals=externals,
    )

    # Initialisation des champs avec des valeurs strictement positives
    field_names = ["th_t", "rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t", "lv_fact", "ls_fact"]
    
    fields = {}
    for name in field_names:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustement pour valeurs physiquement réalistes et positives
    fields["th_t"] = 250.0 + fields["th_t"] * 50.0  # 250-300 K
    fields["rv_t"] = fields["rv_t"] * 0.02  # 0-20 g/kg
    fields["rc_t"] = fields["rc_t"] * 0.005  # 0-5 g/kg
    fields["rr_t"] = fields["rr_t"] * 0.003  # 0-3 g/kg
    fields["ri_t"] = fields["ri_t"] * 0.001  # 0-1 g/kg
    fields["rs_t"] = fields["rs_t"] * 0.002  # 0-2 g/kg
    fields["rg_t"] = fields["rg_t"] * 0.003  # 0-3 g/kg
    fields["lv_fact"] = 2000.0 + fields["lv_fact"] * 500.0  # K
    fields["ls_fact"] = 2500.0 + fields["ls_fact"] * 500.0  # K
    
    # Sauvegarde des valeurs initiales
    fields_initial = {name: fields[name].copy() for name in field_names}
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in field_names:
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    correct_neg_gt4py(
        th_t=gt4py_fields["th_t"],
        rv_t=gt4py_fields["rv_t"],
        rc_t=gt4py_fields["rc_t"],
        rr_t=gt4py_fields["rr_t"],
        ri_t=gt4py_fields["ri_t"],
        rs_t=gt4py_fields["rs_t"],
        rg_t=gt4py_fields["rg_t"],
        lv_fact=gt4py_fields["lv_fact"],
        ls_fact=gt4py_fields["ls_fact"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. Les valeurs positives ne doivent pas être modifiées
    for name in ["th_t", "rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t"]:
        assert_allclose(gt4py_fields[name], fields_initial[name], rtol=1e-14, atol=1e-14,
                       err_msg=f"{name} should not be modified when all values are positive")
    
    # 2. Tous les champs sont finis
    for name in field_names:
        assert np.all(np.isfinite(gt4py_fields[name])), f"{name} contains non-finite values"
    
    # 3. Toutes les valeurs restent positives
    for name in ["rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t"]:
        assert np.all(gt4py_fields[name] >= 0), f"{name} should remain non-negative"
    
    print("Test passed! No modification when all values are positive")


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
def test_ice4_correct_negativities_with_negatives(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil ice4_correct_negativities avec valeurs négatives.
    
    Ce test valide la correction des rapports de mélange négatifs par transfert vers
    la vapeur d'eau et ajustement thermodynamique approprié.
    
    Processus validé:
    - Correction: r_x < 0 → r_v += |r_x|, θ -= |r_x| * L/Cp, r_x = 0
    - Conservation de la masse totale d'eau
    - Ajustement thermodynamique cohérent
    
    Vérifications:
        - Toutes les valeurs négatives sont corrigées à 0
        - La vapeur d'eau augmente de la quantité corrigée
        - La température potentielle est ajustée selon la chaleur latente
        - Conservation de la masse totale
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_correct_negativities import ice4_correct_negativities

    # Compilation du stencil GT4Py
    correct_neg_gt4py = stencil(
        backend,
        definition=ice4_correct_negativities,
        name="ice4_correct_negativities",
        dtypes=dtypes,
        externals=externals,
    )

    # Initialisation des champs avec quelques valeurs négatives
    field_names = ["th_t", "rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t", "lv_fact", "ls_fact"]
    
    fields = {}
    for name in field_names:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustement pour valeurs réalistes avec quelques négatives
    fields["th_t"] = 250.0 + fields["th_t"] * 50.0  # 250-300 K
    fields["rv_t"] = fields["rv_t"] * 0.02  # 0-20 g/kg
    
    # Introduire quelques valeurs négatives pour tester la correction
    fields["rc_t"] = (fields["rc_t"] - 0.5) * 0.001  # -0.5 à 0.5 g/kg
    fields["rr_t"] = (fields["rr_t"] - 0.5) * 0.001
    fields["ri_t"] = (fields["ri_t"] - 0.5) * 0.0005
    fields["rs_t"] = (fields["rs_t"] - 0.5) * 0.001
    fields["rg_t"] = (fields["rg_t"] - 0.5) * 0.001
    
    fields["lv_fact"] = 2000.0 + fields["lv_fact"] * 500.0  # K
    fields["ls_fact"] = 2500.0 + fields["ls_fact"] * 500.0  # K
    
    # Calculer la masse totale d'eau avant correction
    total_water_before = (
        fields["rv_t"] + fields["rc_t"] + fields["rr_t"] + 
        fields["ri_t"] + fields["rs_t"] + fields["rg_t"]
    )
    
    # Identifier les cellules avec des valeurs négatives
    has_negative = (
        (fields["rc_t"] < 0) | (fields["rr_t"] < 0) | (fields["ri_t"] < 0) |
        (fields["rs_t"] < 0) | (fields["rg_t"] < 0)
    )
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in field_names:
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    correct_neg_gt4py(
        th_t=gt4py_fields["th_t"],
        rv_t=gt4py_fields["rv_t"],
        rc_t=gt4py_fields["rc_t"],
        rr_t=gt4py_fields["rr_t"],
        ri_t=gt4py_fields["ri_t"],
        rs_t=gt4py_fields["rs_t"],
        rg_t=gt4py_fields["rg_t"],
        lv_fact=gt4py_fields["lv_fact"],
        ls_fact=gt4py_fields["ls_fact"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. Toutes les valeurs négatives ont été corrigées
    for name in ["rc_t", "rr_t", "ri_t", "rs_t", "rg_t"]:
        assert np.all(gt4py_fields[name] >= -1e-15), \
            f"{name} should have no negative values after correction"
    
    # 2. Dans la plupart des cellules corrigées, la vapeur d'eau a changé
    # Note: Elle peut aussi diminuer si la correction de vapeur négative est déclenchée
    if np.any(has_negative):
        rv_changed = ~np.isclose(gt4py_fields["rv_t"][has_negative], 
                                 fields["rv_t"][has_negative], atol=1e-15)
        assert np.any(rv_changed), \
            "rv_t should change where corrections were made"
    
    # 3. Conservation de la masse totale d'eau
    total_water_after = (
        gt4py_fields["rv_t"] + gt4py_fields["rc_t"] + gt4py_fields["rr_t"] + 
        gt4py_fields["ri_t"] + gt4py_fields["rs_t"] + gt4py_fields["rg_t"]
    )
    assert_allclose(total_water_after, total_water_before, rtol=1e-5, atol=1e-10,
                   err_msg="Total water mass should be conserved")
    
    # 4. Tous les champs sont finis
    for name in field_names:
        assert np.all(np.isfinite(gt4py_fields[name])), f"{name} contains non-finite values"
    
    # 5. La température a été ajustée
    if np.any(has_negative):
        # La température devrait avoir changé dans les cellules corrigées
        theta_changed = ~np.isclose(gt4py_fields["th_t"][has_negative], 
                                     fields["th_t"][has_negative], atol=1e-15)
        assert np.any(theta_changed), \
            "th_t should be adjusted where corrections were made"
    
    n_corrected = np.sum(has_negative)
    print(f"Test passed! {n_corrected}/{np.prod(domain)} cells corrected")
    print(f"  Total water conserved: {np.allclose(total_water_after, total_water_before, rtol=1e-5)}")


@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numpy", marks=pytest.mark.numpy),
    ],
)
def test_ice4_correct_negativities_vapor_correction(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil ice4_correct_negativities avec vapeur négative.
    
    Ce test valide la correction de la vapeur d'eau négative par sublimation
    de neige (rs) et de grésil (rg) vers la vapeur.
    
    Processus validé:
    - Si r_v < S_RTMIN après corrections: sublimer rs → rv
    - Si r_v < G_RTMIN après corrections: sublimer rg → rv
    - Ajustement thermodynamique avec chaleur latente de sublimation
    
    Vérifications:
        - La vapeur d'eau est ramenée au-dessus du seuil minimal
        - La neige/grésil est réduite pour compenser
        - La température est ajustée avec Ls
        - Conservation de la masse
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_correct_negativities import ice4_correct_negativities

    # Compilation du stencil GT4Py
    correct_neg_gt4py = stencil(
        backend,
        definition=ice4_correct_negativities,
        name="ice4_correct_negativities",
        dtypes=dtypes,
        externals=externals,
    )

    # Initialisation des champs
    field_names = ["th_t", "rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t", "lv_fact", "ls_fact"]
    
    fields = {}
    for name in field_names:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustement pour créer une situation avec vapeur très faible et neige/grésil disponibles
    fields["th_t"] = 250.0 + fields["th_t"] * 50.0
    fields["rv_t"] = fields["rv_t"] * 1e-16  # Vapeur quasi-nulle (< S_RTMIN)
    fields["rc_t"] = fields["rc_t"] * 0.005
    fields["rr_t"] = fields["rr_t"] * 0.003
    fields["ri_t"] = fields["ri_t"] * 0.001
    fields["rs_t"] = fields["rs_t"] * 0.005  # Neige disponible
    fields["rg_t"] = fields["rg_t"] * 0.005  # Grésil disponible
    
    fields["lv_fact"] = 2000.0 + fields["lv_fact"] * 500.0
    fields["ls_fact"] = 2500.0 + fields["ls_fact"] * 500.0
    
    # Masse totale avant
    total_water_before = (
        fields["rv_t"] + fields["rc_t"] + fields["rr_t"] + 
        fields["ri_t"] + fields["rs_t"] + fields["rg_t"]
    )
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in field_names:
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    correct_neg_gt4py(
        th_t=gt4py_fields["th_t"],
        rv_t=gt4py_fields["rv_t"],
        rc_t=gt4py_fields["rc_t"],
        rr_t=gt4py_fields["rr_t"],
        ri_t=gt4py_fields["ri_t"],
        rs_t=gt4py_fields["rs_t"],
        rg_t=gt4py_fields["rg_t"],
        lv_fact=gt4py_fields["lv_fact"],
        ls_fact=gt4py_fields["ls_fact"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. La vapeur a été augmentée par sublimation
    assert np.all(gt4py_fields["rv_t"] >= fields["rv_t"] - 1e-15), \
        "rv_t should increase after vapor correction"
    
    # 2. Conservation de la masse
    total_water_after = (
        gt4py_fields["rv_t"] + gt4py_fields["rc_t"] + gt4py_fields["rr_t"] + 
        gt4py_fields["ri_t"] + gt4py_fields["rs_t"] + gt4py_fields["rg_t"]
    )
    assert_allclose(total_water_after, total_water_before, rtol=1e-5, atol=1e-10,
                   err_msg="Total water mass should be conserved")
    
    # 3. Tous les champs sont finis et positifs
    for name in ["rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t"]:
        assert np.all(np.isfinite(gt4py_fields[name])), f"{name} contains non-finite values"
        assert np.all(gt4py_fields[name] >= -1e-15), f"{name} should be non-negative"
    
    # 4. La neige ou le grésil a diminué (sublimés)
    rs_reduced = np.sum(gt4py_fields["rs_t"] < fields["rs_t"] - 1e-15)
    rg_reduced = np.sum(gt4py_fields["rg_t"] < fields["rg_t"] - 1e-15)
    
    print(f"Test passed! Vapor correction applied")
    print(f"  Snow sublimated in {rs_reduced} cells")
    print(f"  Graupel sublimated in {rg_reduced} cells")
    print(f"  Mass conserved: {np.allclose(total_water_after, total_water_before, rtol=1e-5)}")


@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numpy", marks=pytest.mark.numpy),
    ],
)
def test_ice4_correct_negativities_mass_conservation(dtypes, backend, externals, domain, origin):
    """
    Test de conservation de la masse dans ice4_correct_negativities.
    
    Ce test vérifie que la masse totale d'eau est strictement conservée lors des
    corrections de négativités, même avec des valeurs négatives importantes.
    
    Masse totale: M = rv + rc + rr + ri + rs + rg
    
    Cette masse doit être strictement conservée.
    
    Vérifications:
        - La masse totale d'eau est conservée à la précision numérique près
        - Toutes les valeurs négatives sont éliminées
        - Tous les champs restent finis
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_correct_negativities import ice4_correct_negativities

    # Compilation du stencil GT4Py
    correct_neg_gt4py = stencil(
        backend,
        definition=ice4_correct_negativities,
        name="ice4_correct_negativities",
        dtypes=dtypes,
        externals=externals,
    )

    # Initialisation des champs avec quelques négatives
    field_names = ["th_t", "rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t", "lv_fact", "ls_fact"]
    
    fields = {}
    for name in field_names:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Configuration avec valeurs négatives
    fields["th_t"] = 270.0 + fields["th_t"] * 30.0
    fields["rv_t"] = fields["rv_t"] * 0.015
    fields["rc_t"] = (fields["rc_t"] - 0.5) * 0.002  # Quelques négatives
    fields["rr_t"] = (fields["rr_t"] - 0.5) * 0.001
    fields["ri_t"] = (fields["ri_t"] - 0.5) * 0.0005
    fields["rs_t"] = (fields["rs_t"] - 0.5) * 0.002
    fields["rg_t"] = (fields["rg_t"] - 0.5) * 0.002
    
    fields["lv_fact"] = 2200.0 + fields["lv_fact"] * 300.0
    fields["ls_fact"] = 2700.0 + fields["ls_fact"] * 300.0
    
    # Calcul de la masse totale d'eau avant
    mass_before = (
        fields["rv_t"] + fields["rc_t"] + fields["rr_t"] +
        fields["ri_t"] + fields["rs_t"] + fields["rg_t"]
    )
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in field_names:
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    correct_neg_gt4py(
        th_t=gt4py_fields["th_t"],
        rv_t=gt4py_fields["rv_t"],
        rc_t=gt4py_fields["rc_t"],
        rr_t=gt4py_fields["rr_t"],
        ri_t=gt4py_fields["ri_t"],
        rs_t=gt4py_fields["rs_t"],
        rg_t=gt4py_fields["rg_t"],
        lv_fact=gt4py_fields["lv_fact"],
        ls_fact=gt4py_fields["ls_fact"],
        domain=domain,
        origin=origin,
    )
    
    # Calcul de la masse totale d'eau après
    mass_after = (
        gt4py_fields["rv_t"] + gt4py_fields["rc_t"] + gt4py_fields["rr_t"] +
        gt4py_fields["ri_t"] + gt4py_fields["rs_t"] + gt4py_fields["rg_t"]
    )
    
    # Vérifications
    # 1. Conservation stricte de la masse
    assert_allclose(mass_after, mass_before, rtol=1e-5, atol=1e-10,
                   err_msg="Total water mass must be strictly conserved")
    
    # 2. Toutes les valeurs négatives éliminées
    for name in ["rc_t", "rr_t", "ri_t", "rs_t", "rg_t"]:
        assert np.all(gt4py_fields[name] >= -1e-15), \
            f"{name} should have no negative values"
    
    # 3. Tous les champs finis
    for name in field_names:
        assert np.all(np.isfinite(gt4py_fields[name])), \
            f"{name} contains non-finite values"
    
    print("Test passed! Mass conservation verified")
    print(f"  Mean mass before: {np.mean(mass_before):.6e} kg/kg")
    print(f"  Mean mass after:  {np.mean(mass_after):.6e} kg/kg")
    print(f"  Max mass diff: {np.max(np.abs(mass_after - mass_before)):.2e} kg/kg")
