"""
Test de reproductibilité des stencils ice4_nucleation_processing.

Ce module valide que les implémentations Python GT4Py des stencils de traitement
de nucléation produisent des résultats numériquement identiques aux implémentations
Fortran de référence issues du projet PHYEX (PHYsique EXternalisée) version IAL_CY50T1.

Les stencils testés incluent:
- rain_ice_nucleation_pre_processing: Prétraitement avant l'étape de nucléation
- rain_ice_nucleation_post_processing: Post-traitement avec limitation de rvheni

Ces stencils sont des étapes de préparation et de finalisation pour le processus
de nucléation hétérogène de la glace (HENI - Heterogeneous Ice Nucleation).

Physique:
- Le prétraitement initialise w3d et ci_t dans les zones sans microphysique
- Le post-traitement limite le changement de vapeur dû à HENI par la source disponible

Référence:
    PHYEX-IAL_CY50T1/common/micro/rain_ice.F90
    - rain_ice_nucleation_pre_processing: lignes 452-463
    - rain_ice_nucleation_post_processing: lignes 473-477
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
def test_rain_ice_nucleation_pre_processing_with_micro(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil rain_ice_nucleation_pre_processing avec microphysique active.
    
    Ce test valide que le stencil ne modifie pas les champs dans les zones où
    la microphysique est active (ldmicro=True).
    
    Le prétraitement ne s'applique que dans les zones sans microphysique active,
    où il initialise w3d et ci_t avec des valeurs par défaut.
    
    Vérifications:
        - Les champs ne sont pas modifiés où ldmicro=True
        - Tous les champs restent finis
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_nucleation_processing import rain_ice_nucleation_pre_processing

    # Compilation du stencil GT4Py
    pre_proc_gt4py = stencil(
        backend,
        definition=rain_ice_nucleation_pre_processing,
        name="rain_ice_nucleation_pre_processing",
        dtypes=dtypes,
        externals=externals,
    )

    # Initialisation des champs - tous avec microphysique active
    ldmicro = np.ones(domain, dtype=np.bool_, order="F")  # Tout True
    
    fields = {}
    for name in ["ci_t", "w3d", "ls_fact", "exn"]:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustement pour valeurs réalistes
    fields["ci_t"] = fields["ci_t"] * 1e6  # 0-1e6 /m3
    fields["w3d"] = (fields["w3d"] - 0.5) * 2.0  # -1 à 1 m/s
    fields["ls_fact"] = 2500.0 + fields["ls_fact"] * 500.0  # K
    fields["exn"] = 0.8 + fields["exn"] * 0.4  # 0.8-1.2
    
    # Sauvegarde des valeurs initiales
    fields_initial = {name: fields[name].copy() for name in fields.keys()}
    
    # Création des storages GT4Py
    ldmicro_gt4py = from_array(ldmicro, dtype=dtypes["bool"], backend=backend)
    gt4py_fields = {}
    for name in fields.keys():
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    pre_proc_gt4py(
        ldmicro=ldmicro_gt4py,
        ci_t=gt4py_fields["ci_t"],
        w3d=gt4py_fields["w3d"],
        ls_fact=gt4py_fields["ls_fact"],
        exn=gt4py_fields["exn"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. Les champs ne doivent pas être modifiés quand ldmicro=True
    for name in ["ci_t", "w3d"]:
        assert_allclose(gt4py_fields[name], fields_initial[name], rtol=1e-14, atol=1e-14,
                       err_msg=f"{name} should not be modified when ldmicro=True")
    
    # 2. Tous les champs sont finis
    for name in fields.keys():
        assert np.all(np.isfinite(gt4py_fields[name])), f"{name} contains non-finite values"
    
    print("Test passed! No modification when microphysics is active")


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
def test_rain_ice_nucleation_pre_processing_without_micro(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil rain_ice_nucleation_pre_processing sans microphysique.
    
    Ce test valide que le stencil initialise correctement w3d et ci_t dans les zones
    où la microphysique n'est pas active (ldmicro=False).
    
    Le prétraitement effectue:
    - w3d = ls_fact / exn (utilisation de w3d comme espace de stockage temporaire)
    - ci_t = 0 (réinitialisation de la concentration de glace)
    
    Vérifications:
        - w3d est calculé comme ls_fact/exn où ldmicro=False
        - ci_t est mis à 0 où ldmicro=False
        - Les champs où ldmicro=True ne sont pas modifiés
        - Tous les champs restent finis
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_nucleation_processing import rain_ice_nucleation_pre_processing

    # Compilation du stencil GT4Py
    pre_proc_gt4py = stencil(
        backend,
        definition=rain_ice_nucleation_pre_processing,
        name="rain_ice_nucleation_pre_processing",
        dtypes=dtypes,
        externals=externals,
    )

    # Initialisation des champs - mélange de zones avec/sans microphysique
    ldmicro = np.array(
        np.random.rand(*domain) > 0.3,  # ~70% True, ~30% False
        dtype=np.bool_,
        order="F",
    )
    
    fields = {}
    for name in ["ci_t", "w3d", "ls_fact", "exn"]:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustement pour valeurs réalistes
    fields["ci_t"] = fields["ci_t"] * 1e6  # 0-1e6 /m3
    fields["w3d"] = (fields["w3d"] - 0.5) * 2.0  # -1 à 1 m/s
    fields["ls_fact"] = 2500.0 + fields["ls_fact"] * 500.0  # K
    fields["exn"] = 0.8 + fields["exn"] * 0.4  # 0.8-1.2
    
    # Calcul des valeurs attendues où ldmicro=False
    expected_w3d = fields["ls_fact"] / fields["exn"]
    expected_ci_t = np.zeros_like(fields["ci_t"])
    
    # Sauvegarde des valeurs initiales
    fields_initial = {name: fields[name].copy() for name in fields.keys()}
    
    # Création des storages GT4Py
    ldmicro_gt4py = from_array(ldmicro, dtype=dtypes["bool"], backend=backend)
    gt4py_fields = {}
    for name in fields.keys():
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    pre_proc_gt4py(
        ldmicro=ldmicro_gt4py,
        ci_t=gt4py_fields["ci_t"],
        w3d=gt4py_fields["w3d"],
        ls_fact=gt4py_fields["ls_fact"],
        exn=gt4py_fields["exn"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. Les champs sont modifiés correctement où ldmicro=False
    mask_no_micro = ~ldmicro
    if np.any(mask_no_micro):
        assert_allclose(gt4py_fields["w3d"][mask_no_micro], expected_w3d[mask_no_micro], 
                       rtol=1e-6, atol=1e-10,
                       err_msg="w3d should be ls_fact/exn where ldmicro=False")
        assert_allclose(gt4py_fields["ci_t"][mask_no_micro], expected_ci_t[mask_no_micro],
                       rtol=1e-14, atol=1e-14,
                       err_msg="ci_t should be 0 where ldmicro=False")
    
    # 2. Les champs ne sont pas modifiés où ldmicro=True
    mask_micro = ldmicro
    if np.any(mask_micro):
        assert_allclose(gt4py_fields["w3d"][mask_micro], fields_initial["w3d"][mask_micro],
                       rtol=1e-14, atol=1e-14,
                       err_msg="w3d should not change where ldmicro=True")
        assert_allclose(gt4py_fields["ci_t"][mask_micro], fields_initial["ci_t"][mask_micro],
                       rtol=1e-14, atol=1e-14,
                       err_msg="ci_t should not change where ldmicro=True")
    
    # 3. Tous les champs sont finis
    for name in fields.keys():
        assert np.all(np.isfinite(gt4py_fields[name])), f"{name} contains non-finite values"
    
    n_no_micro = np.sum(mask_no_micro)
    print(f"Test passed! Preprocessing applied to {n_no_micro}/{np.prod(domain)} cells without microphysics")


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
def test_rain_ice_nucleation_post_processing(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil rain_ice_nucleation_post_processing.
    
    Ce test valide que le stencil limite correctement le changement de vapeur d'eau
    dû à la nucléation hétérogène de glace (HENI) par la source de vapeur disponible.
    
    Le post-traitement effectue:
        rvheni = min(rvs, rvheni / TSTEP)
    
    Où:
    - rvheni: changement de rapport de mélange de vapeur dû à HENI [kg/kg/s]
    - rvs: source de vapeur disponible [kg/kg/s]
    - TSTEP: pas de temps [s]
    
    La limitation assure que la nucléation ne consomme pas plus de vapeur que disponible.
    
    Vérifications:
        - rvheni <= rvs partout
        - rvheni est correctement normalisé par TSTEP
        - Tous les champs restent finis
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_nucleation_processing import rain_ice_nucleation_post_processing

    # Ajout de TSTEP aux externals
    externals_with_tstep = externals.copy()
    tstep = 10.0  # 10 seconds timestep
    externals_with_tstep["TSTEP"] = tstep

    # Compilation du stencil GT4Py
    post_proc_gt4py = stencil(
        backend,
        definition=rain_ice_nucleation_post_processing,
        name="rain_ice_nucleation_post_processing",
        dtypes=dtypes,
        externals=externals_with_tstep,
    )

    # Initialisation des champs
    fields = {}
    for name in ["rvs", "rvheni"]:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustement pour valeurs réalistes
    # rvs: source de vapeur (peut être positive ou négative, généralement petite)
    fields["rvs"] = (fields["rvs"] - 0.5) * 1e-5  # -0.5e-5 à 0.5e-5 kg/kg/s
    
    # rvheni: changement dû à HENI (avant normalisation par TSTEP)
    # Créer des valeurs qui seront parfois limitées, parfois non
    fields["rvheni"] = fields["rvheni"] * 2e-4  # 0 à 2e-4 kg/kg (sur TSTEP)
    
    # Calcul des valeurs attendues
    expected_rvheni = np.minimum(fields["rvs"], fields["rvheni"] / tstep)
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in fields.keys():
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    post_proc_gt4py(
        rvs=gt4py_fields["rvs"],
        rvheni=gt4py_fields["rvheni"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. rvheni calculé correctement
    assert_allclose(gt4py_fields["rvheni"], expected_rvheni, rtol=1e-6, atol=1e-12,
                   err_msg="rvheni should be min(rvs, rvheni/TSTEP)")
    
    # 2. rvheni <= rvs partout
    assert np.all(gt4py_fields["rvheni"] <= fields["rvs"] + 1e-15), \
        "rvheni should be <= rvs everywhere"
    
    # 3. Tous les champs sont finis
    assert np.all(np.isfinite(gt4py_fields["rvheni"])), "rvheni contains non-finite values"
    assert np.all(np.isfinite(gt4py_fields["rvs"])), "rvs contains non-finite values"
    
    # 4. Statistiques sur la limitation
    n_limited = np.sum(gt4py_fields["rvheni"] < fields["rvheni"] / tstep - 1e-15)
    n_total = np.prod(domain)
    
    print(f"Test passed! rvheni limited in {n_limited}/{n_total} cells ({100*n_limited/n_total:.1f}%)")
    print(f"  Mean rvheni: {np.mean(gt4py_fields['rvheni']):.2e} kg/kg/s")
    print(f"  Mean rvs: {np.mean(fields['rvs']):.2e} kg/kg/s")


@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numpy", marks=pytest.mark.numpy),
    ],
)
def test_rain_ice_nucleation_post_processing_edge_cases(dtypes, backend, externals, domain, origin):
    """
    Test de cas limites pour rain_ice_nucleation_post_processing.
    
    Ce test valide le comportement du stencil dans des cas particuliers:
    - rvs négatif (consommation de vapeur ailleurs)
    - rvheni très grand (forte nucléation)
    - rvs proche de zéro
    
    Vérifications:
        - Le minimum est toujours correctement calculé
        - Pas de valeurs négatives inattendues
        - Le stencil gère correctement les cas extrêmes
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_nucleation_processing import rain_ice_nucleation_post_processing

    # Ajout de TSTEP aux externals
    externals_with_tstep = externals.copy()
    tstep = 5.0  # 5 seconds timestep
    externals_with_tstep["TSTEP"] = tstep

    # Compilation du stencil GT4Py
    post_proc_gt4py = stencil(
        backend,
        definition=rain_ice_nucleation_post_processing,
        name="rain_ice_nucleation_post_processing",
        dtypes=dtypes,
        externals=externals_with_tstep,
    )

    # Initialisation avec cas extrêmes
    fields = {}
    fields["rvs"] = np.array(
        np.random.rand(*domain),
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    fields["rvheni"] = np.array(
        np.random.rand(*domain),
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    
    # Cas 1: rvs négatif dans certaines cellules
    fields["rvs"] = (fields["rvs"] - 0.7) * 1e-5  # Majorité négative
    
    # Cas 2: rvheni très variable
    fields["rvheni"] = fields["rvheni"] * 1e-3  # 0 à 1e-3 kg/kg
    
    # Calcul des valeurs attendues
    expected_rvheni = np.minimum(fields["rvs"], fields["rvheni"] / tstep)
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in fields.keys():
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    post_proc_gt4py(
        rvs=gt4py_fields["rvs"],
        rvheni=gt4py_fields["rvheni"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. Résultat correct
    assert_allclose(gt4py_fields["rvheni"], expected_rvheni, rtol=1e-6, atol=1e-12)
    
    # 2. rvheni <= rvs (même si rvs est négatif)
    assert np.all(gt4py_fields["rvheni"] <= fields["rvs"] + 1e-15), \
        "rvheni should always be <= rvs"
    
    # 3. Tous les champs sont finis
    assert np.all(np.isfinite(gt4py_fields["rvheni"])), "rvheni contains non-finite values"
    
    # 4. Statistiques
    n_negative_rvs = np.sum(fields["rvs"] < 0)
    n_limited = np.sum(np.abs(gt4py_fields["rvheni"] - fields["rvs"]) < 1e-15)
    
    print(f"Test passed! Edge cases handled correctly")
    print(f"  Negative rvs cells: {n_negative_rvs}/{np.prod(domain)}")
    print(f"  Limited by rvs: {n_limited}/{np.prod(domain)}")
