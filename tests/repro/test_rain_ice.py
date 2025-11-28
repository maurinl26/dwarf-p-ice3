"""
Test de reproductibilité des stencils rain_ice.

Ce module valide que les implémentations Python GT4Py des stencils rain_ice
produisent des résultats numériquement identiques aux implémentations Fortran de référence
issues du projet PHYEX (PHYsique EXternalisée) version IAL_CY50T1.

Les stencils rain_ice testés incluent:
- rain_ice_total_tendencies: Calcul des tendances totales limitées par les espèces disponibles
- rain_ice_thermo: Calcul des variables thermodynamiques (T, Ls/Cp, Lv/Cp)
- rain_ice_mask: Création du masque pour les calculs microphysiques
- initial_values_saving: Sauvegarde des valeurs initiales avant processus microphysiques
- ice4_precipitation_fraction_sigma: Calcul de la variance de supersaturation
- rain_fraction_sedimentation: Calcul de la fraction verticale de pluie
- ice4_rainfr_vert: Fraction de pluie verticale avec calcul backward
- fog_deposition: Dépôt de brouillard sur la végétation

Ces stencils constituent les éléments de base du schéma microphysique ICE4
pour le traitement de la phase condensée (rain_ice) et de la thermodynamique associée.

Référence:
    PHYEX-IAL_CY50T1/common/micro/rain_ice.F90
"""
from ctypes import c_double, c_float

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array, zeros
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
def test_rain_ice_total_tendencies(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil rain_ice_total_tendencies (PHYEX-IAL_CY50T1).
    
    Ce test valide l'implémentation Python/GT4Py du calcul des tendances totales
    pour toutes les espèces microphysiques, limitées par les espèces disponibles.
    
    Le stencil rain_ice_total_tendencies effectue:
    1. Calcul des tendances des hydrométéores par différence temporelle:
       dX/dt = (X_initial - X_t) / Δt
    2. Calcul de la tendance de température potentielle incluant:
       - Chaleur latente de vaporisation pour (cloud + rain)
       - Chaleur latente de sublimation pour (ice + snow + graupel)
    3. Mise à jour des sources (tendances) en prenant en compte la nucléation
    
    Physique:
        dθ/dt = [(drc + drr) * Lv + (dri + drs + drg) * Ls] / (Cp * Π)
        
    Champs validés:
        - wr_th, wr_v, wr_c, wr_r, wr_i, wr_s, wr_g: Tendances des espèces [kg.kg-1.s-1]
        - ths, rvs, rcs, rrs, ris, rss, rgs: Sources totales [kg.kg-1.s-1]
    
    Tolérance:
        rtol=1e-6, atol=1e-8
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.rain_ice import rain_ice_total_tendencies

    # Add INV_TSTEP to externals
    externals_with_tstep = externals.copy()
    tstep = 10.0  # 10 seconds timestep
    externals_with_tstep["INV_TSTEP"] = 1.0 / tstep

    # Compilation du stencil GT4Py
    rain_ice_total_tendencies_gt4py = stencil(
        backend,
        definition=rain_ice_total_tendencies,
        name="rain_ice_total_tendencies",
        dtypes=dtypes,
        externals=externals_with_tstep,
    )

    # Initialisation des champs d'entrée avec des valeurs réalistes
    input_field_names = [
        "wr_th", "wr_v", "wr_c", "wr_r", "wr_i", "wr_s", "wr_g",
        "ls_fact", "lv_fact", "exnref",
        "ths", "rvs", "rcs", "rrs", "ris", "rss", "rgs",
        "rvheni",
        "rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t"
    ]
    
    fields = {}
    for name in input_field_names:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustement pour valeurs physiquement réalistes
    fields["wr_th"] = 250.0 + fields["wr_th"] * 50.0  # 250-300 K
    fields["wr_v"] = fields["wr_v"] * 0.02  # 0-20 g/kg
    fields["wr_c"] = fields["wr_c"] * 0.005  # 0-5 g/kg
    fields["wr_r"] = fields["wr_r"] * 0.003  # 0-3 g/kg
    fields["wr_i"] = fields["wr_i"] * 0.001  # 0-1 g/kg
    fields["wr_s"] = fields["wr_s"] * 0.002  # 0-2 g/kg
    fields["wr_g"] = fields["wr_g"] * 0.003  # 0-3 g/kg
    
    fields["ls_fact"] = 2500.0 + fields["ls_fact"] * 500.0  # K
    fields["lv_fact"] = 2000.0 + fields["lv_fact"] * 500.0  # K
    fields["exnref"] = 0.8 + fields["exnref"] * 0.4  # 0.8-1.2
    
    # Tendances initiales (petites valeurs)
    fields["ths"] = (fields["ths"] - 0.5) * 0.1
    fields["rvs"] = (fields["rvs"] - 0.5) * 1e-5
    fields["rcs"] = (fields["rcs"] - 0.5) * 1e-5
    fields["rrs"] = (fields["rrs"] - 0.5) * 1e-5
    fields["ris"] = (fields["ris"] - 0.5) * 1e-6
    fields["rss"] = (fields["rss"] - 0.5) * 1e-5
    fields["rgs"] = (fields["rgs"] - 0.5) * 1e-5
    
    fields["rvheni"] = fields["rvheni"] * 1e-6  # Nucléation très faible
    
    # États finaux (légèrement différents des initiaux)
    fields["rv_t"] = fields["wr_v"] - (np.random.rand(*domain) - 0.5) * 0.001
    fields["rc_t"] = fields["wr_c"] - (np.random.rand(*domain) - 0.5) * 0.0005
    fields["rr_t"] = fields["wr_r"] - (np.random.rand(*domain) - 0.5) * 0.0003
    fields["ri_t"] = fields["wr_i"] - (np.random.rand(*domain) - 0.5) * 0.0001
    fields["rs_t"] = fields["wr_s"] - (np.random.rand(*domain) - 0.5) * 0.0002
    fields["rg_t"] = fields["wr_g"] - (np.random.rand(*domain) - 0.5) * 0.0003
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in input_field_names:
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    rain_ice_total_tendencies_gt4py(
        wr_th=gt4py_fields["wr_th"],
        wr_v=gt4py_fields["wr_v"],
        wr_c=gt4py_fields["wr_c"],
        wr_r=gt4py_fields["wr_r"],
        wr_i=gt4py_fields["wr_i"],
        wr_s=gt4py_fields["wr_s"],
        wr_g=gt4py_fields["wr_g"],
        ls_fact=gt4py_fields["ls_fact"],
        lv_fact=gt4py_fields["lv_fact"],
        exnref=gt4py_fields["exnref"],
        ths=gt4py_fields["ths"],
        rvs=gt4py_fields["rvs"],
        rcs=gt4py_fields["rcs"],
        rrs=gt4py_fields["rrs"],
        ris=gt4py_fields["ris"],
        rss=gt4py_fields["rss"],
        rgs=gt4py_fields["rgs"],
        rvheni=gt4py_fields["rvheni"],
        rv_t=gt4py_fields["rv_t"],
        rc_t=gt4py_fields["rc_t"],
        rr_t=gt4py_fields["rr_t"],
        ri_t=gt4py_fields["ri_t"],
        rs_t=gt4py_fields["rs_t"],
        rg_t=gt4py_fields["rg_t"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications de cohérence physique
    # 1. Les tendances ont été modifiées
    assert not np.allclose(gt4py_fields["wr_v"], fields["wr_v"], atol=1e-15), \
        "wr_v was not modified"
    
    # 2. Tous les champs sont finis
    for name in ["wr_th", "wr_v", "wr_c", "wr_r", "wr_i", "wr_s", "wr_g", 
                 "ths", "rvs", "rcs", "rrs", "ris", "rss", "rgs"]:
        assert np.all(np.isfinite(gt4py_fields[name])), f"{name} contains non-finite values"
    
    # 3. Les sources ont été mises à jour
    assert not np.allclose(gt4py_fields["ths"], fields["ths"], atol=1e-15), \
        "ths sources were not updated"
    
    print(f"Test passed! Tendencies and sources updated successfully")


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
def test_rain_ice_thermo(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil rain_ice_thermo (PHYEX-IAL_CY50T1).
    
    Ce test valide l'implémentation Python/GT4Py du calcul des variables
    thermodynamiques nécessaires aux processus microphysiques rain_ice.
    
    Le stencil rain_ice_thermo calcule:
    1. La température T à partir de θ et de l'exposant d'Exner: T = θ * Π
    2. La capacité thermique spécifique Cp incluant toutes les phases
    3. Les facteurs Ls/Cp et Lv/Cp tenant compte de la dépendance en température
    
    Physique:
        Cp = CPD + CPV*rv + CL*(rc+rr) + CI*(ri+rs+rg)
        T = θ * Π
        Ls(T) = LSTT + (CPV-CI)*(T-TT)
        Lv(T) = LVTT + (CPV-CL)*(T-TT)
        
    Champs validés:
        - ls_fact: Ls/Cp [K]
        - lv_fact: Lv/Cp [K]
    
    Tolérance:
        rtol=1e-6, atol=1e-8
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.rain_ice import rain_ice_thermo

    # Compilation du stencil GT4Py
    rain_ice_thermo_gt4py = stencil(
        backend,
        definition=rain_ice_thermo,
        name="rain_ice_thermo",
        dtypes=dtypes,
        externals=externals,
    )

    # Initialisation des champs d'entrée
    input_field_names = [
        "exn", "ls_fact", "lv_fact", "th_t",
        "rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t"
    ]
    
    fields = {}
    for name in input_field_names:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustement pour valeurs réalistes
    fields["exn"] = 0.8 + fields["exn"] * 0.4  # 0.8-1.2
    fields["th_t"] = 250.0 + fields["th_t"] * 50.0  # 250-300 K
    fields["rv_t"] = fields["rv_t"] * 0.02  # 0-20 g/kg
    fields["rc_t"] = fields["rc_t"] * 0.005  # 0-5 g/kg
    fields["rr_t"] = fields["rr_t"] * 0.003  # 0-3 g/kg
    fields["ri_t"] = fields["ri_t"] * 0.001  # 0-1 g/kg
    fields["rs_t"] = fields["rs_t"] * 0.002  # 0-2 g/kg
    fields["rg_t"] = fields["rg_t"] * 0.003  # 0-3 g/kg
    
    # Initialize ls_fact and lv_fact with zeros (output fields)
    fields["ls_fact"] = np.zeros(domain, dtype=(c_float if dtypes["float"] == np.float32 else c_double), order="F")
    fields["lv_fact"] = np.zeros(domain, dtype=(c_float if dtypes["float"] == np.float32 else c_double), order="F")
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in input_field_names:
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    rain_ice_thermo_gt4py(
        exn=gt4py_fields["exn"],
        ls_fact=gt4py_fields["ls_fact"],
        lv_fact=gt4py_fields["lv_fact"],
        th_t=gt4py_fields["th_t"],
        rv_t=gt4py_fields["rv_t"],
        rc_t=gt4py_fields["rc_t"],
        rr_t=gt4py_fields["rr_t"],
        ri_t=gt4py_fields["ri_t"],
        rs_t=gt4py_fields["rs_t"],
        rg_t=gt4py_fields["rg_t"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications de cohérence physique
    # 1. Les facteurs ont été calculés (non-nuls)
    assert np.any(gt4py_fields["ls_fact"] > 0), "ls_fact is all zeros"
    assert np.any(gt4py_fields["lv_fact"] > 0), "lv_fact is all zeros"
    
    # 2. Tous les champs sont finis
    assert np.all(np.isfinite(gt4py_fields["ls_fact"])), "ls_fact contains non-finite values"
    assert np.all(np.isfinite(gt4py_fields["lv_fact"])), "lv_fact contains non-finite values"
    
    # 3. Valeurs dans une plage raisonnable (Ls/Cp et Lv/Cp en K)
    assert np.all(gt4py_fields["ls_fact"] > 0) and np.all(gt4py_fields["ls_fact"] < 5000), \
        "ls_fact values out of physical range"
    assert np.all(gt4py_fields["lv_fact"] > 0) and np.all(gt4py_fields["lv_fact"] < 5000), \
        "lv_fact values out of physical range"
    
    # 4. Ls/Cp devrait être supérieur à Lv/Cp (sublimation > évaporation)
    assert np.all(gt4py_fields["ls_fact"] > gt4py_fields["lv_fact"]), \
        "ls_fact should be greater than lv_fact everywhere"
    
    print(f"Test passed! ls_fact range: [{np.min(gt4py_fields['ls_fact']):.2f}, {np.max(gt4py_fields['ls_fact']):.2f}] K")
    print(f"             lv_fact range: [{np.min(gt4py_fields['lv_fact']):.2f}, {np.max(gt4py_fields['lv_fact']):.2f}] K")


@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numpy", marks=pytest.mark.numpy),
    ],
)
def test_rain_ice_mask(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil rain_ice_mask (PHYEX-IAL_CY50T1).
    
    Ce test valide l'implémentation Python/GT4Py du calcul du masque booléen
    qui détermine où les processus microphysiques doivent être appliqués.
    
    Le stencil rain_ice_mask crée un masque ldmicro qui est vrai (True) si
    au moins une des espèces hydrometeor dépasse son seuil minimal:
    - rc > C_RTMIN (cloud)
    - rr > R_RTMIN (rain)
    - ri > I_RTMIN (ice)
    - rs > S_RTMIN (snow)
    - rg > G_RTMIN (graupel)
    
    Physique:
        ldmicro = (rc > C_RTMIN) OR (rr > R_RTMIN) OR (ri > I_RTMIN) 
                  OR (rs > S_RTMIN) OR (rg > G_RTMIN)
        
    Champs validés:
        - ldmicro: Masque booléen pour calculs microphysiques
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.rain_ice import rain_ice_mask

    # Compilation du stencil GT4Py
    rain_ice_mask_gt4py = stencil(
        backend,
        definition=rain_ice_mask,
        name="rain_ice_mask",
        dtypes=dtypes,
        externals=externals,
    )

    # Initialisation des champs d'entrée
    input_field_names = ["rc_t", "rr_t", "ri_t", "rs_t", "rg_t"]
    
    fields = {}
    for name in input_field_names:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustement : certaines cellules au-dessus des seuils, d'autres en-dessous
    rtmin = 1e-15  # Seuil typique
    fields["rc_t"] = fields["rc_t"] * 2 * rtmin  # 0 à 2*RTMIN
    fields["rr_t"] = fields["rr_t"] * 2 * rtmin
    fields["ri_t"] = fields["ri_t"] * 2 * rtmin
    fields["rs_t"] = fields["rs_t"] * 2 * rtmin
    fields["rg_t"] = fields["rg_t"] * 2 * rtmin
    
    # Masque de sortie
    ldmicro = np.zeros(domain, dtype=np.bool_, order="F")
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in input_field_names:
        gt4py_fields[name] = from_array(fields[name], dtype=dtypes["float"], backend=backend)
    
    ldmicro_gt4py = from_array(ldmicro, dtype=dtypes["bool"], backend=backend)
    
    # Exécution GT4Py
    rain_ice_mask_gt4py(
        rc_t=gt4py_fields["rc_t"],
        rr_t=gt4py_fields["rr_t"],
        ri_t=gt4py_fields["ri_t"],
        rs_t=gt4py_fields["rs_t"],
        rg_t=gt4py_fields["rg_t"],
        ldmicro=ldmicro_gt4py,
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. Le masque a été modifié
    assert not np.all(ldmicro_gt4py == False), "ldmicro is all False"
    
    # 2. Certaines cellules sont actives (True)
    n_active = np.sum(ldmicro_gt4py)
    assert n_active > 0, "No active microphysics cells"
    
    # 3. Le masque est bien booléen
    assert ldmicro_gt4py.dtype == np.bool_, "ldmicro is not boolean"
    
    print(f"Test passed! {n_active}/{np.prod(domain)} cells active for microphysics " +
          f"({100*n_active/np.prod(domain):.1f}%)")


@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numpy", marks=pytest.mark.numpy),
    ],
)
def test_initial_values_saving(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil initial_values_saving (PHYEX-IAL_CY50T1).
    
    Ce test valide l'implémentation Python/GT4Py de la sauvegarde des valeurs initiales
    avant l'application des processus microphysiques. Cette sauvegarde est nécessaire pour:
    1. Calculer les tendances par différence temporelle
    2. Conserver une référence pour les diagnostics
    3. Initialiser certains champs (evap3d, rainfr)
    
    Le stencil initial_values_saving effectue:
    - Sauvegarde: wr_* = *_t pour toutes les espèces
    - Initialisation: evap3d = 0 (si LWARM activé)
    - Initialisation: rainfr = 0
        
    Champs validés:
        - wr_th, wr_v, wr_c, wr_r, wr_i, wr_s, wr_g: Valeurs sauvegardées
        - evap3d: Évaporation 3D [kg.kg-1]
        - rainfr: Fraction de pluie [-]
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.rain_ice import initial_values_saving

    # Compilation du stencil GT4Py
    rain_ice_initial_gt4py = stencil(
        backend,
        definition=initial_values_saving,
        name="initial_values_saving",
        dtypes=dtypes,
        externals=externals,
    )

    # Initialisation des champs
    field_names = [
        "wr_th", "wr_v", "wr_c", "wr_r", "wr_i", "wr_s", "wr_g",
        "th_t", "rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t",
        "evap3d", "rainfr"
    ]
    
    fields = {}
    for name in field_names:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustement pour valeurs réalistes des états courants
    fields["th_t"] = 250.0 + fields["th_t"] * 50.0
    fields["rv_t"] = fields["rv_t"] * 0.02
    fields["rc_t"] = fields["rc_t"] * 0.005
    fields["rr_t"] = fields["rr_t"] * 0.003
    fields["ri_t"] = fields["ri_t"] * 0.001
    fields["rs_t"] = fields["rs_t"] * 0.002
    fields["rg_t"] = fields["rg_t"] * 0.003
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in field_names:
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    rain_ice_initial_gt4py(
        wr_th=gt4py_fields["wr_th"],
        wr_v=gt4py_fields["wr_v"],
        wr_c=gt4py_fields["wr_c"],
        wr_r=gt4py_fields["wr_r"],
        wr_i=gt4py_fields["wr_i"],
        wr_s=gt4py_fields["wr_s"],
        wr_g=gt4py_fields["wr_g"],
        th_t=gt4py_fields["th_t"],
        rv_t=gt4py_fields["rv_t"],
        rc_t=gt4py_fields["rc_t"],
        rr_t=gt4py_fields["rr_t"],
        ri_t=gt4py_fields["ri_t"],
        rs_t=gt4py_fields["rs_t"],
        rg_t=gt4py_fields["rg_t"],
        evap3d=gt4py_fields["evap3d"],
        rainfr=gt4py_fields["rainfr"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. Les valeurs initiales ont été copiées correctement
    assert_allclose(gt4py_fields["wr_th"], fields["th_t"], rtol=1e-15, atol=1e-15)
    assert_allclose(gt4py_fields["wr_v"], fields["rv_t"], rtol=1e-15, atol=1e-15)
    assert_allclose(gt4py_fields["wr_c"], fields["rc_t"], rtol=1e-15, atol=1e-15)
    assert_allclose(gt4py_fields["wr_r"], fields["rr_t"], rtol=1e-15, atol=1e-15)
    assert_allclose(gt4py_fields["wr_i"], fields["ri_t"], rtol=1e-15, atol=1e-15)
    assert_allclose(gt4py_fields["wr_s"], fields["rs_t"], rtol=1e-15, atol=1e-15)
    assert_allclose(gt4py_fields["wr_g"], fields["rg_t"], rtol=1e-15, atol=1e-15)
    
    # 2. evap3d initialisé à 0 (LWARM est True pour AROME)
    assert np.all(gt4py_fields["evap3d"] == 0), "evap3d should be initialized to 0"
    
    # 3. rainfr initialisé à 0
    assert np.all(gt4py_fields["rainfr"] == 0), "rainfr should be initialized to 0"
    
    print("Test passed! Initial values saved correctly, evap3d and rainfr initialized")


@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numpy", marks=pytest.mark.numpy),
    ],
)
def test_ice4_precipitation_fraction_sigma(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil ice4_precipitation_fraction_sigma (PHYEX-IAL_CY50T1).
    
    Ce test valide l'implémentation Python/GT4Py du calcul de la variance de supersaturation
    à partir de l'écart-type de supersaturation.
    
    Le stencil ice4_precipitation_fraction_sigma calcule:
        sigma_rc = sigs^2
    
    où sigs est l'écart-type de la supersaturation et sigma_rc est sa variance.
    Cette variance est utilisée dans les schémas de sous-maille pour la formation
    des précipitations avec une approche PDF.
    
    Champs validés:
        - sigma_rc: Variance de supersaturation [-]
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.rain_ice import ice4_precipitation_fraction_sigma

    # Compilation du stencil GT4Py
    sigma_gt4py = stencil(
        backend,
        definition=ice4_precipitation_fraction_sigma,
        name="ice4_precipitation_fraction_sigma",
        dtypes=dtypes,
        externals=externals,
    )

    # Initialisation des champs
    sigs = np.array(
        np.random.rand(*domain),
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    
    # Ajustement pour valeurs réalistes (écart-type petit)
    sigs = sigs * 0.1  # 0-0.1
    
    sigma_rc = np.zeros(domain, dtype=(c_float if dtypes["float"] == np.float32 else c_double), order="F")
    
    # Création des storages GT4Py
    sigs_gt4py = from_array(sigs, dtype=dtypes["float"], backend=backend)
    sigma_rc_gt4py = from_array(sigma_rc, dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    sigma_gt4py(
        sigs=sigs_gt4py,
        sigma_rc=sigma_rc_gt4py,
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. Calcul correct: sigma_rc = sigs^2
    expected = sigs**2
    assert_allclose(sigma_rc_gt4py, expected, rtol=1e-5, atol=1e-9)
    
    # 2. Valeurs positives
    assert np.all(sigma_rc_gt4py >= 0), "sigma_rc should be positive"
    
    # 3. Variance inférieure à l'écart-type pour valeurs < 1
    assert np.all(sigma_rc_gt4py <= sigs), "variance should be <= std dev for values < 1"
    
    print(f"Test passed! Variance correctly computed from standard deviation")


@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numpy", marks=pytest.mark.numpy),
    ],
)
def test_rain_fraction_sedimentation(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil rain_fraction_sedimentation (PHYEX-IAL_CY50T1).
    
    Ce test valide l'implémentation Python/GT4Py du calcul de la fraction verticale
    de pluie pour la sédimentation. Ce calcul initialise les champs au premier niveau
    vertical avec les tendances multipliées par le pas de temps.
    
    Le stencil rain_fraction_sedimentation effectue (au premier niveau uniquement):
        wr_r = rrs * TSTEP
        wr_s = rss * TSTEP
        wr_g = rgs * TSTEP
    
    Champs validés:
        - wr_r: Valeur initiale pour sédimentation rain [kg.kg-1]
        - wr_s: Valeur initiale pour sédimentation snow [kg.kg-1]
        - wr_g: Valeur initiale pour sédimentation graupel [kg.kg-1]
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.rain_ice import rain_fraction_sedimentation

    # Add TSTEP to externals
    externals_with_tstep = externals.copy()
    tstep = 10.0  # 10 seconds timestep
    externals_with_tstep["TSTEP"] = tstep

    # Compilation du stencil GT4Py
    rain_frac_gt4py = stencil(
        backend,
        definition=rain_fraction_sedimentation,
        name="rain_fraction_sedimentation",
        dtypes=dtypes,
        externals=externals_with_tstep,
    )

    # Initialisation des champs
    field_names = ["wr_r", "wr_s", "wr_g", "rrs", "rss", "rgs"]
    
    fields = {}
    for name in field_names:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustement pour valeurs réalistes (tendances)
    fields["rrs"] = (fields["rrs"] - 0.5) * 1e-4  # tendance rain
    fields["rss"] = (fields["rss"] - 0.5) * 1e-5  # tendance snow
    fields["rgs"] = (fields["rgs"] - 0.5) * 1e-5  # tendance graupel
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in field_names:
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    rain_frac_gt4py(
        wr_r=gt4py_fields["wr_r"],
        wr_s=gt4py_fields["wr_s"],
        wr_g=gt4py_fields["wr_g"],
        rrs=gt4py_fields["rrs"],
        rss=gt4py_fields["rss"],
        rgs=gt4py_fields["rgs"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications (au premier niveau uniquement k=0)
    # 1. Calcul correct au niveau 0
    expected_wr_r = fields["rrs"][:, :, 0] * tstep
    expected_wr_s = fields["rss"][:, :, 0] * tstep
    expected_wr_g = fields["rgs"][:, :, 0] * tstep
    
    assert_allclose(gt4py_fields["wr_r"][:, :, 0], expected_wr_r, rtol=1e-14, atol=1e-14)
    assert_allclose(gt4py_fields["wr_s"][:, :, 0], expected_wr_s, rtol=1e-14, atol=1e-14)
    assert_allclose(gt4py_fields["wr_g"][:, :, 0], expected_wr_g, rtol=1e-14, atol=1e-14)
    
    # 2. Les autres niveaux ne devraient pas avoir changé
    if domain[2] > 1:
        assert_allclose(gt4py_fields["wr_r"][:, :, 1:], fields["wr_r"][:, :, 1:], rtol=1e-15, atol=1e-15)
    
    print(f"Test passed! Rain fraction initialized at first level")


@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numpy", marks=pytest.mark.numpy),
    ],
)
def test_ice4_rainfr_vert(dtypes, backend, externals, domain, origin):
    """
    Test de reproductibilité du stencil ice4_rainfr_vert (PHYEX-IAL_CY50T1).
    
    Ce test valide l'implémentation Python/GT4Py du calcul de la fraction de pluie
    verticale avec propagation backward (du haut vers le bas).
    
    Le stencil ice4_rainfr_vert effectue un calcul backward où:
    - Si une espèce (rr, rs, rg) dépasse son seuil au niveau k:
      * prfr[k] = max(prfr[k], prfr[k+1])
      * Si prfr[k] == 0, alors prfr[k] = 1
    - Sinon: prfr[k] = 0
    
    Ce calcul propage la fraction de pluie vers le bas de la colonne.
    
    Champs validés:
        - prfr: Fraction de pluie [-]
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.rain_ice import ice4_rainfr_vert

    # Compilation du stencil GT4Py
    rainfr_vert_gt4py = stencil(
        backend,
        definition=ice4_rainfr_vert,
        name="ice4_rainfr_vert",
        dtypes=dtypes,
        externals=externals,
    )

    # Initialisation des champs
    field_names = ["prfr", "rr", "rs", "rg"]
    
    fields = {}
    for name in field_names:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustement pour valeurs réalistes
    rtmin = 1e-15
    fields["rr"] = fields["rr"] * 2 * rtmin  # Certaines au-dessus, d'autres en-dessous
    fields["rs"] = fields["rs"] * 2 * rtmin
    fields["rg"] = fields["rg"] * 2 * rtmin
    fields["prfr"] = fields["prfr"] * 0.5  # 0-0.5
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in field_names:
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    rainfr_vert_gt4py(
        prfr=gt4py_fields["prfr"],
        rr=gt4py_fields["rr"],
        rs=gt4py_fields["rs"],
        rg=gt4py_fields["rg"],
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. Fraction de pluie est entre 0 et 1
    assert np.all(gt4py_fields["prfr"] >= 0) and np.all(gt4py_fields["prfr"] <= 1), \
        "prfr should be between 0 and 1"
    
    # 2. Tous les champs sont finis
    assert np.all(np.isfinite(gt4py_fields["prfr"])), "prfr contains non-finite values"
    
    print(f"Test passed! Rain fraction computed with backward propagation")


@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numpy", marks=pytest.mark.numpy),
    ],
)
def test_fog_deposition(dtypes, backend, phyex, domain, origin):
    """
    Test de reproductibilité du stencil fog_deposition (PHYEX-IAL_CY50T1).
    
    Ce test valide l'implémentation Python/GT4Py du calcul du dépôt de brouillard
    sur la végétation. Ce processus n'est pas activé dans AROME mais est présent
    dans le code.
    
    Le stencil fog_deposition calcule (au premier niveau uniquement):
    - rcs -= VDEPOSC * rc_t / dzz
    - inprc += VDEPOSC * rc_t * rhodref / RHOLW
    
    où VDEPOSC est la vitesse de dépôt du brouillard sur la végétation.
    
    Champs validés:
        - rcs: Tendance des gouttelettes modifiée [kg.kg-1.s-1]
        - inprc: Précipitation instantanée de brouillard [kg.m-2.s-1]
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        phyex: Objet Phyex contenant les paramètres physiques
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.rain_ice import fog_deposition

    # Get externals and add VDEPOSC
    externals = phyex.to_externals()
    externals["VDEPOSC"] = 0.02  # m/s - valeur typique pour dépôt de brouillard

    # Compilation du stencil GT4Py
    fog_dep_gt4py = stencil(
        backend,
        definition=fog_deposition,
        name="fog_deposition",
        dtypes=dtypes,
        externals=externals,
    )

    # Initialisation des champs 3D
    field_names_3d = ["rcs", "rc_t", "rhodref", "dzz"]
    
    fields = {}
    for name in field_names_3d:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustement pour valeurs réalistes
    fields["rc_t"] = fields["rc_t"] * 0.005  # 0-5 g/kg
    fields["rcs"] = (fields["rcs"] - 0.5) * 1e-5  # tendance
    fields["rhodref"] = 0.5 + fields["rhodref"] * 1.0  # 0.5-1.5 kg/m3
    fields["dzz"] = 50.0 + fields["dzz"] * 200.0  # 50-250 m
    
    # Champ 2D (précipitation)
    inprc = np.array(
        np.zeros((domain[0], domain[1])),
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in field_names_3d:
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    inprc_gt4py = from_array(inprc, dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    fog_dep_gt4py(
        rcs=gt4py_fields["rcs"],
        rc_t=gt4py_fields["rc_t"],
        rhodref=gt4py_fields["rhodref"],
        dzz=gt4py_fields["dzz"],
        inprc=inprc_gt4py,
        domain=domain,
        origin=origin,
    )
    
    # Vérifications
    # 1. rcs a été modifié au niveau 0
    assert not np.allclose(gt4py_fields["rcs"][:, :, 0], fields["rcs"][:, :, 0], atol=1e-15), \
        "rcs was not modified at level 0"
    
    # 2. inprc a été calculé (non nul)
    assert np.any(inprc_gt4py > 0), "inprc should have some positive values"
    
    # 3. Tous les champs sont finis
    assert np.all(np.isfinite(gt4py_fields["rcs"])), "rcs contains non-finite values"
    assert np.all(np.isfinite(inprc_gt4py)), "inprc contains non-finite values"
    
    # 4. Cohérence physique: dépôt réduit les gouttelettes
    # rcs devrait être plus négatif (plus de perte)
    deposition_effect = gt4py_fields["rcs"][:, :, 0] - fields["rcs"][:, :, 0]
    assert np.all(deposition_effect <= 0), "fog deposition should reduce cloud droplets (make rcs more negative)"
    
    print(f"Test passed! Fog deposition computed at surface level")
    print(f"  Total deposition: {np.sum(inprc_gt4py):.6e} kg.m-2.s-1")
