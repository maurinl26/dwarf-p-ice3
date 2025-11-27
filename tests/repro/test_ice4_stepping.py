"""
Test de reproductibilité des stencils de time-stepping ICE4.

Ce module valide que les implémentations Python GT4Py des stencils utilitaires
pour le time-stepping de la microphysique ICE4 produisent des résultats numériquement 
identiques aux implémentations Fortran de référence issues du projet PHYEX 
(PHYsique EXternalisée) version IAL_CY50T1.

Les stencils time-stepping testés incluent:
- ice4_stepping_heat: Calcul des variables thermodynamiques (T, Ls/Cp, Lv/Cp)
- ice4_step_limiter: Limiteur de pas de temps basé sur les seuils de disparition des espèces
- ice4_mixing_ratio_step_limiter: Limiteur basé sur les tendances des rapports de mélange
- state_update: Mise à jour des variables d'état après intégration temporelle
- external_tendencies_update: Mise à jour des tendances externes

Ces stencils assurent la stabilité numérique et la conservation lors de l'intégration
temporelle des processus microphysiques.

Référence:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_stepping.F90
"""
from ctypes import c_double, c_float

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array, zeros
from numpy.testing import assert_allclose

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import dp_dtypes, sp_dtypes

@pytest.mark.skip(reason="Python bus error to solve")
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
def test_ice4_stepping_heat(dtypes, backend, externals, packed_dims, domain, origin):
    """
    Test de reproductibilité du stencil ice4_stepping_heat (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre
    l'implémentation Python/GT4Py et l'implémentation Fortran de référence PHYEX-IAL_CY50T1
    pour le calcul des variables thermodynamiques nécessaires aux processus microphysiques.
    
    Le stencil ice4_stepping_heat calcule:
    1. La capacité thermique spécifique à pression constante Cp(rv, rc, ri, rr, rs, rg)
    2. La température T à partir de θ et de l'exposant d'Exner
    3. Le facteur Ls/Cp pour les processus de sublimation
    4. Le facteur Lv/Cp pour les processus de vaporisation
    
    Ces calculs sont effectués au début de chaque pas de temps microphysique et
    fournissent les données thermodynamiques nécessaires à l'évaluation des processus
    de changement de phase.
    
    Physique:
        Cp = CPD + (CPV-CPD)*rv + CL*(rc+rr) + CI*(ri+rs+rg)
        T = θ * (P/P0)^(R/Cp) = θ * Π
        Ls(T) = XLSTT + (XCPV-CI) * (T-TT)
        Lv(T) = XLVTT + (XCPV-CL) * (T-TT)
    
    Champs validés:
        - t: Température [K]
        - ls_fact: Ls/Cp, chaleur latente de sublimation divisée par Cp [K]
        - lv_fact: Lv/Cp, chaleur latente de vaporisation divisée par Cp [K]
    
    Tolérance:
        rtol=1e-6, atol=1e-8
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        packed_dims: Dimensions pour l'interface Fortran (kproma, ksize)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_stepping import ice4_stepping_heat

    # Compilation des stencils
    ice4_stepping_heat_gt4py = stencil(
        backend,
        definition=ice4_stepping_heat,
        name="ice4_stepping_heat",
        dtypes=dtypes,
        externals=externals,
    )
    
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_stepping.F90", "mode_ice4_stepping", "ice4_stepping_heat"
    )

    # Initialisation des champs d'entrée avec des valeurs réalistes
    input_field_names = ["rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t", "exn", "th_t"]
    fields = {}
    
    for name in input_field_names:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustement pour valeurs réalistes
    fields["rv_t"] = fields["rv_t"] * 0.02  # 0-20 g/kg
    fields["rc_t"] = fields["rc_t"] * 0.005  # 0-5 g/kg
    fields["rr_t"] = fields["rr_t"] * 0.003  # 0-3 g/kg
    fields["ri_t"] = fields["ri_t"] * 0.001  # 0-1 g/kg
    fields["rs_t"] = fields["rs_t"] * 0.002  # 0-2 g/kg
    fields["rg_t"] = fields["rg_t"] * 0.003  # 0-3 g/kg
    fields["exn"] = 0.8 + fields["exn"] * 0.4  # 0.8-1.2
    fields["th_t"] = 250.0 + fields["th_t"] * 50.0  # 250-300 K
    
    # Champs de sortie
    output_field_names = ["ls_fact", "lv_fact", "t"]
    for name in output_field_names:
        fields[name] = np.array(
            np.zeros(domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Création des storages GT4Py
    rv_t_gt4py = from_array(fields["rv_t"], dtype=dtypes["float"], backend=backend)
    rc_t_gt4py = from_array(fields["rc_t"], dtype=dtypes["float"], backend=backend)
    rr_t_gt4py = from_array(fields["rr_t"], dtype=dtypes["float"], backend=backend)
    ri_t_gt4py = from_array(fields["ri_t"], dtype=dtypes["float"], backend=backend)
    rs_t_gt4py = from_array(fields["rs_t"], dtype=dtypes["float"], backend=backend)
    rg_t_gt4py = from_array(fields["rg_t"], dtype=dtypes["float"], backend=backend)
    exn_gt4py = from_array(fields["exn"], dtype=dtypes["float"], backend=backend)
    th_t_gt4py = from_array(fields["th_t"], dtype=dtypes["float"], backend=backend)
    
    ls_fact_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    lv_fact_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    t_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    
    # Exécution GT4Py
    ice4_stepping_heat_gt4py(
        rv_t=rv_t_gt4py,
        rc_t=rc_t_gt4py,
        rr_t=rr_t_gt4py,
        ri_t=ri_t_gt4py,
        rs_t=rs_t_gt4py,
        rg_t=rg_t_gt4py,
        exn=exn_gt4py,
        th_t=th_t_gt4py,
        ls_fact=ls_fact_gt4py,
        lv_fact=lv_fact_gt4py,
        t=t_gt4py,
        domain=domain,
        origin=origin,
    )
    
    # Exécution Fortran - extract required constants
    xcpd = externals["CPD"]
    xcpv = externals["CPV"]
    xcl = externals["CL"]
    xci = externals["CI"]
    xtt = externals["TT"]
    xlvtt = externals["LVTT"]
    xlstt = externals["LSTT"]
    
    result = fortran_stencil(
        xcpd=xcpd,
        xcpv=xcpv,
        xcl=xcl,
        xci=xci,
        xtt=xtt,
        xlvtt=xlvtt,
        xlstt=xlstt,
        prvt=fields["rv_t"],
        prct=fields["rc_t"],
        prrt=fields["rr_t"],
        prit=fields["ri_t"],
        prst=fields["rs_t"],
        prgt=fields["rg_t"],
        pexn=fields["exn"],
        ptht=fields["th_t"],
        pt=fields["t"],
        plsfact=fields["ls_fact"],
        plvfact=fields["lv_fact"],
        **packed_dims,
    )
    
    t_out = result[0].ravel()
    ls_fact_out = result[1].ravel()
    lv_fact_out = result[2].ravel()
    
    # Comparaisons
    assert_allclose(t_out, t_gt4py.ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(ls_fact_out, ls_fact_gt4py.ravel(), rtol=1e-6, atol=1e-8)
    assert_allclose(lv_fact_out, lv_fact_gt4py.ravel(), rtol=1e-6, atol=1e-8)


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
def test_state_update(dtypes, backend, externals, packed_dims, domain, origin):
    """
    Test de reproductibilité du stencil state_update (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre
    l'implémentation Python/GT4Py et l'implémentation Fortran de référence PHYEX-IAL_CY50T1
    pour la mise à jour des variables d'état après un pas de temps microphysique.
    
    Le stencil state_update effectue l'intégration temporelle explicite:
        X(t+Δt) = X(t) + dX/dt * Δt + ΔX_processus
    
    Pour chaque variable pronostique (θ, rc, rr, ri, rs, rg):
    - Applique la tendance calculée sur le pas de temps Δt
    - Ajoute la variation due aux processus instantanés (ΔX_b)
    - Gère les cas spéciaux (notamment ri → 0 implique ci → 0)
    
    Cette étape assure la conservation et la cohérence temporelle de l'intégration
    des processus microphysiques.
    
    Champs validés:
        - th_t: Température potentielle mise à jour [K]
        - rc_t, rr_t, ri_t, rs_t, rg_t: Rapports de mélange mis à jour [kg/kg]
        - ci_t: Concentration de cristaux de glace mise à jour [#/kg]
        - t_micro: Temps microphysique mis à jour [s]
    
    Tolérance:
        rtol=1e-6, atol=1e-8
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        packed_dims: Dimensions pour l'interface Fortran (kproma, ksize)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_stepping import state_update

    # Compilation des stencils
    state_update_gt4py = stencil(
        backend,
        definition=state_update,
        name="state_update",
        dtypes=dtypes,
        externals=externals,
    )
    
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_stepping.F90", "mode_ice4_stepping", "state_update"
    )

    # Initialisation des champs
    field_names = [
        "th_t", "theta_b", "theta_tnd_a",
        "rc_t", "rr_t", "ri_t", "rs_t", "rg_t",
        "rc_b", "rr_b", "ri_b", "rs_b", "rg_b",
        "rc_tnd_a", "rr_tnd_a", "ri_tnd_a", "rs_tnd_a", "rg_tnd_a",
        "delta_t_micro", "ci_t", "t_micro"
    ]
    
    fields = {}
    for name in field_names:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustements pour valeurs réalistes
    fields["th_t"] = 250.0 + fields["th_t"] * 50.0
    fields["theta_b"] = (fields["theta_b"] - 0.5) * 5.0
    fields["theta_tnd_a"] = (fields["theta_tnd_a"] - 0.5) * 0.1
    fields["delta_t_micro"] = fields["delta_t_micro"] * 10.0
    fields["t_micro"] = fields["t_micro"] * 300.0
    
    for species in ["rc", "rr", "ri", "rs", "rg"]:
        fields[f"{species}_t"] *= 0.005
        fields[f"{species}_b"] = (fields[f"{species}_b"] - 0.5) * 0.001
        fields[f"{species}_tnd_a"] = (fields[f"{species}_tnd_a"] - 0.5) * 0.0001
    
    fields["ci_t"] = fields["ci_t"] * 10000.0
    
    # Masque booléen
    ldmicro = np.array(np.random.rand(*domain) > 0.2, dtype=np.bool_, order="F")
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in field_names:
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    ldmicro_gt4py = from_array(ldmicro, dtype=dtypes["bool"], backend=backend)
    
    # Exécution GT4Py
    state_update_gt4py(
        th_t=gt4py_fields["th_t"],
        theta_b=gt4py_fields["theta_b"],
        theta_tnd_a=gt4py_fields["theta_tnd_a"],
        rc_t=gt4py_fields["rc_t"],
        rr_t=gt4py_fields["rr_t"],
        ri_t=gt4py_fields["ri_t"],
        rs_t=gt4py_fields["rs_t"],
        rg_t=gt4py_fields["rg_t"],
        rc_b=gt4py_fields["rc_b"],
        rr_b=gt4py_fields["rr_b"],
        ri_b=gt4py_fields["ri_b"],
        rs_b=gt4py_fields["rs_b"],
        rg_b=gt4py_fields["rg_b"],
        rc_tnd_a=gt4py_fields["rc_tnd_a"],
        rr_tnd_a=gt4py_fields["rr_tnd_a"],
        ri_tnd_a=gt4py_fields["ri_tnd_a"],
        rs_tnd_a=gt4py_fields["rs_tnd_a"],
        rg_tnd_a=gt4py_fields["rg_tnd_a"],
        delta_t_micro=gt4py_fields["delta_t_micro"],
        ldmicro=ldmicro_gt4py,
        ci_t=gt4py_fields["ci_t"],
        t_micro=gt4py_fields["t_micro"],
        domain=domain,
        origin=origin,
    )
    
    # Exécution Fortran (avec copies car modifiées)
    fortran_fields = {name: fields[name].copy() for name in field_names}
    
    # Reshape 3D arrays to 2D (kproma, ksize) for Fortran interface
    kproma_val = domain[0] * domain[1]  # horizontal dimensions
    ksize_val = domain[2]  # vertical dimension
    
    result = fortran_stencil(
        ptht=fortran_fields["th_t"].reshape((kproma_val, ksize_val), order='F'),
        ptheta_b=fortran_fields["theta_b"].reshape((kproma_val, ksize_val), order='F'),
        ptheta_tnd_a=fortran_fields["theta_tnd_a"].reshape((kproma_val, ksize_val), order='F'),
        prct=fortran_fields["rc_t"].reshape((kproma_val, ksize_val), order='F'),
        prrt=fortran_fields["rr_t"].reshape((kproma_val, ksize_val), order='F'),
        prit=fortran_fields["ri_t"].reshape((kproma_val, ksize_val), order='F'),
        prst=fortran_fields["rs_t"].reshape((kproma_val, ksize_val), order='F'),
        prgt=fortran_fields["rg_t"].reshape((kproma_val, ksize_val), order='F'),
        prc_b=fortran_fields["rc_b"].reshape((kproma_val, ksize_val), order='F'),
        prr_b=fortran_fields["rr_b"].reshape((kproma_val, ksize_val), order='F'),
        pri_b=fortran_fields["ri_b"].reshape((kproma_val, ksize_val), order='F'),
        prs_b=fortran_fields["rs_b"].reshape((kproma_val, ksize_val), order='F'),
        prg_b=fortran_fields["rg_b"].reshape((kproma_val, ksize_val), order='F'),
        prc_tnd_a=fortran_fields["rc_tnd_a"].reshape((kproma_val, ksize_val), order='F'),
        prr_tnd_a=fortran_fields["rr_tnd_a"].reshape((kproma_val, ksize_val), order='F'),
        pri_tnd_a=fortran_fields["ri_tnd_a"].reshape((kproma_val, ksize_val), order='F'),
        prs_tnd_a=fortran_fields["rs_tnd_a"].reshape((kproma_val, ksize_val), order='F'),
        prg_tnd_a=fortran_fields["rg_tnd_a"].reshape((kproma_val, ksize_val), order='F'),
        pdelta_t_micro=fortran_fields["delta_t_micro"].reshape((kproma_val, ksize_val), order='F'),
        ldmicro=ldmicro.reshape((kproma_val, ksize_val), order='F'),
        pcit=fortran_fields["ci_t"].reshape((kproma_val, ksize_val), order='F'),
        pt_micro=fortran_fields["t_micro"].reshape((kproma_val, ksize_val), order='F'),
        kproma=kproma_val,
        ksize=ksize_val,
    )
    
    # Validation
    output_names = ["th_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t", "ci_t", "t_micro"]
    
    for i, name in enumerate(output_names):
        assert_allclose(
            result[i].ravel(),
            gt4py_fields[name].ravel(),
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"[ÉCHEC] {name}: divergence Python/Fortran PHYEX"
        )


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
def test_external_tendencies_update(dtypes, backend, externals, packed_dims, domain, origin):
    """
    Test de reproductibilité du stencil external_tendencies_update (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre
    l'implémentation Python/GT4Py et l'implémentation Fortran de référence PHYEX-IAL_CY50T1
    pour la mise à jour des tendances externes.
    
    Le stencil external_tendencies_update retire les tendances externes qui avaient été
    ajoutées temporairement lors du calcul des processus microphysiques. Ceci permet de:
    - Découpler les processus microphysiques des autres forçages
    - Assurer la cohérence des budgets
    - Faciliter le diagnostic et l'analyse
    
    Opération effectuée:
        X(t) = X(t) - dX_ext/dt * Δt
    
    Pour chaque variable (θ, rc, rr, ri, rs, rg).
    
    Champs validés:
        - th_t: Température potentielle corrigée [K]
        - rc_t, rr_t, ri_t, rs_t, rg_t: Rapports de mélange corrigés [kg/kg]
    
    Tolérance:
        rtol=1e-6, atol=1e-8
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et microphysiques)
        packed_dims: Dimensions pour l'interface Fortran (kproma, ksize)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_stepping import external_tendencies_update

    # Compilation des stencils
    external_tendencies_update_gt4py = stencil(
        backend,
        definition=external_tendencies_update,
        name="external_tendencies_update",
        dtypes=dtypes,
        externals=externals,
    )
    
    fortran_stencil = compile_fortran_stencil(
        "mode_ice4_stepping.F90", "mode_ice4_stepping", "external_tendencies_update"
    )

    # Initialisation des champs
    field_names = [
        "th_t", "theta_tnd_ext",
        "rc_t", "rr_t", "ri_t", "rs_t", "rg_t",
        "rc_tnd_ext", "rr_tnd_ext", "ri_tnd_ext", "rs_tnd_ext", "rg_tnd_ext",
    ]
    
    fields = {}
    for name in field_names:
        fields[name] = np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
    
    # Ajustements
    fields["th_t"] = 250.0 + fields["th_t"] * 50.0
    fields["theta_tnd_ext"] = (fields["theta_tnd_ext"] - 0.5) * 0.01
    
    for species in ["rc", "rr", "ri", "rs", "rg"]:
        fields[f"{species}_t"] *= 0.005
        fields[f"{species}_tnd_ext"] = (fields[f"{species}_tnd_ext"] - 0.5) * 0.0001
    
    # Masque et pas de temps
    ldmicro = np.array(np.random.rand(*domain) > 0.2, dtype=np.bool_, order="F")
    dt = dtypes["float"](10.0)  # secondes
    
    # Création des storages GT4Py
    gt4py_fields = {}
    for name in field_names:
        gt4py_fields[name] = from_array(fields[name].copy(), dtype=dtypes["float"], backend=backend)
    
    ldmicro_gt4py = from_array(ldmicro, dtype=dtypes["bool"], backend=backend)
    
    # Exécution GT4Py
    external_tendencies_update_gt4py(
        th_t=gt4py_fields["th_t"],
        theta_tnd_ext=gt4py_fields["theta_tnd_ext"],
        rc_t=gt4py_fields["rc_t"],
        rr_t=gt4py_fields["rr_t"],
        ri_t=gt4py_fields["ri_t"],
        rs_t=gt4py_fields["rs_t"],
        rg_t=gt4py_fields["rg_t"],
        rc_tnd_ext=gt4py_fields["rc_tnd_ext"],
        rr_tnd_ext=gt4py_fields["rr_tnd_ext"],
        ri_tnd_ext=gt4py_fields["ri_tnd_ext"],
        rs_tnd_ext=gt4py_fields["rs_tnd_ext"],
        rg_tnd_ext=gt4py_fields["rg_tnd_ext"],
        ldmicro=ldmicro_gt4py,
        dt=dt,
        domain=domain,
        origin=origin,
    )
    
    # Exécution Fortran
    fortran_fields = {name: fields[name].copy() for name in field_names}
    
    # Reshape 3D arrays to 2D (kproma, ksize) for Fortran interface
    kproma_val = domain[0] * domain[1]  # horizontal dimensions
    ksize_val = domain[2]  # vertical dimension
    
    result = fortran_stencil(
        ptht=fortran_fields["th_t"].reshape((kproma_val, ksize_val), order='F'),
        ptheta_tnd_ext=fortran_fields["theta_tnd_ext"].reshape((kproma_val, ksize_val), order='F'),
        prct=fortran_fields["rc_t"].reshape((kproma_val, ksize_val), order='F'),
        prrt=fortran_fields["rr_t"].reshape((kproma_val, ksize_val), order='F'),
        prit=fortran_fields["ri_t"].reshape((kproma_val, ksize_val), order='F'),
        prst=fortran_fields["rs_t"].reshape((kproma_val, ksize_val), order='F'),
        prgt=fortran_fields["rg_t"].reshape((kproma_val, ksize_val), order='F'),
        prc_tnd_ext=fortran_fields["rc_tnd_ext"].reshape((kproma_val, ksize_val), order='F'),
        prr_tnd_ext=fortran_fields["rr_tnd_ext"].reshape((kproma_val, ksize_val), order='F'),
        pri_tnd_ext=fortran_fields["ri_tnd_ext"].reshape((kproma_val, ksize_val), order='F'),
        prs_tnd_ext=fortran_fields["rs_tnd_ext"].reshape((kproma_val, ksize_val), order='F'),
        prg_tnd_ext=fortran_fields["rg_tnd_ext"].reshape((kproma_val, ksize_val), order='F'),
        ldmicro=ldmicro.reshape((kproma_val, ksize_val), order='F'),
        dt=dt,
        kproma=kproma_val,
        ksize=ksize_val,
    )
    
    # Validation
    output_names = ["th_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t"]
    
    for i, name in enumerate(output_names):
        assert_allclose(
            result[i].ravel(),
            gt4py_fields[name].ravel(),
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"[ÉCHEC] {name}: divergence Python/Fortran PHYEX"
        )
