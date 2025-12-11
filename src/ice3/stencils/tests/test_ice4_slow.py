# -*- coding: utf-8 -*-
"""
Test de reproductibilité du stencil ice4_slow par rapport à PHYEX-IAL_CY50T1.

Ce module valide que l'implémentation Python GT4Py des processus lents de la 
microphysique ICE4 produit des résultats numériquement identiques à l'implémentation 
Fortran de référence issue du projet PHYEX (PHYsique EXternalisée) version IAL_CY50T1.

Les processus lents incluent:
1. RCHONI - Nucléation homogène (formation de glace à partir d'eau surfondue)
2. RVDEPS - Déposition de vapeur sur la neige
3. RIAGGS - Agrégation de glace sur la neige
4. RIAUTS - Autoconversion de glace en neige
5. RVDEPG - Déposition de vapeur sur le graupel

Ces processus sont appelés "lents" car leur échelle de temps caractéristique est 
plus longue que celle des processus "rapides" (rimming, shedding, etc.).

Référence:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_slow.F90
"""
from ctypes import c_double, c_float

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array, zeros
from numpy.testing import assert_allclose

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import dp_dtypes, sp_dtypes


@pytest.mark.parametrize("ldsoft", [False, True])
@pytest.mark.parametrize("dtypes", [sp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("debug", marks=pytest.mark.debug),
        pytest.param("numpy", marks=pytest.mark.numpy),
        pytest.param("gt:cpu_ifirst", marks=pytest.mark.cpu),
        pytest.param("gt:gpu", marks=pytest.mark.gpu),
    ],
)
def test_ice4_slow(dtypes, backend, externals, packed_dims, domain, origin, ldsoft):
    """
    Test de reproductibilité du stencil ice4_slow (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre:
    - L'implémentation Python/GT4Py des processus microphysiques lents
    - L'implémentation Fortran de référence PHYEX-IAL_CY50T1
    
    Les processus microphysiques lents calculent:
    
    1. RCHONI - Nucléation homogène:
       - Active pour T < -35°C et rct > C_RTMIN
       - Congélation quasi-instantanée des gouttelettes surfondues
       - Formule: min(1000, HON * rhodref * rct * exp(ALPHA3 * (T-TT) - BETA3))
    
    2. RVDEPS - Déposition sur la neige:
       - Active pour rvt > V_RTMIN et rst > S_RTMIN
       - Croissance des flocons par déposition de vapeur
       - Dépend de SSI (sursaturation), lbdas (paramètre de pente), ventilation
    
    3. RIAGGS - Agrégation sur la neige:
       - Active pour rit > I_RTMIN et rst > S_RTMIN
       - Collision et collage des cristaux de glace sur les flocons
       - Processus thermodépendant (exp(COLEXIS * (T-TT)))
    
    4. RIAUTS - Autoconversion glace → neige:
       - Active pour hli_hri > I_RTMIN (glace dans les bas nuages)
       - Conversion automatique au-delà d'un seuil critique
       - Seuil dépendant de la température: CRIAUTI(T)
    
    5. RVDEPG - Déposition sur le graupel:
       - Active pour rvt > V_RTMIN et rgt > G_RTMIN
       - Similaire à RVDEPS mais pour le graupel
       - Croissance par déposition de vapeur
    
    Champs vérifiés:
        - rc_honi_tnd: Taux de nucléation homogène [kg/kg/s]
        - rv_deps_tnd: Taux de déposition sur neige [kg/kg/s]
        - ri_aggs_tnd: Taux d'agrégation sur neige [kg/kg/s]
        - ri_auts_tnd: Taux d'autoconversion glace→neige [kg/kg/s]
        - rv_depg_tnd: Taux de déposition sur graupel [kg/kg/s]
    
    Tolérance:
        rtol=1e-6, atol=1e-8
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et paramètres microphysiques)
        packed_dims: Dimensions pour l'interface Fortran (kproma, ksize)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_slow import ice4_slow

    # Compilation du stencil GT4Py
    ice4_slow_gt4py = stencil(
        backend,
        definition=ice4_slow,
        name="ice4_slow",
        dtypes=dtypes,
        externals=externals,
    )

    # Compilation du stencil Fortran de référence
    ice4_slow_fortran = compile_fortran_stencil(
        "mode_ice4_slow.F90", "mode_ice4_slow", "ice4_slow"
    )

    # =========================================================================
    # Initialisation des champs d'entrée
    # =========================================================================
    
    # Champs scalaires 3D d'entrée
    FloatFieldsIJK_Input_Names = [
        "rhodref",    # Densité de référence [kg/m³]
        "t",          # Température [K]
        "ssi",        # Sursaturation par rapport à la glace [-]
        "rvt",        # Rapport de mélange de vapeur d'eau [kg/kg]
        "rct",        # Rapport de mélange d'eau nuageuse [kg/kg]
        "rit",        # Rapport de mélange de glace [kg/kg]
        "rst",        # Rapport de mélange de neige [kg/kg]
        "rgt",        # Rapport de mélange de graupel [kg/kg]
        "lbdas",      # Paramètre de pente de la distribution de neige [m^-1]
        "lbdag",      # Paramètre de pente de la distribution de graupel [m^-1]
        "ai",         # Fonction thermodynamique pour déposition [-]
        "cj",         # Coefficient de ventilation [-]
        "hli_hcf",    # Fraction nuageuse des bas nuages [-]
        "hli_hri",    # Rapport de mélange de glace dans bas nuages [kg/kg]
    ]

    FloatFieldsIJK_Input = {
        name: np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Input_Names
    }

    # Ajustement des valeurs pour avoir des données réalistes
    # Température entre 200K et 280K (très froid à tempéré)
    FloatFieldsIJK_Input["t"] = FloatFieldsIJK_Input["t"] * 80.0 + 200.0
    
    # Densité de référence entre 0.3 et 1.5 kg/m³
    FloatFieldsIJK_Input["rhodref"] = FloatFieldsIJK_Input["rhodref"] * 1.2 + 0.3
    
    # Sursaturation par rapport à la glace entre -0.5 et 0.5
    FloatFieldsIJK_Input["ssi"] = (FloatFieldsIJK_Input["ssi"] - 0.5) * 1.0
    
    # Rapports de mélange (valeurs petites, typiquement < 0.01)
    FloatFieldsIJK_Input["rvt"] = FloatFieldsIJK_Input["rvt"] * 0.015
    FloatFieldsIJK_Input["rct"] = FloatFieldsIJK_Input["rct"] * 0.005
    FloatFieldsIJK_Input["rit"] = FloatFieldsIJK_Input["rit"] * 0.001
    FloatFieldsIJK_Input["rst"] = FloatFieldsIJK_Input["rst"] * 0.002
    FloatFieldsIJK_Input["rgt"] = FloatFieldsIJK_Input["rgt"] * 0.003
    
    # Paramètres de pente (valeurs typiques: 10^3 à 10^6 m^-1)
    FloatFieldsIJK_Input["lbdas"] = FloatFieldsIJK_Input["lbdas"] * 9e5 + 1e5
    FloatFieldsIJK_Input["lbdag"] = FloatFieldsIJK_Input["lbdag"] * 9e5 + 1e5
    
    # Fonction thermodynamique ai (valeurs typiques: 1e9 à 1e11)
    FloatFieldsIJK_Input["ai"] = FloatFieldsIJK_Input["ai"] * 9e10 + 1e10
    
    # Coefficient de ventilation cj (valeurs typiques: 0 à 10)
    FloatFieldsIJK_Input["cj"] = FloatFieldsIJK_Input["cj"] * 10.0
    
    # Fraction nuageuse (0 à 1)
    FloatFieldsIJK_Input["hli_hcf"] = FloatFieldsIJK_Input["hli_hcf"]
    
    # Rapport de mélange de glace dans bas nuages (petites valeurs)
    FloatFieldsIJK_Input["hli_hri"] = FloatFieldsIJK_Input["hli_hri"] * 0.001

    # Champs de sortie
    FloatFieldsIJK_Output_Names = [
        "rc_honi_tnd",  # Nucléation homogène
        "rv_deps_tnd",  # Déposition sur neige
        "ri_aggs_tnd",  # Agrégation sur neige
        "ri_auts_tnd",  # Autoconversion glace→neige
        "rv_depg_tnd",  # Déposition sur graupel
    ]

    FloatFieldsIJK_Output = {
        name: np.array(
            np.zeros(domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Output_Names
    }

    # Champ booléen pour activer/désactiver le calcul par colonne
    BoolFieldsIJK = {
        "ldcompute": np.array(
            np.random.rand(*domain) > 0.2,  # ~80% des points activés
            dtype=np.bool_,
            order="F",
        )
    }

    # =========================================================================
    # Conversion en storages GT4Py
    # =========================================================================
    
    # Champs d'entrée
    ldcompute_gt4py = from_array(
        BoolFieldsIJK["ldcompute"], dtype=dtypes["bool"], backend=backend
    )
    rhodref_gt4py = from_array(
        FloatFieldsIJK_Input["rhodref"], dtype=dtypes["float"], backend=backend
    )
    t_gt4py = from_array(
        FloatFieldsIJK_Input["t"], dtype=dtypes["float"], backend=backend
    )
    ssi_gt4py = from_array(
        FloatFieldsIJK_Input["ssi"], dtype=dtypes["float"], backend=backend
    )
    rvt_gt4py = from_array(
        FloatFieldsIJK_Input["rvt"], dtype=dtypes["float"], backend=backend
    )
    rct_gt4py = from_array(
        FloatFieldsIJK_Input["rct"], dtype=dtypes["float"], backend=backend
    )
    rit_gt4py = from_array(
        FloatFieldsIJK_Input["rit"], dtype=dtypes["float"], backend=backend
    )
    rst_gt4py = from_array(
        FloatFieldsIJK_Input["rst"], dtype=dtypes["float"], backend=backend
    )
    rgt_gt4py = from_array(
        FloatFieldsIJK_Input["rgt"], dtype=dtypes["float"], backend=backend
    )
    lbdas_gt4py = from_array(
        FloatFieldsIJK_Input["lbdas"], dtype=dtypes["float"], backend=backend
    )
    lbdag_gt4py = from_array(
        FloatFieldsIJK_Input["lbdag"], dtype=dtypes["float"], backend=backend
    )
    ai_gt4py = from_array(
        FloatFieldsIJK_Input["ai"], dtype=dtypes["float"], backend=backend
    )
    cj_gt4py = from_array(
        FloatFieldsIJK_Input["cj"], dtype=dtypes["float"], backend=backend
    )
    hli_hcf_gt4py = from_array(
        FloatFieldsIJK_Input["hli_hcf"], dtype=dtypes["float"], backend=backend
    )
    hli_hri_gt4py = from_array(
        FloatFieldsIJK_Input["hli_hri"], dtype=dtypes["float"], backend=backend
    )
    
    # Champs de sortie (initialisés à zéro)
    rc_honi_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rv_deps_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    ri_aggs_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    ri_auts_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rv_depg_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)

    # =========================================================================
    # Exécution du stencil Python GT4Py
    # =========================================================================
    
    ice4_slow_gt4py(
        ldsoft=ldsoft,
        ldcompute=ldcompute_gt4py,
        rhodref=rhodref_gt4py,
        t=t_gt4py,
        ssi=ssi_gt4py,
        rvt=rvt_gt4py,
        rct=rct_gt4py,
        rit=rit_gt4py,
        rst=rst_gt4py,
        rgt=rgt_gt4py,
        lbdas=lbdas_gt4py,
        lbdag=lbdag_gt4py,
        ai=ai_gt4py,
        cj=cj_gt4py,
        hli_hcf=hli_hcf_gt4py,
        hli_hri=hli_hri_gt4py,
        rc_honi_tnd=rc_honi_tnd_gt4py,
        rv_deps_tnd=rv_deps_tnd_gt4py,
        ri_aggs_tnd=ri_aggs_tnd_gt4py,
        ri_auts_tnd=ri_auts_tnd_gt4py,
        rv_depg_tnd=rv_depg_tnd_gt4py,
        domain=domain,
        origin=origin,
    )

    # =========================================================================
    # Exécution de la référence Fortran PHYEX
    # =========================================================================
    
    # Extraction des paramètres externes pour Fortran
    xtt = externals["TT"]                  # Température de référence (273.15 K)
    v_rtmin = externals["V_RTMIN"]         # Seuil minimal de vapeur d'eau
    c_rtmin = externals["C_RTMIN"]         # Seuil minimal d'eau nuageuse
    i_rtmin = externals["I_RTMIN"]         # Seuil minimal de glace
    s_rtmin = externals["S_RTMIN"]         # Seuil minimal de neige
    g_rtmin = externals["G_RTMIN"]         # Seuil minimal de graupel
    xexiaggs = externals["EXIAGGS"]        # Exposant pour agrégation
    xfiaggs = externals["FIAGGS"]          # Facteur pour agrégation
    xcolexis = externals["COLEXIS"]        # Exposant de collision
    xtimauti = externals["TIMAUTI"]        # Facteur de temps pour autoconversion
    xcriauti = externals["CRIAUTI"]        # Seuil critique pour autoconversion
    xacriauti = externals["ACRIAUTI"]      # Paramètre A pour CRIAUTI(T)
    xbcriauti = externals["BCRIAUTI"]      # Paramètre B pour CRIAUTI(T)
    xcexvt = externals["CEXVT"]            # Exposant de vitesse terminale
    xtexauti = externals["TEXAUTI"]        # Exposant de température pour autoconversion
    x0depg = externals["O0DEPG"]           # Paramètre 0 pour déposition sur graupel
    x1depg = externals["O1DEPG"]           # Paramètre 1 pour déposition sur graupel
    xex0depg = externals["EX0DEPG"]        # Exposant 0 pour déposition sur graupel
    xex1depg = externals["EX1DEPG"]        # Exposant 1 pour déposition sur graupel
    xhon = externals["HON"]                # Paramètre pour nucléation homogène
    xalpha3 = externals["ALPHA3"]          # Paramètre alpha3 pour nucléation
    xbeta3 = externals["BETA3"]            # Paramètre beta3 pour nucléation
    x0deps = externals["O0DEPS"]           # Paramètre 0 pour déposition sur neige
    x1deps = externals["O1DEPS"]           # Paramètre 1 pour déposition sur neige
    xex1deps = externals["EX1DEPS"]        # Exposant 1 pour déposition sur neige
    xex0deps = externals["EX0DEPS"]        # Exposant 0 pour déposition sur neige
    
    # Aplatissement des champs 3D en 1D pour Fortran (ordre Fortran)
    ldcompute_flat = BoolFieldsIJK["ldcompute"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rhodref_flat = FloatFieldsIJK_Input["rhodref"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    t_flat = FloatFieldsIJK_Input["t"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    ssi_flat = FloatFieldsIJK_Input["ssi"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rvt_flat = FloatFieldsIJK_Input["rvt"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rct_flat = FloatFieldsIJK_Input["rct"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rit_flat = FloatFieldsIJK_Input["rit"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rst_flat = FloatFieldsIJK_Input["rst"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rgt_flat = FloatFieldsIJK_Input["rgt"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    lbdas_flat = FloatFieldsIJK_Input["lbdas"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    lbdag_flat = FloatFieldsIJK_Input["lbdag"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    ai_flat = FloatFieldsIJK_Input["ai"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    cj_flat = FloatFieldsIJK_Input["cj"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    hli_hcf_flat = FloatFieldsIJK_Input["hli_hcf"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    hli_hri_flat = FloatFieldsIJK_Input["hli_hri"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    
    # Champs de sortie (copies car ils sont modifiés)
    rc_honi_tnd_flat = FloatFieldsIJK_Output["rc_honi_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rv_deps_tnd_flat = FloatFieldsIJK_Output["rv_deps_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    ri_aggs_tnd_flat = FloatFieldsIJK_Output["ri_aggs_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    ri_auts_tnd_flat = FloatFieldsIJK_Output["ri_auts_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rv_depg_tnd_flat = FloatFieldsIJK_Output["rv_depg_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    
    # Appel de la routine Fortran
    ice4_slow_fortran(
        xtt=xtt,
        v_rtmin=v_rtmin,
        c_rtmin=c_rtmin,
        i_rtmin=i_rtmin,
        s_rtmin=s_rtmin,
        g_rtmin=g_rtmin,
        xexiaggs=xexiaggs,
        xfiaggs=xfiaggs,
        xcolexis=xcolexis,
        xtimauti=xtimauti,
        xcriauti=xcriauti,
        xacriauti=xacriauti,
        xbcriauti=xbcriauti,
        xcexvt=xcexvt,
        xtexauti=xtexauti,
        x0depg=x0depg,
        x1depg=x1depg,
        xex0depg=xex0depg,
        xex1depg=xex1depg,
        xhon=xhon,
        xalpha3=xalpha3,
        xbeta3=xbeta3,
        x0deps=x0deps,
        x1deps=x1deps,
        xex1deps=xex1deps,
        xex0deps=xex0deps,
        ldsoft=ldsoft,
        ldcompute=ldcompute_flat,
        prhodref=rhodref_flat,
        pt=t_flat,
        pssi=ssi_flat,
        prvt=rvt_flat,
        prct=rct_flat,
        prit=rit_flat,
        prst=rst_flat,
        prgt=rgt_flat,
        plbdas=lbdas_flat,
        plbdag=lbdag_flat,
        pai=ai_flat,
        pcj=cj_flat,
        phli_hcf=hli_hcf_flat,
        phli_hri=hli_hri_flat,
        prchoni=rc_honi_tnd_flat,
        prvdeps=rv_deps_tnd_flat,
        priaggs=ri_aggs_tnd_flat,
        priauts=ri_auts_tnd_flat,
        prvdepg=rv_depg_tnd_flat,
        **packed_dims,
    )

    # =========================================================================
    # VALIDATION DE LA REPRODUCTIBILITÉ - Comparaison Python vs Fortran PHYEX
    # =========================================================================
    
    print("\n" + "="*80)
    print("TEST DE REPRODUCTIBILITÉ: ice4_slow.py vs PHYEX-IAL_CY50T1")
    print("="*80)
    print(f"Backend: {backend}")
    print(f"Précision: {'simple' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domaine: {domain[0]}x{domain[1]}x{domain[2]}")
    print("="*80)
    
    # Reshape des sorties Python pour comparaison
    rc_honi_tnd_py = rc_honi_tnd_gt4py.reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rv_deps_tnd_py = rv_deps_tnd_gt4py.reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    ri_aggs_tnd_py = ri_aggs_tnd_gt4py.reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    ri_auts_tnd_py = ri_auts_tnd_gt4py.reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rv_depg_tnd_py = rv_depg_tnd_gt4py.reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    
    # ------------------------------------------------------------------------
    # Validation de la nucléation homogène (RCHONI)
    # ------------------------------------------------------------------------
    print("\n1. RCHONI - Nucléation homogène")
    print("-" * 80)
    print(f"  Python - min: {rc_honi_tnd_py.min():.6e}, max: {rc_honi_tnd_py.max():.6e}")
    print(f"  Fortran - min: {rc_honi_tnd_flat.min():.6e}, max: {rc_honi_tnd_flat.max():.6e}")
    
    diff = np.abs(rc_honi_tnd_py - rc_honi_tnd_flat)
    rel_diff = np.where(
        np.abs(rc_honi_tnd_flat) > 1e-10,
        diff / np.abs(rc_honi_tnd_flat),
        0.0
    )
    
    print(f"  Différence absolue - max: {diff.max():.6e}, moyenne: {diff.mean():.6e}")
    print(f"  Différence relative - max: {rel_diff.max():.6e}, "
          f"moyenne: {rel_diff[rel_diff>0].mean() if np.any(rel_diff>0) else 0:.6e}")
    
    assert_allclose(
        rc_honi_tnd_flat,
        rc_honi_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] Nucléation homogène (rc_honi_tnd): divergence Python/Fortran PHYEX"
    )
    print("  ✓ RCHONI : OK")
    
    # ------------------------------------------------------------------------
    # Validation de la déposition sur neige (RVDEPS)
    # ------------------------------------------------------------------------
    print("\n2. RVDEPS - Déposition sur neige")
    print("-" * 80)
    print(f"  Python - min: {rv_deps_tnd_py.min():.6e}, max: {rv_deps_tnd_py.max():.6e}")
    print(f"  Fortran - min: {rv_deps_tnd_flat.min():.6e}, max: {rv_deps_tnd_flat.max():.6e}")
    
    diff = np.abs(rv_deps_tnd_py - rv_deps_tnd_flat)
    rel_diff = np.where(
        np.abs(rv_deps_tnd_flat) > 1e-10,
        diff / np.abs(rv_deps_tnd_flat),
        0.0
    )
    
    print(f"  Différence absolue - max: {diff.max():.6e}, moyenne: {diff.mean():.6e}")
    print(f"  Différence relative - max: {rel_diff.max():.6e}, "
          f"moyenne: {rel_diff[rel_diff>0].mean() if np.any(rel_diff>0) else 0:.6e}")
    
    assert_allclose(
        rv_deps_tnd_flat,
        rv_deps_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] Déposition sur neige (rv_deps_tnd): divergence Python/Fortran PHYEX"
    )
    print("  ✓ RVDEPS : OK")
    
    # ------------------------------------------------------------------------
    # Validation de l'agrégation sur neige (RIAGGS)
    # ------------------------------------------------------------------------
    print("\n3. RIAGGS - Agrégation sur neige")
    print("-" * 80)
    print(f"  Python - min: {ri_aggs_tnd_py.min():.6e}, max: {ri_aggs_tnd_py.max():.6e}")
    print(f"  Fortran - min: {ri_aggs_tnd_flat.min():.6e}, max: {ri_aggs_tnd_flat.max():.6e}")
    
    diff = np.abs(ri_aggs_tnd_py - ri_aggs_tnd_flat)
    rel_diff = np.where(
        np.abs(ri_aggs_tnd_flat) > 1e-10,
        diff / np.abs(ri_aggs_tnd_flat),
        0.0
    )
    
    print(f"  Différence absolue - max: {diff.max():.6e}, moyenne: {diff.mean():.6e}")
    print(f"  Différence relative - max: {rel_diff.max():.6e}, "
          f"moyenne: {rel_diff[rel_diff>0].mean() if np.any(rel_diff>0) else 0:.6e}")
    
    assert_allclose(
        ri_aggs_tnd_flat,
        ri_aggs_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] Agrégation sur neige (ri_aggs_tnd): divergence Python/Fortran PHYEX"
    )
    print("  ✓ RIAGGS : OK")
    
    # ------------------------------------------------------------------------
    # Validation de l'autoconversion glace→neige (RIAUTS)
    # ------------------------------------------------------------------------
    print("\n4. RIAUTS - Autoconversion glace→neige")
    print("-" * 80)
    print(f"  Python - min: {ri_auts_tnd_py.min():.6e}, max: {ri_auts_tnd_py.max():.6e}")
    print(f"  Fortran - min: {ri_auts_tnd_flat.min():.6e}, max: {ri_auts_tnd_flat.max():.6e}")
    
    diff = np.abs(ri_auts_tnd_py - ri_auts_tnd_flat)
    rel_diff = np.where(
        np.abs(ri_auts_tnd_flat) > 1e-10,
        diff / np.abs(ri_auts_tnd_flat),
        0.0
    )
    
    print(f"  Différence absolue - max: {diff.max():.6e}, moyenne: {diff.mean():.6e}")
    print(f"  Différence relative - max: {rel_diff.max():.6e}, "
          f"moyenne: {rel_diff[rel_diff>0].mean() if np.any(rel_diff>0) else 0:.6e}")
    
    assert_allclose(
        ri_auts_tnd_flat,
        ri_auts_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] Autoconversion glace→neige (ri_auts_tnd): divergence Python/Fortran PHYEX"
    )
    print("  ✓ RIAUTS : OK")
    
    # ------------------------------------------------------------------------
    # Validation de la déposition sur graupel (RVDEPG)
    # ------------------------------------------------------------------------
    print("\n5. RVDEPG - Déposition sur graupel")
    print("-" * 80)
    print(f"  Python - min: {rv_depg_tnd_py.min():.6e}, max: {rv_depg_tnd_py.max():.6e}")
    print(f"  Fortran - min: {rv_depg_tnd_flat.min():.6e}, max: {rv_depg_tnd_flat.max():.6e}")
    
    diff = np.abs(rv_depg_tnd_py - rv_depg_tnd_flat)
    rel_diff = np.where(
        np.abs(rv_depg_tnd_flat) > 1e-10,
        diff / np.abs(rv_depg_tnd_flat),
        0.0
    )
    
    print(f"  Différence absolue - max: {diff.max():.6e}, moyenne: {diff.mean():.6e}")
    print(f"  Différence relative - max: {rel_diff.max():.6e}, "
          f"moyenne: {rel_diff[rel_diff>0].mean() if np.any(rel_diff>0) else 0:.6e}")
    
    assert_allclose(
        rv_depg_tnd_flat,
        rv_depg_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] Déposition sur graupel (rv_depg_tnd): divergence Python/Fortran PHYEX"
    )
    print("  ✓ RVDEPG : OK")
    
    # ========================================================================
    # Statistiques globales
    # ========================================================================
    print("\n" + "="*80)
    print("STATISTIQUES DES PROCESSUS LENTS")
    print("="*80)
    
    n_total = domain[0] * domain[1] * domain[2]
    
    # Points actifs pour chaque processus
    n_honi = np.sum(rc_honi_tnd_py > 1e-10)
    n_deps = np.sum(rv_deps_tnd_py > 1e-10)
    n_aggs = np.sum(ri_aggs_tnd_py > 1e-10)
    n_auts = np.sum(ri_auts_tnd_py > 1e-10)
    n_depg = np.sum(rv_depg_tnd_py > 1e-10)
    
    print(f"\nPoints actifs (tendance > 1e-10):")
    print(f"  RCHONI (nucléation homogène):     {n_honi:6d}/{n_total} ({100.0*n_honi/n_total:5.1f}%)")
    print(f"  RVDEPS (déposition neige):        {n_deps:6d}/{n_total} ({100.0*n_deps/n_total:5.1f}%)")
    print(f"  RIAGGS (agrégation neige):        {n_aggs:6d}/{n_total} ({100.0*n_aggs/n_total:5.1f}%)")
    print(f"  RIAUTS (autoconversion):          {n_auts:6d}/{n_total} ({100.0*n_auts/n_total:5.1f}%)")
    print(f"  RVDEPG (déposition graupel):      {n_depg:6d}/{n_total} ({100.0*n_depg/n_total:5.1f}%)")
    
    # Statistiques de température pour nucléation homogène
    if n_honi > 0:
        t_flat = FloatFieldsIJK_Input["t"].reshape(domain[0] * domain[1] * domain[2], order="F")
        t_honi = t_flat[rc_honi_tnd_py > 1e-10]
        print(f"\nTempérature dans zones de nucléation homogène:")
        print(f"  min={t_honi.min():.2f}K, max={t_honi.max():.2f}K, moyenne={t_honi.mean():.2f}K")
    
    print("\n" + "="*80)
    print("SUCCÈS: Reproductibilité validée!")
    print("Le stencil Python GT4Py ice4_slow reproduit fidèlement PHYEX-IAL_CY50T1")
