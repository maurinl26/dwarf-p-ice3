# -*- coding: utf-8 -*-
"""
Test de reproductibilité du stencil ice4_fast_ri par rapport à PHYEX-IAL_CY50T1.

Ce module valide que l'implémentation Python GT4Py du processus rapide Bergeron-Findeisen
de la microphysique ICE4 produit des résultats numériquement identiques à l'implémentation 
Fortran de référence issue du projet PHYEX (PHYsique EXternalisée) version IAL_CY50T1.

Le processus Bergeron-Findeisen (RCBERI) représente:
- L'évaporation des gouttelettes d'eau nuageuse (cloud droplets)
- La déposition simultanée sur les cristaux de glace pristine
- Actif uniquement en conditions de sursaturation par rapport à la glace (SSI > 0)
- Nécessite la présence simultanée d'eau nuageuse et de glace pristine

Ce processus est dit "rapide" car son échelle de temps caractéristique est plus courte
que celle des processus lents (nucléation, agrégation, etc.).

Référence:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_ri.F90
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
def test_ice4_fast_ri(dtypes, backend, externals, packed_dims, domain, origin, ldsoft):
    """
    Test de reproductibilité du stencil ice4_fast_ri (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre:
    - L'implémentation Python/GT4Py du processus Bergeron-Findeisen
    - L'implémentation Fortran de référence PHYEX-IAL_CY50T1
    
    Le processus Bergeron-Findeisen calcule:
    
    RCBERI - Effet Bergeron-Findeisen:
       - Actif pour SSI > 0, rct > C_RTMIN, rit > I_RTMIN, cit > 1e-20
       - Évaporation d'eau nuageuse et déposition sur cristaux de glace
       - Dépend de lambda_i (paramètre de pente de la distribution de glace)
       - Formule: (SSI/(rhodref*AI)) * CIT * (O0DEPI/lbda_i + O2DEPI*CJ²/lbda_i^(DI+2))
       - où lbda_i = min(1e8, LBI*(rhodref*rit/cit)^LBEXI)
    
    Champs vérifiés:
        - rc_beri_tnd: Taux d'évaporation par effet Bergeron-Findeisen [kg/kg/s]
    
    Tolérance:
        rtol=1e-6, atol=1e-8
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et paramètres microphysiques)
        packed_dims: Dimensions pour l'interface Fortran (kproma, ksize)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
        ldsoft: Indicateur de mode soft (désactive le calcul pour tests)
    """
    from ice3.stencils.ice4_fast_ri import ice4_fast_ri

    # Compilation du stencil GT4Py
    ice4_fast_ri_gt4py = stencil(
        backend,
        definition=ice4_fast_ri,
        name="ice4_fast_ri",
        dtypes=dtypes,
        externals=externals,
    )

    # Compilation du stencil Fortran de référence
    ice4_fast_ri_fortran = compile_fortran_stencil(
        "mode_ice4_fast_ri.F90", "mode_ice4_fast_ri", "ice4_fast_ri"
    )

    # =========================================================================
    # Initialisation des champs d'entrée
    # =========================================================================
    
    # Champs scalaires 3D d'entrée
    FloatFieldsIJK_Input_Names = [
        "rhodref",    # Densité de référence [kg/m³]
        "ai",         # Fonction thermodynamique [-]
        "cj",         # Coefficient de ventilation [-]
        "cit",        # Concentration de glace pristine [#/m³]
        "ssi",        # Sursaturation par rapport à la glace [-]
        "rct",        # Rapport de mélange d'eau nuageuse [kg/kg]
        "rit",        # Rapport de mélange de glace pristine [kg/kg]
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
    # Densité de référence entre 0.3 et 1.5 kg/m³
    FloatFieldsIJK_Input["rhodref"] = FloatFieldsIJK_Input["rhodref"] * 1.2 + 0.3
    
    # Fonction thermodynamique ai (valeurs typiques: 1e9 à 1e11)
    FloatFieldsIJK_Input["ai"] = FloatFieldsIJK_Input["ai"] * 9e10 + 1e10
    
    # Coefficient de ventilation cj (valeurs typiques: 0 à 10)
    FloatFieldsIJK_Input["cj"] = FloatFieldsIJK_Input["cj"] * 10.0
    
    # Concentration de glace pristine (valeurs typiques: 1e3 à 1e6 #/m³)
    FloatFieldsIJK_Input["cit"] = FloatFieldsIJK_Input["cit"] * 9.99e5 + 1e3
    
    # Sursaturation par rapport à la glace entre -0.2 et 0.2
    FloatFieldsIJK_Input["ssi"] = (FloatFieldsIJK_Input["ssi"] - 0.5) * 0.4
    
    # Rapports de mélange (valeurs petites, typiquement < 0.01)
    FloatFieldsIJK_Input["rct"] = FloatFieldsIJK_Input["rct"] * 0.005  # eau nuageuse
    FloatFieldsIJK_Input["rit"] = FloatFieldsIJK_Input["rit"] * 0.002  # glace pristine

    # Champs de sortie
    FloatFieldsIJK_Output_Names = [
        "rc_beri_tnd",  # Effet Bergeron-Findeisen
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
    ai_gt4py = from_array(
        FloatFieldsIJK_Input["ai"], dtype=dtypes["float"], backend=backend
    )
    cj_gt4py = from_array(
        FloatFieldsIJK_Input["cj"], dtype=dtypes["float"], backend=backend
    )
    cit_gt4py = from_array(
        FloatFieldsIJK_Input["cit"], dtype=dtypes["float"], backend=backend
    )
    ssi_gt4py = from_array(
        FloatFieldsIJK_Input["ssi"], dtype=dtypes["float"], backend=backend
    )
    rct_gt4py = from_array(
        FloatFieldsIJK_Input["rct"], dtype=dtypes["float"], backend=backend
    )
    rit_gt4py = from_array(
        FloatFieldsIJK_Input["rit"], dtype=dtypes["float"], backend=backend
    )
    
    # Champs de sortie (initialisés à zéro)
    rc_beri_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)

    # =========================================================================
    # Exécution du stencil Python GT4Py
    # =========================================================================
    
    ice4_fast_ri_gt4py(
        ldsoft=ldsoft,
        ldcompute=ldcompute_gt4py,
        rhodref=rhodref_gt4py,
        ai=ai_gt4py,
        cj=cj_gt4py,
        cit=cit_gt4py,
        ssi=ssi_gt4py,
        rct=rct_gt4py,
        rit=rit_gt4py,
        rc_beri_tnd=rc_beri_tnd_gt4py,
        domain=domain,
        origin=origin,
    )

    # =========================================================================
    # Exécution de la référence Fortran PHYEX
    # =========================================================================
    
    # Extraction des paramètres externes pour Fortran
    xlbi = externals["LBI"]                # Paramètre de pente de la glace
    xlbexi = externals["LBEXI"]            # Exposant de pente de la glace
    xdi = externals["DI"]                  # Paramètre de dimension
    x0depi = externals["O0DEPI"]           # Paramètre 0 pour déposition
    x2depi = externals["O2DEPI"]           # Paramètre 2 pour déposition
    c_rtmin = externals["C_RTMIN"]         # Seuil minimal eau nuageuse
    i_rtmin = externals["I_RTMIN"]         # Seuil minimal glace
    
    # Aplatissement des champs 3D en 1D pour Fortran (ordre Fortran)
    ldcompute_flat = BoolFieldsIJK["ldcompute"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rhodref_flat = FloatFieldsIJK_Input["rhodref"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    ai_flat = FloatFieldsIJK_Input["ai"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    cj_flat = FloatFieldsIJK_Input["cj"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    cit_flat = FloatFieldsIJK_Input["cit"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    ssi_flat = FloatFieldsIJK_Input["ssi"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rct_flat = FloatFieldsIJK_Input["rct"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rit_flat = FloatFieldsIJK_Input["rit"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    
    # Champs de sortie (copies car ils sont modifiés)
    rc_beri_tnd_flat = FloatFieldsIJK_Output["rc_beri_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    
    # Appel de la routine Fortran
    ice4_fast_ri_fortran(
        xlbi=xlbi,
        xlbexi=xlbexi,
        xdi=xdi,
        x0depi=x0depi,
        x2depi=x2depi,
        c_rtmin=c_rtmin,
        i_rtmin=i_rtmin,
        ldsoft=ldsoft,
        ldcompute=ldcompute_flat,
        prhodref=rhodref_flat,
        pai=ai_flat,
        pcj=cj_flat,
        pcit=cit_flat,
        pssi=ssi_flat,
        prct=rct_flat,
        prit=rit_flat,
        prcberi=rc_beri_tnd_flat,
        **packed_dims,
    )

    # =========================================================================
    # VALIDATION DE LA REPRODUCTIBILITÉ - Comparaison Python vs Fortran PHYEX
    # =========================================================================
    
    print("\n" + "="*80)
    print("TEST DE REPRODUCTIBILITÉ: ice4_fast_ri.py vs PHYEX-IAL_CY50T1")
    print("="*80)
    print(f"Backend: {backend}")
    print(f"Précision: {'simple' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domaine: {domain[0]}x{domain[1]}x{domain[2]}")
    print("="*80)
    
    # Reshape des sorties Python pour comparaison
    rc_beri_tnd_py = rc_beri_tnd_gt4py.reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    
    # ------------------------------------------------------------------------
    # Validation de l'effet Bergeron-Findeisen (RCBERI)
    # ------------------------------------------------------------------------
    print("\n1. RCBERI - Effet Bergeron-Findeisen")
    print("-" * 80)
    print(f"  Python - min: {rc_beri_tnd_py.min():.6e}, max: {rc_beri_tnd_py.max():.6e}")
    print(f"  Fortran - min: {rc_beri_tnd_flat.min():.6e}, max: {rc_beri_tnd_flat.max():.6e}")
    
    diff = np.abs(rc_beri_tnd_py - rc_beri_tnd_flat)
    rel_diff = np.where(
        np.abs(rc_beri_tnd_flat) > 1e-10,
        diff / np.abs(rc_beri_tnd_flat),
        0.0
    )
    
    print(f"  Différence absolue - max: {diff.max():.6e}, moyenne: {diff.mean():.6e}")
    print(f"  Différence relative - max: {rel_diff.max():.6e}, "
          f"moyenne: {rel_diff[rel_diff>0].mean() if np.any(rel_diff>0) else 0:.6e}")
    
    assert_allclose(
        rc_beri_tnd_flat,
        rc_beri_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] Effet Bergeron-Findeisen (rc_beri_tnd): divergence Python/Fortran PHYEX"
    )
    print("  ✓ RCBERI : OK")
    
    # ========================================================================
    # Statistiques globales
    # ========================================================================
    print("\n" + "="*80)
    print("STATISTIQUES DU PROCESSUS BERGERON-FINDEISEN")
    print("="*80)
    
    n_total = domain[0] * domain[1] * domain[2]
    
    # Points actifs pour le processus
    n_beri = np.sum(rc_beri_tnd_py > 1e-10)
    
    print(f"\nPoints actifs (tendance > 1e-10):")
    print(f"  RCBERI (Bergeron-Findeisen):      {n_beri:6d}/{n_total} ({100.0*n_beri/n_total:5.1f}%)")
    
    # Statistiques de sursaturation pour les processus actifs
    if n_beri > 0:
        ssi_flat = FloatFieldsIJK_Input["ssi"].reshape(domain[0] * domain[1] * domain[2], order="F")
        ssi_beri = ssi_flat[rc_beri_tnd_py > 1e-10]
        print(f"\nSursaturation dans zones Bergeron-Findeisen:")
        print(f"  min={ssi_beri.min():.4f}, max={ssi_beri.max():.4f}, moyenne={ssi_beri.mean():.4f}")
        
        # Statistiques des rapports de mélange
        rct_flat = FloatFieldsIJK_Input["rct"].reshape(domain[0] * domain[1] * domain[2], order="F")
        rit_flat = FloatFieldsIJK_Input["rit"].reshape(domain[0] * domain[1] * domain[2], order="F")
        rct_beri = rct_flat[rc_beri_tnd_py > 1e-10]
        rit_beri = rit_flat[rc_beri_tnd_py > 1e-10]
        
        print(f"\nRapports de mélange dans zones actives:")
        print(f"  Eau nuageuse (rct): min={rct_beri.min():.6e}, max={rct_beri.max():.6e}, moyenne={rct_beri.mean():.6e}")
        print(f"  Glace pristine (rit): min={rit_beri.min():.6e}, max={rit_beri.max():.6e}, moyenne={rit_beri.mean():.6e}")
    
    print("\n" + "="*80)
    print("SUCCÈS: Reproductibilité validée!")
    print("Le stencil Python GT4Py ice4_fast_ri reproduit fidèlement PHYEX-IAL_CY50T1")
    print("="*80)
