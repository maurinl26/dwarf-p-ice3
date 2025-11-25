# -*- coding: utf-8 -*-
"""
Test de reproductibilité du stencil ice4_rrhong par rapport à PHYEX-IAL_CY50T1.

Ce module valide que l'implémentation Python GT4Py du schéma de congélation spontanée
de la pluie (RRHONG - Rain Rain HONGelation) produit des résultats numériquement identiques
à l'implémentation Fortran de référence issue du projet PHYEX (PHYsique EXternalisée)
version IAL_CY50T1.

Le processus RRHONG représente la congélation spontanée des gouttes de pluie lorsque
la température descend en dessous de -35°C. C'est un processus microphysique important
pour la conversion de la pluie en graupel dans les parties froides des nuages.

Référence:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_rrhong.F90
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
def test_ice4_rrhong(dtypes, backend, externals, packed_dims, domain, origin):
    """
    Test de reproductibilité du stencil de congélation spontanée RRHONG (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre:
    - L'implémentation Python/GT4Py du processus de congélation spontanée
    - L'implémentation Fortran de référence PHYEX-IAL_CY50T1
    
    Le processus RRHONG (Rain Rain HONgelation) calcule le taux de congélation spontanée
    des gouttes de pluie lorsque la température devient inférieure à -35°C. Ce processus
    est activé uniquement dans des conditions très froides et lorsque la pluie est présente.
    
    Configuration testée:
        - Seuil de température: T < -35°C (TT - 35.0)
        - Seuil de pluie: rrt > R_RTMIN
        - LFEEDBACKT: Rétroaction thermodynamique sur la congélation
        
    Physique du processus:
        Si T < -35°C ET rrt > R_RTMIN ET ldcompute = True:
            rrhong_mr = rrt (toute la pluie gèle)
            
            Si LFEEDBACKT = True:
                Limitation thermodynamique pour éviter de franchir -35°C:
                rrhong_mr = min(rrhong_mr, max(0, ((TT-35)/exn - tht)/(lsfact-lvfact)))
                
        Sinon:
            rrhong_mr = 0 (pas de congélation)
    
    Champs vérifiés:
        - rrhong_mr: Taux de congélation spontanée [kg/kg] (conversion pluie -> graupel)
    
    Tolérance:
        rtol=1e-6, atol=1e-8
    
    Args:
        dtypes: Dictionnaire des types (simple/double précision)
        backend: Backend GT4Py (debug, numpy, cpu, gpu)
        externals: Paramètres externes (constantes physiques et options)
        packed_dims: Dimensions pour l'interface Fortran (kproma, ksize)
        domain: Taille du domaine de calcul
        origin: Origine du domaine GT4Py
    """
    from ice3.stencils.ice4_rrhong import ice4_rrhong

    # Compilation du stencil GT4Py
    ice4_rrhong_gt4py = stencil(
        backend,
        definition=ice4_rrhong,
        name="ice4_rrhong",
        dtypes=dtypes,
        externals=externals,
    )

    # Compilation du stencil Fortran de référence
    ice4_rrhong_fortran = compile_fortran_stencil(
        "mode_ice4_rrhong.F90", "mode_ice4_rrhong", "ice4_rrhong"
    )

    # =========================================================================
    # Initialisation des champs d'entrée
    # =========================================================================
    
    # Champs scalaires 3D
    FloatFieldsIJK_Names = [
        "t",          # Température [K]
        "exn",        # Fonction d'Exner (P/P0)^(R/Cp) [-]
        "lvfact",     # Chaleur latente de vaporisation [J/kg]
        "lsfact",     # Chaleur latente de sublimation [J/kg]
        "tht",        # Température potentielle [K]
        "rrt",        # Rapport de mélange de la pluie [kg/kg]
        "rrhong_mr",  # Taux de congélation (sortie) [kg/kg]
    ]

    FloatFieldsIJK = {
        name: np.array(
            np.random.rand(*domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Names
    }

    # Ajustement des températures pour avoir des valeurs réalistes
    # Température entre 200K et 300K
    FloatFieldsIJK["t"] = FloatFieldsIJK["t"] * 100.0 + 200.0
    
    # Température potentielle entre 250K et 350K
    FloatFieldsIJK["tht"] = FloatFieldsIJK["tht"] * 100.0 + 250.0
    
    # Exner function autour de 1.0
    FloatFieldsIJK["exn"] = FloatFieldsIJK["exn"] * 0.2 + 0.9
    
    # Chaleurs latentes réalistes (environ 2.5e6 J/kg pour Lv, 2.8e6 J/kg pour Ls)
    FloatFieldsIJK["lvfact"] = FloatFieldsIJK["lvfact"] * 0.5e6 + 2.0e6
    FloatFieldsIJK["lsfact"] = FloatFieldsIJK["lsfact"] * 0.5e6 + 2.5e6
    
    # Rapport de mélange de pluie (valeurs petites, typiquement < 0.01)
    FloatFieldsIJK["rrt"] = FloatFieldsIJK["rrt"] * 0.01

    # Champ booléen pour activer/désactiver le calcul par colonne
    BoolFieldsIJK = {
        "ldcompute": np.array(
            np.random.rand(*domain) > 0.3,  # ~70% des points activés
            dtype=np.bool_,
            order="F",
        )
    }

    # =========================================================================
    # Conversion en storages GT4Py
    # =========================================================================
    
    ldcompute_gt4py = from_array(
        BoolFieldsIJK["ldcompute"], dtype=dtypes["bool"], backend=backend
    )
    t_gt4py = from_array(
        FloatFieldsIJK["t"], dtype=dtypes["float"], backend=backend
    )
    exn_gt4py = from_array(
        FloatFieldsIJK["exn"], dtype=dtypes["float"], backend=backend
    )
    lvfact_gt4py = from_array(
        FloatFieldsIJK["lvfact"], dtype=dtypes["float"], backend=backend
    )
    lsfact_gt4py = from_array(
        FloatFieldsIJK["lsfact"], dtype=dtypes["float"], backend=backend
    )
    tht_gt4py = from_array(
        FloatFieldsIJK["tht"], dtype=dtypes["float"], backend=backend
    )
    rrt_gt4py = from_array(
        FloatFieldsIJK["rrt"], dtype=dtypes["float"], backend=backend
    )
    
    # Champ de sortie (initialisé à zéro)
    rrhong_mr_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)

    # =========================================================================
    # Exécution du stencil Python GT4Py
    # =========================================================================
    
    ice4_rrhong_gt4py(
        ldcompute=ldcompute_gt4py,
        t=t_gt4py,
        exn=exn_gt4py,
        lvfact=lvfact_gt4py,
        lsfact=lsfact_gt4py,
        tht=tht_gt4py,
        rrt=rrt_gt4py,
        rrhong_mr=rrhong_mr_gt4py,
        domain=domain,
        origin=origin,
    )

    # =========================================================================
    # Exécution de la référence Fortran PHYEX
    # =========================================================================
    
    # Extraction des paramètres externes pour Fortran
    xtt = externals["TT"]           # Température de référence (273.15 K)
    r_rtmin = externals["R_RTMIN"]  # Seuil minimal de pluie
    lfeedbackt = externals["LFEEDBACKT"]  # Rétroaction thermodynamique
    
    # Aplatissement des champs 3D en 1D pour Fortran (ordre Fortran)
    ldcompute_flat = BoolFieldsIJK["ldcompute"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    t_flat = FloatFieldsIJK["t"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    exn_flat = FloatFieldsIJK["exn"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    lvfact_flat = FloatFieldsIJK["lvfact"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    lsfact_flat = FloatFieldsIJK["lsfact"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    tht_flat = FloatFieldsIJK["tht"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rrt_flat = FloatFieldsIJK["rrt"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    
    # Appel de la routine Fortran
    prrhong_mr_fortran = ice4_rrhong_fortran(
        xtt=xtt,
        r_rtmin=r_rtmin,
        lfeedbackt=lfeedbackt,
        ldcompute=ldcompute_flat,
        pexn=exn_flat,
        plvfact=lvfact_flat,
        plsfact=lsfact_flat,
        pt=t_flat,
        prrt=rrt_flat,
        ptht=tht_flat,
        **packed_dims,
    )

    # =========================================================================
    # VALIDATION DE LA REPRODUCTIBILITÉ - Comparaison Python vs Fortran PHYEX
    # =========================================================================
    
    print("\n" + "="*80)
    print("TEST DE REPRODUCTIBILITÉ: ice4_rrhong.py vs PHYEX-IAL_CY50T1")
    print("="*80)
    print(f"Backend: {backend}")
    print(f"Précision: {'simple' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domaine: {domain[0]}x{domain[1]}x{domain[2]}")
    print("="*80)
    
    # ------------------------------------------------------------------------
    # Validation du taux de congélation spontanée
    # ------------------------------------------------------------------------
    rrhong_mr_py = rrhong_mr_gt4py.reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    
    # Statistiques pour diagnostic
    print("\nStatistiques du champ rrhong_mr:")
    print(f"  Python - min: {rrhong_mr_py.min():.6e}, max: {rrhong_mr_py.max():.6e}")
    print(f"  Fortran - min: {prrhong_mr_fortran.min():.6e}, max: {prrhong_mr_fortran.max():.6e}")
    
    # Calcul des différences
    diff = np.abs(rrhong_mr_py - prrhong_mr_fortran)
    rel_diff = np.where(
        np.abs(prrhong_mr_fortran) > 1e-10,
        diff / np.abs(prrhong_mr_fortran),
        0.0
    )
    
    print(f"\nDifférences:")
    print(f"  Absolue - max: {diff.max():.6e}, moyenne: {diff.mean():.6e}")
    print(f"  Relative - max: {rel_diff.max():.6e}, moyenne: {rel_diff[rel_diff>0].mean() if np.any(rel_diff>0) else 0:.6e}")
    
    # Validation numérique
    assert_allclose(
        prrhong_mr_fortran,
        rrhong_mr_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] Taux de congélation spontanée (rrhong_mr): divergence Python/Fortran PHYEX"
    )
    print("\n✓ rrhong_mr (taux de congélation spontanée) : OK")
    
    # Informations supplémentaires
    n_freezing = np.sum(rrhong_mr_py > 1e-10)
    n_total = domain[0] * domain[1] * domain[2]
    pct_freezing = 100.0 * n_freezing / n_total
    
    print(f"\nPoints avec congélation active: {n_freezing}/{n_total} ({pct_freezing:.1f}%)")
    
    print("\n" + "="*80)
    print("SUCCÈS: Reproductibilité validée!")
    print("Le stencil Python GT4Py reproduit fidèlement PHYEX-IAL_CY50T1")
    print("="*80 + "\n")
