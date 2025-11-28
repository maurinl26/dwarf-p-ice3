# -*- coding: utf-8 -*-
"""
Test de reproductibilité du stencil ice4_warm par rapport à PHYEX-IAL_CY50T1.

Ce module valide que l'implémentation Python GT4Py des processus chauds de la 
microphysique ICE4 produit des résultats numériquement identiques à l'implémentation 
Fortran de référence issue du projet PHYEX (PHYsique EXternalisée) version IAL_CY50T1.

Les processus chauds incluent:
1. RCAUTR - Autoconversion d'eau nuageuse en pluie (au-dessus d'un seuil critique)
2. RCACCR - Accrétion d'eau nuageuse par la pluie (collection des gouttelettes)
3. RREVAV - Évaporation de la pluie (en ciel clair ou sous-saturé)

Ces processus sont appelés "chauds" car ils ne concernent que la phase liquide
(eau nuageuse et pluie), sans intervention de la phase glacée.

Référence:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_warm.F90
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
def test_ice4_warm(dtypes, backend, externals, packed_dims, domain, origin, ldsoft):
    """
    Test de reproductibilité du stencil ice4_warm (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre:
    - L'implémentation Python/GT4Py des processus microphysiques chauds
    - L'implémentation Fortran de référence PHYEX-IAL_CY50T1
    
    Les processus microphysiques chauds calculent:
    
    1. RCAUTR - Autoconversion eau nuageuse → pluie:
       - Active pour hlc_hrc > C_RTMIN et hlc_hcf > 0
       - Conversion automatique au-delà du seuil critique CRIAUTC
       - Formule: TIMAUTC * max(0, hlc_hrc - hlc_hcf * CRIAUTC / rhodref)
    
    2. RCACCR - Accrétion par la pluie:
       - Active pour rct > C_RTMIN et rrt > R_RTMIN
       - Collection des gouttelettes nuageuses par les gouttes de pluie
       - Formule: FCACCR * rct * lbdar^EXCACCR * rhodref^(-CEXVT)
    
    3. RREVAV - Évaporation de la pluie:
       - Active pour rrt > R_RTMIN et conditions sous-saturées
       - Dépend du mode SUBG_RR_EVAP (none, clfr, prfr)
       - Calcul avec température non saturée pour le mode clfr/prfr
    
    Champs vérifiés:
        - rcautr: Taux d'autoconversion rc→rr [kg/kg/s]
        - rcaccr: Taux d'accrétion rc par rr [kg/kg/s]
        - rrevav: Taux d'évaporation de rr [kg/kg/s]
    
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
    from ice3.stencils.ice4_warm import ice4_warm

    # Compilation du stencil GT4Py
    ice4_warm_gt4py = stencil(
        backend,
        definition=ice4_warm,
        name="ice4_warm",
        dtypes=dtypes,
        externals=externals,
    )

    # Compilation du stencil Fortran de référence
    ice4_warm_fortran = compile_fortran_stencil(
        "mode_ice4_warm.F90", "mode_ice4_warm", "ice4_warm"
    )

    # =========================================================================
    # Initialisation des champs d'entrée
    # =========================================================================
    
    # Champs scalaires 3D d'entrée
    FloatFieldsIJK_Input_Names = [
        "rhodref",     # Densité de référence [kg/m³]
        "t",           # Température [K]
        "pres",        # Pression absolue [Pa]
        "tht",         # Température potentielle [K]
        "lbdar",       # Paramètre de pente de la distribution de pluie [m^-1]
        "lbdar_rf",    # Paramètre de pente pour la fraction de pluie [m^-1]
        "ka",          # Conductivité thermique de l'air [J/m/s/K]
        "dv",          # Diffusivité de la vapeur d'eau [m²/s]
        "cj",          # Coefficient de ventilation [-]
        "hlc_hcf",     # Fraction de nuages hauts dans la grille [-]
        "hlc_hrc",     # Contenu en eau liquide des nuages hauts [kg/kg]
        "cf",          # Fraction nuageuse [-]
        "rf",          # Fraction de pluie [-]
        "rvt",         # Rapport de mélange de vapeur d'eau [kg/kg]
        "rct",         # Rapport de mélange d'eau nuageuse [kg/kg]
        "rrt",         # Rapport de mélange de pluie [kg/kg]
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
    # Température entre 260K et 300K (froid à chaud)
    FloatFieldsIJK_Input["t"] = FloatFieldsIJK_Input["t"] * 40.0 + 260.0
    
    # Température potentielle (légèrement supérieure à T)
    FloatFieldsIJK_Input["tht"] = FloatFieldsIJK_Input["tht"] * 40.0 + 265.0
    
    # Pression entre 50000 et 100000 Pa (500-1000 hPa)
    FloatFieldsIJK_Input["pres"] = FloatFieldsIJK_Input["pres"] * 50000.0 + 50000.0
    
    # Densité de référence entre 0.3 et 1.5 kg/m³
    FloatFieldsIJK_Input["rhodref"] = FloatFieldsIJK_Input["rhodref"] * 1.2 + 0.3
    
    # Rapports de mélange (valeurs petites, typiquement < 0.02)
    FloatFieldsIJK_Input["rvt"] = FloatFieldsIJK_Input["rvt"] * 0.015  # vapeur
    FloatFieldsIJK_Input["rct"] = FloatFieldsIJK_Input["rct"] * 0.005  # nuages
    FloatFieldsIJK_Input["rrt"] = FloatFieldsIJK_Input["rrt"] * 0.010  # pluie
    
    # Paramètres de pente (valeurs typiques: 10^3 à 10^6 m^-1)
    FloatFieldsIJK_Input["lbdar"] = FloatFieldsIJK_Input["lbdar"] * 9e5 + 1e5
    FloatFieldsIJK_Input["lbdar_rf"] = FloatFieldsIJK_Input["lbdar_rf"] * 9e5 + 1e5
    
    # Conductivité thermique (valeurs typiques: 0.01-0.03 J/m/s/K)
    FloatFieldsIJK_Input["ka"] = FloatFieldsIJK_Input["ka"] * 0.02 + 0.01
    
    # Diffusivité (valeurs typiques: 1e-5 à 3e-5 m²/s)
    FloatFieldsIJK_Input["dv"] = FloatFieldsIJK_Input["dv"] * 2e-5 + 1e-5
    
    # Coefficient de ventilation (valeurs typiques: 0 à 10)
    FloatFieldsIJK_Input["cj"] = FloatFieldsIJK_Input["cj"] * 10.0
    
    # Fractions (0 à 1)
    FloatFieldsIJK_Input["hlc_hcf"] = FloatFieldsIJK_Input["hlc_hcf"]
    FloatFieldsIJK_Input["cf"] = FloatFieldsIJK_Input["cf"]
    FloatFieldsIJK_Input["rf"] = FloatFieldsIJK_Input["rf"]
    
    # Contenu en eau liquide des nuages hauts (petites valeurs)
    FloatFieldsIJK_Input["hlc_hrc"] = FloatFieldsIJK_Input["hlc_hrc"] * 0.005

    # Champs de sortie
    FloatFieldsIJK_Output_Names = [
        "rcautr",  # Autoconversion rc→rr
        "rcaccr",  # Accrétion rc par rr
        "rrevav",  # Évaporation de rr
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
    pres_gt4py = from_array(
        FloatFieldsIJK_Input["pres"], dtype=dtypes["float"], backend=backend
    )
    tht_gt4py = from_array(
        FloatFieldsIJK_Input["tht"], dtype=dtypes["float"], backend=backend
    )
    lbdar_gt4py = from_array(
        FloatFieldsIJK_Input["lbdar"], dtype=dtypes["float"], backend=backend
    )
    lbdar_rf_gt4py = from_array(
        FloatFieldsIJK_Input["lbdar_rf"], dtype=dtypes["float"], backend=backend
    )
    ka_gt4py = from_array(
        FloatFieldsIJK_Input["ka"], dtype=dtypes["float"], backend=backend
    )
    dv_gt4py = from_array(
        FloatFieldsIJK_Input["dv"], dtype=dtypes["float"], backend=backend
    )
    cj_gt4py = from_array(
        FloatFieldsIJK_Input["cj"], dtype=dtypes["float"], backend=backend
    )
    hlc_hcf_gt4py = from_array(
        FloatFieldsIJK_Input["hlc_hcf"], dtype=dtypes["float"], backend=backend
    )
    hlc_hrc_gt4py = from_array(
        FloatFieldsIJK_Input["hlc_hrc"], dtype=dtypes["float"], backend=backend
    )
    cf_gt4py = from_array(
        FloatFieldsIJK_Input["cf"], dtype=dtypes["float"], backend=backend
    )
    rf_gt4py = from_array(
        FloatFieldsIJK_Input["rf"], dtype=dtypes["float"], backend=backend
    )
    rvt_gt4py = from_array(
        FloatFieldsIJK_Input["rvt"], dtype=dtypes["float"], backend=backend
    )
    rct_gt4py = from_array(
        FloatFieldsIJK_Input["rct"], dtype=dtypes["float"], backend=backend
    )
    rrt_gt4py = from_array(
        FloatFieldsIJK_Input["rrt"], dtype=dtypes["float"], backend=backend
    )
    
    # Champs de sortie (initialisés à zéro)
    rcautr_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rcaccr_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rrevav_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)

    # =========================================================================
    # Exécution du stencil Python GT4Py
    # =========================================================================
    
    ice4_warm_gt4py(
        ldsoft=ldsoft,
        ldcompute=ldcompute_gt4py,
        rhodref=rhodref_gt4py,
        t=t_gt4py,
        pres=pres_gt4py,
        tht=tht_gt4py,
        lbdar=lbdar_gt4py,
        lbdar_rf=lbdar_rf_gt4py,
        ka=ka_gt4py,
        dv=dv_gt4py,
        cj=cj_gt4py,
        hlc_hcf=hlc_hcf_gt4py,
        hlc_hrc=hlc_hrc_gt4py,
        cf=cf_gt4py,
        rf=rf_gt4py,
        rvt=rvt_gt4py,
        rct=rct_gt4py,
        rrt=rrt_gt4py,
        rcautr=rcautr_gt4py,
        rcaccr=rcaccr_gt4py,
        rrevav=rrevav_gt4py,
        domain=domain,
        origin=origin,
    )

    # =========================================================================
    # Exécution de la référence Fortran PHYEX
    # =========================================================================
    
    # Extraction des paramètres externes pour Fortran
    xalpw = externals["ALPW"]              # Constante saturation vapeur
    xbetaw = externals["BETAW"]            # Constante saturation vapeur
    xgamw = externals["GAMW"]              # Constante saturation vapeur
    xepsilo = externals["EPSILO"]          # Rapport masses molaires
    xlvtt = externals["LVTT"]              # Chaleur latente vaporisation
    xcpv = externals["CPV"]                # Cp vapeur
    xcl = externals["CL"]                  # Cp liquide
    xtt = externals["TT"]                  # Température de référence (273.15 K)
    xrv = externals["RV"]                  # Constante gaz vapeur
    xcpd = externals["CPD"]                # Cp air sec
    xtimautc = externals["TIMAUTC"]        # Temps autoconversion
    xcriautc = externals["CRIAUTC"]        # Seuil critique autoconversion
    xfcaccr = externals["FCACCR"]          # Facteur accrétion
    xexcaccr = externals["EXCACCR"]        # Exposant accrétion
    x0evar = externals["O0EVAR"]           # Paramètre 0 évaporation
    x1evar = externals["O1EVAR"]           # Paramètre 1 évaporation
    xex0evar = externals["EX0EVAR"]        # Exposant 0 évaporation
    xex1evar = externals["EX1EVAR"]        # Exposant 1 évaporation
    c_rtmin = externals["C_RTMIN"]         # Seuil minimal eau nuageuse
    r_rtmin = externals["R_RTMIN"]         # Seuil minimal pluie
    xcexvt = externals["CEXVT"]            # Exposant vitesse terminale
    
    # Paramètre pour le mode d'évaporation (converti en string pour Fortran)
    subg_rr_evap = externals["SUBG_RR_EVAP"]
    if subg_rr_evap == 0:
        hsubg_rr_evap = "none"
    elif subg_rr_evap == 1:
        hsubg_rr_evap = "clfr"
    elif subg_rr_evap == 2:
        hsubg_rr_evap = "prfr"
    else:
        hsubg_rr_evap = "none"
    
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
    pres_flat = FloatFieldsIJK_Input["pres"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    tht_flat = FloatFieldsIJK_Input["tht"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    lbdar_flat = FloatFieldsIJK_Input["lbdar"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    lbdar_rf_flat = FloatFieldsIJK_Input["lbdar_rf"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    ka_flat = FloatFieldsIJK_Input["ka"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    dv_flat = FloatFieldsIJK_Input["dv"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    cj_flat = FloatFieldsIJK_Input["cj"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    hlc_hcf_flat = FloatFieldsIJK_Input["hlc_hcf"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    hlc_hrc_flat = FloatFieldsIJK_Input["hlc_hrc"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    cf_flat = FloatFieldsIJK_Input["cf"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rf_flat = FloatFieldsIJK_Input["rf"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rvt_flat = FloatFieldsIJK_Input["rvt"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rct_flat = FloatFieldsIJK_Input["rct"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rrt_flat = FloatFieldsIJK_Input["rrt"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    
    # Champs de sortie (copies car ils sont modifiés)
    rcautr_flat = FloatFieldsIJK_Output["rcautr"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rcaccr_flat = FloatFieldsIJK_Output["rcaccr"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rrevav_flat = FloatFieldsIJK_Output["rrevav"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    
    # Appel de la routine Fortran
    ice4_warm_fortran(
        xalpw=xalpw,
        xbetaw=xbetaw,
        xgamw=xgamw,
        xepsilo=xepsilo,
        xlvtt=xlvtt,
        xcpv=xcpv,
        xcl=xcl,
        xtt=xtt,
        xrv=xrv,
        xcpd=xcpd,
        xtimautc=xtimautc,
        xcriautc=xcriautc,
        xfcaccr=xfcaccr,
        xexcaccr=xexcaccr,
        x0evar=x0evar,
        x1evar=x1evar,
        xex0evar=xex0evar,
        xex1evar=xex1evar,
        c_rtmin=c_rtmin,
        r_rtmin=r_rtmin,
        xcexvt=xcexvt,
        ldsoft=ldsoft,
        ldcompute=ldcompute_flat,
        hsubg_rr_evap=hsubg_rr_evap,
        prhodref=rhodref_flat,
        pt=t_flat,
        ppres=pres_flat,
        ptht=tht_flat,
        plbdar=lbdar_flat,
        plbdar_rf=lbdar_rf_flat,
        pka=ka_flat,
        pdv=dv_flat,
        pcj=cj_flat,
        phlc_hcf=hlc_hcf_flat,
        phlc_hrc=hlc_hrc_flat,
        pcf=cf_flat,
        prf=rf_flat,
        prvt=rvt_flat,
        prct=rct_flat,
        prrt=rrt_flat,
        prcautr=rcautr_flat,
        prcaccr=rcaccr_flat,
        prrevav=rrevav_flat,
        **packed_dims,
    )

    # =========================================================================
    # VALIDATION DE LA REPRODUCTIBILITÉ - Comparaison Python vs Fortran PHYEX
    # =========================================================================
    
    print("\n" + "="*80)
    print("TEST DE REPRODUCTIBILITÉ: ice4_warm.py vs PHYEX-IAL_CY50T1")
    print("="*80)
    print(f"Backend: {backend}")
    print(f"Précision: {'simple' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domaine: {domain[0]}x{domain[1]}x{domain[2]}")
    print(f"Mode évaporation: {hsubg_rr_evap}")
    print("="*80)
    
    # Reshape des sorties Python pour comparaison
    rcautr_py = rcautr_gt4py.reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rcaccr_py = rcaccr_gt4py.reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rrevav_py = rrevav_gt4py.reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    
    # ------------------------------------------------------------------------
    # Validation de l'autoconversion (RCAUTR)
    # ------------------------------------------------------------------------
    print("\n1. RCAUTR - Autoconversion eau nuageuse → pluie")
    print("-" * 80)
    print(f"  Python - min: {rcautr_py.min():.6e}, max: {rcautr_py.max():.6e}")
    print(f"  Fortran - min: {rcautr_flat.min():.6e}, max: {rcautr_flat.max():.6e}")
    
    diff = np.abs(rcautr_py - rcautr_flat)
    rel_diff = np.where(
        np.abs(rcautr_flat) > 1e-10,
        diff / np.abs(rcautr_flat),
        0.0
    )
    
    print(f"  Différence absolue - max: {diff.max():.6e}, moyenne: {diff.mean():.6e}")
    print(f"  Différence relative - max: {rel_diff.max():.6e}, "
          f"moyenne: {rel_diff[rel_diff>0].mean() if np.any(rel_diff>0) else 0:.6e}")
    
    assert_allclose(
        rcautr_flat,
        rcautr_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] Autoconversion (rcautr): divergence Python/Fortran PHYEX"
    )
    print("  ✓ RCAUTR : OK")
    
    # ------------------------------------------------------------------------
    # Validation de l'accrétion (RCACCR)
    # ------------------------------------------------------------------------
    print("\n2. RCACCR - Accrétion eau nuageuse par pluie")
    print("-" * 80)
    print(f"  Python - min: {rcaccr_py.min():.6e}, max: {rcaccr_py.max():.6e}")
    print(f"  Fortran - min: {rcaccr_flat.min():.6e}, max: {rcaccr_flat.max():.6e}")
    
    diff = np.abs(rcaccr_py - rcaccr_flat)
    rel_diff = np.where(
        np.abs(rcaccr_flat) > 1e-10,
        diff / np.abs(rcaccr_flat),
        0.0
    )
    
    print(f"  Différence absolue - max: {diff.max():.6e}, moyenne: {diff.mean():.6e}")
    print(f"  Différence relative - max: {rel_diff.max():.6e}, "
          f"moyenne: {rel_diff[rel_diff>0].mean() if np.any(rel_diff>0) else 0:.6e}")
    
    assert_allclose(
        rcaccr_flat,
        rcaccr_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] Accrétion (rcaccr): divergence Python/Fortran PHYEX"
    )
    print("  ✓ RCACCR : OK")
    
    # ------------------------------------------------------------------------
    # Validation de l'évaporation (RREVAV)
    # ------------------------------------------------------------------------
    print("\n3. RREVAV - Évaporation de la pluie")
    print("-" * 80)
    print(f"  Python - min: {rrevav_py.min():.6e}, max: {rrevav_py.max():.6e}")
    print(f"  Fortran - min: {rrevav_flat.min():.6e}, max: {rrevav_flat.max():.6e}")
    
    diff = np.abs(rrevav_py - rrevav_flat)
    rel_diff = np.where(
        np.abs(rrevav_flat) > 1e-10,
        diff / np.abs(rrevav_flat),
        0.0
    )
    
    print(f"  Différence absolue - max: {diff.max():.6e}, moyenne: {diff.mean():.6e}")
    print(f"  Différence relative - max: {rel_diff.max():.6e}, "
          f"moyenne: {rel_diff[rel_diff>0].mean() if np.any(rel_diff>0) else 0:.6e}")
    
    assert_allclose(
        rrevav_flat,
        rrevav_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] Évaporation (rrevav): divergence Python/Fortran PHYEX"
    )
    print("  ✓ RREVAV : OK")
    
    # ========================================================================
    # Statistiques globales
    # ========================================================================
    print("\n" + "="*80)
    print("STATISTIQUES DES PROCESSUS CHAUDS")
    print("="*80)
    
    n_total = domain[0] * domain[1] * domain[2]
    
    # Points actifs pour chaque processus
    n_autr = np.sum(rcautr_py > 1e-10)
    n_accr = np.sum(rcaccr_py > 1e-10)
    n_evap = np.sum(rrevav_py > 1e-10)
    
    print(f"\nPoints actifs (tendance > 1e-10):")
    print(f"  RCAUTR (autoconversion):          {n_autr:6d}/{n_total} ({100.0*n_autr/n_total:5.1f}%)")
    print(f"  RCACCR (accrétion):               {n_accr:6d}/{n_total} ({100.0*n_accr/n_total:5.1f}%)")
    print(f"  RREVAV (évaporation):             {n_evap:6d}/{n_total} ({100.0*n_evap/n_total:5.1f}%)")
    
    # Statistiques de température pour les processus
    if n_autr > 0 or n_accr > 0 or n_evap > 0:
        t_flat = FloatFieldsIJK_Input["t"].reshape(domain[0] * domain[1] * domain[2], order="F")
        
        if n_autr > 0:
            t_autr = t_flat[rcautr_py > 1e-10]
            print(f"\nTempérature dans zones d'autoconversion:")
            print(f"  min={t_autr.min():.2f}K, max={t_autr.max():.2f}K, moyenne={t_autr.mean():.2f}K")
        
        if n_accr > 0:
            t_accr = t_flat[rcaccr_py > 1e-10]
            print(f"\nTempérature dans zones d'accrétion:")
            print(f"  min={t_accr.min():.2f}K, max={t_accr.max():.2f}K, moyenne={t_accr.mean():.2f}K")
        
        if n_evap > 0:
            t_evap = t_flat[rrevav_py > 1e-10]
            print(f"\nTempérature dans zones d'évaporation:")
            print(f"  min={t_evap.min():.2f}K, max={t_evap.max():.2f}K, moyenne={t_evap.mean():.2f}K")
    
    print("\n" + "="*80)
    print("SUCCÈS: Reproductibilité validée!")
    print("Le stencil Python GT4Py ice4_warm reproduit fidèlement PHYEX-IAL_CY50T1")
    print("="*80)
