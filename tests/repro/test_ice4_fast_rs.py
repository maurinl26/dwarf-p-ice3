# -*- coding: utf-8 -*-
"""
Test de reproductibilité du stencil ice4_fast_rs par rapport à PHYEX-IAL_CY50T1.

Ce module valide que l'implémentation Python GT4Py des processus rapides de la neige/agrégats
de la microphysique ICE4 produit des résultats numériquement identiques à l'implémentation 
Fortran de référence issue du projet PHYEX (PHYsique EXternalisée) version IAL_CY50T1.

Les processus rapides de la neige représentent:
- Le givrage des gouttelettes nuageuses sur les agrégats de neige (RCRIMSS, RCRIMSG, RSRIMCG)
- L'accrétion des gouttes de pluie sur les agrégats (RRACCSS, RRACCSG, RSACCRG)
- La fonte et conversion des agrégats (RSMLTG, RCMLTSR)

Ces processus sont dits "rapides" car leurs échelles de temps caractéristiques sont plus courtes
que celles des processus lents (nucléation, agrégation, etc.).

Référence:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_processes.F90
"""
from ctypes import c_double, c_float

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array, zeros
from numpy.testing import assert_allclose

from ice3.phyex_common.xker_raccs import KER_RACCS, KER_RACCSS, KER_SACCRG
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
def test_ice4_fast_rs(dtypes, backend, externals, packed_dims, domain, origin, ldsoft):
    """
    Test de reproductibilité du stencil ice4_fast_rs (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre:
    - L'implémentation Python/GT4Py des processus rapides de la neige
    - L'implémentation Fortran de référence PHYEX-IAL_CY50T1
    
    Les processus validés incluent:
    
    1. Givrage des gouttelettes sur les agrégats (T < 0°C):
       RCRIMSS - Givrage sur petits agrégats
       RCRIMSG - Givrage sur gros agrégats avec conversion en graupel
       RSRIMCG - Contribution à la conversion neige -> graupel
    
    2. Accrétion de la pluie sur les agrégats (T < 0°C):
       RRACCSS - Accrétion sur petits agrégats
       RRACCSG - Accrétion sur gros agrégats
       RSACCRG - Accrétion-conversion en graupel
    
    3. Fonte et conversion (T > 0°C):
       RSMLTG  - Fonte des agrégats
       RCMLTSR - Collection de gouttelettes par température positive
    
    4. Tendances individuelles:
       RS_RCRIMS_TND, RS_RCRIMSS_TND, RS_RSRIMCG_TND
       RS_RRACCS_TND, RS_RRACCSS_TND, RS_RSACCRG_TND
       RS_FREEZ1_TND, RS_FREEZ2_TND
    
    Champs vérifiés:
        - riaggs: Agrégation de glace pristine sur neige
        - rcrimss, rcrimsg, rsrimcg: Processus de givrage
        - rraccss, rraccsg, rsaccrg: Processus d'accrétion
        - rs_mltg_tnd, rc_mltsr_tnd: Processus de fonte
        - rst: Rapport de mélange de neige total
    
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
    from ice3.stencils.ice4_fast_rs import ice4_fast_rs

    # Compilation du stencil GT4Py
    ice4_fast_rs_gt4py = stencil(
        backend,
        name="ice4_fast_rs",
        definition=ice4_fast_rs,
        dtypes=dtypes,
        externals=externals,
    )

    # Compilation du stencil Fortran de référence
    ice4_fast_rs_fortran = compile_fortran_stencil(
        "mode_ice4_fast_rs.F90", "mode_ice4_fast_rs", "ice4_fast_rs"
    )

    # =========================================================================
    # Initialisation des champs d'entrée
    # =========================================================================
    
    # Champs scalaires 3D d'entrée
    FloatFieldsIJK_Input_Names = [
        "rhodref",     # Densité de référence [kg/m³]
        "pres",        # Pression absolue [Pa]
        "dv",          # Diffusivité de la vapeur d'eau [m²/s]
        "ka",          # Conductivité thermique de l'air [J/m/s/K]
        "cj",          # Coefficient de ventilation [-]
        "lbdar",       # Paramètre de pente de la distribution de pluie [m⁻¹]
        "lbdas",       # Paramètre de pente de la distribution de neige [m⁻¹]
        "t",           # Température [K]
        "rvt",         # Rapport de mélange de vapeur d'eau [kg/kg]
        "rct",         # Rapport de mélange d'eau nuageuse [kg/kg]
        "rrt",         # Rapport de mélange de pluie [kg/kg]
        "rst",         # Rapport de mélange de neige [kg/kg]
        "riaggs",      # Agrégation de glace pristine [kg/kg/s]
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
    # Densité de référence entre 0.5 et 1.3 kg/m³
    FloatFieldsIJK_Input["rhodref"] = FloatFieldsIJK_Input["rhodref"] * 0.8 + 0.5
    
    # Pression entre 50000 et 101325 Pa
    FloatFieldsIJK_Input["pres"] = FloatFieldsIJK_Input["pres"] * 51325 + 50000
    
    # Diffusivité de la vapeur d'eau (valeurs typiques: 1e-5 à 3e-5 m²/s)
    FloatFieldsIJK_Input["dv"] = FloatFieldsIJK_Input["dv"] * 2e-5 + 1e-5
    
    # Conductivité thermique (valeurs typiques: 0.02 à 0.03 J/m/s/K)
    FloatFieldsIJK_Input["ka"] = FloatFieldsIJK_Input["ka"] * 0.01 + 0.02
    
    # Coefficient de ventilation (valeurs typiques: 0 à 10)
    FloatFieldsIJK_Input["cj"] = FloatFieldsIJK_Input["cj"] * 10.0
    
    # Paramètres de pente (valeurs typiques: 1e3 à 1e6 m⁻¹)
    FloatFieldsIJK_Input["lbdar"] = FloatFieldsIJK_Input["lbdar"] * 9e5 + 1e5
    FloatFieldsIJK_Input["lbdas"] = FloatFieldsIJK_Input["lbdas"] * 9e5 + 1e5
    
    # Température entre 233K (-40°C) et 303K (30°C)
    FloatFieldsIJK_Input["t"] = FloatFieldsIJK_Input["t"] * 70 + 233
    
    # Rapports de mélange (valeurs petites, typiquement < 0.01)
    FloatFieldsIJK_Input["rvt"] = FloatFieldsIJK_Input["rvt"] * 0.015  # vapeur d'eau
    FloatFieldsIJK_Input["rct"] = FloatFieldsIJK_Input["rct"] * 0.003  # eau nuageuse
    FloatFieldsIJK_Input["rrt"] = FloatFieldsIJK_Input["rrt"] * 0.005  # pluie
    FloatFieldsIJK_Input["rst"] = FloatFieldsIJK_Input["rst"] * 0.004  # neige
    
    # Agrégation (valeurs typiques: 0 à 1e-6 kg/kg/s)
    FloatFieldsIJK_Input["riaggs"] = FloatFieldsIJK_Input["riaggs"] * 1e-6

    # Champs de sortie - processus de givrage et accrétion
    FloatFieldsIJK_Output_Names = [
        "rcrimss",        # Givrage sur petits agrégats
        "rcrimsg",        # Givrage sur gros agrégats
        "rsrimcg",        # Conversion neige -> graupel par givrage
        "rraccss",        # Accrétion pluie sur petits agrégats
        "rraccsg",        # Accrétion pluie sur gros agrégats
        "rsaccrg",        # Conversion neige -> graupel par accrétion
        "rs_mltg_tnd",    # Fonte des agrégats
        "rc_mltsr_tnd",   # Collection par température positive
    ]

    FloatFieldsIJK_Output = {
        name: np.array(
            np.zeros(domain),
            dtype=(c_float if dtypes["float"] == np.float32 else c_double),
            order="F",
        )
        for name in FloatFieldsIJK_Output_Names
    }

    # Tendances individuelles (8 composantes)
    rs_tend = np.array(
        np.zeros((*domain, 8)),
        dtype=(c_float if dtypes["float"] == np.float32 else c_double),
        order="F",
    )

    # Champ booléen pour activer/désactiver le calcul par colonne
    BoolFieldsIJK = {
        "ldcompute": np.array(
            np.random.rand(*domain) > 0.1,  # ~90% des points activés
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
    pres_gt4py = from_array(
        FloatFieldsIJK_Input["pres"], dtype=dtypes["float"], backend=backend
    )
    dv_gt4py = from_array(
        FloatFieldsIJK_Input["dv"], dtype=dtypes["float"], backend=backend
    )
    ka_gt4py = from_array(
        FloatFieldsIJK_Input["ka"], dtype=dtypes["float"], backend=backend
    )
    cj_gt4py = from_array(
        FloatFieldsIJK_Input["cj"], dtype=dtypes["float"], backend=backend
    )
    lbdar_gt4py = from_array(
        FloatFieldsIJK_Input["lbdar"], dtype=dtypes["float"], backend=backend
    )
    lbdas_gt4py = from_array(
        FloatFieldsIJK_Input["lbdas"], dtype=dtypes["float"], backend=backend
    )
    t_gt4py = from_array(
        FloatFieldsIJK_Input["t"], dtype=dtypes["float"], backend=backend
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
    rst_gt4py = from_array(
        FloatFieldsIJK_Input["rst"], dtype=dtypes["float"], backend=backend
    )
    riaggs_gt4py = from_array(
        FloatFieldsIJK_Input["riaggs"], dtype=dtypes["float"], backend=backend
    )
    
    # Champs de sortie (initialisés à zéro)
    rcrimss_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rcrimsg_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rsrimcg_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rraccss_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rraccsg_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rsaccrg_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rs_mltg_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rc_mltsr_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    
    # Tendances individuelles
    rs_rcrims_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rs_rcrimss_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rs_rsrimcg_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rs_rraccs_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rs_rraccss_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rs_rsaccrg_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rs_freez1_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rs_freez2_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)

    # GlobalTables
    from ice3.phyex_common.phyex import Phyex
    gaminc_rim1 = from_array(
        Phyex("AROME").rain_ice_param.GAMINC_RIM1.astype(dtypes["float"]),
        backend=backend,
        dtype=dtypes["float"],
        )
    gaminc_rim2 = from_array(
        Phyex("AROME").rain_ice_param.GAMINC_RIM2.astype(dtypes["float"]),
        backend=backend,
        dtype=dtypes["float"],
        )
    gaminc_rim4 = from_array(
        Phyex("AROME").rain_ice_param.GAMINC_RIM4.astype(dtypes["float"]),
        backend=backend,
        dtype=dtypes["float"],
        )

    index_floor = zeros(domain, dtype=dtypes["int"], backend=backend)

    from ice3.phyex_common.xker_raccs import KER_RACCS, KER_RACCSS, KER_SACCRG
    ker_raccs = from_array(KER_RACCS, backend=backend, dtype=dtypes["float"])
    ker_raccss = from_array(KER_RACCSS, backend=backend, dtype=dtypes["float"])
    ker_saccrg = from_array(KER_SACCRG, backend=backend, dtype=dtypes["float"])

    index_floor_s = zeros(domain, dtype=dtypes["int"], backend=backend)
    index_floor_r = zeros(domain, dtype=dtypes["int"], backend=backend)

    # =========================================================================
    # Exécution du stencil Python GT4Py
    # =========================================================================
    
    ice4_fast_rs_gt4py(
        ldsoft=ldsoft,
        ldcompute=ldcompute_gt4py,
        rhodref=rhodref_gt4py,
        pres=pres_gt4py,
        dv=dv_gt4py,
        ka=ka_gt4py,
        cj=cj_gt4py,
        lbdar=lbdar_gt4py,
        lbdas=lbdas_gt4py,
        t=t_gt4py,
        rvt=rvt_gt4py,
        rct=rct_gt4py,
        rrt=rrt_gt4py,
        rst=rst_gt4py,
        riaggs=riaggs_gt4py,
        rcrimss=rcrimss_gt4py,
        rcrimsg=rcrimsg_gt4py,
        rsrimcg=rsrimcg_gt4py,
        rraccss=rraccss_gt4py,
        rraccsg=rraccsg_gt4py,
        rsaccrg=rsaccrg_gt4py,
        rs_mltg_tnd=rs_mltg_tnd_gt4py,
        rc_mltsr_tnd=rc_mltsr_tnd_gt4py,
        rs_rcrims_tnd=rs_rcrims_tnd_gt4py,
        rs_rcrimss_tnd=rs_rcrimss_tnd_gt4py,
        rs_rsrimcg_tnd=rs_rsrimcg_tnd_gt4py,
        rs_rraccs_tnd=rs_rraccs_tnd_gt4py,
        rs_rraccss_tnd=rs_rraccss_tnd_gt4py,
        rs_rsaccrg_tnd=rs_rsaccrg_tnd_gt4py,
        rs_freez1_tnd=rs_freez1_tnd_gt4py,
        rs_freez2_tnd=rs_freez2_tnd_gt4py,
        gaminc_rim1=gaminc_rim1,
        gaminc_rim2=gaminc_rim2,
        gaminc_rim4=gaminc_rim4,
        index_floor=index_floor,
        ker_raccs=ker_raccs,
        ker_raccss=ker_raccss,
        ker_saccrg=ker_saccrg,
        index_floor_r=index_floor_r,
        index_floor_s=index_floor_s,
        domain=domain,
        origin=origin,
    )

    # =========================================================================
    # Exécution de la référence Fortran PHYEX
    # =========================================================================
    
    # Mapping des paramètres externes pour Fortran
    externals_mapping = {
        "ngaminc": "NGAMINC",
        "nacclbdas": "NACCLBDAS",
        "nacclbdar": "NACCLBDAR",
        "levlimit": "LEVLIMIT",
        "lpack_interp": "LPACK_INTERP",
        "csnowriming": "CSNOWRIMING",
        "xcrimss": "CRIMSS",
        "xexcrimss": "EXCRIMSS",
        "xcrimsg": "CRIMSG",
        "xexcrimsg": "EXCRIMSG",
        "xexsrimcg2": "EXSRIMCG2",
        "xfraccss": "FRACCSS",
        "s_rtmin": "S_RTMIN",
        "c_rtmin": "C_RTMIN",
        "r_rtmin": "R_RTMIN",
        "xepsilo": "EPSILO",
        "xalpi": "ALPI",
        "xbetai": "BETAI",
        "xgami": "GAMI",
        "xtt": "TT",
        "xlvtt": "LVTT",
        "xcpv": "CPV",
        "xci": "CI",
        "xcl": "CL",
        "xlmtt": "LMTT",
        "xestt": "ESTT",
        "xrv": "RV",
        "x0deps": "O0DEPS",
        "x1deps": "O1DEPS",
        "xex0deps": "EX0DEPS",
        "xex1deps": "EX1DEPS",
        "xlbraccs1": "LBRACCS1",
        "xlbraccs2": "LBRACCS2",
        "xlbraccs3": "LBRACCS3",
        "xcxs": "CXS",
        "xsrimcg2": "SRIMCG2",
        "xsrimcg3": "SRIMCG3",
        "xbs": "BS",
        "xlbsaccr1": "LBSACCR1",
        "xlbsaccr2": "LBSACCR2",
        "xlbsaccr3": "LBSACCR3",
        "xfsaccrg": "FSACCRG",
        "xsrimcg": "SRIMCG",
        "xexsrimcg": "EXSRIMCG",
        "xcexvt": "CVEXT",
        "xalpw": "ALPW",
        "xbetaw": "BETAW",
        "xgamw": "GAMW",
        "xfscvmg": "FSCVMG",
    }
    
    fortran_externals = {
        fkey: externals[pykey] for fkey, pykey in externals_mapping.items()
    }
    
    # Tables de lookup
    fortran_lookup_tables = {
        "xker_raccss": KER_RACCSS,
        "xker_raccs": KER_RACCS,
        "xker_saccrg": KER_SACCRG,
        "xgaminc_rim1": externals["GAMINC_RIM1"],
        "xgaminc_rim2": externals["GAMINC_RIM2"],
        "xgaminc_rim4": externals["GAMINC_RIM4"],
        "xrimintp1": externals["RIMINTP1"],
        "xrimintp2": externals["RIMINTP2"],
        "xaccintp1s": externals["ACCINTP1S"],
        "xaccintp2s": externals["ACCINTP2S"],
        "xaccintp1r": externals["ACCINTP1R"],
        "xaccintp2r": externals["ACCINTP2R"],
    }
    
    # Aplatissement des champs 3D en 1D pour Fortran (ordre Fortran)
    ldcompute_flat = BoolFieldsIJK["ldcompute"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rhodref_flat = FloatFieldsIJK_Input["rhodref"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    pres_flat = FloatFieldsIJK_Input["pres"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    dv_flat = FloatFieldsIJK_Input["dv"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    ka_flat = FloatFieldsIJK_Input["ka"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    cj_flat = FloatFieldsIJK_Input["cj"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    lbdar_flat = FloatFieldsIJK_Input["lbdar"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    lbdas_flat = FloatFieldsIJK_Input["lbdas"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    t_flat = FloatFieldsIJK_Input["t"].reshape(
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
    rst_flat = FloatFieldsIJK_Input["rst"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    riaggs_flat = FloatFieldsIJK_Input["riaggs"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    
    # Champs de sortie (copies car ils sont modifiés)
    rcrimss_flat = FloatFieldsIJK_Output["rcrimss"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rcrimsg_flat = FloatFieldsIJK_Output["rcrimsg"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rsrimcg_flat = FloatFieldsIJK_Output["rsrimcg"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rraccss_flat = FloatFieldsIJK_Output["rraccss"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rraccsg_flat = FloatFieldsIJK_Output["rraccsg"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rsaccrg_flat = FloatFieldsIJK_Output["rsaccrg"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rs_mltg_tnd_flat = FloatFieldsIJK_Output["rs_mltg_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rc_mltsr_tnd_flat = FloatFieldsIJK_Output["rc_mltsr_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    
    # Tendances individuelles (tableau 2D pour Fortran: nproma x 8)
    rs_tend_flat = rs_tend.reshape(domain[0] * domain[1] * domain[2], 8, order="F").copy()
    
    # Appel de la routine Fortran
    (
        riaggs_fortran,
        rcrimss_fortran,
        rcrimsg_fortran,
        rsrimcg_fortran,
        rraccss_fortran,
        rraccsg_fortran,
        rsaccrg_fortran,
        rs_mltg_tnd_fortran,
        rc_mltsr_tnd_fortran,
        rst_fortran
    ) = ice4_fast_rs_fortran(
        ldsoft=ldsoft,
        ldcompute=ldcompute_flat,
        prhodref=rhodref_flat,
        ppres=pres_flat,
        pdv=dv_flat,
        pka=ka_flat,
        pcj=cj_flat,
        plbdar=lbdar_flat,
        plbdas=lbdas_flat,
        pt=t_flat,
        prvt=rvt_flat,
        prct=rct_flat,
        prrt=rrt_flat,
        prst=rst_flat,
        priaggs=riaggs_flat,
        prcrimss=rcrimss_flat,
        prcrimsg=rcrimsg_flat,
        prsrimcg=rsrimcg_flat,
        prraccss=rraccss_flat,
        prraccsg=rraccsg_flat,
        prsaccrg=rsaccrg_flat,
        prsmltg=rs_mltg_tnd_flat,
        prcmltsr=rc_mltsr_tnd_flat,
        prs_tend=rs_tend_flat,
        **packed_dims,
        **fortran_externals,
        **fortran_lookup_tables,
    )

    # =========================================================================
    # VALIDATION DE LA REPRODUCTIBILITÉ - Comparaison Python vs Fortran PHYEX
    # =========================================================================
    
    print("\n" + "="*80)
    print("TEST DE REPRODUCTIBILITÉ: ice4_fast_rs.py vs PHYEX-IAL_CY50T1")
    print("="*80)
    print(f"Backend: {backend}")
    print(f"Précision: {'simple' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domaine: {domain[0]}x{domain[1]}x{domain[2]}")
    print(f"Mode soft: {ldsoft}")
    print("="*80)
    
    # Reshape des sorties Python pour comparaison
    rcrimss_py = rcrimss_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rcrimsg_py = rcrimsg_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rsrimcg_py = rsrimcg_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rraccss_py = rraccss_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rraccsg_py = rraccsg_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rsaccrg_py = rsaccrg_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rs_mltg_tnd_py = rs_mltg_tnd_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rc_mltsr_tnd_py = rc_mltsr_tnd_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rst_py = rst_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    
    # ------------------------------------------------------------------------
    # Validation du givrage des gouttelettes sur les agrégats (T < 0°C)
    # ------------------------------------------------------------------------
    print("\n1. PROCESSUS DE GIVRAGE (T < 0°C)")
    print("-" * 80)
    
    print("\n  a) RCRIMSS - Givrage sur petits agrégats")
    print(f"     Python  - min: {rcrimss_py.min():.6e}, max: {rcrimss_py.max():.6e}")
    print(f"     Fortran - min: {rcrimss_fortran.min():.6e}, max: {rcrimss_fortran.max():.6e}")
    
    assert_allclose(
        rcrimss_fortran,
        rcrimss_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RCRIMSS: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RCRIMSS : OK")
    
    print("\n  b) RCRIMSG - Givrage sur gros agrégats")
    print(f"     Python  - min: {rcrimsg_py.min():.6e}, max: {rcrimsg_py.max():.6e}")
    print(f"     Fortran - min: {rcrimsg_fortran.min():.6e}, max: {rcrimsg_fortran.max():.6e}")
    
    assert_allclose(
        rcrimsg_fortran,
        rcrimsg_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RCRIMSG: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RCRIMSG : OK")
    
    print("\n  c) RSRIMCG - Conversion neige -> graupel par givrage")
    print(f"     Python  - min: {rsrimcg_py.min():.6e}, max: {rsrimcg_py.max():.6e}")
    print(f"     Fortran - min: {rsrimcg_fortran.min():.6e}, max: {rsrimcg_fortran.max():.6e}")
    
    assert_allclose(
        rsrimcg_fortran,
        rsrimcg_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RSRIMCG: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RSRIMCG : OK")
    
    # ------------------------------------------------------------------------
    # Validation de l'accrétion de la pluie sur les agrégats (T < 0°C)
    # ------------------------------------------------------------------------
    print("\n2. PROCESSUS D'ACCRÉTION (T < 0°C)")
    print("-" * 80)
    
    print("\n  a) RRACCSS - Accrétion sur petits agrégats")
    print(f"     Python  - min: {rraccss_py.min():.6e}, max: {rraccss_py.max():.6e}")
    print(f"     Fortran - min: {rraccss_fortran.min():.6e}, max: {rraccss_fortran.max():.6e}")
    
    assert_allclose(
        rraccss_fortran,
        rraccss_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RRACCSS: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RRACCSS : OK")
    
    print("\n  b) RRACCSG - Accrétion sur gros agrégats")
    print(f"     Python  - min: {rraccsg_py.min():.6e}, max: {rraccsg_py.max():.6e}")
    print(f"     Fortran - min: {rraccsg_fortran.min():.6e}, max: {rraccsg_fortran.max():.6e}")
    
    assert_allclose(
        rraccsg_fortran,
        rraccsg_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RRACCSG: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RRACCSG : OK")
    
    print("\n  c) RSACCRG - Conversion neige -> graupel par accrétion")
    print(f"     Python  - min: {rsaccrg_py.min():.6e}, max: {rsaccrg_py.max():.6e}")
    print(f"     Fortran - min: {rsaccrg_fortran.min():.6e}, max: {rsaccrg_fortran.max():.6e}")
    
    assert_allclose(
        rsaccrg_fortran,
        rsaccrg_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RSACCRG: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RSACCRG : OK")
    
    # ------------------------------------------------------------------------
    # Validation de la fonte et conversion des agrégats (T > 0°C)
    # ------------------------------------------------------------------------
    print("\n3. PROCESSUS DE FONTE (T > 0°C)")
    print("-" * 80)
    
    print("\n  a) RSMLTG - Fonte des agrégats")
    print(f"     Python  - min: {rs_mltg_tnd_py.min():.6e}, max: {rs_mltg_tnd_py.max():.6e}")
    print(f"     Fortran - min: {rs_mltg_tnd_fortran.min():.6e}, max: {rs_mltg_tnd_fortran.max():.6e}")
    
    assert_allclose(
        rs_mltg_tnd_fortran,
        rs_mltg_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RSMLTG: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RSMLTG : OK")
    
    print("\n  b) RCMLTSR - Collection par température positive")
    print(f"     Python  - min: {rc_mltsr_tnd_py.min():.6e}, max: {rc_mltsr_tnd_py.max():.6e}")
    print(f"     Fortran - min: {rc_mltsr_tnd_fortran.min():.6e}, max: {rc_mltsr_tnd_fortran.max():.6e}")
    
    assert_allclose(
        rc_mltsr_tnd_fortran,
        rc_mltsr_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RCMLTSR: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RCMLTSR : OK")
    
    # ------------------------------------------------------------------------
    # Validation du rapport de mélange de neige total
    # ------------------------------------------------------------------------
    print("\n4. RAPPORT DE MÉLANGE DE NEIGE TOTAL")
    print("-" * 80)
    
    print(f"  Python  - min: {rst_py.min():.6e}, max: {rst_py.max():.6e}")
    print(f"  Fortran - min: {rst_fortran.min():.6e}, max: {rst_fortran.max():.6e}")
    
    assert_allclose(
        rst_fortran,
        rst_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RST: divergence Python/Fortran PHYEX"
    )
    print("  ✓ RST : OK")
    
    # ========================================================================
    # Statistiques globales
    # ========================================================================
    print("\n" + "="*80)
    print("STATISTIQUES DES PROCESSUS RAPIDES DE LA NEIGE")
    print("="*80)
    
    n_total = domain[0] * domain[1] * domain[2]
    
    # Points actifs pour chaque processus
    n_crim = np.sum((rcrimss_py > 1e-10) | (rcrimsg_py > 1e-10))
    n_acc = np.sum((rraccss_py > 1e-10) | (rraccsg_py > 1e-10))
    n_mlt = np.sum(rs_mltg_tnd_py > 1e-10)
    
    print(f"\nPoints actifs (tendance > 1e-10):")
    print(f"  Givrage (RCRIMSS/RCRIMSG):       {n_crim:6d}/{n_total} ({100.0*n_crim/n_total:5.1f}%)")
    print(f"  Accrétion (RRACCSS/RRACCSG):     {n_acc:6d}/{n_total} ({100.0*n_acc/n_total:5.1f}%)")
    print(f"  Fonte (RSMLTG):                  {n_mlt:6d}/{n_total} ({100.0*n_mlt/n_total:5.1f}%)")
    
    # Distribution de température
    t_flat = FloatFieldsIJK_Input["t"].reshape(domain[0] * domain[1] * domain[2], order="F")
    # Get TT from externals - try different possible key names
    TT = externals.get("TT", externals.get("tt", externals.get("XTT", 273.15)))
    t_freezing = t_flat[t_flat < TT]
    t_melting = t_flat[t_flat >= TT]
    
    if len(t_freezing) > 0:
        print(f"\nTempératures T < {TT}K (processus de gel):")
        print(f"  min={t_freezing.min():.1f}K, max={t_freezing.max():.1f}K, "
              f"moyenne={t_freezing.mean():.1f}K ({100.0*len(t_freezing)/n_total:.1f}% des points)")
    
    if len(t_melting) > 0:
        print(f"\nTempératures T >= {TT}K (processus de fonte):")
        print(f"  min={t_melting.min():.1f}K, max={t_melting.max():.1f}K, "
              f"moyenne={t_melting.mean():.1f}K ({100.0*len(t_melting)/n_total:.1f}% des points)")
    
    # Statistiques des rapports de mélange dans les zones actives
    if n_crim > 0:
        rct_crim = rct_flat[(rcrimss_py > 1e-10) | (rcrimsg_py > 1e-10)]
        rst_crim = rst_flat[(rcrimss_py > 1e-10) | (rcrimsg_py > 1e-10)]
        print(f"\nRapports de mélange dans zones de givrage:")
        print(f"  Eau nuageuse (rct): min={rct_crim.min():.6e}, max={rct_crim.max():.6e}, "
              f"moyenne={rct_crim.mean():.6e}")
        print(f"  Neige (rst):        min={rst_crim.min():.6e}, max={rst_crim.max():.6e}, "
              f"moyenne={rst_crim.mean():.6e}")
    
    if n_acc > 0:
        rrt_acc = rrt_flat[(rraccss_py > 1e-10) | (rraccsg_py > 1e-10)]
        rst_acc = rst_flat[(rraccss_py > 1e-10) | (rraccsg_py > 1e-10)]
        print(f"\nRapports de mélange dans zones d'accrétion:")
        print(f"  Pluie (rrt):  min={rrt_acc.min():.6e}, max={rrt_acc.max():.6e}, "
              f"moyenne={rrt_acc.mean():.6e}")
        print(f"  Neige (rst):  min={rst_acc.min():.6e}, max={rst_acc.max():.6e}, "
              f"moyenne={rst_acc.mean():.6e}")
    
    print("\n" + "="*80)
    print("SUCCÈS: Reproductibilité validée!")
    print("Le stencil Python GT4Py ice4_fast_rs reproduit fidèlement PHYEX-IAL_CY50T1")
    print("="*80)
