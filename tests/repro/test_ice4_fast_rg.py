# -*- coding: utf-8 -*-
"""
Test de reproductibilité du stencil ice4_fast_rg par rapport à PHYEX-IAL_CY50T1.

Ce module valide que l'implémentation Python GT4Py des processus rapides du graupel
de la microphysique ICE4 produit des résultats numériquement identiques à l'implémentation 
Fortran de référence issue du projet PHYEX (PHYsique EXternalisée) version IAL_CY50T1.

Les processus rapides du graupel représentent:
- Le givrage par congélation de contact de la pluie (RICFRRG, RRCFRIG, RICFRR)
- La croissance du graupel par collection (RCDRY, RIDRY, RSDRY, RRDRY)
- La croissance humide du graupel (RIWET, RSWET)
- Les taux de congélation (FREEZ1, FREEZ2)
- La fonte du graupel (RGMLTR)

Ces processus sont dits "rapides" car leurs échelles de temps caractéristiques sont plus courtes
que celles des processus lents (nucléation, agrégation, etc.).

Référence:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_rg.F90
"""
from ctypes import c_double, c_float

import numpy as np
import pytest
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import from_array, zeros
from numpy.testing import assert_allclose

from ice3.phyex_common.xker_raccs import KER_SACCRG
from ice3.phyex_common.xker_rdryg import KER_RDRYG
from ice3.phyex_common.xker_sdryg import KER_SDRYG
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
def test_ice4_fast_rg(dtypes, backend, externals, packed_dims, domain, origin, ldsoft):
    """
    Test de reproductibilité du stencil ice4_fast_rg (PHYEX-IAL_CY50T1).
    
    Ce test valide la correspondance bit-à-bit (à la tolérance numérique près) entre:
    - L'implémentation Python/GT4Py des processus rapides du graupel
    - L'implémentation Fortran de référence PHYEX-IAL_CY50T1
    
    Les processus validés incluent:
    
    1. Congélation de contact de la pluie (T < 0°C):
       RICFRRG - Congélation de contact (glace pristine)
       RRCFRIG - Congélation de contact (pluie)
       RICFRR  - Congélation de contact limitée
    
    2. Croissance sèche du graupel (T < 0°C):
       RG_RCDRY_TND - Collection de gouttelettes nuageuses
       RG_RIDRY_TND - Collection de glace pristine
       RG_RSDRY_TND - Collection de neige
       RG_RRDRY_TND - Collection de pluie
    
    3. Croissance humide du graupel (T < 0°C):
       RG_RIWET_TND - Croissance humide (glace)
       RG_RSWET_TND - Croissance humide (neige)
    
    4. Taux de congélation:
       RG_FREEZ1_TND - Taux de congélation 1
       RG_FREEZ2_TND - Taux de congélation 2
    
    5. Fonte du graupel (T > 0°C):
       RGMLTR - Fonte du graupel en pluie
    
    Champs vérifiés:
        - ricfrrg, rrcfrig, ricfrr: Congélation de contact
        - rg_rcdry_tnd, rg_ridry_tnd, rg_rsdry_tnd, rg_rrdry_tnd: Croissance sèche
        - rg_riwet_tnd, rg_rswet_tnd: Croissance humide
        - rg_freez1_tnd, rg_freez2_tnd: Taux de congélation
        - rgmltr: Fonte
    
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
    from ice3.stencils.ice4_fast_rg import ice4_fast_rg

    # Compilation du stencil GT4Py
    ice4_fast_rg_gt4py = stencil(
        backend,
        name="ice4_fast_rg",
        definition=ice4_fast_rg,
        dtypes=dtypes,
        externals=externals,
    )

    # Compilation du stencil Fortran de référence
    ice4_fast_rg_fortran = compile_fortran_stencil(
        "mode_ice4_fast_rg.F90", "mode_ice4_fast_rg", "ice4_fast_rg"
    )

    # =========================================================================
    # Initialisation des champs d'entrée
    # =========================================================================
    
    # Champs scalaires 3D d'entrée
    FloatFieldsIJK_Input_Names = [
        "t",           # Température [K]
        "rhodref",     # Densité de référence [kg/m³]
        "pres",        # Pression absolue [Pa]
        "rvt",         # Rapport de mélange de vapeur d'eau [kg/kg]
        "rrt",         # Rapport de mélange de pluie [kg/kg]
        "rit",         # Rapport de mélange de glace pristine [kg/kg]
        "rgt",         # Rapport de mélange de graupel [kg/kg]
        "rct",         # Rapport de mélange d'eau nuageuse [kg/kg]
        "rst",         # Rapport de mélange de neige [kg/kg]
        "cit",         # Concentration de glace pristine [#/m³]
        "ka",          # Conductivité thermique de l'air [J/m/s/K]
        "dv",          # Diffusivité de la vapeur d'eau [m²/s]
        "cj",          # Coefficient de ventilation [-]
        "lbdar",       # Paramètre de pente de la distribution de pluie [m⁻¹]
        "lbdas",       # Paramètre de pente de la distribution de neige [m⁻¹]
        "lbdag",       # Paramètre de pente de la distribution de graupel [m⁻¹]
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
    # Température entre 233K (-40°C) et 303K (30°C)
    FloatFieldsIJK_Input["t"] = FloatFieldsIJK_Input["t"] * 70 + 233
    
    # Densité de référence entre 0.5 et 1.3 kg/m³
    FloatFieldsIJK_Input["rhodref"] = FloatFieldsIJK_Input["rhodref"] * 0.8 + 0.5
    
    # Pression entre 50000 et 101325 Pa
    FloatFieldsIJK_Input["pres"] = FloatFieldsIJK_Input["pres"] * 51325 + 50000
    
    # Rapports de mélange (valeurs petites, typiquement < 0.01)
    FloatFieldsIJK_Input["rvt"] = FloatFieldsIJK_Input["rvt"] * 0.015  # vapeur d'eau
    FloatFieldsIJK_Input["rrt"] = FloatFieldsIJK_Input["rrt"] * 0.005  # pluie
    FloatFieldsIJK_Input["rit"] = FloatFieldsIJK_Input["rit"] * 0.002  # glace pristine
    FloatFieldsIJK_Input["rgt"] = FloatFieldsIJK_Input["rgt"] * 0.006  # graupel
    FloatFieldsIJK_Input["rct"] = FloatFieldsIJK_Input["rct"] * 0.003  # eau nuageuse
    FloatFieldsIJK_Input["rst"] = FloatFieldsIJK_Input["rst"] * 0.004  # neige
    
    # Concentration de glace pristine (valeurs typiques: 1e3 à 1e6 #/m³)
    FloatFieldsIJK_Input["cit"] = FloatFieldsIJK_Input["cit"] * 9.99e5 + 1e3
    
    # Conductivité thermique (valeurs typiques: 0.02 à 0.03 J/m/s/K)
    FloatFieldsIJK_Input["ka"] = FloatFieldsIJK_Input["ka"] * 0.01 + 0.02
    
    # Diffusivité de la vapeur d'eau (valeurs typiques: 1e-5 à 3e-5 m²/s)
    FloatFieldsIJK_Input["dv"] = FloatFieldsIJK_Input["dv"] * 2e-5 + 1e-5
    
    # Coefficient de ventilation (valeurs typiques: 0 à 10)
    FloatFieldsIJK_Input["cj"] = FloatFieldsIJK_Input["cj"] * 10.0
    
    # Paramètres de pente (valeurs typiques: 1e3 à 1e6 m⁻¹)
    FloatFieldsIJK_Input["lbdar"] = FloatFieldsIJK_Input["lbdar"] * 9e5 + 1e5
    FloatFieldsIJK_Input["lbdas"] = FloatFieldsIJK_Input["lbdas"] * 9e5 + 1e5
    FloatFieldsIJK_Input["lbdag"] = FloatFieldsIJK_Input["lbdag"] * 9e5 + 1e5

    # Champs de sortie
    FloatFieldsIJK_Output_Names = [
        "ricfrrg",        # Congélation de contact (glace pristine)
        "rrcfrig",        # Congélation de contact (pluie)
        "ricfrr",         # Congélation de contact limitée
        "rg_rcdry_tnd",   # Croissance sèche (cloud)
        "rg_ridry_tnd",   # Croissance sèche (ice)
        "rg_rsdry_tnd",   # Croissance sèche (snow)
        "rg_rrdry_tnd",   # Croissance sèche (rain)
        "rg_riwet_tnd",   # Croissance humide (ice)
        "rg_rswet_tnd",   # Croissance humide (snow)
        "rg_freez1_tnd",  # Taux de congélation 1
        "rg_freez2_tnd",  # Taux de congélation 2
        "rgmltr",         # Fonte du graupel
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
            np.random.rand(*domain) > 0.1,  # ~90% des points activés
            dtype=np.bool_,
            order="F",
        )
    }

    # Champs d'index pour les tables de lookup
    IntFieldsIJK = {
        "index_floor_s": np.ones(domain, dtype=np.int32, order="F"),
        "index_floor_g": np.ones(domain, dtype=np.int32, order="F"),
        "index_floor_r": np.ones(domain, dtype=np.int32, order="F"),
    }

    # =========================================================================
    # Conversion en storages GT4Py
    # =========================================================================
    
    # Champs d'entrée
    ldcompute_gt4py = from_array(
        BoolFieldsIJK["ldcompute"], dtype=dtypes["bool"], backend=backend
    )
    t_gt4py = from_array(
        FloatFieldsIJK_Input["t"], dtype=dtypes["float"], backend=backend
    )
    rhodref_gt4py = from_array(
        FloatFieldsIJK_Input["rhodref"], dtype=dtypes["float"], backend=backend
    )
    pres_gt4py = from_array(
        FloatFieldsIJK_Input["pres"], dtype=dtypes["float"], backend=backend
    )
    rvt_gt4py = from_array(
        FloatFieldsIJK_Input["rvt"], dtype=dtypes["float"], backend=backend
    )
    rrt_gt4py = from_array(
        FloatFieldsIJK_Input["rrt"], dtype=dtypes["float"], backend=backend
    )
    rit_gt4py = from_array(
        FloatFieldsIJK_Input["rit"], dtype=dtypes["float"], backend=backend
    )
    rgt_gt4py = from_array(
        FloatFieldsIJK_Input["rgt"], dtype=dtypes["float"], backend=backend
    )
    rct_gt4py = from_array(
        FloatFieldsIJK_Input["rct"], dtype=dtypes["float"], backend=backend
    )
    rst_gt4py = from_array(
        FloatFieldsIJK_Input["rst"], dtype=dtypes["float"], backend=backend
    )
    cit_gt4py = from_array(
        FloatFieldsIJK_Input["cit"], dtype=dtypes["float"], backend=backend
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
    lbdar_gt4py = from_array(
        FloatFieldsIJK_Input["lbdar"], dtype=dtypes["float"], backend=backend
    )
    lbdas_gt4py = from_array(
        FloatFieldsIJK_Input["lbdas"], dtype=dtypes["float"], backend=backend
    )
    lbdag_gt4py = from_array(
        FloatFieldsIJK_Input["lbdag"], dtype=dtypes["float"], backend=backend
    )
    
    # Champs de sortie (initialisés à zéro)
    ricfrrg_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rrcfrig_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    ricfrr_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rg_rcdry_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rg_ridry_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rg_rsdry_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rg_rrdry_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rg_riwet_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rg_rswet_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rg_freez1_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rg_freez2_tnd_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    rgmltr_gt4py = zeros(domain, dtype=dtypes["float"], backend=backend)
    
    # Champs d'index
    index_floor_s_gt4py = from_array(
        IntFieldsIJK["index_floor_s"], dtype=dtypes["int"], backend=backend
    )
    index_floor_g_gt4py = from_array(
        IntFieldsIJK["index_floor_g"], dtype=dtypes["int"], backend=backend
    )
    index_floor_r_gt4py = from_array(
        IntFieldsIJK["index_floor_r"], dtype=dtypes["int"], backend=backend
    )

    # =========================================================================
    # Exécution du stencil Python GT4Py
    # =========================================================================
    
    ice4_fast_rg_gt4py(
        ldsoft=ldsoft,
        ldcompute=ldcompute_gt4py,
        t=t_gt4py,
        rhodref=rhodref_gt4py,
        pres=pres_gt4py,
        rvt=rvt_gt4py,
        rrt=rrt_gt4py,
        rit=rit_gt4py,
        rgt=rgt_gt4py,
        rct=rct_gt4py,
        rst=rst_gt4py,
        cit=cit_gt4py,
        ka=ka_gt4py,
        dv=dv_gt4py,
        cj=cj_gt4py,
        lbdar=lbdar_gt4py,
        lbdas=lbdas_gt4py,
        lbdag=lbdag_gt4py,
        ricfrrg=ricfrrg_gt4py,
        rrcfrig=rrcfrig_gt4py,
        ricfrr=ricfrr_gt4py,
        rg_rcdry_tnd=rg_rcdry_tnd_gt4py,
        rg_ridry_tnd=rg_ridry_tnd_gt4py,
        rg_rsdry_tnd=rg_rsdry_tnd_gt4py,
        rg_rrdry_tnd=rg_rrdry_tnd_gt4py,
        rg_riwet_tnd=rg_riwet_tnd_gt4py,
        rg_rswet_tnd=rg_rswet_tnd_gt4py,
        rg_freez1_tnd=rg_freez1_tnd_gt4py,
        rg_freez2_tnd=rg_freez2_tnd_gt4py,
        rgmltr=rgmltr_gt4py,
        ker_sdryg=KER_SDRYG,
        ker_rdryg=KER_RDRYG,
        index_floor_s=index_floor_s_gt4py,
        index_floor_g=index_floor_g_gt4py,
        index_floor_r=index_floor_r_gt4py,
        domain=domain,
        origin=origin,
    )

    # =========================================================================
    # Exécution de la référence Fortran PHYEX
    # =========================================================================
    
    # Mapping des paramètres externes pour Fortran
    externals_mapping = {
        "lpack_interp": "LPACK_INTERP",
        "xlbsdryg1": "LBSDRYG1",
        "xlbsdryg2": "LBSDRYG2",
        "xlbsdryg3": "LBSDRYG3",
        "xlbrdryg1": "LBRDRYG1",
        "xlbrdryg2": "LBRDRYG2",
        "xlbrdryg3": "LBRDRYG3",
        "xfcdryg": "FCDRYG",
        "xfidryg": "FIDRYG",
        "xfsdryg": "FSDRYG",
        "xfrdryg": "FRDRYG",
        "xcolexig": "COLEXIG",
        "xcolig": "COLIG",
        "xcolsg": "COLSG",
        "xcolexsg": "COLEXSG",
        "xexicfrr": "EXICFRR",
        "xexrcfri": "EXRCFRI",
        "xicfrr": "ICFRR",
        "xrcfri": "RCFRI",
        "lcrflimit": "LCRFLIMIT",
        "lnullwetg": "LNULLWETG",
        "lwetgpost": "LWETGPOST",
        "levlimit": "LEVLIMIT",
        "c_rtmin": "C_RTMIN",
        "g_rtmin": "G_RTMIN",
        "r_rtmin": "R_RTMIN",
        "i_rtmin": "I_RTMIN",
        "s_rtmin": "S_RTMIN",
        "xtt": "TT",
        "xlvtt": "LVTT",
        "xlmtt": "LMTT",
        "xcpv": "CPV",
        "xci": "CI",
        "xcl": "CL",
        "xepsilo": "EPSILO",
        "xestt": "ESTT",
        "xrv": "RV",
        "xalpi": "ALPI",
        "xbetai": "BETAI",
        "xgami": "GAMI",
        "xalpw": "ALPW",
        "xbetaw": "BETAW",
        "xgamw": "GAMW",
        "x0depg": "O0DEPG",
        "x1depg": "O1DEPG",
        "xex0depg": "EX0DEPG",
        "xex1depg": "EX1DEPG",
        "xcxg": "CXG",
        "xcxs": "CXS",
        "xdg": "DG",
        "xbs": "BS",
        "xcexvt": "CEXVT",
        "krr": "NRR",
        "ndrylbdag": "NDRYLBDAG",
        "ndrylbdas": "NDRYLBDAS",
        "ndrylbdar": "NDRYLBDAR"
    }
    
    fortran_externals = {
        fkey: externals[pykey] for fkey, pykey in externals_mapping.items()
    }
    
    # Tables de lookup
    fortran_lookup_tables = {
        "xker_sdryg": KER_SDRYG,
        "xker_rdryg": KER_RDRYG,
        "xdryintp1g": externals["DRYINTP1G"],
        "xdryintp2g": externals["DRYINTP2G"],
        "xdryintp1s": externals["DRYINTP1S"],
        "xdryintp2s": externals["DRYINTP2S"],
        "xdryintp1r": externals["DRYINTP1R"],
        "xdryintp2r": externals["DRYINTP2R"],
    }
    
    # Aplatissement des champs 3D en 1D pour Fortran (ordre Fortran)
    ldcompute_flat = BoolFieldsIJK["ldcompute"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    t_flat = FloatFieldsIJK_Input["t"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rhodref_flat = FloatFieldsIJK_Input["rhodref"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    pres_flat = FloatFieldsIJK_Input["pres"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rvt_flat = FloatFieldsIJK_Input["rvt"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rrt_flat = FloatFieldsIJK_Input["rrt"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rit_flat = FloatFieldsIJK_Input["rit"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rgt_flat = FloatFieldsIJK_Input["rgt"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rct_flat = FloatFieldsIJK_Input["rct"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    rst_flat = FloatFieldsIJK_Input["rst"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    cit_flat = FloatFieldsIJK_Input["cit"].reshape(
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
    lbdar_flat = FloatFieldsIJK_Input["lbdar"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    lbdas_flat = FloatFieldsIJK_Input["lbdas"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    lbdag_flat = FloatFieldsIJK_Input["lbdag"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    )
    
    # Champs de sortie (copies car ils sont modifiés)
    ricfrrg_flat = FloatFieldsIJK_Output["ricfrrg"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rrcfrig_flat = FloatFieldsIJK_Output["rrcfrig"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    ricfrr_flat = FloatFieldsIJK_Output["ricfrr"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rg_rcdry_tnd_flat = FloatFieldsIJK_Output["rg_rcdry_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rg_ridry_tnd_flat = FloatFieldsIJK_Output["rg_ridry_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rg_rsdry_tnd_flat = FloatFieldsIJK_Output["rg_rsdry_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rg_rrdry_tnd_flat = FloatFieldsIJK_Output["rg_rrdry_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rg_riwet_tnd_flat = FloatFieldsIJK_Output["rg_riwet_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rg_rswet_tnd_flat = FloatFieldsIJK_Output["rg_rswet_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rg_freez1_tnd_flat = FloatFieldsIJK_Output["rg_freez1_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rg_freez2_tnd_flat = FloatFieldsIJK_Output["rg_freez2_tnd"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    rgmltr_flat = FloatFieldsIJK_Output["rgmltr"].reshape(
        domain[0] * domain[1] * domain[2], order="F"
    ).copy()
    
    # Additional output arrays required by Fortran (but not used in Python GT4Py)
    prgsi_flat = np.zeros(domain[0] * domain[1] * domain[2], 
                          dtype=(c_float if dtypes["float"] == np.float32 else c_double),
                          order="F")
    prgsi_mr_flat = np.zeros(domain[0] * domain[1] * domain[2],
                             dtype=(c_float if dtypes["float"] == np.float32 else c_double),
                             order="F")
    prg_tend_flat = np.zeros((domain[0] * domain[1] * domain[2], 8),
                             dtype=(c_float if dtypes["float"] == np.float32 else c_double),
                             order="F")
    
    # Appel de la routine Fortran
    # Convert c_int to plain int for packed_dims
    packed_dims_int = {k: int(v) if hasattr(v, 'value') else v for k, v in packed_dims.items()}
    
    (
        ricfrrg_fortran,
        rrcfrig_fortran,
        ricfrr_fortran,
        rg_rcdry_tnd_fortran,
        rg_ridry_tnd_fortran,
        rg_rsdry_tnd_fortran,
        rg_rrdry_tnd_fortran,
        rg_riwet_tnd_fortran,
        rg_rswet_tnd_fortran,
        rg_freez1_tnd_fortran,
        rg_freez2_tnd_fortran,
        rgmltr_fortran,
    ) = ice4_fast_rg_fortran(
        ldsoft=ldsoft,
        ldcompute=ldcompute_flat,
        pt=t_flat,
        prhodref=rhodref_flat,
        ppres=pres_flat,
        prvt=rvt_flat,
        prrt=rrt_flat,
        prit=rit_flat,
        prgt=rgt_flat,
        prct=rct_flat,
        prst=rst_flat,
        pcit=cit_flat,
        pka=ka_flat,
        pdv=dv_flat,
        pcj=cj_flat,
        plbdar=lbdar_flat,
        plbdas=lbdas_flat,
        plbdag=lbdag_flat,
        pricfrrg=ricfrrg_flat,
        prrcfrig=rrcfrig_flat,
        pricfrr=ricfrr_flat,
        prgmltr=rgmltr_flat,
        prgsi=prgsi_flat,
        prgsi_mr=prgsi_mr_flat,
        prg_tend=prg_tend_flat,
        **packed_dims_int,
        **fortran_externals,
        **fortran_lookup_tables,
    )

    # =========================================================================
    # VALIDATION DE LA REPRODUCTIBILITÉ - Comparaison Python vs Fortran PHYEX
    # =========================================================================
    
    print("\n" + "="*80)
    print("TEST DE REPRODUCTIBILITÉ: ice4_fast_rg.py vs PHYEX-IAL_CY50T1")
    print("="*80)
    print(f"Backend: {backend}")
    print(f"Précision: {'simple' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domaine: {domain[0]}x{domain[1]}x{domain[2]}")
    print(f"Mode soft: {ldsoft}")
    print("="*80)
    
    # Reshape des sorties Python pour comparaison
    ricfrrg_py = ricfrrg_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rrcfrig_py = rrcfrig_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    ricfrr_py = ricfrr_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rg_rcdry_tnd_py = rg_rcdry_tnd_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rg_ridry_tnd_py = rg_ridry_tnd_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rg_rsdry_tnd_py = rg_rsdry_tnd_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rg_rrdry_tnd_py = rg_rrdry_tnd_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rg_riwet_tnd_py = rg_riwet_tnd_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rg_rswet_tnd_py = rg_rswet_tnd_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rg_freez1_tnd_py = rg_freez1_tnd_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rg_freez2_tnd_py = rg_freez2_tnd_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    rgmltr_py = rgmltr_gt4py.reshape(domain[0] * domain[1] * domain[2], order="F")
    
    # ------------------------------------------------------------------------
    # Validation de la congélation de contact de la pluie (T < 0°C)
    # ------------------------------------------------------------------------
    print("\n1. CONGÉLATION DE CONTACT DE LA PLUIE (T < 0°C)")
    print("-" * 80)
    
    print("\n  a) RICFRRG - Congélation de contact (glace pristine)")
    print(f"     Python  - min: {ricfrrg_py.min():.6e}, max: {ricfrrg_py.max():.6e}")
    print(f"     Fortran - min: {ricfrrg_fortran.min():.6e}, max: {ricfrrg_fortran.max():.6e}")
    
    assert_allclose(
        ricfrrg_fortran,
        ricfrrg_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RICFRRG: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RICFRRG : OK")
    
    print("\n  b) RRCFRIG - Congélation de contact (pluie)")
    print(f"     Python  - min: {rrcfrig_py.min():.6e}, max: {rrcfrig_py.max():.6e}")
    print(f"     Fortran - min: {rrcfrig_fortran.min():.6e}, max: {rrcfrig_fortran.max():.6e}")
    
    assert_allclose(
        rrcfrig_fortran,
        rrcfrig_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RRCFRIG: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RRCFRIG : OK")
    
    print("\n  c) RICFRR - Congélation de contact limitée")
    print(f"     Python  - min: {ricfrr_py.min():.6e}, max: {ricfrr_py.max():.6e}")
    print(f"     Fortran - min: {ricfrr_fortran.min():.6e}, max: {ricfrr_fortran.max():.6e}")
    
    assert_allclose(
        ricfrr_fortran,
        ricfrr_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RICFRR: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RICFRR : OK")
    
    # ------------------------------------------------------------------------
    # Validation de la croissance sèche du graupel (T < 0°C)
    # ------------------------------------------------------------------------
    print("\n2. CROISSANCE SÈCHE DU GRAUPEL (T < 0°C)")
    print("-" * 80)
    
    print("\n  a) RG_RCDRY_TND - Collection de gouttelettes nuageuses")
    print(f"     Python  - min: {rg_rcdry_tnd_py.min():.6e}, max: {rg_rcdry_tnd_py.max():.6e}")
    print(f"     Fortran - min: {rg_rcdry_tnd_fortran.min():.6e}, max: {rg_rcdry_tnd_fortran.max():.6e}")
    
    assert_allclose(
        rg_rcdry_tnd_fortran,
        rg_rcdry_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RG_RCDRY_TND: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RG_RCDRY_TND : OK")
    
    print("\n  b) RG_RIDRY_TND - Collection de glace pristine")
    print(f"     Python  - min: {rg_ridry_tnd_py.min():.6e}, max: {rg_ridry_tnd_py.max():.6e}")
    print(f"     Fortran - min: {rg_ridry_tnd_fortran.min():.6e}, max: {rg_ridry_tnd_fortran.max():.6e}")
    
    assert_allclose(
        rg_ridry_tnd_fortran,
        rg_ridry_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RG_RIDRY_TND: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RG_RIDRY_TND : OK")
    
    print("\n  c) RG_RSDRY_TND - Collection de neige")
    print(f"     Python  - min: {rg_rsdry_tnd_py.min():.6e}, max: {rg_rsdry_tnd_py.max():.6e}")
    print(f"     Fortran - min: {rg_rsdry_tnd_fortran.min():.6e}, max: {rg_rsdry_tnd_fortran.max():.6e}")
    
    assert_allclose(
        rg_rsdry_tnd_fortran,
        rg_rsdry_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RG_RSDRY_TND: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RG_RSDRY_TND : OK")
    
    print("\n  d) RG_RRDRY_TND - Collection de pluie")
    print(f"     Python  - min: {rg_rrdry_tnd_py.min():.6e}, max: {rg_rrdry_tnd_py.max():.6e}")
    print(f"     Fortran - min: {rg_rrdry_tnd_fortran.min():.6e}, max: {rg_rrdry_tnd_fortran.max():.6e}")
    
    assert_allclose(
        rg_rrdry_tnd_fortran,
        rg_rrdry_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RG_RRDRY_TND: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RG_RRDRY_TND : OK")
    
    # ------------------------------------------------------------------------
    # Validation de la croissance humide du graupel (T < 0°C)
    # ------------------------------------------------------------------------
    print("\n3. CROISSANCE HUMIDE DU GRAUPEL (T < 0°C)")
    print("-" * 80)
    
    print("\n  a) RG_RIWET_TND - Croissance humide (glace)")
    print(f"     Python  - min: {rg_riwet_tnd_py.min():.6e}, max: {rg_riwet_tnd_py.max():.6e}")
    print(f"     Fortran - min: {rg_riwet_tnd_fortran.min():.6e}, max: {rg_riwet_tnd_fortran.max():.6e}")
    
    assert_allclose(
        rg_riwet_tnd_fortran,
        rg_riwet_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RG_RIWET_TND: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RG_RIWET_TND : OK")
    
    print("\n  b) RG_RSWET_TND - Croissance humide (neige)")
    print(f"     Python  - min: {rg_rswet_tnd_py.min():.6e}, max: {rg_rswet_tnd_py.max():.6e}")
    print(f"     Fortran - min: {rg_rswet_tnd_fortran.min():.6e}, max: {rg_rswet_tnd_fortran.max():.6e}")
    
    assert_allclose(
        rg_rswet_tnd_fortran,
        rg_rswet_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RG_RSWET_TND: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RG_RSWET_TND : OK")
    
    # ------------------------------------------------------------------------
    # Validation des taux de congélation
    # ------------------------------------------------------------------------
    print("\n4. TAUX DE CONGÉLATION")
    print("-" * 80)
    
    print("\n  a) RG_FREEZ1_TND - Taux de congélation 1")
    print(f"     Python  - min: {rg_freez1_tnd_py.min():.6e}, max: {rg_freez1_tnd_py.max():.6e}")
    print(f"     Fortran - min: {rg_freez1_tnd_fortran.min():.6e}, max: {rg_freez1_tnd_fortran.max():.6e}")
    
    assert_allclose(
        rg_freez1_tnd_fortran,
        rg_freez1_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RG_FREEZ1_TND: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RG_FREEZ1_TND : OK")
    
    print("\n  b) RG_FREEZ2_TND - Taux de congélation 2")
    print(f"     Python  - min: {rg_freez2_tnd_py.min():.6e}, max: {rg_freez2_tnd_py.max():.6e}")
    print(f"     Fortran - min: {rg_freez2_tnd_fortran.min():.6e}, max: {rg_freez2_tnd_fortran.max():.6e}")
    
    assert_allclose(
        rg_freez2_tnd_fortran,
        rg_freez2_tnd_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RG_FREEZ2_TND: divergence Python/Fortran PHYEX"
    )
    print("     ✓ RG_FREEZ2_TND : OK")
    
    # ------------------------------------------------------------------------
    # Validation de la fonte du graupel (T > 0°C)
    # ------------------------------------------------------------------------
    print("\n5. FONTE DU GRAUPEL (T > 0°C)")
    print("-" * 80)
    
    print(f"  Python  - min: {rgmltr_py.min():.6e}, max: {rgmltr_py.max():.6e}")
    print(f"  Fortran - min: {rgmltr_fortran.min():.6e}, max: {rgmltr_fortran.max():.6e}")
    
    assert_allclose(
        rgmltr_fortran,
        rgmltr_py,
        rtol=1e-6,
        atol=1e-8,
        err_msg="[ÉCHEC] RGMLTR: divergence Python/Fortran PHYEX"
    )
    print("  ✓ RGMLTR : OK")
    
    # ========================================================================
    # Statistiques globales
    # ========================================================================
    print("\n" + "="*80)
    print("STATISTIQUES DES PROCESSUS RAPIDES DU GRAUPEL")
    print("="*80)
    
    n_total = domain[0] * domain[1] * domain[2]
    
    # Points actifs pour chaque processus
    n_cfrz = np.sum((ricfrrg_py > 1e-10) | (rrcfrig_py > 1e-10))
    n_dry = np.sum((rg_rcdry_tnd_py > 1e-10) | (rg_ridry_tnd_py > 1e-10) | 
                   (rg_rsdry_tnd_py > 1e-10) | (rg_rrdry_tnd_py > 1e-10))
    n_wet = np.sum((rg_riwet_tnd_py > 1e-10) | (rg_rswet_tnd_py > 1e-10))
    n_mlt = np.sum(rgmltr_py > 1e-10)
    
    print(f"\nPoints actifs (tendance > 1e-10):")
    print(f"  Congélation de contact:         {n_cfrz:6d}/{n_total} ({100.0*n_cfrz/n_total:5.1f}%)")
    print(f"  Croissance sèche:               {n_dry:6d}/{n_total} ({100.0*n_dry/n_total:5.1f}%)")
    print(f"  Croissance humide:              {n_wet:6d}/{n_total} ({100.0*n_wet/n_total:5.1f}%)")
    print(f"  Fonte (RGMLTR):                 {n_mlt:6d}/{n_total} ({100.0*n_mlt/n_total:5.1f}%)")
    
    # Distribution de température
    t_flat = FloatFieldsIJK_Input["t"].reshape(domain[0] * domain[1] * domain[2], order="F")
    t_freezing = t_flat[t_flat < externals["TT"]]
    t_melting = t_flat[t_flat >= externals["TT"]]
    
    if len(t_freezing) > 0:
        print(f"\nTempératures T < {externals['TT']}K (processus de gel):")
        print(f"  min={t_freezing.min():.1f}K, max={t_freezing.max():.1f}K, "
              f"moyenne={t_freezing.mean():.1f}K ({100.0*len(t_freezing)/n_total:.1f}% des points)")
    
    if len(t_melting) > 0:
        print(f"\nTempératures T >= {externals['TT']}K (processus de fonte):")
        print(f"  min={t_melting.min():.1f}K, max={t_melting.max():.1f}K, "
              f"moyenne={t_melting.mean():.1f}K ({100.0*len(t_melting)/n_total:.1f}% des points)")
    
    # Statistiques des rapports de mélange dans les zones actives
    if n_dry > 0:
        rgt_dry = rgt_flat[(rg_rcdry_tnd_py > 1e-10) | (rg_ridry_tnd_py > 1e-10) | 
                           (rg_rsdry_tnd_py > 1e-10) | (rg_rrdry_tnd_py > 1e-10)]
        print(f"\nRapports de mélange dans zones de croissance sèche:")
        print(f"  Graupel (rgt): min={rgt_dry.min():.6e}, max={rgt_dry.max():.6e}, "
              f"moyenne={rgt_dry.mean():.6e}")
    
    if n_mlt > 0:
        rgt_mlt = rgt_flat[rgmltr_py > 1e-10]
        t_mlt = t_flat[rgmltr_py > 1e-10]
        print(f"\nRapports de mélange dans zones de fonte:")
        print(f"  Graupel (rgt): min={rgt_mlt.min():.6e}, max={rgt_mlt.max():.6e}, "
              f"moyenne={rgt_mlt.mean():.6e}")
        print(f"  Température:   min={t_mlt.min():.1f}K, max={t_mlt.max():.1f}K, "
              f"moyenne={t_mlt.mean():.1f}K")
    
    print("\n" + "="*80)
    print("SUCCÈS: Reproductibilité validée!")
    print("Le stencil Python GT4Py ice4_fast_rg reproduit fidèlement PHYEX-IAL_CY50T1")
