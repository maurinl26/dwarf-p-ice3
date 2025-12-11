# -*- coding: utf-8 -*-
"""
Test de reproductibilité des processus rapides du graupel (ICE4_FAST_RG).

Ce module valide que les implémentations Python des processus rapides du graupel
de la microphysique ICE4 produisent des résultats numériquement identiques à 
l'implémentation Fortran de référence issue du projet PHYEX (PHYsique EXternalisée) 
version IAL_CY50T1.

Suite à la modularisation des stencils, ce test valide maintenant chaque processus
séparément en utilisant les kernels individuels:
- rain_contact_freezing: Congélation de contact de la pluie
- cloud_pristine_collection_graupel: Collection de gouttelettes et glace pristine
- graupel_melting: Fonte du graupel

Référence:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_rg.F90
"""
from ctypes import c_double, c_float

import numpy as np
import pytest
from numpy.testing import assert_allclose
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import zeros, from_array, ones

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import dp_dtypes, sp_dtypes


@pytest.mark.parametrize("ldsoft", [False, True])
@pytest.mark.parametrize("lcrflimit", [False, True])
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
def test_rain_contact_freezing(backend, dtypes, packed_dims, domain, origin, externals, ldsoft, lcrflimit):
    """
    Test de reproductibilité: Congélation de contact de la pluie.
    
    Valide les processus suivants (T < 0°C):
    - RICFRRG: Collection de glace pristine par la pluie → graupel
    - RRCFRIG: Congélation de la pluie par contact avec glace pristine
    - PRICFRR: Limitation de collection si bilan thermique limitant
    
    La congélation de contact représente un processus rapide où les gouttes
    de pluie se congelent au contact avec les cristaux de glace pristine,
    formant du graupel. Le paramètre LCRFLIMIT active la limitation basée
    sur le bilan thermique.
    
    Tolérance: rtol=1e-6, atol=1e-10
    """
    # Compilation du kernel Fortran de référence
    ice4_rain_contact_freezing_fortran = compile_fortran_stencil(
        "mode_ice4_fast_rg_kernels.F90",
        "mode_ice4_fast_rg_kernels",
        "ice4_rain_contact_freezing"
    )
    
    # =========================================================================
    # Initialisation des champs d'entrée
    # =========================================================================
    n_points = domain[0] * domain[1] * domain[2]
    
    # Génération de champs d'entrée réalistes
    prhodref = np.random.rand(n_points) * 0.8 + 0.5  # 0.5-1.3 kg/m³
    plbdar = np.random.rand(n_points) * 9e5 + 1e5    # 1e5-1e6 m⁻¹
    pt = np.random.rand(n_points) * 70 + 233         # 233-303 K
    prit = np.random.rand(n_points) * 0.002          # 0-0.002 kg/kg
    prrt = np.random.rand(n_points) * 0.005          # 0-0.005 kg/kg
    pcit = np.random.rand(n_points) * 9.99e5 + 1e3   # 1e3-1e6 #/m³
    
    # Masque de calcul (~90% de points actifs)
    ldcompute = np.random.rand(n_points) > 0.1
    
    # Champs de sortie
    dtype = c_float if dtypes["float"] == np.float32 else c_double
    pricfrrg = np.zeros(n_points, dtype=dtype)
    prrcfrig = np.zeros(n_points, dtype=dtype)  
    pricfrr = np.zeros(n_points, dtype=dtype)
    
    # Conversion en ordre Fortran
    prhodref_f = np.asfortranarray(prhodref.astype(dtype))
    plbdar_f = np.asfortranarray(plbdar.astype(dtype))
    pt_f = np.asfortranarray(pt.astype(dtype))
    prit_f = np.asfortranarray(prit.astype(dtype))
    prrt_f = np.asfortranarray(prrt.astype(dtype))
    pcit_f = np.asfortranarray(pcit.astype(dtype))
    ldcompute_f = np.asfortranarray(ldcompute)
    pricfrrg_f = np.asfortranarray(pricfrrg)
    prrcfrig_f = np.asfortranarray(prrcfrig)
    pricfrr_f = np.asfortranarray(pricfrr)
    
    # =========================================================================
    # Appel de la référence Fortran PHYEX
    # =========================================================================
    pricfrrg_fortran, prrcfrig_fortran, pricfrr_fortran = ice4_rain_contact_freezing_fortran(
        ldsoft=ldsoft,
        lcrflimit=lcrflimit,
        ldcompute=ldcompute_f,
        i_rtmin=externals["I_RTMIN"],
        r_rtmin=externals["R_RTMIN"],
        xicfrr=externals["ICFRR"],
        xexicfrr=externals["EXICFRR"],
        xcexvt=externals["CEXVT"],
        xrcfri=externals["RCFRI"],
        xexrcfri=externals["EXRCFRI"],
        xtt=externals["TT"],
        xci=externals["CI"],
        xcl=externals["CL"],
        xlvtt=externals["LVTT"],
        prhodref=prhodref_f,
        plbdar=plbdar_f,
        pt=pt_f,
        prit=prit_f,
        prrt=prrt_f,
        pcit=pcit_f,
        pricfrrg=pricfrrg_f,
        prrcfrig=prrcfrig_f,
        pricfrr=pricfrr_f,
        **packed_dims
    )
    
    # =========================================================================
    # Appel de l'implémentation Python GT4Py
    # =========================================================================
    from ice3.stencils.ice4_fast_rg import rain_contact_freezing

    stencil_rain_contact_freezing = stencil(
        backend=backend,
        definition=rain_contact_freezing,
        name="rain_contact_freezing",
        dtypes=dtypes,
        externals=externals
    )

    # Conversion des champs d'entrée en storages GT4Py
    rhodref = from_array(prhodref.reshape(domain), dtype=dtypes["float"], backend=backend)
    lbdar = from_array(plbdar.reshape(domain), dtype=dtypes["float"], backend=backend)
    t = from_array(pt.reshape(domain), dtype=dtypes["float"], backend=backend)
    rit = from_array(prit.reshape(domain), dtype=dtypes["float"], backend=backend)
    rrt = from_array(prrt.reshape(domain), dtype=dtypes["float"], backend=backend)
    cit = from_array(pcit.reshape(domain), dtype=dtypes["float"], backend=backend)

    ldcompute_gt = ones(shape=domain, dtype=dtypes["bool"], backend=backend)

    # Champs de sortie
    ricfrrg = zeros(shape=domain, dtype=dtypes["float"], backend=backend)
    rrcfrig = zeros(shape=domain, dtype=dtypes["float"], backend=backend)
    ricfrr = zeros(shape=domain, dtype=dtypes["float"], backend=backend)

    # Appel du stencil GT4Py
    stencil_rain_contact_freezing(
        ldcompute=ldcompute_gt,
        prhodref=rhodref,
        plbdar=lbdar,
        pt=t,
        prit=rit,
        prrt=rrt,
        pcit=cit,
        pricfrrg=ricfrrg,
        prrcfrig=rrcfrig,
        pricfrr=ricfrr,
        ldsoft=ldsoft,
        lcrflimit=lcrflimit,
        domain=domain,
        origin=origin
    )

    # Extraction des résultats Python
    pricfrrg_python = np.asarray(ricfrrg)
    prrcfrig_python = np.asarray(rrcfrig)
    pricfrr_python = np.asarray(ricfrr)
    
    # =========================================================================
    # VALIDATION - Comparaison Python vs Fortran PHYEX
    # =========================================================================
    print("\n" + "="*80)
    print("TEST DE REPRODUCTIBILITÉ: Congélation de contact de la pluie")
    print("="*80)
    print(f"Paramètres: LDSOFT={ldsoft}, LCRFLIMIT={lcrflimit}")
    print(f"Précision: {'simple' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domaine: {n_points} points")
    print("-" * 80)
    
    print("\nRICFRRG (collection glace pristine par pluie):")
    print(f"  Fortran - min: {pricfrrg_fortran.min():.6e}, max: {pricfrrg_fortran.max():.6e}")
    print(f"  Python  - min: {pricfrrg_python.min():.6e}, max: {pricfrrg_python.max():.6e}")
    assert_allclose(pricfrrg_python, pricfrrg_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="[ÉCHEC] RICFRRG: divergence Python/Fortran")
    print("  ✓ RICFRRG: OK")
    
    print("\nRRCFRIG (congélation pluie par contact):")
    print(f"  Fortran - min: {prrcfrig_fortran.min():.6e}, max: {prrcfrig_fortran.max():.6e}")
    print(f"  Python  - min: {prrcfrig_python.min():.6e}, max: {prrcfrig_python.max():.6e}")
    assert_allclose(prrcfrig_python, prrcfrig_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="[ÉCHEC] RRCFRIG: divergence Python/Fortran")
    print("  ✓ RRCFRIG: OK")
    
    print("\nPRICFRR (collection limitée par bilan thermique):")
    print(f"  Fortran - min: {pricfrr_fortran.min():.6e}, max: {pricfrr_fortran.max():.6e}")
    print(f"  Python  - min: {pricfrr_python.min():.6e}, max: {pricfrr_python.max():.6e}")
    assert_allclose(pricfrr_python, pricfrr_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="[ÉCHEC] PRICFRR: divergence Python/Fortran")
    print("  ✓ PRICFRR: OK")
    
    # Statistiques
    n_active = np.sum((pricfrrg_fortran > 1e-10) | (prrcfrig_fortran > 1e-10))
    print(f"\nPoints actifs: {n_active}/{n_points} ({100.0*n_active/n_points:.1f}%)")
    
    print("\n" + "="*80)
    print("SUCCÈS: Reproductibilité validée pour la congélation de contact")
    print("="*80)


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
def test_cloud_pristine_collection(backend, dtypes, packed_dims, domain, origin, externals, ldsoft):
    """
    Test de reproductibilité: Collection gouttelettes et glace pristine sur graupel.
    
    Valide les processus suivants (T < 0°C):
    - PRCDRYG_TEND: Collection sèche de gouttelettes nuageuses sur graupel
    - PRIDRYG_TEND: Collection sèche de glace pristine sur graupel
    - PRIWETG_TEND: Taux de croissance humide associé à collection glace
    
    Ces processus représentent la croissance du graupel par collection de
    particules nuageuses plus petites. La distinction sec/humide dépend du
    bilan thermique à la surface de la particule.
    
    Tolérance: rtol=1e-6, atol=1e-10
    """
    # Compilation du kernel Fortran de référence
    ice4_cloud_pristine_collection_fortran = compile_fortran_stencil(
        "mode_ice4_fast_rg_kernels.F90",
        "mode_ice4_fast_rg_kernels",
        "ice4_cloud_pristine_collection"
    )
    
    # =========================================================================
    # Initialisation des champs d'entrée
    # =========================================================================
    n_points = domain[0] * domain[1] * domain[2]
    
    # Génération de champs d'entrée réalistes
    prhodref = np.random.rand(n_points) * 0.8 + 0.5  # 0.5-1.3 kg/m³
    plbdag = np.random.rand(n_points) * 9e5 + 1e5    # 1e5-1e6 m⁻¹
    pt = np.random.rand(n_points) * 70 + 233         # 233-303 K
    prct = np.random.rand(n_points) * 0.003          # 0-0.003 kg/kg
    prit = np.random.rand(n_points) * 0.002          # 0-0.002 kg/kg
    prgt = np.random.rand(n_points) * 0.006          # 0-0.006 kg/kg
    
    # Masque de calcul
    ldcompute = np.random.rand(n_points) > 0.1
    
    # Champs de sortie
    dtype = c_float if dtypes["float"] == np.float32 else c_double
    prcdryg_tend = np.zeros(n_points, dtype=dtype)
    pridryg_tend = np.zeros(n_points, dtype=dtype)
    priwetg_tend = np.zeros(n_points, dtype=dtype)
    
    # Conversion en ordre Fortran
    prhodref_f = np.asfortranarray(prhodref.astype(dtype))
    plbdag_f = np.asfortranarray(plbdag.astype(dtype))
    pt_f = np.asfortranarray(pt.astype(dtype))
    prct_f = np.asfortranarray(prct.astype(dtype))
    prit_f = np.asfortranarray(prit.astype(dtype))
    prgt_f = np.asfortranarray(prgt.astype(dtype))
    ldcompute_f = np.asfortranarray(ldcompute)
    prcdryg_tend_f = np.asfortranarray(prcdryg_tend)
    pridryg_tend_f = np.asfortranarray(pridryg_tend)
    priwetg_tend_f = np.asfortranarray(priwetg_tend)
    
    # =========================================================================
    # Appel de la référence Fortran PHYEX
    # =========================================================================
    prcdryg_tend_fortran, pridryg_tend_fortran, priwetg_tend_fortran = (
        ice4_cloud_pristine_collection_fortran(
            ldsoft=ldsoft,
            ldcompute=ldcompute_f,
            c_rtmin=externals["C_RTMIN"],
            i_rtmin=externals["I_RTMIN"],
            g_rtmin=externals["G_RTMIN"],
            xtt=externals["TT"],
            xfcdryg=externals["FCDRYG"],
            xfidryg=externals["FIDRYG"],
            xcolig=externals["COLIG"],
            xcolexig=externals["COLEXIG"],
            xcxg=externals["CXG"],
            xdg=externals["DG"],
            xcexvt=externals["CEXVT"],
            prhodref=prhodref_f,
            plbdag=plbdag_f,
            pt=pt_f,
            prct=prct_f,
            prit=prit_f,
            prgt=prgt_f,
            prcdryg_tend=prcdryg_tend_f,
            pridryg_tend=pridryg_tend_f,
            priwetg_tend=priwetg_tend_f,
            **packed_dims
        )
    )
    
    # =========================================================================
    # Appel de l'implémentation Python GT4Py
    # =========================================================================
    from ice3.stencils.ice4_fast_rg import cloud_pristine_collection_graupel

    stencil_cloud_pristine_collection = stencil(
        backend=backend,
        definition=cloud_pristine_collection_graupel,
        name="cloud_pristine_collection_graupel",
        dtypes=dtypes,
        externals=externals
    )

    # Conversion des champs d'entrée en storages GT4Py
    rhodref = from_array(prhodref.reshape(domain), dtype=dtypes["float"], backend=backend)
    lbdag = from_array(plbdag.reshape(domain), dtype=dtypes["float"], backend=backend)
    t = from_array(pt.reshape(domain), dtype=dtypes["float"], backend=backend)
    rct = from_array(prct.reshape(domain), dtype=dtypes["float"], backend=backend)
    rit = from_array(prit.reshape(domain), dtype=dtypes["float"], backend=backend)
    rgt = from_array(prgt.reshape(domain), dtype=dtypes["float"], backend=backend)

    ldcompute_gt = ones(shape=domain, dtype=dtypes["bool"], backend=backend)

    # Champs de sortie
    rcdryg_tend = zeros(shape=domain, dtype=dtypes["float"], backend=backend)
    ridryg_tend = zeros(shape=domain, dtype=dtypes["float"], backend=backend)
    riwetg_tend = zeros(shape=domain, dtype=dtypes["float"], backend=backend)

    # Appel du stencil GT4Py
    stencil_cloud_pristine_collection(
        ldcompute=ldcompute_gt,
        prhodref=rhodref,
        plbdag=lbdag,
        pt=t,
        prct=rct,
        prit=rit,
        prgt=rgt,
        rcdryg_tend=rcdryg_tend,
        ridryg_tend=ridryg_tend,
        riwetg_tend=riwetg_tend,
        ldsoft=ldsoft,
        domain=domain,
        origin=origin
    )

    # Extraction des résultats Python
    prcdryg_tend_python = np.asarray(rcdryg_tend)
    pridryg_tend_python = np.asarray(ridryg_tend)
    priwetg_tend_python = np.asarray(riwetg_tend)
    
    # =========================================================================
    # VALIDATION - Comparaison Python vs Fortran PHYEX
    # =========================================================================
    print("\n" + "="*80)
    print("TEST DE REPRODUCTIBILITÉ: Collection gouttelettes/glace sur graupel")
    print("="*80)
    print(f"Paramètres: LDSOFT={ldsoft}")
    print(f"Précision: {'simple' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domaine: {n_points} points")
    print("-" * 80)
    
    print("\nPRCDRYG_TEND (collection gouttelettes nuageuses):")
    print(f"  Fortran - min: {prcdryg_tend_fortran.min():.6e}, max: {prcdryg_tend_fortran.max():.6e}")
    print(f"  Python  - min: {prcdryg_tend_python.min():.6e}, max: {prcdryg_tend_python.max():.6e}")
    assert_allclose(prcdryg_tend_python, prcdryg_tend_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="[ÉCHEC] PRCDRYG_TEND: divergence Python/Fortran")
    print("  ✓ PRCDRYG_TEND: OK")
    
    print("\nPRIDRYG_TEND (collection sèche glace pristine):")
    print(f"  Fortran - min: {pridryg_tend_fortran.min():.6e}, max: {pridryg_tend_fortran.max():.6e}")
    print(f"  Python  - min: {pridryg_tend_python.min():.6e}, max: {pridryg_tend_python.max():.6e}")
    assert_allclose(pridryg_tend_python, pridryg_tend_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="[ÉCHEC] PRIDRYG_TEND: divergence Python/Fortran")
    print("  ✓ PRIDRYG_TEND: OK")
    
    print("\nPRIWETG_TEND (croissance humide glace):")
    print(f"  Fortran - min: {priwetg_tend_fortran.min():.6e}, max: {priwetg_tend_fortran.max():.6e}")
    print(f"  Python  - min: {priwetg_tend_python.min():.6e}, max: {priwetg_tend_python.max():.6e}")
    assert_allclose(priwetg_tend_python, priwetg_tend_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="[ÉCHEC] PRIWETG_TEND: divergence Python/Fortran")
    print("  ✓ PRIWETG_TEND: OK")
    
    # Statistiques
    n_cloud = np.sum(prcdryg_tend_fortran > 1e-10)
    n_ice = np.sum(pridryg_tend_fortran > 1e-10)
    print(f"\nPoints actifs (cloud): {n_cloud}/{n_points} ({100.0*n_cloud/n_points:.1f}%)")
    print(f"Points actifs (ice): {n_ice}/{n_points} ({100.0*n_ice/n_points:.1f}%)")
    
    print("\n" + "="*80)
    print("SUCCÈS: Reproductibilité validée pour collection cloud/ice")
    print("="*80)


@pytest.mark.parametrize("ldsoft", [False, True])
@pytest.mark.parametrize("levlimit", [False, True])
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
def test_graupel_melting(backend, dtypes, packed_dims, domain, origin, externals, ldsoft, levlimit):
    """
    Test de reproductibilité: Fonte du graupel.
    
    Valide le processus suivant (T > 0°C):
    - PRGMLTR: Taux de fonte du graupel en pluie au-dessus de 0°C
    
    La fonte du graupel est contrôlée par:
    - La conduction thermique et diffusion de vapeur
    - La chaleur libérée par collection de gouttelettes et pluie
    - Le paramètre LEVLIMIT limite la pression de vapeur à saturation
    
    Tolérance: rtol=1e-6, atol=1e-10
    """
    # Compilation du kernel Fortran de référence
    ice4_graupel_melting_fortran = compile_fortran_stencil(
        "mode_ice4_fast_rg_kernels.F90",
        "mode_ice4_fast_rg_kernels",
        "ice4_graupel_melting"
    )
    
    # =========================================================================
    # Initialisation des champs d'entrée
    # =========================================================================
    n_points = domain[0] * domain[1] * domain[2]
    
    # Génération de champs d'entrée réalistes
    prhodref = np.random.rand(n_points) * 0.8 + 0.5  # 0.5-1.3 kg/m³
    ppres = np.random.rand(n_points) * 51325 + 50000 # 50000-101325 Pa
    pdv = np.random.rand(n_points) * 2e-5 + 1e-5     # 1e-5-3e-5 m²/s
    pka = np.random.rand(n_points) * 0.01 + 0.02     # 0.02-0.03 J/m/s/K
    pcj = np.random.rand(n_points) * 10.0            # 0-10
    plbdag = np.random.rand(n_points) * 9e5 + 1e5    # 1e5-1e6 m⁻¹
    pt = np.random.rand(n_points) * 70 + 233         # 233-303 K
    prvt = np.random.rand(n_points) * 0.015          # 0-0.015 kg/kg
    prgt = np.random.rand(n_points) * 0.006          # 0-0.006 kg/kg
    
    # Tendances nécessaires pour le calcul de fonte
    prcdryg_tend = np.random.rand(n_points) * 1e-6
    prrdryg_tend = np.random.rand(n_points) * 1e-6
    
    # Masque de calcul
    ldcompute = np.random.rand(n_points) > 0.1
    
    # Champ de sortie
    dtype = c_float if dtypes["float"] == np.float32 else c_double
    prgmltr = np.zeros(n_points, dtype=dtype)
    
    # Conversion en ordre Fortran
    prhodref_f = np.asfortranarray(prhodref.astype(dtype))
    ppres_f = np.asfortranarray(ppres.astype(dtype))
    pdv_f = np.asfortranarray(pdv.astype(dtype))
    pka_f = np.asfortranarray(pka.astype(dtype))
    pcj_f = np.asfortranarray(pcj.astype(dtype))
    plbdag_f = np.asfortranarray(plbdag.astype(dtype))
    pt_f = np.asfortranarray(pt.astype(dtype))
    prvt_f = np.asfortranarray(prvt.astype(dtype))
    prgt_f = np.asfortranarray(prgt.astype(dtype))
    prcdryg_tend_f = np.asfortranarray(prcdryg_tend.astype(dtype))
    prrdryg_tend_f = np.asfortranarray(prrdryg_tend.astype(dtype))
    ldcompute_f = np.asfortranarray(ldcompute)
    prgmltr_f = np.asfortranarray(prgmltr)
    
    # =========================================================================
    # Appel de la référence Fortran PHYEX
    # =========================================================================
    prgmltr_fortran = ice4_graupel_melting_fortran(
        ldsoft=ldsoft,
        levlimit=levlimit,
        ldcompute=ldcompute_f,
        g_rtmin=externals["G_RTMIN"],
        xtt=externals["TT"],
        xepsilo=externals["EPSILO"],
        xalpw=externals["ALPW"],
        xbetaw=externals["BETAW"],
        xgamw=externals["GAMW"],
        xlvtt=externals["LVTT"],
        xcpv=externals["CPV"],
        xcl=externals["CL"],
        xestt=externals["ESTT"],
        xrv=externals["RV"],
        xlmtt=externals["LMTT"],
        x0depg=externals["O0DEPG"],
        x1depg=externals["O1DEPG"],
        xex0depg=externals["EX0DEPG"],
        xex1depg=externals["EX1DEPG"],
        prhodref=prhodref_f,
        ppres=ppres_f,
        pdv=pdv_f,
        pka=pka_f,
        pcj=pcj_f,
        plbdag=plbdag_f,
        pt=pt_f,
        prvt=prvt_f,
        prgt=prgt_f,
        prcdryg_tend=prcdryg_tend_f,
        prrdryg_tend=prrdryg_tend_f,
        prgmltr=prgmltr_f,
        **packed_dims
    )
    
    # =========================================================================
    # Appel de l'implémentation Python GT4Py
    # =========================================================================
    from ice3.stencils.ice4_fast_rg import graupel_melting

    stencil_graupel_melting = stencil(
        backend=backend,
        definition=graupel_melting,
        name="graupel_melting",
        dtypes=dtypes,
        externals=externals
    )

    # Conversion des champs d'entrée en storages GT4Py
    rhodref = from_array(prhodref.reshape(domain), dtype=dtypes["float"], backend=backend)
    pres = from_array(ppres.reshape(domain), dtype=dtypes["float"], backend=backend)
    dv = from_array(pdv.reshape(domain), dtype=dtypes["float"], backend=backend)
    ka = from_array(pka.reshape(domain), dtype=dtypes["float"], backend=backend)
    cj = from_array(pcj.reshape(domain), dtype=dtypes["float"], backend=backend)
    lbdag = from_array(plbdag.reshape(domain), dtype=dtypes["float"], backend=backend)
    t = from_array(pt.reshape(domain), dtype=dtypes["float"], backend=backend)
    rvt = from_array(prvt.reshape(domain), dtype=dtypes["float"], backend=backend)
    rgt = from_array(prgt.reshape(domain), dtype=dtypes["float"], backend=backend)
    rcdryg_tend = from_array(prcdryg_tend.reshape(domain), dtype=dtypes["float"], backend=backend)
    rrdryg_tend = from_array(prrdryg_tend.reshape(domain), dtype=dtypes["float"], backend=backend)

    ldcompute_gt = ones(shape=domain, dtype=dtypes["bool"], backend=backend)

    # Champ de sortie
    rgmltr = zeros(shape=domain, dtype=dtypes["float"], backend=backend)

    # Appel du stencil GT4Py
    stencil_graupel_melting(
        ldcompute=ldcompute_gt,
        prhodref=rhodref,
        ppres=pres,
        pdv=dv,
        pka=ka,
        pcj=cj,
        plbdag=lbdag,
        pt=t,
        prvt=rvt,
        prgt=rgt,
        prgmltr=rgmltr,
        rcdryg_tend=rcdryg_tend,
        rrdryg_tend=rrdryg_tend,
        ldsoft=ldsoft,
        levlimit=levlimit,
        domain=domain, 
        origin=origin
    )

    # Extraction des résultats Python
    prgmltr_python = np.asarray(rgmltr)
    
    # =========================================================================
    # VALIDATION - Comparaison Python vs Fortran PHYEX
    # =========================================================================
    print("\n" + "="*80)
    print("TEST DE REPRODUCTIBILITÉ: Fonte du graupel")
    print("="*80)
    print(f"Paramètres: LDSOFT={ldsoft}, LEVLIMIT={levlimit}")
    print(f"Précision: {'simple' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domaine: {n_points} points")
    print("-" * 80)
    
    print("\nPRGMLTR (taux de fonte du graupel):")
    print(f"  Fortran - min: {prgmltr_fortran.min():.6e}, max: {prgmltr_fortran.max():.6e}")
    print(f"  Python  - min: {prgmltr_python.min():.6e}, max: {prgmltr_python.max():.6e}")
    assert_allclose(prgmltr_python, prgmltr_fortran, rtol=1e-6, atol=1e-10,
                   err_msg="[ÉCHEC] PRGMLTR: divergence Python/Fortran")
    print("  ✓ PRGMLTR: OK")
    
    # Statistiques
    n_melting = np.sum(prgmltr_fortran > 1e-10)
    print(f"\nPoints actifs (fonte): {n_melting}/{n_points} ({100.0*n_melting/n_points:.1f}%)")
    
    # Distribution de température dans zones de fonte
    t_melting = pt[prgmltr_fortran > 1e-10]
    if len(t_melting) > 0:
        print(f"Températures zone de fonte:")
        print(f"  min={t_melting.min():.1f}K, max={t_melting.max():.1f}K, "
              f"moyenne={t_melting.mean():.1f}K")
    
    print("\n" + "="*80)
    print("SUCCÈS: Reproductibilité validée pour la fonte du graupel")
    print("="*80)
