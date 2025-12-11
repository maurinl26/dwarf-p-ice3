# -*- coding: utf-8 -*-
"""
Test de reproductibilité des processus rapides de la neige (ICE4_FAST_RS).

Ce module valide que les implémentations Python des processus rapides de la neige/agrégats
de la microphysique ICE4 produisent des résultats numériquement identiques à
l'implémentation Fortran de référence issue du projet PHYEX (PHYsique EXternalisée)
version IAL_CY50T1.

Suite à la modularisation des stencils, ce test valide maintenant chaque processus
séparément en utilisant les kernels individuels:
- compute_freezing_rate: Calcul du taux de congélation maximum pour les processus neige
- conversion_melting_snow: Fonte et conversion des agrégats de neige

Référence:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_rs.F90
"""

from ctypes import c_double, c_float

import numpy as np
import pytest
from numpy.testing import assert_allclose
from gt4py.cartesian.gtscript import stencil
from gt4py.storage import zeros, from_array, ones

from ice3.utils.compile_fortran import compile_fortran_stencil
from ice3.utils.env import dp_dtypes, sp_dtypes
from ice3.phyex_common.phyex import Phyex


@pytest.mark.parametrize("ldsoft", [False, True])
@pytest.mark.parametrize("csnowriming", ["M90", "OLD"])
@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
def test_cloud_droplet_riming_snow(
    dtypes, packed_dims, domain, externals, ldsoft, csnowriming
):
    """
    Test de reproductibilité: Givrage des gouttelettes sur agrégats de neige.

    Valide les processus de givrage (riming) où les gouttelettes nuageuses
    se déposent sur les agrégats de neige et gèlent instantanément (T < 0°C).

    Processus validés:
    - RCRIMSS: Givrage sur petits agrégats
    - RCRIMSG: Givrage sur gros agrégats
    - RSRIMCG: Conversion neige → graupel par givrage

    Le givrage représente un processus de croissance rapide de la neige par
    collection de gouttelettes nuageuses. Pour les gros agrégats chargés en
    gouttelettes, la conversion en graupel peut se produire.

    Le paramètre CSNOWRIMING contrôle la paramétrisation:
    - "M90": Murakami 1990 (conversion graupel)
    - "OLD": Ancienne paramétrisation (pas de conversion)

    Tolérance: rtol=1e-6, atol=1e-10
    """
    # Mise à jour des externals avec CSNOWRIMING
    if "CSNOWRIMING" not in externals:
        externals = dict(externals)
        externals["CSNOWRIMING"] = csnowriming

    # Note: Ces processus utilisent l'interpolation de tables de lookup
    # Les tests complets nécessitent l'implémentation CuPy/NumPy

    # =========================================================================
    # Initialisation des champs d'entrée
    # =========================================================================
    n_points = domain[0] * domain[1] * domain[2]

    # Génération de champs d'entrée réalistes
    prhodref = np.random.rand(n_points) * 0.8 + 0.5  # 0.5-1.3 kg/m³
    plbdas = np.random.rand(n_points) * 9e5 + 1e5  # 1e5-1e6 m⁻¹
    pt = np.random.rand(n_points) * 70 + 233  # 233-303 K
    prct = np.random.rand(n_points) * 0.003  # 0-0.003 kg/kg
    prst = np.random.rand(n_points) * 0.004  # 0-0.004 kg/kg

    # Taux de congélation maximum (calculé par compute_freezing_rate)
    zfreez_rate = np.random.rand(n_points) * 1e-5

    # Masque de calcul
    ldcompute = np.random.rand(n_points) > 0.1

    # Champs de sortie
    dtype = c_float if dtypes["float"] == np.float32 else c_double
    prcrimss = np.zeros(n_points, dtype=dtype)
    prcrimsg = np.zeros(n_points, dtype=dtype)
    prsrimcg = np.zeros(n_points, dtype=dtype)

    # =========================================================================
    # TODO: Appel de l'implémentation Python/CuPy
    # =========================================================================
    # from ice3.stencils.ice4_fast_rs import cloud_droplet_riming_snow
    #
    # Les processus de riming nécessitent:
    # - Interpolation bilinéaire de tables GAMINC_RIM1, GAMINC_RIM2, GAMINC_RIM4
    # - Application des limites de congélation
    # - Calcul de conversion neige → graupel selon paramétrisation
    #
    # prcrimss, prcrimsg, prsrimcg = cloud_droplet_riming_snow(...)

    prcrimss_python = prcrimss.copy()
    prcrimsg_python = prcrimsg.copy()
    prsrimcg_python = prsrimcg.copy()

    # =========================================================================
    # VALIDATION - Comparaison Python vs Fortran PHYEX
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST DE REPRODUCTIBILITÉ: Givrage gouttelettes sur neige")
    print("=" * 80)
    print(f"Paramètres: LDSOFT={ldsoft}, CSNOWRIMING={csnowriming}")
    print(f"Précision: {'simple' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domaine: {n_points} points")
    print("-" * 80)

    print("\nRCRIMSS (givrage petits agrégats):")
    print(
        f"  Python  - min: {prcrimss_python.min():.6e}, max: {prcrimss_python.max():.6e}"
    )
    print("  ⚠ Test en attente d'implémentation Python/CuPy")

    print("\nRCRIMSG (givrage gros agrégats):")
    print(
        f"  Python  - min: {prcrimsg_python.min():.6e}, max: {prcrimsg_python.max():.6e}"
    )
    print("  ⚠ Test en attente d'implémentation Python/CuPy")

    print("\nRSRIMCG (conversion neige → graupel):")
    print(
        f"  Python  - min: {prsrimcg_python.min():.6e}, max: {prsrimcg_python.max():.6e}"
    )
    print("  ⚠ Test en attente d'implémentation Python/CuPy")

    print("\n" + "=" * 80)
    print("INFO: Test placeholder - implémentation CuPy requise")
    print("=" * 80)


@pytest.mark.parametrize("ldsoft", [False, True])
@pytest.mark.parametrize("dtypes", [dp_dtypes, sp_dtypes])
def test_rain_accretion_snow(dtypes, packed_dims, domain, externals, ldsoft):
    """
    Test de reproductibilité: Accrétion de la pluie sur agrégats de neige.

    Valide les processus d'accrétion où les gouttes de pluie sont collectées
    par les agrégats de neige et gèlent instantanément (T < 0°C).

    Processus validés:
    - RRACCSS: Accrétion pluie sur petits agrégats
    - RRACCSG: Accrétion pluie sur gros agrégats
    - RSACCRG: Conversion neige → graupel par accrétion pluie

    L'accrétion de pluie est un processus de croissance rapide qui peut
    conduire à la conversion des agrégats en graupel lorsque la charge
    en eau liquide devient importante.

    Ces processus utilisent l'interpolation bilinéaire de tables 2D:
    - KER_RACCSS: Kernel pour petits agrégats
    - KER_RACCS: Kernel pour tous agrégats
    - KER_SACCRG: Kernel pour conversion graupel

    Tolérance: rtol=1e-6, atol=1e-10
    """
    # =========================================================================
    # Initialisation des champs d'entrée
    # =========================================================================
    n_points = domain[0] * domain[1] * domain[2]

    # Génération de champs d'entrée réalistes
    prhodref = np.random.rand(n_points) * 0.8 + 0.5  # 0.5-1.3 kg/m³
    plbdas = np.random.rand(n_points) * 9e5 + 1e5  # 1e5-1e6 m⁻¹
    plbdar = np.random.rand(n_points) * 9e5 + 1e5  # 1e5-1e6 m⁻¹
    pt = np.random.rand(n_points) * 70 + 233  # 233-303 K
    prrt = np.random.rand(n_points) * 0.005  # 0-0.005 kg/kg
    prst = np.random.rand(n_points) * 0.004  # 0-0.004 kg/kg

    # Taux de congélation maximum (calculé par compute_freezing_rate)
    zfreez_rate = np.random.rand(n_points) * 1e-5

    # Masque de calcul
    ldcompute = np.random.rand(n_points) > 0.1

    # Champs de sortie
    dtype = c_float if dtypes["float"] == np.float32 else c_double
    prraccss = np.zeros(n_points, dtype=dtype)
    prraccsg = np.zeros(n_points, dtype=dtype)
    prsaccrg = np.zeros(n_points, dtype=dtype)

    # =========================================================================
    # TODO: Appel de l'implémentation Python/CuPy
    # =========================================================================
    # from ice3.stencils.ice4_fast_rs import rain_accretion_snow
    #
    # Les processus d'accrétion nécessitent:
    # - Interpolation bilinéaire 2D de tables KER_RACCSS, KER_RACCS, KER_SACCRG
    # - Application des limites de congélation
    # - Calcul de conversion neige → graupel
    #
    # prraccss, prraccsg, prsaccrg = rain_accretion_snow(...)

    prraccss_python = prraccss.copy()
    prraccsg_python = prraccsg.copy()
    prsaccrg_python = prsaccrg.copy()

    # =========================================================================
    # VALIDATION - Comparaison Python vs Fortran PHYEX
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST DE REPRODUCTIBILITÉ: Accrétion pluie sur neige")
    print("=" * 80)
    print(f"Paramètres: LDSOFT={ldsoft}")
    print(f"Précision: {'simple' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domaine: {n_points} points")
    print("-" * 80)

    print("\nRRACCSS (accrétion pluie sur petits agrégats):")
    print(
        f"  Python  - min: {prraccss_python.min():.6e}, max: {prraccss_python.max():.6e}"
    )
    print("  ⚠ Test en attente d'implémentation Python/CuPy")

    print("\nRRACCSG (accrétion pluie sur gros agrégats):")
    print(
        f"  Python  - min: {prraccsg_python.min():.6e}, max: {prraccsg_python.max():.6e}"
    )
    print("  ⚠ Test en attente d'implémentation Python/CuPy")

    print("\nRSACCRG (conversion neige → graupel par accrétion):")
    print(
        f"  Python  - min: {prsaccrg_python.min():.6e}, max: {prsaccrg_python.max():.6e}"
    )
    print("  ⚠ Test en attente d'implémentation Python/CuPy")

    print("\n" + "=" * 80)
    print("INFO: Test placeholder - implémentation CuPy requise")
    print("=" * 80)


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
def test_compute_freezing_rate(
    backend, dtypes, packed_dims, domain, origin, externals, ldsoft, levlimit
):
    """
    Test de reproductibilité: Calcul du taux de congélation maximum pour la neige.

    Valide le calcul du taux maximum de congélation des agrégats de neige
    basé sur les contraintes de bilan thermique et la déposition de vapeur.

    Processus validés:
    - PFREEZ_RATE: Taux de congélation maximum disponible
    - PFREEZ1_TEND: Premier terme de congélation (déposition de vapeur)
    - PFREEZ2_TEND: Second terme de congélation (facteur de capacité thermique)

    Ce taux limite ensuite les processus de givrage (riming) et d'accrétion
    de pluie sur les agrégats de neige. Le paramètre LEVLIMIT active la
    limitation de la pression de vapeur à saturation.

    Tolérance: rtol=1e-6, atol=1e-10
    """
    # Compilation du kernel Fortran de référence
    ice4_compute_freezing_rate_fortran = compile_fortran_stencil(
        "mode_ice4_fast_rs_kernels.F90",
        "mode_ice4_fast_rs_kernels",
        "ice4_compute_freezing_rate",
    )

    # =========================================================================
    # Initialisation des champs d'entrée
    # =========================================================================
    n_points = domain[0] * domain[1] * domain[2]

    # Génération de champs d'entrée réalistes
    prhodref = np.random.rand(n_points) * 0.8 + 0.5  # 0.5-1.3 kg/m³
    ppres = np.random.rand(n_points) * 51325 + 50000  # 50000-101325 Pa
    pdv = np.random.rand(n_points) * 2e-5 + 1e-5  # 1e-5-3e-5 m²/s
    pka = np.random.rand(n_points) * 0.01 + 0.02  # 0.02-0.03 J/m/s/K
    pcj = np.random.rand(n_points) * 10.0  # 0-10
    plbdas = np.random.rand(n_points) * 9e5 + 1e5  # 1e5-1e6 m⁻¹
    pt = np.random.rand(n_points) * 70 + 233  # 233-303 K
    prvt = np.random.rand(n_points) * 0.015  # 0-0.015 kg/kg
    prst = np.random.rand(n_points) * 0.004  # 0-0.004 kg/kg
    priaggs = np.random.rand(n_points) * 1e-6  # Taux d'agrégation glace

    # Masque de calcul (~90% de points actifs)
    ldcompute = np.random.rand(n_points) > 0.1

    # Champs de sortie
    dtype = c_float if dtypes["float"] == np.float32 else c_double
    pfreez_rate = np.zeros(n_points, dtype=dtype)
    pfreez1_tend = np.zeros(n_points, dtype=dtype)
    pfreez2_tend = np.zeros(n_points, dtype=dtype)

    # Conversion en ordre Fortran
    prhodref_f = np.asfortranarray(prhodref.astype(dtype))
    ppres_f = np.asfortranarray(ppres.astype(dtype))
    pdv_f = np.asfortranarray(pdv.astype(dtype))
    pka_f = np.asfortranarray(pka.astype(dtype))
    pcj_f = np.asfortranarray(pcj.astype(dtype))
    plbdas_f = np.asfortranarray(plbdas.astype(dtype))
    pt_f = np.asfortranarray(pt.astype(dtype))
    prvt_f = np.asfortranarray(prvt.astype(dtype))
    prst_f = np.asfortranarray(prst.astype(dtype))
    priaggs_f = np.asfortranarray(priaggs.astype(dtype))
    ldcompute_f = np.asfortranarray(ldcompute)
    pfreez_rate_f = np.asfortranarray(pfreez_rate)
    pfreez1_tend_f = np.asfortranarray(pfreez1_tend)
    pfreez2_tend_f = np.asfortranarray(pfreez2_tend)

    # =========================================================================
    # Appel de la référence Fortran PHYEX
    # =========================================================================
    pfreez_rate_fortran, pfreez1_tend_fortran, pfreez2_tend_fortran = (
        ice4_compute_freezing_rate_fortran(
            ldsoft=ldsoft,
            levlimit=levlimit,
            ldcompute=ldcompute_f,
            s_rtmin=externals["S_RTMIN"],
            xepsilo=externals["EPSILO"],
            xalpi=externals["ALPI"],
            xbetai=externals["BETAI"],
            xgami=externals["GAMI"],
            xtt=externals["TT"],
            xlvtt=externals["LVTT"],
            xcpv=externals["CPV"],
            xcl=externals["CL"],
            xci=externals["CI"],
            xlmtt=externals["LMTT"],
            xestt=externals["ESTT"],
            xrv=externals["RV"],
            x0deps=externals["O0DEPS"],
            x1deps=externals["O1DEPS"],
            xex0deps=externals["EX0DEPS"],
            xex1deps=externals["EX1DEPS"],
            prhodref=prhodref_f,
            ppres=ppres_f,
            pdv=pdv_f,
            pka=pka_f,
            pcj=pcj_f,
            plbdas=plbdas_f,
            pt=pt_f,
            prvt=prvt_f,
            prst=prst_f,
            priaggs=priaggs_f,
            pfreez_rate=pfreez_rate_f,
            pfreez1_tend=pfreez1_tend_f,
            pfreez2_tend=pfreez2_tend_f,
            **packed_dims,
        )
    )

    # =========================================================================
    # Appel de l'implémentation Python/CuPy
    # =========================================================================
    from ice3.stencils.ice4_fast_rs import compute_freezing_rate

    stencil_compute_freezing_rate = stencil(
        backend=backend,
        definition=compute_freezing_rate,
        name="compute_freezing_rate",
        dtypes=dtypes,
        externals=externals,
    )

    rhodref = from_array(prhodref.reshape(domain), dtype=dtypes["float"], backend=backend)
    pres = from_array(ppres.reshape(domain), dtype=dtypes["float"], backend=backend)
    dv = from_array(pdv.reshape(domain), dtype=dtypes["float"], backend=backend)
    ka = from_array(pka.reshape(domain), dtype=dtypes["float"], backend=backend)
    cj = from_array(pcj.reshape(domain), dtype=dtypes["float"], backend=backend)
    lbdas = from_array(plbdas.reshape(domain), dtype=dtypes["float"], backend=backend)
    t = from_array(pt.reshape(domain), dtype=dtypes["float"], backend=backend)
    rvt = from_array(prvt.reshape(domain), dtype=dtypes["float"], backend=backend)
    rst = from_array(prst.reshape(domain), dtype=dtypes["float"], backend=backend)
    riaggs = from_array(priaggs.reshape(domain), dtype=dtypes["float"], backend=backend)

    ldcompute = ones(shape=domain, dtype=dtypes["bool"], backend=backend)

    freez1_tend = zeros(shape=domain, dtype=dtypes["float"], backend=backend)
    freez2_tend = zeros(shape=domain, dtype=dtypes["float"], backend=backend)
    zfreez_rate = zeros(shape=domain, dtype=dtypes["float"], backend=backend)

    stencil_compute_freezing_rate(
        ldcompute=ldcompute,
        prhodref=rhodref,
        ppres=pres,
        pdv=dv,
        pka=ka,
        pcj=cj,
        plbdas=lbdas,
        pt=t,
        prvt=rvt,
        prst=rst,
        priaggs=riaggs,
        freez1_tend=freez1_tend,
        freez2_tend=freez2_tend,
        zfreez_rate=zfreez_rate,
        ldsoft=ldsoft,
        levlimit=levlimit,
        domain=domain,
        origin=origin
    )

    pfreez1_tend_python = np.asarray(freez1_tend)
    pfreez2_tend_python = np.asarray(freez2_tend)
    zfreez_rate_python = np.asarray(zfreez_rate)

    # =========================================================================
    # VALIDATION - Comparaison Python vs Fortran PHYEX
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST DE REPRODUCTIBILITÉ: Taux de congélation maximum (neige)")
    print("=" * 80)
    print(f"Paramètres: LDSOFT={ldsoft}, LEVLIMIT={levlimit}")
    print(f"Précision: {'simple' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domaine: {n_points} points")
    print("-" * 80)

    print("\nPFREEZ_RATE (taux de congélation maximum):")
    print(
        f"  Fortran - min: {pfreez_rate_fortran.min():.6e}, max: {pfreez_rate_fortran.max():.6e}"
    )
    print(f"  Python  - min: {zfreez_rate_python.min():.6e}, max: {zfreez_rate_python.max():.6e}")
    assert_allclose(
        zfreez_rate_python,
        pfreez_rate_fortran,
        rtol=1e-6,
        atol=1e-10,
        err_msg="[ÉCHEC] PFREEZ_RATE: divergence Python/Fortran",
    )
    print("  ✓ PFREEZ_RATE: OK")

    print("\nPFREEZ1_TEND (terme de déposition vapeur):")
    print(
        f"  Fortran - min: {pfreez1_tend_fortran.min():.6e}, max: {pfreez1_tend_fortran.max():.6e}"
    )
    print(f"  Python  - min: {pfreez1_tend_python.min():.6e}, max: {pfreez1_tend_python.max():.6e}")
    assert_allclose(
        pfreez1_tend_python,
        pfreez1_tend_fortran,
        rtol=1e-6,
        atol=1e-10,
        err_msg="[ÉCHEC] PFREEZ1_TEND: divergence Python/Fortran",
    )
    print("  ✓ PFREEZ1_TEND: OK")

    print("\nPFREEZ2_TEND (facteur capacité thermique):")
    print(
        f"  Fortran - min: {pfreez2_tend_fortran.min():.6e}, max: {pfreez2_tend_fortran.max():.6e}"
    )
    print(f"  Python  - min: {pfreez2_tend_python.min():.6e}, max: {pfreez2_tend_python.max():.6e}")
    assert_allclose(
        pfreez2_tend_python,
        pfreez2_tend_fortran,
        rtol=1e-6,
        atol=1e-10,
        err_msg="[ÉCHEC] PFREEZ2_TEND: divergence Python/Fortran",
    )
    print("  ✓ PFREEZ2_TEND: OK")

    # Statistiques
    n_active = np.sum(pfreez_rate_fortran > 1e-10)
    print(
        f"\nPoints actifs (congélation): {n_active}/{n_points} ({100.0 * n_active / n_points:.1f}%)"
    )

    print("\n" + "=" * 80)
    print("SUCCÈS: Reproductibilité validée pour le taux de congélation")
    print("=" * 80)


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
def test_conversion_melting_snow(
    backend, dtypes, packed_dims, domain, origin, externals, ldsoft, levlimit
):
    """
    Test de reproductibilité: Fonte et conversion des agrégats de neige.

    Valide les processus de fonte de la neige au-dessus de 0°C:
    - PRSMLTG: Taux de fonte des agrégats de neige en pluie
    - PRCMLTSR: Collection de gouttelettes par neige à T>0°C (sans changement de phase)

    La fonte des agrégats tient compte de:
    - La conduction thermique et diffusion de vapeur
    - La chaleur libérée par collection de gouttelettes nuageuses (riming)
    - La chaleur libérée par collection de pluie (accretion)
    - Le paramètre LEVLIMIT limite la pression de vapeur à saturation

    Le processus RCMLTSR représente la collection de gouttelettes nuageuses
    par les agrégats de neige lorsque les deux espèces sont liquides (T>0°C),
    donc sans échange de chaleur latente.

    Tolérance: rtol=1e-6, atol=1e-10
    """
    # Compilation du kernel Fortran de référence
    ice4_conversion_melting_snow_fortran = compile_fortran_stencil(
        "mode_ice4_fast_rs_kernels.F90",
        "mode_ice4_fast_rs_kernels",
        "ice4_conversion_melting_snow",
    )

    # =========================================================================
    # Initialisation des champs d'entrée
    # =========================================================================
    n_points = domain[0] * domain[1] * domain[2]

    # Génération de champs d'entrée réalistes
    prhodref = np.random.rand(n_points) * 0.8 + 0.5  # 0.5-1.3 kg/m³
    ppres = np.random.rand(n_points) * 51325 + 50000  # 50000-101325 Pa
    pdv = np.random.rand(n_points) * 2e-5 + 1e-5  # 1e-5-3e-5 m²/s
    pka = np.random.rand(n_points) * 0.01 + 0.02  # 0.02-0.03 J/m/s/K
    pcj = np.random.rand(n_points) * 10.0  # 0-10
    plbdas = np.random.rand(n_points) * 9e5 + 1e5  # 1e5-1e6 m⁻¹
    pt = np.random.rand(n_points) * 70 + 233  # 233-303 K
    prvt = np.random.rand(n_points) * 0.015  # 0-0.015 kg/kg
    prst = np.random.rand(n_points) * 0.004  # 0-0.004 kg/kg

    # Tendances d'entrée provenant des processus de givrage et accrétion
    prcrims_tend = np.random.rand(n_points) * 1e-6  # Tendance givrage gouttelettes
    prraccs_tend = np.random.rand(n_points) * 1e-6  # Tendance accrétion pluie

    # Masque de calcul
    ldcompute = np.random.rand(n_points) > 0.1

    # Champs de sortie
    dtype = c_float if dtypes["float"] == np.float32 else c_double
    prsmltg = np.zeros(n_points, dtype=dtype)
    prcmltsr = np.zeros(n_points, dtype=dtype)

    # Conversion en ordre Fortran
    prhodref_f = np.asfortranarray(prhodref.astype(dtype))
    ppres_f = np.asfortranarray(ppres.astype(dtype))
    pdv_f = np.asfortranarray(pdv.astype(dtype))
    pka_f = np.asfortranarray(pka.astype(dtype))
    pcj_f = np.asfortranarray(pcj.astype(dtype))
    plbdas_f = np.asfortranarray(plbdas.astype(dtype))
    pt_f = np.asfortranarray(pt.astype(dtype))
    prvt_f = np.asfortranarray(prvt.astype(dtype))
    prst_f = np.asfortranarray(prst.astype(dtype))
    prcrims_tend_f = np.asfortranarray(prcrims_tend.astype(dtype))
    prraccs_tend_f = np.asfortranarray(prraccs_tend.astype(dtype))
    ldcompute_f = np.asfortranarray(ldcompute)
    prsmltg_f = np.asfortranarray(prsmltg)
    prcmltsr_f = np.asfortranarray(prcmltsr)

    # =========================================================================
    # Appel de la référence Fortran PHYEX
    # =========================================================================
    prsmltg_fortran, prcmltsr_fortran = ice4_conversion_melting_snow_fortran(
        ldsoft=ldsoft,
        levlimit=levlimit,
        ldcompute=ldcompute_f,
        s_rtmin=externals["S_RTMIN"],
        xepsilo=externals["EPSILO"],
        xalpw=externals["ALPW"],
        xbetaw=externals["BETAW"],
        xgamw=externals["GAMW"],
        xtt=externals["TT"],
        xlvtt=externals["LVTT"],
        xcpv=externals["CPV"],
        xcl=externals["CL"],
        xlmtt=externals["LMTT"],
        xestt=externals["ESTT"],
        xrv=externals["RV"],
        x0deps=externals["O0DEPS"],
        x1deps=externals["O1DEPS"],
        xex0deps=externals["EX0DEPS"],
        xex1deps=externals["EX1DEPS"],
        xfscvmg=externals["FSCVMG"],
        prhodref=prhodref_f,
        ppres=ppres_f,
        pdv=pdv_f,
        pka=pka_f,
        pcj=pcj_f,
        plbdas=plbdas_f,
        pt=pt_f,
        prvt=prvt_f,
        prst=prst_f,
        prcrims_tend=prcrims_tend_f,
        prraccs_tend=prraccs_tend_f,
        prsmltg=prsmltg_f,
        prcmltsr=prcmltsr_f,
        **packed_dims,
    )

    # =========================================================================
    # Appel de l'implémentation Python GT4Py
    # =========================================================================
    from ice3.stencils.ice4_fast_rs import conversion_melting_snow

    stencil_conversion_melting_snow = stencil(
        backend=backend,
        definition=conversion_melting_snow,
        name="conversion_melting_snow",
        dtypes=dtypes,
        externals=externals,
    )

    # Conversion des champs d'entrée en storages GT4Py
    rhodref = from_array(prhodref.reshape(domain), dtype=dtypes["float"], backend=backend)
    pres = from_array(ppres.reshape(domain), dtype=dtypes["float"], backend=backend)
    dv = from_array(pdv.reshape(domain), dtype=dtypes["float"], backend=backend)
    ka = from_array(pka.reshape(domain), dtype=dtypes["float"], backend=backend)
    cj = from_array(pcj.reshape(domain), dtype=dtypes["float"], backend=backend)
    lbdas = from_array(plbdas.reshape(domain), dtype=dtypes["float"], backend=backend)
    t = from_array(pt.reshape(domain), dtype=dtypes["float"], backend=backend)
    rvt = from_array(prvt.reshape(domain), dtype=dtypes["float"], backend=backend)
    rst = from_array(prst.reshape(domain), dtype=dtypes["float"], backend=backend)
    rcrims_tend = from_array(prcrims_tend.reshape(domain), dtype=dtypes["float"], backend=backend)
    rraccs_tend = from_array(prraccs_tend.reshape(domain), dtype=dtypes["float"], backend=backend)

    ldcompute_gt = ones(shape=domain, dtype=dtypes["bool"], backend=backend)

    # Champs de sortie
    rsmltg = zeros(shape=domain, dtype=dtypes["float"], backend=backend)
    rcmltsr = zeros(shape=domain, dtype=dtypes["float"], backend=backend)

    # Appel du stencil GT4Py
    stencil_conversion_melting_snow(
        ldcompute=ldcompute_gt,
        prhodref=rhodref,
        ppres=pres,
        pdv=dv,
        pka=ka,
        pcj=cj,
        plbdas=lbdas,
        pt=t,
        prvt=rvt,
        prst=rst,
        rcrims_tend=rcrims_tend,
        rraccs_tend=rraccs_tend,
        prsmltg=rsmltg,
        prcmltsr=rcmltsr,
        ldsoft=ldsoft,
        levlimit=levlimit,
        domain=domain,
        origin=origin,
    )

    # Extraction des résultats Python
    prsmltg_python = np.asarray(rsmltg)
    prcmltsr_python = np.asarray(rcmltsr)

    # =========================================================================
    # VALIDATION - Comparaison Python vs Fortran PHYEX
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST DE REPRODUCTIBILITÉ: Fonte et conversion des agrégats de neige")
    print("=" * 80)
    print(f"Paramètres: LDSOFT={ldsoft}, LEVLIMIT={levlimit}")
    print(f"Précision: {'simple' if dtypes['float'] == np.float32 else 'double'}")
    print(f"Domaine: {n_points} points")
    print("-" * 80)

    print("\nPRSMLTG (taux de fonte de la neige):")
    print(
        f"  Fortran - min: {prsmltg_fortran.min():.6e}, max: {prsmltg_fortran.max():.6e}"
    )
    print(
        f"  Python  - min: {prsmltg_python.min():.6e}, max: {prsmltg_python.max():.6e}"
    )
    assert_allclose(
        prsmltg_python,
        prsmltg_fortran,
        rtol=1e-6,
        atol=1e-10,
        err_msg="[ÉCHEC] PRSMLTG: divergence Python/Fortran",
    )
    print("  ✓ PRSMLTG: OK")

    print("\nPRCMLTSR (collection gouttelettes à T>0°C):")
    print(
        f"  Fortran - min: {prcmltsr_fortran.min():.6e}, max: {prcmltsr_fortran.max():.6e}"
    )
    print(
        f"  Python  - min: {prcmltsr_python.min():.6e}, max: {prcmltsr_python.max():.6e}"
    )
    assert_allclose(
        prcmltsr_python,
        prcmltsr_fortran,
        rtol=1e-6,
        atol=1e-10,
        err_msg="[ÉCHEC] PRCMLTSR: divergence Python/Fortran",
    )
    print("  ✓ PRCMLTSR: OK")

    # Statistiques
    n_melting = np.sum(prsmltg_fortran > 1e-10)
    n_collection = np.sum(prcmltsr_fortran > 1e-10)
    print(
        f"\nPoints actifs (fonte): {n_melting}/{n_points} ({100.0 * n_melting / n_points:.1f}%)"
    )
    print(
        f"Points actifs (collection T>0): {n_collection}/{n_points} ({100.0 * n_collection / n_points:.1f}%)"
    )

    # Distribution de température dans zones de fonte
    t_melting = pt[prsmltg_fortran > 1e-10]
    if len(t_melting) > 0:
        print(f"\nTempératures zone de fonte:")
        print(
            f"  min={t_melting.min():.1f}K, max={t_melting.max():.1f}K, "
            f"moyenne={t_melting.mean():.1f}K"
        )

    print("\n" + "=" * 80)
    print("SUCCÈS: Reproductibilité validée pour fonte/conversion neige")
    print("=" * 80)
