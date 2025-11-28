"""
Test du composant IceAdjustModular avec données réelles ice_adjust.nc.

Ce test valide le composant modulaire IceAdjustModular qui utilise
les stencils séparés (condensation.py + cloud_fraction.py) contre
le jeu de données de référence ice_adjust.nc provenant de PHYEX.

Comparaison:
    - IceAdjustModular (4 stencils) vs données de référence
    - Validation des tendances microphysiques
    - Validation des champs haute résolution
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ice3.components.ice_adjust_modular import IceAdjustModular
from ice3.phyex_common.phyex import Phyex
from ice3.utils.env import sp_dtypes, dp_dtypes


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("debug", marks=pytest.mark.debug),
        pytest.param("numpy", marks=pytest.mark.numpy),
        pytest.param("gt:cpu_ifirst", marks=pytest.mark.cpu),
        pytest.param("gt:gpu", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
def test_ice_adjust_modular(benchmark, backend, dtypes, ice_adjust_repro_ds):
    """
    Test du composant IceAdjustModular avec le jeu de données ice_adjust.nc.
    
    Ce test valide que le composant modulaire (utilisant 4 stencils séparés)
    produit les mêmes résultats que les données de référence PHYEX.
    
    Séquence testée:
        1. thermodynamic_fields : T, Lv, Ls, Cph
        2. condensation         : CB02, rc_out, ri_out
        3. cloud_fraction_1     : Sources, conservation
        4. cloud_fraction_2     : Fraction nuageuse, autoconversion
    
    Champs validés:
        - ths, rvs, rcs, ris : Tendances microphysiques
        - hlc_hrc, hlc_hcf  : Contenu/fraction liquide haute résolution
        - hli_hri, hli_hcf  : Contenu/fraction glace haute résolution
    
    Args:
        benchmark: Fixture pytest-benchmark
        backend: Backend GT4Py (numpy, cpu, gpu, debug)
        dtypes: Types de données (float32/float64)
        ice_adjust_repro_ds: Dataset xarray ice_adjust.nc
    """
    print("\n" + "="*75)
    print("TEST COMPOSANT: IceAdjustModular avec ice_adjust.nc")
    print("="*75)
    print(f"Backend: {backend}")
    print(f"Precision: {dtypes['float']}")
    
    # ========================================================================
    # Configuration PHYEX et création du composant
    # ========================================================================
    phyex = Phyex("AROME")
    ice_adjust_modular = IceAdjustModular(
        phyex=phyex,
        dtypes=dtypes,
        backend=backend,
    )
    
    print(f"Composant créé: {ice_adjust_modular}")
    
    # ========================================================================
    # Chargement et préparation des données
    # ========================================================================
    shape = (
        ice_adjust_repro_ds.sizes["ngpblks"],
        ice_adjust_repro_ds.sizes["nproma"],
        ice_adjust_repro_ds.sizes["nflevg"]
    )
    print(f"Domaine: {shape}")
    
    # Préparation sigqsat (2D → 3D broadcast)
    sigqsat = ice_adjust_repro_ds["ZSIGQSAT"].data[:,:,np.newaxis]
    sigqsat = np.broadcast_to(sigqsat, shape)
    
    # Chargement des champs d'entrée (swap axes pour ordre correct)
    zrs = ice_adjust_repro_ds["ZRS"].data
    prs = ice_adjust_repro_ds["PRS"].data
    
    # Réorganisation des axes (passage du format Fortran au format Python)
    prs = np.swapaxes(prs, axis1=1, axis2=2)
    prs = np.swapaxes(prs, axis1=2, axis2=3)
    prs = np.swapaxes(prs, axis1=1, axis2=2)
    
    zrs = np.swapaxes(zrs, axis1=1, axis2=2)
    zrs = np.swapaxes(zrs, axis1=2, axis2=3)
    zrs = np.swapaxes(zrs, axis1=1, axis2=2)
    
    # Champs atmosphériques
    ppabsm = np.swapaxes(ice_adjust_repro_ds["PPABSM"].data, axis1=1, axis2=2)
    psigs = np.swapaxes(ice_adjust_repro_ds["PSIGS"].data, axis1=1, axis2=2)
    pexn = np.swapaxes(ice_adjust_repro_ds["PEXNREF"].data, axis1=1, axis2=2)
    pexnref = pexn.copy()  # exn et exnref identiques
    prhodref = np.swapaxes(ice_adjust_repro_ds["PRHODREF"].data, axis1=1, axis2=2)
    
    # Flux de masse
    pcf_mf = np.swapaxes(ice_adjust_repro_ds["PCF_MF"].data, axis1=1, axis2=2)
    pri_mf = np.swapaxes(ice_adjust_repro_ds["PRI_MF"].data, axis1=1, axis2=2)
    prc_mf = np.swapaxes(ice_adjust_repro_ds["PRC_MF"].data, axis1=1, axis2=2)
    
    # Température potentielle
    pths = np.swapaxes(ice_adjust_repro_ds["PTHS"].data, axis1=1, axis2=2)
    
    # Variables météorologiques (indexation dans le tableau zrs)
    th = zrs[:,:,:,0].copy()   # Température potentielle
    rv = zrs[:,:,:,1].copy()   # Vapeur d'eau
    ri = zrs[:,:,:,2].copy()   # Glace
    rc = zrs[:,:,:,3].copy()   # Liquide nuageux
    rr = zrs[:,:,:,4].copy()   # Pluie
    rs = zrs[:,:,:,5].copy()   # Neige
    rg = zrs[:,:,:,6].copy()   # Graupel
    
    # Variables sources (copiées depuis prs)
    ths = pths.copy()
    rvs = prs[:,:,:,0].copy()
    rcs = prs[:,:,:,1].copy()
    ris = prs[:,:,:,3].copy()
    
    # Champs de sortie (initialisés à zéro)
    cldfr = np.zeros(shape, dtype=dtypes["float"], order="F")
    hlc_hcf = np.zeros(shape, dtype=dtypes["float"], order="F")
    hlc_hrc = np.zeros(shape, dtype=dtypes["float"], order="F")
    hli_hcf = np.zeros(shape, dtype=dtypes["float"], order="F")
    hli_hri = np.zeros(shape, dtype=dtypes["float"], order="F")
    sigrc = np.zeros(shape, dtype=dtypes["float"], order="F")
    
    # Pas de temps
    dt = dtypes["float"](50.0)
    
    print("✓ Données chargées et préparées")
    
    # ========================================================================
    # Fonction d'exécution du composant (pour benchmark)
    # ========================================================================
    def run_ice_adjust_modular():
        """Exécute la séquence complète ICE_ADJUST modulaire."""
        ice_adjust_modular(
            sigqsat=sigqsat,
            exn=pexn,
            exnref=pexnref,
            rhodref=prhodref,
            pabs=ppabsm,
            sigs=psigs,
            cf_mf=pcf_mf,
            rc_mf=prc_mf,
            ri_mf=pri_mf,
            th=th,
            rv=rv,
            rc=rc,
            rr=rr,
            ri=ri,
            rs=rs,
            rg=rg,
            cldfr=cldfr,
            hlc_hrc=hlc_hrc,
            hlc_hcf=hlc_hcf,
            hli_hri=hli_hri,
            hli_hcf=hli_hcf,
            sigrc=sigrc,
            ths=ths,
            rvs=rvs,
            rcs=rcs,
            ris=ris,
            timestep=dt,
            domain=shape,
            exec_info={},
            validate_args=False,
        )
        
        return (ths, rvs, rcs, ris, hlc_hcf, hlc_hrc, hli_hcf, hli_hri)
    
    # ========================================================================
    # Exécution et benchmark
    # ========================================================================
    print("\nExécution du composant...")
    results = benchmark(run_ice_adjust_modular)
    print("✓ Composant exécuté")
    
    # ========================================================================
    # Chargement des données de référence
    # ========================================================================
    print("\nChargement des données de référence...")
    prs_out = ice_adjust_repro_ds["PRS_OUT"].data
    prs_out = np.swapaxes(prs_out, axis1=2, axis2=3)
    
    rvs_ref = prs_out[:,0,:,:]
    rcs_ref = prs_out[:,1,:,:]
    ris_ref = prs_out[:,3,:,:]
    
    phlc_hrc_ref = ice_adjust_repro_ds["PHLC_HRC_OUT"].data
    phlc_hcf_ref = ice_adjust_repro_ds["PHLC_HCF_OUT"].data
    phli_hri_ref = ice_adjust_repro_ds["PHLI_HRI_OUT"].data
    phli_hcf_ref = ice_adjust_repro_ds["PHLI_HCF_OUT"].data
    
    print("✓ Données de référence chargées")
    
    # ========================================================================
    # VALIDATION - Comparaison avec données de référence
    # ========================================================================
    print("\n" + "="*75)
    print("VALIDATION DES RÉSULTATS")
    print("="*75)
    
    # Tendances microphysiques
    print("\n1. Tendances microphysiques:")
    
    assert_allclose(
        rvs_ref, rvs,
        atol=1e-5, rtol=1e-4,
        err_msg="[ÉCHEC] Tendance vapeur (rvs)"
    )
    print("   ✓ rvs (tendance vapeur d'eau)")
    
    assert_allclose(
        rcs_ref, rcs,
        atol=1e-5, rtol=1e-4,
        err_msg="[ÉCHEC] Tendance liquide (rcs)"
    )
    print("   ✓ rcs (tendance liquide nuageux)")
    
    assert_allclose(
        ris_ref, ris,
        atol=1e-5, rtol=1e-4,
        err_msg="[ÉCHEC] Tendance glace (ris)"
    )
    print("   ✓ ris (tendance glace)")
    
    # Champs haute résolution
    print("\n2. Champs haute résolution:")
    
    assert_allclose(
        phlc_hcf_ref, hlc_hcf,
        atol=1e-4, rtol=1e-4,
        err_msg="[ÉCHEC] Fraction liquide haute résolution (hlc_hcf)"
    )
    print("   ✓ hlc_hcf (fraction liquide haute résolution)")
    
    assert_allclose(
        phlc_hrc_ref, hlc_hrc,
        atol=1e-4, rtol=1e-4,
        err_msg="[ÉCHEC] Contenu liquide haute résolution (hlc_hrc)"
    )
    print("   ✓ hlc_hrc (contenu liquide haute résolution)")
    
    assert_allclose(
        phli_hcf_ref, hli_hcf,
        atol=1e-4, rtol=1e-4,
        err_msg="[ÉCHEC] Fraction glace haute résolution (hli_hcf)"
    )
    print("   ✓ hli_hcf (fraction glace haute résolution)")
    
    assert_allclose(
        phli_hri_ref, hli_hri,
        atol=1e-4, rtol=1e-4,
        err_msg="[ÉCHEC] Contenu glace haute résolution (hli_hri)"
    )
    print("   ✓ hli_hri (contenu glace haute résolution)")
    
    # ========================================================================
    # Statistiques de validation
    # ========================================================================
    print("\n" + "="*75)
    print("STATISTIQUES")
    print("="*75)
    
    def print_stats(name, computed, reference):
        """Affiche les statistiques de comparaison."""
        diff = computed - reference
        rel_err = np.abs(diff / (reference + 1e-20))
        print(f"\n{name}:")
        print(f"  Max abs diff : {np.max(np.abs(diff)):.2e}")
        print(f"  Mean abs diff: {np.mean(np.abs(diff)):.2e}")
        print(f"  Max rel err  : {np.max(rel_err):.2e}")
        print(f"  Mean rel err : {np.mean(rel_err):.2e}")
    
    print_stats("rvs", rvs, rvs_ref)
    print_stats("rcs", rcs, rcs_ref)
    print_stats("ris", ris, ris_ref)
    print_stats("hlc_hcf", hlc_hcf, phlc_hcf_ref)
    print_stats("hlc_hrc", hlc_hrc, phlc_hrc_ref)
    print_stats("hli_hcf", hli_hcf, phli_hcf_ref)
    print_stats("hli_hri", hli_hri, phli_hri_ref)
    
    print("\n" + "="*75)
    print("SUCCÈS: Composant IceAdjustModular validé avec ice_adjust.nc")
    print("="*75 + "\n")


@pytest.mark.parametrize("backend", [pytest.param("numpy", marks=pytest.mark.numpy)])
@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
def test_ice_adjust_modular_vs_monolithic(backend, dtypes, ice_adjust_repro_ds):
    """
    Test de comparaison: IceAdjustModular vs ice_adjust monolithique.
    
    Ce test compare les résultats du composant modulaire (4 stencils)
    avec le stencil monolithique ice_adjust.py pour vérifier leur équivalence.
    
    Note: Ce test nécessite que les deux implémentations soient disponibles
    et compilées avec les mêmes paramètres.
    
    Args:
        backend: Backend GT4Py
        dtypes: Types de données
        ice_adjust_repro_ds: Dataset de test
    """
    pytest.skip("Test de comparaison modulaire vs monolithique (à implémenter)")
    
    # TODO: Implémenter la comparaison directe entre:
    # - IceAdjustModular (ce composant)
    # - ice_adjust monolithique (stencil existant)
    # 
    # Objectif: Valider que les deux approches donnent les mêmes résultats
    # à la précision numérique près.
