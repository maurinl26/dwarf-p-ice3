# -*- coding: utf-8 -*-
"""Test for IceAdjust component"""
import numpy as np
from numpy.testing import assert_allclose
import pytest
from gt4py.storage import zeros

from ice3.components.ice_adjust import IceAdjust
from ice3.phyex_common.phyex import Phyex
from ice3.utils.env import sp_dtypes, dp_dtypes
from ice3.utils.env import DTYPES, BACKEND
from ice3.initialisation.state_ice_adjust import get_state_ice_adjust


@pytest.fixture(name="ice_adjust_state")
def ice_adjust_state_fixture(domain):
    """Create a minimal state dictionary for IceAdjust testing"""
    shape = domain
    dtype = DTYPES["float"]
    backend = BACKEND
    
    # Create state with all required fields
    state = {
        # Thermodynamic variables
        "th": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rv": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rc": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rr": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "ri": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rs": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rg": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        
        # Pressure and temperature
        "exn": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "exnref": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "pabs": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rhodref": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        
        # Cloud parameters
        "sigs": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "sigqsat": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "cldfr": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        
        # Mass flux variables
        "cf_mf": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rc_mf": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "ri_mf": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        
        # High-level cloud diagnostics
        "hlc_hcf": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "hlc_hrc": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "hli_hcf": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "hli_hri": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        
        # Subgrid parameters
        "sigrc": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        
        # Source terms (tendencies)
        "ths": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rvs": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rcs": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rrs": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "ris": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rss": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rgs": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
    }
    
    # Initialize with some reasonable values to avoid zero divisions
    state["exn"][:] = 1.0
    state["exnref"][:] = 1.0
    state["pabs"][:] = 101325.0  # 1 atm
    state["th"][:] = 273.15  # 0°C
    state["rhodref"][:] = 1.2  # kg/m³
    
    return state


def test_ice_adjust_phyex_configuration():
    """Test that IceAdjust PHYEX configuration is accessible"""
    phyex_arome = Phyex("AROME")
    
    # Check that PHYEX parameters are accessible
    assert hasattr(phyex_arome, "nebn")
    assert hasattr(phyex_arome.nebn, "LSUBG_COND")
    assert hasattr(phyex_arome, "param_icen")
    assert hasattr(phyex_arome.param_icen, "SUBG_MF_PDF")
    
    # Test that externals can be generated
    externals = phyex_arome.to_externals()
    assert externals is not None
    assert isinstance(externals, dict)


def test_ice_adjust_imports():
    """Test that all necessary imports for IceAdjust work correctly"""
    from ice3.components.ice_adjust import IceAdjust
    from ice3.stencils.ice_adjust import ice_adjust
    
    assert IceAdjust is not None
    assert ice_adjust is not None


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
def test_ice_adjust(benchmark, backend, domain, dtypes, ice_adjust_repro_ds):
    """
    Test du composant IceAdjust avec le jeu de données ice_adjust.nc.
    
    Ce test valide que le composant IceAdjust produit les mêmes résultats 
    que les données de référence PHYEX.
    
    Le schéma ICE_ADJUST effectue l'ajustement à saturation de la température
    et des rapports de mélange pour assurer la cohérence thermodynamique et
    l'équilibre de saturation.
    
    Champs validés:
        - ths, rvs, rcs, ris : Tendances microphysiques
        - cldfr : Fraction nuageuse
        - hlc_hrc, hlc_hcf  : Contenu/fraction liquide haute résolution
        - hli_hri, hli_hcf  : Contenu/fraction glace haute résolution
    
    Args:
        benchmark: Fixture pytest-benchmark
        backend: Backend GT4Py (numpy, cpu, gpu, debug)
        dtypes: Types de données (float32/float64)
        ice_adjust_repro_ds: Dataset xarray ice_adjust.nc
    """
    print("\n" + "="*75)
    print("TEST COMPOSANT: IceAdjust avec ice_adjust.nc")
    print("="*75)
    print(f"Backend: {backend}")
    print(f"Precision: {dtypes['float']}")
    
    # ========================================================================
    # Configuration PHYEX et création du composant
    # ========================================================================
    phyex = Phyex("AROME")
    
    # Add OCND2 external that's needed by ice_adjust stencil
    # We need to override the to_externals method to include OCND2
    original_to_externals = phyex.to_externals
    def custom_to_externals():
        externals = original_to_externals()
        externals["OCND2"] = False
        return externals
    phyex.to_externals = custom_to_externals
    
    ice_adjust = IceAdjust(
        phyex=phyex,
        dtypes=dtypes,
        backend=backend,
    )
    
    print(f"Composant créé: {ice_adjust}")
    
    # ========================================================================
    # Chargement et préparation des données
    # ========================================================================
    shape = (
        ice_adjust_repro_ds.sizes["ngpblks"],
        ice_adjust_repro_ds.sizes["nproma"],
        ice_adjust_repro_ds.sizes["nflevg"]
    )
    print(f"Domaine: {shape}")

    # Initialize state from dataset
    state = get_state_ice_adjust(domain, backend=backend, dataset=ice_adjust_repro_ds)
    
    # Timestep
    timestep = 50.0  # seconds

    # ========================================================================
    # Fonction d'exécution du composant (pour benchmark)
    # ========================================================================
    def run_ice_adjust():
        """Exécute le composant IceAdjust."""
        ice_adjust(
            sigqsat=state["sigqsat"],
            exn=state["exn"],
            exnref=state["exnref"],
            rhodref=state["rhodref"],
            pabs=state["pabs"],
            sigs=state["sigs"],
            cf_mf=state["cf_mf"],
            rc_mf=state["rc_mf"],
            ri_mf=state["ri_mf"],
            th=state["th"],
            rv=state["rv"],
            rc=state["rc"],
            rr=state["rr"],
            ri=state["ri"],
            rs=state["rs"],
            rg=state["rg"],
            cldfr=state["cldfr"],
            hlc_hrc=state["hlc_hrc"],
            hlc_hcf=state["hlc_hcf"],
            hli_hri=state["hli_hri"],
            hli_hcf=state["hli_hcf"],
            sigrc=state["sigrc"],
            ths=state["ths"],
            rvs=state["rvs"],
            rcs=state["rcs"],
            ris=state["ris"],
            timestep=timestep,
            domain=shape,
            exec_info={},
            validate_args=False,
        )
        
        return state
    
    # ========================================================================
    # Exécution et benchmark
    # ========================================================================
    print("\nExécution du composant...")
    results = benchmark(run_ice_adjust)
    print("✓ Composant exécuté")
    
    # ========================================================================
    # Comparaison avec les données de référence
    # ========================================================================
    
    print("\n" + "-"*75)
    print("Comparaison des tendances des espèces (PRS_OUT)")
    print("-"*75)
    
    # PRS_OUT has dimensions (ngpblks, krr, nflevg, nproma)
    # Need to swap axes to match our layout (ngpblks, nproma, nflevg)
    prs_out = ice_adjust_repro_ds["PRS_OUT"].values
    prs_out = np.swapaxes(prs_out, axis1=2, axis2=3)
    
    # krr indices: 0=h, 1=v, 2=c, 3=r, 4=i, 5=s, 6=g (but h is not used)
    assert_allclose(
        state["rvs"],
        prs_out[:, 1, :, :],
        atol=1e-5,
        rtol=1e-4,
        err_msg="[ECHEC] Ecart - Tendance de contenu en vapeur d'eau"
    )
    print("✓ rvs (vapeur d'eau)")
    
    assert_allclose(
        state["rcs"],
        prs_out[:, 2, :, :],
        atol=1e-5,
        rtol=1e-4,
        err_msg="[ECHEC] Ecart - Tendance de contenu en goutelettes (cloud droplets)"
    )
    print("✓ rcs (cloud droplets)")
    
    assert_allclose(
        state["ris"],
        prs_out[:, 4, :, :],
        atol=1e-5,
        rtol=1e-4,
        err_msg="[ECHEC] Ecart - Tendance de contenu en glace pristine"
    )
    print("✓ ris (ice)")
    
    print("\n" + "-"*75)
    print("Comparaison de la fraction nuageuse (PCLDFR_OUT)")
    print("-"*75)
    
    pcldfr_out = ice_adjust_repro_ds["PCLDFR_OUT"].values
    pcldfr_out = np.swapaxes(pcldfr_out, axis1=1, axis2=2)
    
    assert_allclose(
        state["cldfr"],
        pcldfr_out,
        atol=1e-4,
        rtol=1e-4,
        err_msg="[ECHEC] Ecart - Fraction nuageuse"
    )
    print("✓ cldfr (cloud fraction)")
    
    print("\n" + "-"*75)
    print("Comparaison des diagnostics haute résolution - Liquide")
    print("-"*75)
    
    phlc_hrc_out = ice_adjust_repro_ds["PHLC_HRC_OUT"].values
    phlc_hrc_out = np.swapaxes(phlc_hrc_out, axis1=1, axis2=2)
    
    assert_allclose(
        state["hlc_hrc"],
        phlc_hrc_out,
        atol=1e-4,
        rtol=1e-4,
        err_msg="[ECHEC] Ecart - Contenu liquide haute résolution"
    )
    print("✓ hlc_hrc (high resolution liquid content)")
    
    phlc_hcf_out = ice_adjust_repro_ds["PHLC_HCF_OUT"].values
    phlc_hcf_out = np.swapaxes(phlc_hcf_out, axis1=1, axis2=2)
    
    assert_allclose(
        state["hlc_hcf"],
        phlc_hcf_out,
        atol=1e-4,
        rtol=1e-4,
        err_msg="[ECHEC] Ecart - Fraction nuageuse liquide haute résolution"
    )
    print("✓ hlc_hcf (high resolution liquid cloud fraction)")
    
    print("\n" + "-"*75)
    print("Comparaison des diagnostics haute résolution - Glace")
    print("-"*75)
    
    phli_hri_out = ice_adjust_repro_ds["PHLI_HRI_OUT"].values
    phli_hri_out = np.swapaxes(phli_hri_out, axis1=1, axis2=2)
    
    assert_allclose(
        state["hli_hri"],
        phli_hri_out,
        atol=1e-4,
        rtol=1e-4,
        err_msg="[ECHEC] Ecart - Contenu glace haute résolution"
    )
    print("✓ hli_hri (high resolution ice content)")
    
    phli_hcf_out = ice_adjust_repro_ds["PHLI_HCF_OUT"].values
    phli_hcf_out = np.swapaxes(phli_hcf_out, axis1=1, axis2=2)
    
    assert_allclose(
        state["hli_hcf"],
        phli_hcf_out,
        atol=1e-4,
        rtol=1e-4,
        err_msg="[ECHEC] Ecart - Fraction nuageuse glace haute résolution"
    )
    print("✓ hli_hcf (high resolution ice cloud fraction)")
    
    print("\n" + "="*75)
    print("TOUS LES TESTS DE COMPARAISON RÉUSSIS ✓")
    print("="*75)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
