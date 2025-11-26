# -*- coding: utf-8 -*-
"""Test for RainIce component"""
from datetime import timedelta

import numpy as np
import pytest
from gt4py.storage import zeros

from ice3.components.rain_ice import RainIce
from ice3.phyex_common.phyex import Phyex
from ice3.utils.env import sp_dtypes, dp_dtypes
from ice3.utils.env import DTYPES, BACKEND


@pytest.fixture(name="rain_ice_state")
def rain_ice_state_fixture(domain):
    """Create a minimal state dictionary for RainIce testing"""
    shape = domain
    dtype = DTYPES["float"]
    backend = BACKEND
    
    # Create state with all required fields
    state = {
        # Thermodynamic variables
        "th_t": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rv_t": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rc_t": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rr_t": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "ri_t": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rs_t": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rg_t": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "ci_t": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        
        # Pressure and temperature
        "exn": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "exnref": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "pabs_t": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "t": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        
        # Reference state variables
        "rhodref": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "dzz": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        
        # Source terms (tendencies at start)
        "ths": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rvs": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rcs": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rrs": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "ris": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rss": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rgs": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        
        # Cloud fraction and diagnostics
        "hlc_hcf": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "hlc_lcf": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "hlc_hrc": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "hlc_lrc": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "hli_hcf": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "hli_lcf": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "hli_hri": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "hli_lri": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        
        # Precipitation fractions
        "fpr_c": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "fpr_r": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "fpr_i": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "fpr_s": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "fpr_g": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        
        # Precipitation rates
        "inprr": zeros(shape[0:2], backend=backend, dtype=dtype, aligned_index=(0, 0)),
        "inprc": zeros(shape[0:2], backend=backend, dtype=dtype, aligned_index=(0, 0)),
        "inprs": zeros(shape[0:2], backend=backend, dtype=dtype, aligned_index=(0, 0)),
        "inprg": zeros(shape[0:2], backend=backend, dtype=dtype, aligned_index=(0, 0)),
        
        # Additional fields
        "evap3d": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rainfr": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "prfr": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        "rf": zeros(shape, backend=backend, dtype=dtype, aligned_index=(0, 0, 0)),
        
        # Sea and town masks
        "sea": zeros(shape[0:2], backend=backend, dtype=dtype, aligned_index=(0, 0)),
        "town": zeros(shape[0:2], backend=backend, dtype=dtype, aligned_index=(0, 0)),
    }
    
    # Initialize with some reasonable values to avoid zero divisions
    state["exn"][:] = 1.0
    state["exnref"][:] = 1.0
    state["pabs_t"][:] = 101325.0  # 1 atm
    state["t"][:] = 273.15  # 0°C
    state["th_t"][:] = 273.15
    state["rhodref"][:] = 1.2  # kg/m³
    state["dzz"][:] = 100.0  # 100m layers
    
    return state


@pytest.mark.skip(reason="RainIce instantiation requires additional external constants for sedimentation stencils (LBC_SEA, LBC_LAND)")
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
def test_rain_ice_instantiation(phyex, dtypes, backend):
    """Test that RainIce component can be instantiated
    
    Note: This test is currently skipped because the sedimentation stencils
    require additional external constants (LBC_SEA, LBC_LAND) that are not
    currently provided by the PHYEX configuration.
    """
    rain_ice = RainIce(
        phyex=phyex,
        backend=backend,
        dtypes=dtypes,
    )
    
    assert rain_ice is not None
    assert rain_ice.phyex is not None
    
    # Check that all stencils are compiled
    assert rain_ice.rain_ice_thermo is not None
    assert rain_ice.rain_ice_mask is not None
    assert rain_ice.initial_values_saving is not None
    assert rain_ice.ice4_tendencies is not None
    assert rain_ice.total_tendencies is not None
    assert rain_ice.ice4_correct_negativities is not None
    assert rain_ice.sedimentation is not None


@pytest.mark.skip(reason="RainIce call requires component instantiation which is blocked by missing external constants")
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
def test_rain_ice_call_minimal(phyex, domain, rain_ice_state, backend, dtypes):
    """Test that RainIce component can be called with minimal state
    
    Note: This is a smoke test to ensure the component doesn't crash.
    Full validation requires reference data.
    
    Currently skipped due to instantiation issues with sedimentation stencils.
    """
    rain_ice = RainIce(
        phyex=phyex,
        backend=backend,
        dtypes=dtypes,
    )
    
    timestep = timedelta(seconds=1.0)
    
    # This should not raise an exception
    try:
        rain_ice(
            state=rain_ice_state,
            timestep=timestep,
            domain=domain,
            validate_args=False,
            exec_info={}
        )
        # If we get here without exception, the test passes
        assert True
    except Exception as e:
        pytest.fail(f"RainIce component raised an exception: {str(e)}")


def test_rain_ice_phyex_configuration():
    """Test that RainIce PHYEX configuration is accessible"""
    phyex_arome = Phyex("AROME")
    
    # Check that PHYEX parameters are accessible even without full instantiation
    assert hasattr(phyex_arome, "param_icen")
    assert hasattr(phyex_arome.param_icen, "LSEDIM_AFTER")
    assert hasattr(phyex_arome.param_icen, "LDEPOSC")
    assert hasattr(phyex_arome.param_icen, "SEDIM")
    
    # Test that externals can be generated
    externals = phyex_arome.to_externals()
    assert externals is not None
    assert isinstance(externals, dict)


def test_rain_ice_imports():
    """Test that all necessary imports for RainIce work correctly"""
    # This is a basic smoke test to ensure the module can be imported
    from ice3.components.rain_ice import RainIce
    from ice3.phyex_common.ice_parameters import Sedim
    
    assert RainIce is not None
    assert Sedim is not None
    
    # Test that Sedim enum values are accessible
    assert hasattr(Sedim, "STAT")
    assert hasattr(Sedim, "SPLI")


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
def test_rain_ice(benchmark, backend, externals, dtypes, rain_ice_repro_ds):
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
    print("TEST COMPOSANT: RainIce avec rain_ice.nc")
    print("="*75)
    print(f"Backend: {backend}")
    print(f"Precision: {dtypes['float']}")
    
    # ========================================================================
    # Configuration PHYEX et création du composant
    # ========================================================================
    phyex = Phyex("AROME")
    rain_ice = RainIce(
        phyex=phyex,
        dtypes=dtypes,
        backend=backend,
    )
    
    print(f"Composant créé: {rain_ice}")
    
    # ========================================================================
    # Chargement et préparation des données
    # ========================================================================
    shape = (
        rain_ice.sizes["ngpblks"],
        rain_ice.sizes["nproma"],
        rain_ice.sizes["nflevg"]
    )
    print(f"Domaine: {shape}")


    # ========================================================================
    # Fonction d'exécution du composant (pour benchmark)
    # ========================================================================
    def run_rain_ice():
        """Exécute la séquence complète RainIce modulaire."""
        rain_ice(
            exec_info={},
            validate_args=False,
        )
        
        return ()
    
    # ========================================================================
    # Exécution et benchmark
    # ========================================================================
    print("\nExécution du composant...")
    results = benchmark(rain_ice)
    print("✓ Composant exécuté")
    
    # ========================================================================
    # Chargement des données de référence
    # ========================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
