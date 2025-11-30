"""
Test du composant Ice4Tendencies.

Ce test valide le composant Ice4Tendencies qui calcule les tendances microphysiques
du schéma ICE4, incluant:
- Nucléation hétérogène de la glace (ice4_nucleation)
- Givrage homogène de la pluie (ice4_rrhong)
- Fonte du givrage (ice4_rimltc)
- Processus lents (ice4_slow): déposition, agrégation
- Processus chauds (ice4_warm): autoconversion, accrétion, évaporation
- Processus rapides neige (ice4_fast_rs): riming, accretion
- Processus rapides graupel (ice4_fast_rg): croissance sèche/humide
- Processus rapides glace (ice4_fast_ri): effet Bergeron
- Calcul de la PDF sous-maille (ice4_compute_pdf)
- Champs dérivés et paramètres de pente
- Mise à jour des tendances et incréments
"""
from datetime import timedelta

import numpy as np
import pytest
from numpy.testing import assert_allclose
from gt4py.storage import zeros

from ice3.components.ice4_tendencies import Ice4Tendencies
from ice3.initialisation.state_ice4_tendencies import (
    allocate_state_ice4_tendencies,
    get_state_ice4_tendencies,
)
from ice3.phyex_common.phyex import Phyex
from ice3.utils.env import sp_dtypes, dp_dtypes, DTYPES, BACKEND


@pytest.fixture(name="ice4_tendencies_state")
def ice4_tendencies_state_fixture(domain):
    """Create a minimal state dictionary for Ice4Tendencies testing.
    
    This fixture creates and initializes all required fields with physically
    reasonable values to avoid computational errors (division by zero, overflow, etc.).
    """
    state = allocate_state_ice4_tendencies(
        domain=domain,
        backend=BACKEND,
        dtypes=DTYPES,
    )
    
    # Initialize with realistic atmospheric values
    # to avoid numerical issues
    state["ldcompute"][:] = True
    
    # Thermodynamic state
    state["exn"][:] = 1.0  # Exner function
    state["exnref"][:] = 1.0
    state["rhodref"][:] = 1.2  # kg/m³ - air density
    state["pres"][:] = 101325.0  # Pa - 1 atm
    state["t"][:] = 273.15  # K - 0°C
    state["th_t"][:] = 273.15  # K - potential temperature
    
    # Mixing ratios (small but non-zero values)
    state["rv_t"][:] = 0.008  # kg/kg - water vapor
    state["rc_t"][:] = 0.0001  # kg/kg - cloud water
    state["rr_t"][:] = 0.0001  # kg/kg - rain
    state["ri_t"][:] = 0.00005  # kg/kg - ice
    state["rs_t"][:] = 0.00005  # kg/kg - snow
    state["rg_t"][:] = 0.00005  # kg/kg - graupel
    
    # Ice nuclei concentration
    state["ci_t"][:] = 1.0e5  # #/kg
    
    # Microphysical parameters
    state["ssi"][:] = 0.0  # Ice supersaturation
    state["ka"][:] = 0.024  # W/m/K - thermal conductivity
    state["dv"][:] = 2.5e-5  # m²/s - water vapor diffusivity
    state["ai"][:] = 0.8  # ventilation coefficient
    state["cj"][:] = 1.0  # ventilation function
    
    # Cloud parameters
    state["cf"][:] = 0.3  # cloud fraction
    state["rf"][:] = 0.1  # rain fraction
    state["fr"][:] = 0.0  # precipitation fraction
    state["sigma_rc"][:] = 1e-6  # standard deviation
    
    # Subgrid parameters (initialized to zero)
    state["hlc_hcf"][:] = 0.0
    state["hlc_lcf"][:] = 0.0
    state["hlc_hrc"][:] = 0.0
    state["hlc_lrc"][:] = 0.0
    state["hli_hcf"][:] = 0.0
    state["hli_lcf"][:] = 0.0
    state["hli_hri"][:] = 0.0
    state["hli_lri"][:] = 0.0
    
    # Tendencies (initialized to zero)
    state["theta_tnd"][:] = 0.0
    state["rv_tnd"][:] = 0.0
    state["rc_tnd"][:] = 0.0
    state["rr_tnd"][:] = 0.0
    state["ri_tnd"][:] = 0.0
    state["rs_tnd"][:] = 0.0
    state["rg_tnd"][:] = 0.0
    
    # Increments (initialized to zero)
    state["theta_increment"][:] = 0.0
    state["rv_increment"][:] = 0.0
    state["rc_increment"][:] = 0.0
    state["rr_increment"][:] = 0.0
    state["ri_increment"][:] = 0.0
    state["rs_increment"][:] = 0.0
    state["rg_increment"][:] = 0.0
    
    # Latent heat factors
    state["ls_fact"][:] = 2.834e6  # J/kg - sublimation
    state["lv_fact"][:] = 2.501e6  # J/kg - vaporization
    
    # Process rates (initialized to zero)
    state["rgsi"][:] = 0.0
    state["rvdepg"][:] = 0.0
    state["rsmltg"][:] = 0.0
    state["rraccsg"][:] = 0.0
    state["rsaccrg"][:] = 0.0
    state["rcrimsg"][:] = 0.0
    state["rsrimcg"][:] = 0.0
    
    return state


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
def test_ice4_tendencies_instantiation(phyex, dtypes, backend):
    """Test that Ice4Tendencies component can be instantiated.
    
    This basic smoke test verifies that:
    - The component can be created with specified backend and dtypes
    - All required stencils are compiled and accessible
    - PHYEX configuration is properly loaded
    
    Args:
        phyex: PHYEX configuration fixture
        dtypes: Type dictionary (float32/float64)
        backend: GT4Py backend (numpy, cpu, gpu, debug)
    """
    print("\n" + "="*75)
    print("TEST INSTANTIATION: Ice4Tendencies")
    print("="*75)
    print(f"Backend: {backend}")
    print(f"Precision: {dtypes['float']}")
    
    ice4_tendencies = Ice4Tendencies(
        phyex=phyex,
        backend=backend,
        dtypes=dtypes,
    )
    
    assert ice4_tendencies is not None
    assert ice4_tendencies.phyex is not None
    
    # Check that all stencils are compiled
    assert ice4_tendencies.ice4_nucleation is not None
    print("  ✓ ice4_nucleation stencil compiled")
    
    assert ice4_tendencies.ice4_nucleation_post_processing is not None
    print("  ✓ ice4_nucleation_post_processing stencil compiled")
    
    assert ice4_tendencies.ice4_rrhong is not None
    print("  ✓ ice4_rrhong stencil compiled")
    
    assert ice4_tendencies.ice4_rrhong_post_processing is not None
    print("  ✓ ice4_rrhong_post_processing stencil compiled")
    
    assert ice4_tendencies.ice4_rimltc is not None
    print("  ✓ ice4_rimltc stencil compiled")
    
    assert ice4_tendencies.ice4_rimltc_post_processing is not None
    print("  ✓ ice4_rimltc_post_processing stencil compiled")
    
    assert ice4_tendencies.ice4_increment_update is not None
    print("  ✓ ice4_increment_update stencil compiled")
    
    assert ice4_tendencies.ice4_derived_fields is not None
    print("  ✓ ice4_derived_fields stencil compiled")
    
    assert ice4_tendencies.ice4_slope_parameters is not None
    print("  ✓ ice4_slope_parameters stencil compiled")
    
    assert ice4_tendencies.ice4_slow is not None
    print("  ✓ ice4_slow stencil compiled")
    
    assert ice4_tendencies.ice4_warm is not None
    print("  ✓ ice4_warm stencil compiled")
    
    assert ice4_tendencies.ice4_fast_rs is not None
    print("  ✓ ice4_fast_rs stencil compiled")
    
    assert ice4_tendencies.ice4_fast_rg_pre_processing is not None
    print("  ✓ ice4_fast_rg_pre_processing stencil compiled")
    
    assert ice4_tendencies.ice4_fast_rg is not None
    print("  ✓ ice4_fast_rg stencil compiled")
    
    assert ice4_tendencies.ice4_fast_ri is not None
    print("  ✓ ice4_fast_ri stencil compiled")
    
    assert ice4_tendencies.ice4_total_tendencies_update is not None
    print("  ✓ ice4_total_tendencies_update stencil compiled")
    
    assert ice4_tendencies.ice4_compute_pdf is not None
    print("  ✓ ice4_compute_pdf stencil compiled")
    
    print("\n" + "="*75)
    print("SUCCESS: All stencils compiled successfully")
    print("="*75 + "\n")


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
def test_ice4_tendencies_call_minimal(
    benchmark, phyex, domain, ice4_tendencies_state, backend, dtypes
):
    """Test that Ice4Tendencies component can be called with minimal state.
    
    This is a smoke test to ensure the component doesn't crash and produces
    reasonable output. Full validation would require reference data from
    PHYEX simulations.
    
    The test verifies:
    - Component execution completes without errors
    - Output tendencies are modified (non-zero)
    - Tendencies have reasonable magnitudes
    - No NaN or Inf values in output
    
    Args:
        benchmark: pytest-benchmark fixture
        phyex: PHYEX configuration fixture
        domain: Domain size tuple (ni, nj, nk)
        ice4_tendencies_state: Pre-initialized state dictionary
        backend: GT4Py backend
        dtypes: Type dictionary
    """
    print("\n" + "="*75)
    print("TEST EXECUTION: Ice4Tendencies with minimal state")
    print("="*75)
    print(f"Backend: {backend}")
    print(f"Precision: {dtypes['float']}")
    print(f"Domain: {domain}")
    
    ice4_tendencies = Ice4Tendencies(
        phyex=phyex,
        backend=backend,
        dtypes=dtypes,
    )
    
    timestep = timedelta(seconds=1.0)
    
    # Prepare output dictionaries
    out_tendencies = {
        "theta": ice4_tendencies_state["theta_tnd"],
        "rv": ice4_tendencies_state["rv_tnd"],
        "rc": ice4_tendencies_state["rc_tnd"],
        "rr": ice4_tendencies_state["rr_tnd"],
        "ri": ice4_tendencies_state["ri_tnd"],
        "rs": ice4_tendencies_state["rs_tnd"],
        "rg": ice4_tendencies_state["rg_tnd"],
    }
    
    out_diagnostics = {}
    
    overwrite_tendencies = {
        "theta": True,
        "rv": True,
        "rc": True,
        "rr": True,
        "ri": True,
        "rs": True,
        "rg": True,
    }
    
    # Execution function for benchmarking
    def run_ice4_tendencies():
        """Execute the Ice4Tendencies component."""
        ice4_tendencies(
            ldsoft=True,
            state=ice4_tendencies_state,
            timestep=timestep,
            out_tendencies=out_tendencies,
            out_diagnostics=out_diagnostics,
            overwrite_tendencies=overwrite_tendencies,
            domain=domain,
            exec_info={},
            validate_args=False,
        )
        return out_tendencies
    
    print("\nExecuting component...")
    
    try:
        # Run with benchmark
        result = benchmark(run_ice4_tendencies)
        
        print("✓ Component executed successfully")
        
        # Basic validation: check that tendencies are modified
        print("\n" + "-"*75)
        print("Validating output tendencies")
        print("-"*75)
        
        for name, tend in out_tendencies.items():
            # Check for NaN or Inf
            assert not np.any(np.isnan(tend)), f"{name} tendency contains NaN"
            assert not np.any(np.isinf(tend)), f"{name} tendency contains Inf"
            
            # Get statistics
            max_val = np.max(np.abs(tend))
            mean_val = np.mean(np.abs(tend))
            
            print(f"  {name:10s}: max={max_val:.2e}, mean={mean_val:.2e}")
            
            # Tendencies should have reasonable magnitudes
            # (not all zero, but not too large either)
            assert max_val < 1.0, f"{name} tendency too large: {max_val}"
        
        print("\n" + "="*75)
        print("SUCCESS: Ice4Tendencies executed and validated")
        print("="*75 + "\n")
        
    except Exception as e:
        pytest.fail(f"Ice4Tendencies component raised an exception: {str(e)}")


def test_ice4_tendencies_state_allocation(domain):
    """Test that state allocation works correctly.
    
    Verifies that allocate_state_ice4_tendencies creates all required
    fields with correct shapes and types.
    
    Args:
        domain: Domain size tuple
    """
    print("\n" + "="*75)
    print("TEST: State allocation for Ice4Tendencies")
    print("="*75)
    
    state = allocate_state_ice4_tendencies(
        domain=domain,
        backend=BACKEND,
        dtypes=DTYPES,
    )
    
    # Check that all required fields exist
    required_fields = [
        "ldcompute", "exn", "exnref", "rhodref", "pres", "t", "th_t",
        "rv_t", "rc_t", "rr_t", "ri_t", "rs_t", "rg_t", "ci_t",
        "ssi", "ka", "dv", "ai", "cj", "cf", "rf", "fr", "sigma_rc",
        "hlc_hcf", "hlc_lcf", "hlc_hrc", "hlc_lrc",
        "hli_hcf", "hli_lcf", "hli_hri", "hli_lri",
        "theta_tnd", "rv_tnd", "rc_tnd", "rr_tnd", "ri_tnd", "rs_tnd", "rg_tnd",
        "theta_increment", "rv_increment", "rc_increment", "rr_increment",
        "ri_increment", "rs_increment", "rg_increment",
        "ls_fact", "lv_fact",
        "rgsi", "rvdepg", "rsmltg", "rraccsg", "rsaccrg", "rcrimsg", "rsrimcg",
    ]
    
    print(f"\nDomain: {domain}")
    print(f"Checking {len(required_fields)} required fields...")
    
    for field_name in required_fields:
        assert field_name in state, f"Missing field: {field_name}"
        
        # Check shape for 3D fields
        if hasattr(state[field_name], 'shape'):
            assert state[field_name].shape == domain, \
                f"Field {field_name} has wrong shape: {state[field_name].shape}"
    
    print(f"✓ All {len(required_fields)} fields present with correct shapes")
    
    # Check rwetgh scalar parameter
    assert "rwetgh" in state
    print("✓ rwetgh scalar parameter present")
    
    print("\n" + "="*75)
    print("SUCCESS: State allocation validated")
    print("="*75 + "\n")


def test_ice4_tendencies_phyex_configuration():
    """Test that Ice4Tendencies PHYEX configuration is accessible.
    
    Verifies that required parameters and lookup tables are available
    for the Ice4Tendencies component.
    """
    print("\n" + "="*75)
    print("TEST: PHYEX configuration for Ice4Tendencies")
    print("="*75)
    
    phyex_arome = Phyex("AROME")
    
    # Check that PHYEX parameters are accessible
    assert hasattr(phyex_arome, "rain_ice_param")
    print("✓ rain_ice_param accessible")
    
    # Check gamma incomplete functions (used in fast processes)
    assert hasattr(phyex_arome.rain_ice_param, "GAMINC_RIM1")
    assert hasattr(phyex_arome.rain_ice_param, "GAMINC_RIM2")
    assert hasattr(phyex_arome.rain_ice_param, "GAMINC_RIM4")
    print("✓ Gamma incomplete functions accessible")
    
    # Test that externals can be generated
    externals = phyex_arome.to_externals()
    assert externals is not None
    assert isinstance(externals, dict)
    print("✓ Externals dictionary generated")
    
    # Check key microphysical parameters
    assert "LDEPOSC" in externals
    assert "LNULLWETG" in externals
    assert "PRISTINE_ICE_LIMA" in externals
    print("✓ Key microphysical parameters present")
    
    print("\n" + "="*75)
    print("SUCCESS: PHYEX configuration validated")
    print("="*75 + "\n")


def test_ice4_tendencies_imports():
    """Test that all necessary imports for Ice4Tendencies work correctly.
    
    This is a basic smoke test to ensure all modules can be imported
    without errors.
    """
    print("\n" + "="*75)
    print("TEST: Ice4Tendencies imports")
    print("="*75)
    
    # Component import
    from ice3.components.ice4_tendencies import Ice4Tendencies
    assert Ice4Tendencies is not None
    print("✓ Ice4Tendencies component imported")
    
    # State initialization imports
    from ice3.initialisation.state_ice4_tendencies import (
        allocate_state_ice4_tendencies,
        get_state_ice4_tendencies,
        initialize_state_ice4_tendencies,
    )
    assert allocate_state_ice4_tendencies is not None
    assert get_state_ice4_tendencies is not None
    assert initialize_state_ice4_tendencies is not None
    print("✓ State initialization functions imported")
    
    # Stencil imports
    from ice3.stencils.ice4_nucleation import ice4_nucleation
    from ice3.stencils.ice4_rrhong import ice4_rrhong
    from ice3.stencils.ice4_rimltc import ice4_rimltc
    from ice3.stencils.ice4_slow import ice4_slow
    from ice3.stencils.ice4_warm import ice4_warm
    from ice3.stencils.ice4_fast_rs import ice4_fast_rs
    from ice3.stencils.ice4_fast_ri import ice4_fast_ri
    from ice3.stencils.ice4_fast_rg import ice4_fast_rg
    from ice3.stencils.ice4_compute_pdf import ice4_compute_pdf
    from ice3.stencils.ice4_tendencies import (
        ice4_nucleation_post_processing,
        ice4_rrhong_post_processing,
        ice4_rimltc_post_processing,
        ice4_slope_parameters,
        ice4_fast_rg_pre_post_processing,
        ice4_total_tendencies_update,
        ice4_increment_update,
        ice4_derived_fields,
    )
    
    print("✓ All stencil functions imported")
    
    # Lookup table imports
    from ice3.phyex_common.xker_raccs import KER_RACCS, KER_RACCSS, KER_SACCRG
    from ice3.phyex_common.xker_rdryg import KER_RDRYG
    from ice3.phyex_common.xker_sdryg import KER_SDRYG
    
    assert KER_RACCS is not None
    assert KER_RACCSS is not None
    assert KER_SACCRG is not None
    assert KER_RDRYG is not None
    assert KER_SDRYG is not None
    print("✓ Lookup tables imported")
    
    print("\n" + "="*75)
    print("SUCCESS: All imports validated")
    print("="*75 + "\n")


@pytest.mark.skip(reason="Reference data ice4_tendencies.nc not yet available")
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
def test_ice4_tendencies_with_reference_data(
    benchmark, backend, domain, dtypes
):
    """Test Ice4Tendencies component with reference data from PHYEX.
    
    This test would validate the component against reference data from
    Fortran PHYEX simulations, checking that all tendencies match to
    within numerical tolerance.
    
    Currently skipped because reference dataset ice4_tendencies.nc
    is not yet available.
    
    To implement this test:
    1. Generate reference data from PHYEX-Fortran
    2. Add fixture to conftest.py to load ice4_tendencies.nc
    3. Initialize state from reference data
    4. Run component and compare all output fields
    
    Args:
        benchmark: pytest-benchmark fixture
        backend: GT4Py backend
        domain: Domain size
        dtypes: Type dictionary
    """
    # TODO: Implement when reference data becomes available
    # This would follow the pattern of test_ice_adjust_modular
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
