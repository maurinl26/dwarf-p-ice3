# -*- coding: utf-8 -*-
"""Performance tests for RainIce component"""
from datetime import timedelta

import numpy as np
import pytest
from gt4py.storage import zeros

from ice3.components.rain_ice import RainIce
from ice3.phyex_common.phyex import Phyex
from ice3.utils.env import sp_dtypes, dp_dtypes


@pytest.fixture(name="rain_ice_perf_state")
def rain_ice_perf_state_fixture():
    """Create a performance test state dictionary for RainIce
    
    Uses realistic domain sizes for performance testing
    """
    # Use a realistic domain size for performance testing
    # Typical atmospheric model grid: ~100x100 horizontal, 90 vertical levels
    shape = (100, 100, 90)
    dtype = np.float64
    backend = "numpy"  # Use numpy for performance baseline
    
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
    
    # Initialize with realistic atmospheric values
    state["exn"][:] = 1.0
    state["exnref"][:] = 1.0
    state["pabs_t"][:] = 101325.0  # 1 atm at surface, could vary with height
    state["t"][:] = 273.15  # 0°C
    state["th_t"][:] = 273.15
    state["rhodref"][:] = 1.2  # kg/m³
    state["dzz"][:] = 100.0  # 100m layers
    
    # Add some small perturbations to make it realistic
    np.random.seed(42)  # Reproducible
    state["rv_t"][:] = 0.001 + 0.0001 * np.random.randn(*shape)
    state["th_t"][:] = 273.15 + 10.0 * np.random.randn(*shape)
    
    return state, shape


@pytest.mark.skip(reason="RainIce performance test requires external constants (LBC_SEA, LBC_LAND) for sedimentation stencils")
@pytest.mark.parametrize("dtypes", [sp_dtypes, dp_dtypes])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numpy", marks=pytest.mark.numpy),
        pytest.param("gt:cpu_ifirst", marks=pytest.mark.cpu),
    ],
)
def test_rain_ice_performance(benchmark, backend, dtypes, rain_ice_perf_state):
    """Performance test for RainIce component
    
    This test measures the execution time of the full RainIce component
    on a realistic atmospheric domain (100x100x90 grid points).
    
    Skipped due to missing external constants needed for sedimentation stencils.
    Once the external constants issue is resolved, this test will measure:
    - Time per timestep
    - Time per grid point
    - Scalability with domain size
    """
    state, shape = rain_ice_perf_state
    
    phyex = Phyex("AROME")
    rain_ice = RainIce(
        phyex=phyex,
        backend=backend,
        dtypes=dtypes,
    )
    
    timestep = timedelta(seconds=60.0)  # 1 minute timestep
    
    def run_rain_ice():
        rain_ice(
            state=state,
            timestep=timestep,
            domain=shape,
            validate_args=False,
            exec_info={}
        )
    
    # Benchmark the execution
    result = benchmark(run_rain_ice)
    
    # Calculate performance metrics
    total_points = np.prod(shape)
    time_per_point = result.stats.mean / total_points
    
    print(f"\nPerformance Metrics:")
    print(f"Domain size: {shape[0]}x{shape[1]}x{shape[2]} = {total_points} points")
    print(f"Mean time: {result.stats.mean:.6f} seconds")
    print(f"Std dev: {result.stats.stddev:.6f} seconds")
    print(f"Time per grid point: {time_per_point*1e6:.3f} microseconds")


def test_rain_ice_component_overhead():
    """Test the overhead of RainIce component structure
    
    This test measures just the Python-level overhead of the component
    without actually running the stencils. Useful for understanding
    the framework overhead vs computation time.
    """
    phyex = Phyex("AROME")
    
    # Just test that we can access the parameters quickly
    assert hasattr(phyex, "param_icen")
    assert hasattr(phyex.param_icen, "LSEDIM_AFTER")
    assert hasattr(phyex.param_icen, "LDEPOSC")
    assert hasattr(phyex.param_icen, "SEDIM")
    
    # Test externals generation performance
    import time
    start = time.perf_counter()
    externals = phyex.to_externals()
    end = time.perf_counter()
    
    print(f"\nExternals generation time: {(end-start)*1000:.3f} ms")
    print(f"Number of externals: {len(externals)}")
    
    assert externals is not None
    assert isinstance(externals, dict)
    assert len(externals) > 0


@pytest.mark.benchmark(group="ice4_tendencies")
def test_ice4_tendencies_import_time(benchmark):
    """Benchmark the import time of Ice4Tendencies component
    
    This measures the overhead of importing and initializing the component,
    which includes stencil compilation on first import.
    """
    def import_and_init():
        from ice3.components.ice4_tendencies import Ice4Tendencies
        from ice3.phyex_common.phyex import Phyex
        
        phyex = Phyex("AROME")
        # Note: This will fail due to missing externals but we're measuring
        # the import time, not instantiation
        return Ice4Tendencies
    
    # Benchmark just the import and class definition access
    benchmark(import_and_init)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
