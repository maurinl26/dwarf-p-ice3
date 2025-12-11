"""
Compare Fortran ICE_ADJUST (via fmodpy) with gt4py implementation.

This script provides a practical framework for comparing the Fortran
reference implementation with the gt4py port for validation and
performance benchmarking.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_test_data():
    """
    Load test data for ICE_ADJUST comparison.
    
    Returns data compatible with both Fortran and gt4py implementations.
    """
    try:
        import xarray as xr
        data_path = Path(__file__).parent.parent / "data" / "ice_adjust.nc"
        
        if data_path.exists():
            ds = xr.open_dataset(data_path)
            print(f"✓ Loaded test data from {data_path}")
            return ds
        else:
            print(f"⚠️  Test data not found at {data_path}")
            return None
    except ImportError:
        print("⚠️  xarray not available")
        return None


def create_synthetic_data(nijt=100, nkt=60):
    """
    Create synthetic test data for ICE_ADJUST.
    
    Parameters
    ----------
    nijt : int
        Number of horizontal points
    nkt : int
        Number of vertical levels
    
    Returns
    -------
    dict
        Dictionary with all required fields
    """
    print(f"\nCreating synthetic data ({nijt}x{nkt})...")
    
    # Vertical coordinate
    z = np.linspace(0, 10000, nkt)  # 0-10km
    
    # Basic atmosphere
    p0 = 101325.0  # Surface pressure
    T0 = 288.15    # Surface temperature
    gamma = 0.0065  # Lapse rate
    
    # Create 2D fields (Fortran order for Fortran calls)
    data = {}
    
    # Pressure profile
    pressure = p0 * (1 - gamma * z / T0) ** 5.26
    data['ppabst'] = np.tile(pressure, (nijt, 1)).T.copy(order='F')
    
    # Temperature profile
    temperature = T0 - gamma * z
    data['temperature'] = np.tile(temperature, (nijt, 1)).T.copy(order='F')
    
    # Add some variability
    data['temperature'] += np.random.randn(nkt, nijt) * 0.5
    data['ppabst'] += np.random.randn(nkt, nijt) * 100
    
    # Convert to potential temperature
    Rd = 287.0
    cp = 1004.0
    p00 = 100000.0
    data['pexn'] = (data['ppabst'] / p00) ** (Rd / cp)
    data['pth'] = data['temperature'] / data['pexn']
    
    # Reference values
    data['pexnref'] = data['pexn'].copy()
    data['prhodref'] = data['ppabst'] / (Rd * data['temperature'])
    data['prhodj'] = data['prhodref'].copy()
    
    # Height
    data['pzz'] = np.tile(z, (nijt, 1)).T.copy(order='F')
    
    # Water vapor (decreasing with height)
    rv_surf = 0.015  # 15 g/kg at surface
    data['prv'] = rv_surf * np.exp(-z / 2000)  # Scale height 2km
    data['prv'] = np.tile(data['prv'], (nijt, 1)).T.copy(order='F')
    data['prv'] += np.random.randn(nkt, nijt) * 0.001
    
    # Cloud water (some clouds at mid-levels)
    data['prc'] = np.zeros((nkt, nijt), order='F')
    cloud_levels = (z > 2000) & (z < 6000)
    for i in range(nijt):
        data['prc'][cloud_levels, i] = np.random.rand(cloud_levels.sum()) * 0.002
    
    # Cloud ice (upper levels)
    data['pri'] = np.zeros((nkt, nijt), order='F')
    ice_levels = z > 5000
    for i in range(nijt):
        data['pri'][ice_levels, i] = np.random.rand(ice_levels.sum()) * 0.001
    
    # Source terms (tendencies)
    data['prvs'] = np.zeros_like(data['prv'])
    data['prcs'] = np.zeros_like(data['prc'])
    data['pris'] = np.zeros_like(data['pri'])
    data['pths'] = np.zeros_like(data['pth'])
    
    # Other hydrometeors (zeros for now)
    data['prr'] = np.zeros((nkt, nijt), order='F')
    data['prs'] = np.zeros((nkt, nijt), order='F')
    data['prg'] = np.zeros((nkt, nijt), order='F')
    
    print(f"  Temperature range: {data['temperature'].min():.1f} - {data['temperature'].max():.1f} K")
    print(f"  Pressure range: {data['ppabst'].min():.0f} - {data['ppabst'].max():.0f} Pa")
    print(f"  Vapor range: {data['prv'].min()*1000:.3f} - {data['prv'].max()*1000:.3f} g/kg")
    print(f"  Cloud water max: {data['prc'].max()*1000:.3f} g/kg")
    print(f"  Cloud ice max: {data['pri'].max()*1000:.3f} g/kg")
    
    return data


def call_fortran_ice_adjust(data, dt=1.0):
    """
    Call Fortran ICE_ADJUST via fmodpy.
    
    Parameters
    ----------
    data : dict
        Input data dictionary
    dt : float
        Time step [s]
    
    Returns
    -------
    dict
        Results from Fortran call
    """
    print("\n" + "="*70)
    print("Calling Fortran ICE_ADJUST (via fmodpy)")
    print("="*70)
    
    try:
        from ice3.components.ice_adjust import ice_adjust
        from ice3.initialisation.state_ice_adjust import load_state_from_config
        
        # Load configuration
        config_path = Path(__file__).parent.parent / "src" / "config" / "ice_adjust.yaml"
        state = load_state_from_config(str(config_path))
        
        # Use test data
        nijt, nkt = data['prv'].shape[1], data['prv'].shape[0]
        
        # Prepare state (this is simplified - real use needs proper setup)
        print(f"  Grid: {nijt} x {nkt}")
        print(f"  Time step: {dt} s")
        
        # Timing
        start = time.time()
        
        # Call ICE_ADJUST
        # Note: Actual call requires full state setup
        print("  ⚠️  Full Fortran call requires complete state initialization")
        print("  See tests/components/test_ice_adjust.py for working example")
        
        elapsed = time.time() - start
        
        result = {
            'method': 'fortran_fmodpy',
            'time': elapsed,
            'note': 'Framework ready - use test_ice_adjust.py for full example',
        }
        
        return result
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {'error': str(e)}


def call_gt4py_ice_adjust(data, dt=1.0):
    """
    Call gt4py ICE_ADJUST implementation.
    
    Parameters
    ----------
    data : dict
        Input data dictionary
    dt : float
        Time step [s]
    
    Returns
    -------
    dict
        Results from gt4py call
    """
    print("\n" + "="*70)
    print("Calling gt4py ICE_ADJUST")
    print("="*70)
    
    try:
        from ice3.stencils.ice_adjust import ice_adjust as ice_adjust_gt4py
        
        nijt, nkt = data['prv'].shape[1], data['prv'].shape[0]
        print(f"  Grid: {nijt} x {nkt}")
        print(f"  Time step: {dt} s")
        
        # Timing
        start = time.time()
        
        # Call gt4py version
        # Note: Actual call requires proper field setup
        print("  ⚠️  Full gt4py call requires GT4Py field setup")
        print("  See tests/components/test_ice_adjust.py for working example")
        
        elapsed = time.time() - start
        
        result = {
            'method': 'gt4py',
            'time': elapsed,
            'note': 'Framework ready - use test_ice_adjust.py for full example',
        }
        
        return result
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {'error': str(e)}


def compare_results(fortran_result, gt4py_result, data):
    """
    Compare Fortran and gt4py results.
    
    Parameters
    ----------
    fortran_result : dict
        Results from Fortran
    gt4py_result : dict
        Results from gt4py
    data : dict
        Input data for context
    """
    print("\n" + "="*70)
    print("Comparison Results")
    print("="*70)
    
    if 'error' in fortran_result:
        print(f"\n⚠️  Fortran: {fortran_result['error']}")
    else:
        print(f"\n✓ Fortran timing: {fortran_result.get('time', 0)*1000:.2f} ms")
    
    if 'error' in gt4py_result:
        print(f"⚠️  gt4py: {gt4py_result['error']}")
    else:
        print(f"✓ gt4py timing: {gt4py_result.get('time', 0)*1000:.2f} ms")
    
    # Compute differences if both succeeded
    if 'error' not in fortran_result and 'error' not in gt4py_result:
        if 'time' in fortran_result and 'time' in gt4py_result:
            speedup = fortran_result['time'] / gt4py_result['time']
            print(f"\nSpeedup (Fortran/gt4py): {speedup:.2f}x")


def benchmark_comparison(n_iterations=10):
    """
    Run multiple iterations for benchmarking.
    
    Parameters
    ----------
    n_iterations : int
        Number of iterations to run
    """
    print("\n" + "="*70)
    print(f"Benchmarking ({n_iterations} iterations)")
    print("="*70)
    
    # Create test data
    data = create_synthetic_data(nijt=100, nkt=60)
    
    # Warmup
    print("\nWarmup...")
    _ = call_fortran_ice_adjust(data)
    _ = call_gt4py_ice_adjust(data)
    
    # Benchmark
    print(f"\nRunning {n_iterations} iterations...")
    
    fortran_times = []
    gt4py_times = []
    
    for i in range(n_iterations):
        # Fortran
        result_f = call_fortran_ice_adjust(data)
        if 'time' in result_f:
            fortran_times.append(result_f['time'])
        
        # gt4py
        result_g = call_gt4py_ice_adjust(data)
        if 'time' in result_g:
            gt4py_times.append(result_g['time'])
        
        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{n_iterations}")
    
    # Statistics
    if fortran_times and gt4py_times:
        print(f"\nFortran: {np.mean(fortran_times)*1000:.2f} ± {np.std(fortran_times)*1000:.2f} ms")
        print(f"gt4py:   {np.mean(gt4py_times)*1000:.2f} ± {np.std(gt4py_times)*1000:.2f} ms")
        print(f"Speedup: {np.mean(fortran_times)/np.mean(gt4py_times):.2f}x")


def main():
    """Run comparison framework."""
    print("="*70)
    print(" Fortran vs gt4py Comparison Framework")
    print("="*70)
    
    print("\nThis framework demonstrates how to compare Fortran and gt4py")
    print("implementations of ICE_ADJUST.")
    
    # Try to load real data
    test_data = load_test_data()
    
    # Use synthetic data if real data not available
    if test_data is None:
        data = create_synthetic_data(nijt=50, nkt=30)
    else:
        print("Using loaded test data")
        data = test_data
    
    # Call both implementations
    fortran_result = call_fortran_ice_adjust(data, dt=1.0)
    gt4py_result = call_gt4py_ice_adjust(data, dt=1.0)
    
    # Compare
    compare_results(fortran_result, gt4py_result, data)
    
    # Summary
    print("\n" + "="*70)
    print("Framework Summary")
    print("="*70)
    print("\nTo use this comparison framework:")
    print("  1. Ensure ICE_ADJUST is working in tests/components/test_ice_adjust.py")
    print("  2. Adapt this script to use the test infrastructure")
    print("  3. Run comparisons with real atmospheric data")
    print("  4. Validate numerical accuracy")
    print("  5. Benchmark performance")
    
    print("\nExample working test:")
    print("  pytest tests/components/test_ice_adjust.py -v")
    
    print("\nFor Cython wrapper:")
    print("  from ice3.cython_bindings.ice_adjust_wrapper import get_module_info")
    print("  # Currently uses fmodpy backend for simplicity")
    print("  # Full Cython binding requires Fortran wrapper subroutine")
    
    print("\nKey Differences:")
    print("  Fortran (fmodpy): Direct call to F90, complex type handling")
    print("  gt4py: Python-based, can target CPU/GPU, easier to modify")
    print("  Cython: Hybrid approach, C-level performance with Python ease")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
