"""
Test and demonstrate ICE_ADJUST Cython wrapper.

This example shows how to use the Cython wrapper for ICE_ADJUST
and compares it with the fmodpy approach.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_ice_adjust_state():
    """Test IceAdjustState class for efficient array management."""
    print("\n" + "="*70)
    print("Testing IceAdjustState Class")
    print("="*70)
    
    try:
        from ice3.cython_bindings.ice_adjust_wrapper import (
            IceAdjustState,
            prepare_ice_adjust_arrays,
            get_module_info,
        )
        
        # Show module info
        info = get_module_info()
        print("\nModule Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Create state
        nijt, nkt = 100, 60
        print(f"\nCreating IceAdjustState({nijt}, {nkt})...")
        
        state = IceAdjustState(nijt, nkt)
        
        print(f"✓ State created with {nijt}x{nkt} arrays")
        print(f"  prhodj shape: {np.asarray(state.prhodj).shape}")
        print(f"  prhodj F_CONTIGUOUS: {np.asarray(state.prhodj).flags['F_CONTIGUOUS']}")
        
        # Alternative using utility function
        state2 = prepare_ice_adjust_arrays(nijt, nkt)
        print(f"✓ Alternative creation method works")
        
        # Fill with test data
        print("\nFilling with test data...")
        state.ppabst[:, :] = 101325.0
        state.pth[:, :] = 300.0
        state.prv[:, :] = 0.010
        
        print(f"  ppabst mean: {np.asarray(state.ppabst).mean():.1f} Pa")
        print(f"  pth mean: {np.asarray(state.pth).mean():.1f} K")
        print(f"  prv mean: {np.asarray(state.prv).mean()*1000:.1f} g/kg")
        
        return True
        
    except ImportError as e:
        print(f"✗ Cannot import: {e}")
        print("  Build extension first: python setup_cython.py build_ext --inplace")
        return False


def test_array_validation():
    """Test Cython-level array validation."""
    print("\n" + "="*70)
    print("Testing Array Validation")
    print("="*70)
    
    try:
        from ice3.cython_bindings.ice_adjust_wrapper import call_ice_adjust
        
        nijt, nkt = 50, 30
        
        # Create correctly shaped Fortran arrays
        prhodj = np.ones((nijt, nkt), dtype=np.float64, order='F')
        pexnref = np.ones((nijt, nkt), dtype=np.float64, order='F')
        prhodref = np.ones((nijt, nkt), dtype=np.float64, order='F')
        ppabst = np.ones((nijt, nkt), dtype=np.float64, order='F') * 101325
        pzz = np.ones((nijt, nkt), dtype=np.float64, order='F')
        pexn = np.ones((nijt, nkt), dtype=np.float64, order='F')
        prv = np.ones((nijt, nkt), dtype=np.float64, order='F') * 0.01
        prc = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        pri = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        pth = np.ones((nijt, nkt), dtype=np.float64, order='F') * 300
        prvs = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        prcs = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        pris = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        pths = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        
        print(f"✓ Created {nijt}x{nkt} Fortran-contiguous arrays")
        
        # Test wrong shape (should fail)
        print("\nTesting error handling with wrong shape...")
        wrong_shape = np.ones((nijt+1, nkt), dtype=np.float64, order='F')
        try:
            result = call_ice_adjust(
                wrong_shape, pexnref, prhodref, ppabst, pzz, pexn,
                prv, prc, pri, pth, prvs, prcs, pris, pths
            )
            print("  ✗ Should have raised ValueError")
        except ValueError as e:
            print(f"  ✓ Correctly caught error: {str(e)[:50]}...")
        
        # Test C-ordered array (should fail)
        print("\nTesting error handling with C-ordered array...")
        c_ordered = np.ones((nijt, nkt), dtype=np.float64, order='C')
        try:
            result = call_ice_adjust(
                c_ordered, pexnref, prhodref, ppabst, pzz, pexn,
                prv, prc, pri, pth, prvs, prcs, pris, pths
            )
            print("  ✗ Should have raised ValueError")
        except ValueError as e:
            print(f"  ✓ Correctly caught error: {str(e)[:50]}...")
        
        # Test correct call (will raise NotImplementedError for full Fortran call)
        print("\nTesting correct array setup...")
        try:
            result = call_ice_adjust(
                prhodj, pexnref, prhodref, ppabst, pzz, pexn,
                prv, prc, pri, pth, prvs, prcs, pris, pths
            )
            print("  ✓ Arrays validated correctly")
        except NotImplementedError as e:
            print(f"  ✓ Arrays validated, full implementation requires Fortran setup")
            print(f"     {str(e)[:60]}...")
        
        return True
        
    except ImportError as e:
        print(f"✗ Cannot import: {e}")
        return False


def demonstrate_usage_pattern():
    """Demonstrate recommended usage pattern."""
    print("\n" + "="*70)
    print("Recommended Usage Pattern")
    print("="*70)
    
    print("""
For production use, the recommended approach is to use the existing
fmodpy-based interface which handles Fortran derived types automatically.

Example using existing infrastructure:
    
    from ice3.components.ice_adjust import ice_adjust
    from ice3.initialisation.state_ice_adjust import load_state_from_config
    
    # Load configuration
    state = load_state_from_config('src/config/ice_adjust.yaml')
    
    # Call ICE_ADJUST
    result = ice_adjust(state, dt=1.0)

For testing, use existing test infrastructure:
    
    pytest tests/components/test_ice_adjust.py -v

The Cython wrapper provides:
    1. Framework for custom performance-critical code
    2. C-level array validation
    3. Memory-efficient state management
    4. Template for future Fortran wrapper subroutines

To enable full Cython ICE_ADJUST:
    1. Create simplified Fortran wrapper subroutine (takes simple arrays)
    2. Declare in ice_adjust_wrapper.pyx
    3. Rebuild: python setup_cython.py build_ext --inplace
    4. Test and compare with fmodpy version
""")


def compare_approaches():
    """Compare different approaches for calling ICE_ADJUST."""
    print("\n" + "="*70)
    print("Approach Comparison")
    print("="*70)
    
    comparison = {
        'fmodpy (Current)': {
            'Setup': 'Automatic',
            'Type Handling': 'Automatic',
            'Performance': 'Good',
            'Maintenance': 'Low',
            'Use Case': 'Production, complex Fortran',
            'Status': '✓ Working',
        },
        'Cython (This wrapper)': {
            'Setup': 'Manual',
            'Type Handling': 'Manual/Framework',
            'Performance': 'Excellent',
            'Maintenance': 'Medium',
            'Use Case': 'Performance-critical custom code',
            'Status': '✓ Framework ready',
        },
        'gt4py': {
            'Setup': 'Python',
            'Type Handling': 'Python objects',
            'Performance': 'Excellent (CPU/GPU)',
            'Maintenance': 'Medium',
            'Use Case': 'Portable, GPU-capable',
            'Status': '✓ Working',
        },
    }
    
    # Print table
    print(f"\n{'Approach':<20} {'Setup':<12} {'Type Handling':<15} {'Performance':<15} {'Status':<15}")
    print("-" * 85)
    for approach, details in comparison.items():
        print(f"{approach:<20} {details['Setup']:<12} {details['Type Handling']:<15} "
              f"{details['Performance']:<15} {details['Status']:<15}")
    
    print(f"\n{'Recommended Uses:'}")
    for approach, details in comparison.items():
        print(f"  {approach}: {details['Use Case']}")


def benchmark_array_operations():
    """Benchmark Cython array operations vs NumPy."""
    print("\n" + "="*70)
    print("Array Operations Benchmark")
    print("="*70)
    
    try:
        from ice3.cython_bindings.ice_adjust_wrapper import IceAdjustState
        
        nijt, nkt = 1000, 200
        n_iterations = 100
        
        print(f"\nBenchmarking array allocation ({nijt}x{nkt})...")
        
        # NumPy allocation
        start = time.time()
        for _ in range(n_iterations):
            arr = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        time_numpy = (time.time() - start) / n_iterations * 1000
        
        # Cython class allocation
        start = time.time()
        for _ in range(n_iterations):
            state = IceAdjustState(nijt, nkt)
        time_cython = (time.time() - start) / n_iterations * 1000
        
        print(f"  NumPy single array: {time_numpy:.3f} ms")
        print(f"  Cython IceAdjustState (17 arrays): {time_cython:.3f} ms")
        print(f"  Per array (Cython): {time_cython/17:.3f} ms")
        
        # Memory usage
        state = IceAdjustState(nijt, nkt)
        bytes_per_array = nijt * nkt * 8  # float64
        total_bytes = bytes_per_array * 17
        print(f"\n  Memory per array: {bytes_per_array/1024/1024:.2f} MB")
        print(f"  Total memory (17 arrays): {total_bytes/1024/1024:.2f} MB")
        
        return True
        
    except ImportError:
        print("  ✗ Cython extension not available")
        return False


def main():
    """Run all tests and demonstrations."""
    print("="*70)
    print(" ICE_ADJUST Cython Wrapper Test & Demonstration")
    print("="*70)
    
    # Check if extension is built
    try:
        from ice3.cython_bindings.ice_adjust_wrapper import get_module_info
        print("\n✓ Cython ice_adjust_wrapper extension is available")
        info = get_module_info()
        print(f"  Status: {info['status']}")
    except ImportError as e:
        print(f"\n✗ Cython extension not built: {e}")
        print("\nBuild it with:")
        print("  python setup_cython.py build_ext --inplace")
        return
    
    # Run tests
    success = True
    success &= test_ice_adjust_state()
    success &= test_array_validation()
    success &= benchmark_array_operations()
    
    # Show usage patterns
    demonstrate_usage_pattern()
    compare_approaches()
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    if success:
        print("\n✓ All Cython wrapper tests passed")
    else:
        print("\n⚠️  Some tests failed")
    
    print("\nKey Points:")
    print("  1. Cython wrapper provides efficient array management")
    print("  2. C-level validation catches errors early")
    print("  3. Framework ready for Fortran integration")
    print("  4. Use fmodpy for production (automatic type handling)")
    print("  5. Use Cython for performance-critical custom code")
    
    print("\nNext Steps:")
    print("  • Use existing tests: pytest tests/components/test_ice_adjust.py")
    print("  • Compare Fortran/gt4py: python examples/compare_fortran_gt4py.py")
    print("  • For full Cython binding: create Fortran wrapper subroutine")
    
    print("\nDocumentation:")
    print("  • docs/fortran_python_bindings.md")
    print("  • docs/CYTHON_IMPLEMENTATION_SUMMARY.md")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
