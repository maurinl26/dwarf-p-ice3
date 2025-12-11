"""
Standalone Cython demonstration without Fortran dependencies.

This shows Cython's performance benefits for pure numerical operations.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def benchmark_performance():
    """Benchmark Cython vs NumPy performance."""
    print("\n" + "="*70)
    print("Cython vs NumPy Performance Comparison")
    print("="*70)
    
    try:
        from ice3.cython_bindings.condensation_wrapper import (
            vectorized_saturation_pressure,
            get_cython_info,
            prepare_fortran_array,
            check_fortran_array,
        )
        
        # Show module info
        info = get_cython_info()
        print("\nCython Module Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # NumPy equivalent
        def numpy_saturation_pressure(T):
            TT = 273.15
            ALVTT = 2.8345e6
            RV = 461.5
            return 611.2 * np.exp(ALVTT / RV * (1.0/TT - 1.0/T))
        
        # Test sizes
        sizes = [(100, 60), (500, 100), (1000, 200)]
        
        print("\n" + "-"*70)
        print("Performance Benchmarks")
        print("-"*70)
        
        for shape in sizes:
            temperature = np.random.uniform(250, 310, shape)
            n_iter = 100
            
            # Cython benchmark
            try:
                _ = vectorized_saturation_pressure(temperature)
                start = time.time()
                for _ in range(n_iter):
                    result_cython = vectorized_saturation_pressure(temperature)
                time_cython = (time.time() - start) / n_iter * 1000
                
                # NumPy benchmark
                _ = numpy_saturation_pressure(temperature)
                start = time.time()
                for _ in range(n_iter):
                    result_numpy = numpy_saturation_pressure(temperature)
                time_numpy = (time.time() - start) / n_iter * 1000
                
                speedup = time_numpy / time_cython
                elements = shape[0] * shape[1]
                
                print(f"\nShape {shape} ({elements:,} elements):")
                print(f"  NumPy:  {time_numpy:.3f} ms  ({elements/time_numpy*1000/1e6:.2f} M elem/s)")
                print(f"  Cython: {time_cython:.3f} ms  ({elements/time_cython*1000/1e6:.2f} M elem/s)")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Max difference: {np.abs(result_cython - result_numpy).max():.2e}")
                
            except Exception as e:
                print(f"\n✗ Cython test failed for {shape}: {e}")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ Cannot import Cython module: {e}")
        return False


def demonstrate_utilities():
    """Demonstrate Cython utility functions."""
    print("\n" + "="*70)
    print("Cython Utility Functions Demo")
    print("="*70)
    
    try:
        from ice3.cython_bindings.condensation_wrapper import (
            prepare_fortran_array,
            check_fortran_array,
        )
        
        print("\n1. Creating Fortran-ordered arrays:")
        arr_f = prepare_fortran_array((100, 60))
        print(f"   Shape: {arr_f.shape}")
        print(f"   F_CONTIGUOUS: {arr_f.flags['F_CONTIGUOUS']}")
        print(f"   C_CONTIGUOUS: {arr_f.flags['C_CONTIGUOUS']}")
        
        print("\n2. Checking array layout:")
        try:
            check_fortran_array(arr_f, "test_array")
            print("   ✓ Array is Fortran-contiguous")
        except ValueError as e:
            print(f"   ✗ {e}")
        
        print("\n3. Testing C-ordered array:")
        arr_c = np.zeros((100, 60), order='C')
        try:
            check_fortran_array(arr_c, "c_array")
            print("   ✓ Array is Fortran-contiguous")
        except ValueError as e:
            print(f"   ✗ {e}")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ Cannot import utilities: {e}")
        return False


def show_cython_features():
    """Explain Cython features."""
    print("\n" + "="*70)
    print("Cython Features Demonstrated in This Project")
    print("="*70)
    
    features = [
        ("Type Annotations", "Static typing for variables (cdef int, cdef double)"),
        ("nogil", "Release Python GIL for true multi-threading"),
        ("Inline Functions", "cdef inline for zero-overhead function calls"),
        ("MemoryViews", "Efficient array access without Python overhead"),
        ("C Library Calls", "Direct calls to C math functions (exp, log, sqrt)"),
        ("Fortran Interface", "Can interface with Fortran via C ABI"),
    ]
    
    for feature, description in features:
        print(f"\n  • {feature}")
        print(f"    {description}")


def main():
    """Run all demonstrations."""
    print("="*70)
    print(" Cython Standalone Demonstration")
    print("="*70)
    
    # Check if extension is available
    ext_path = Path("src/ice3/cython_bindings/condensation_wrapper.cpython-312-x86_64-linux-gnu.so")
    if ext_path.exists():
        print(f"\n✓ Cython extension found")
    else:
        print(f"\n✗ Cython extension not built")
        print("  Run: python setup_cython.py build_ext --inplace")
        return
    
    # Run demonstrations
    success = benchmark_performance()
    
    if success:
        demonstrate_utilities()
        show_cython_features()
        
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        print("\n✓ Cython extension is working correctly")
        print("\nKey Benefits of Cython:")
        print("  • C-level performance for numerical code")
        print("  • Easy integration with NumPy arrays")
        print("  • Can release the GIL for parallelism")
        print("  • Gradual optimization (start with Python, add types)")
        print("  • Seamless integration with existing Python code")
        
        print("\nWhen to Use Cython:")
        print("  ✓ Performance-critical loops")
        print("  ✓ Interfacing with C/Fortran libraries")
        print("  ✓ Custom numerical algorithms")
        print("  ✓ Need to release GIL for threading")
        
        print("\nLearn More:")
        print("  • docs/fortran_python_bindings.md")
        print("  • Cython Tutorial: https://cython.readthedocs.io")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("Could not run full demonstration")
        print("Build the Cython extension first:")
        print("  python setup_cython.py build_ext --inplace")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
