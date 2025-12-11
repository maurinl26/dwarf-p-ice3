"""
Benchmark comparing fmodpy vs Cython for Fortran bindings.

This demonstrates the performance characteristics and usage patterns
of different binding approaches.
"""

import numpy as np
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def benchmark_cython_saturation_pressure():
    """Benchmark the Cython saturation pressure calculation."""
    print("\n" + "="*70)
    print("Benchmarking Cython: Saturation Vapor Pressure Calculation")
    print("="*70)
    
    try:
        from ice3.cython_bindings.condensation_wrapper import (
            vectorized_saturation_pressure,
            get_cython_info
        )
        
        # Show Cython module info
        info = get_cython_info()
        print(f"\nCython Module Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Create test data
        sizes = [(100, 60), (500, 100), (1000, 200)]
        
        for shape in sizes:
            temperature = np.random.uniform(250, 310, shape)  # 250-310 K
            
            # Warm up
            _ = vectorized_saturation_pressure(temperature)
            
            # Benchmark
            n_iterations = 100
            start = time.time()
            for _ in range(n_iterations):
                result = vectorized_saturation_pressure(temperature)
            end = time.time()
            
            time_per_call = (end - start) / n_iterations * 1000  # ms
            elements = shape[0] * shape[1]
            throughput = elements / (time_per_call / 1000) / 1e6  # Million elements/sec
            
            print(f"\nShape {shape}:")
            print(f"  Time per call: {time_per_call:.3f} ms")
            print(f"  Throughput: {throughput:.2f} M elements/sec")
            print(f"  Result shape: {result.shape}")
            print(f"  Result range: [{result.min():.2f}, {result.max():.2f}] Pa")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ Cython extension not available: {e}")
        print("  Build it first: python setup_cython.py build_ext --inplace")
        return False


def benchmark_numpy_equivalent():
    """Benchmark equivalent NumPy vectorized operation for comparison."""
    print("\n" + "="*70)
    print("Benchmarking NumPy: Saturation Vapor Pressure Calculation")
    print("="*70)
    
    def numpy_saturation_pressure(T):
        """Pure NumPy implementation (vectorized)."""
        TT = 273.15
        ALVTT = 2.8345e6
        RV = 461.5
        return 611.2 * np.exp(ALVTT / RV * (1.0/TT - 1.0/T))
    
    sizes = [(100, 60), (500, 100), (1000, 200)]
    
    for shape in sizes:
        temperature = np.random.uniform(250, 310, shape)
        
        # Warm up
        _ = numpy_saturation_pressure(temperature)
        
        # Benchmark
        n_iterations = 100
        start = time.time()
        for _ in range(n_iterations):
            result = numpy_saturation_pressure(temperature)
        end = time.time()
        
        time_per_call = (end - start) / n_iterations * 1000  # ms
        elements = shape[0] * shape[1]
        throughput = elements / (time_per_call / 1000) / 1e6  # Million elements/sec
        
        print(f"\nShape {shape}:")
        print(f"  Time per call: {time_per_call:.3f} ms")
        print(f"  Throughput: {throughput:.2f} M elements/sec")


def demonstrate_cython_usage():
    """Demonstrate how to use the Cython bindings."""
    print("\n" + "="*70)
    print("Cython Usage Example")
    print("="*70)
    
    try:
        from ice3.cython_bindings.condensation_wrapper import (
            call_condensation_cython,
            check_fortran_array,
            prepare_fortran_array,
        )
        
        # Create test data
        nijt, nkt = 100, 60
        print(f"\nCreating test data: ({nijt}, {nkt})")
        
        # Use the utility functions
        ppabs = prepare_fortran_array((nijt, nkt))
        pzz = prepare_fortran_array((nijt, nkt))
        prhodref = prepare_fortran_array((nijt, nkt))
        pt = prepare_fortran_array((nijt, nkt))
        prv_in = prepare_fortran_array((nijt, nkt))
        prc_in = prepare_fortran_array((nijt, nkt))
        pri_in = prepare_fortran_array((nijt, nkt))
        
        # Fill with realistic values
        ppabs[:] = 101325.0 + np.random.rand(nijt, nkt) * 10000
        pzz[:] = np.linspace(0, 10000, nkt)[np.newaxis, :]
        prhodref[:] = 1.2
        pt[:] = 273.15 + 20 + np.random.rand(nijt, nkt) * 10
        prv_in[:] = 0.01 + np.random.rand(nijt, nkt) * 0.005
        prc_in[:] = 0.001 * np.random.rand(nijt, nkt)
        pri_in[:] = 0.0005 * np.random.rand(nijt, nkt)
        
        # Check arrays are Fortran-contiguous
        print("\nChecking array alignment:")
        for name, arr in [('ppabs', ppabs), ('pt', pt), ('prv_in', prv_in)]:
            try:
                check_fortran_array(arr, name)
                print(f"  ✓ {name}: Fortran-contiguous")
            except ValueError as e:
                print(f"  ✗ {name}: {e}")
        
        # Call the Cython wrapper
        print("\nCalling Cython wrapper...")
        result = call_condensation_cython(
            ppabs, pzz, prhodref, pt,
            prv_in, prc_in, pri_in
        )
        
        print("\nResults:")
        for key, value in result.items():
            print(f"  {key}: shape={value.shape}, "
                  f"min={value.min():.6f}, max={value.max():.6f}")
        
        print("\n✓ Cython wrapper executed successfully")
        return True
        
    except ImportError as e:
        print(f"\n✗ Cython extension not available: {e}")
        return False


def compare_array_memory_layout():
    """Compare memory layout and access patterns."""
    print("\n" + "="*70)
    print("Memory Layout Comparison: C vs Fortran Order")
    print("="*70)
    
    shape = (1000, 500)
    
    # C-ordered (row-major, Python default)
    c_array = np.zeros(shape, dtype=np.float64, order='C')
    
    # Fortran-ordered (column-major, Fortran default)
    f_array = np.zeros(shape, dtype=np.float64, order='F')
    
    print(f"\nArray shape: {shape}")
    print(f"Total elements: {shape[0] * shape[1]:,}")
    print(f"Memory size: {c_array.nbytes / 1024 / 1024:.2f} MB")
    
    print(f"\nC-ordered (Python default):")
    print(f"  Contiguous in memory: rows")
    print(f"  C_CONTIGUOUS: {c_array.flags['C_CONTIGUOUS']}")
    print(f"  F_CONTIGUOUS: {c_array.flags['F_CONTIGUOUS']}")
    print(f"  Strides: {c_array.strides}")
    
    print(f"\nFortran-ordered (Fortran default):")
    print(f"  Contiguous in memory: columns")
    print(f"  C_CONTIGUOUS: {f_array.flags['C_CONTIGUOUS']}")
    print(f"  F_CONTIGUOUS: {f_array.flags['F_CONTIGUOUS']}")
    print(f"  Strides: {f_array.strides}")
    
    print(f"\n⚠️  Key Point:")
    print(f"  Fortran expects F_CONTIGUOUS arrays for optimal performance")
    print(f"  Always create arrays with order='F' or use np.asfortranarray()")


def main():
    """Run all benchmarks and examples."""
    print("\n" + "="*70)
    print(" Cython Binding Benchmarks and Examples")
    print("="*70)
    
    # Check if extension is built
    built_files = list(Path("src/ice3/cython_bindings").glob("*.so"))
    if built_files:
        print(f"\n✓ Found compiled extension: {built_files[0].name}")
    else:
        print("\n✗ Cython extension not found!")
        print("  Build it with: python setup_cython.py build_ext --inplace")
        print("\n  Continuing with available demonstrations...")
    
    # Run demonstrations
    compare_array_memory_layout()
    
    # Try Cython benchmarks
    cython_available = benchmark_cython_saturation_pressure()
    
    # NumPy baseline
    benchmark_numpy_equivalent()
    
    # Usage example
    if cython_available:
        demonstrate_cython_usage()
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Cython provides C-level performance for numerical operations")
    print("  2. 'nogil' allows releasing the GIL for true parallelism")
    print("  3. Inline functions eliminate Python overhead")
    print("  4. Memory layout (C vs Fortran order) matters for performance")
    print("  5. Type annotations enable compile-time optimizations")
    
    if cython_available:
        print("\n✓ Cython is fully functional on your system")
    else:
        print("\n⚠️  Build Cython extension to see full comparison")
    
    print("\nLearn more:")
    print("  - docs/fortran_python_bindings.md")
    print("  - Cython documentation: https://cython.readthedocs.io")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
