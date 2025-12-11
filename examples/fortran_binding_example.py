"""
Example demonstrating Fortran-Python bindings in dwarf-p-ice3.

This shows how to use the different binding approaches available.
"""

import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_fmodpy_binding():
    """
    Example using fmodpy (recommended approach).
    
    This is the current method used in the project.
    """
    print("\n" + "="*60)
    print("Example 1: Using fmodpy (Recommended)")
    print("="*60)
    
    try:
        from ice3.utils.compile_fortran import compile_fortran_stencil
        
        # Compile a simple Fortran stencil
        logger.info("Compiling Fortran stencil with fmodpy...")
        
        # Example: compile mode_condensation
        condensation = compile_fortran_stencil(
            fortran_script="mode_condensation.F90",
            fortran_module="mode_condensation",
            fortran_stencil="condensation"
        )
        
        logger.info("✓ Successfully compiled Fortran stencil")
        logger.info(f"  Callable: {condensation}")
        logger.info("  Ready to use with proper parameters")
        
        # Note: Actual usage requires proper initialization of all parameters
        # See tests/repro/test_condensation.py for complete example
        
    except Exception as e:
        logger.error(f"✗ Error with fmodpy binding: {e}")
        logger.info("  Make sure fmodpy is installed: pip install fmodpy")


def example_ctypes_binding():
    """
    Example using ctypes for direct library access.
    
    This shows how to access the compiled shared library.
    """
    print("\n" + "="*60)
    print("Example 2: Using ctypes")  
    print("="*60)
    
    try:
        from ice3.fortran_bindings import find_fortran_library, FortranArray
        
        # Find the compiled library
        lib_path = find_fortran_library()
        logger.info(f"✓ Found Fortran library: {lib_path}")
        
        # Demonstrate array preparation
        nijt, nkt = 10, 5
        logger.info(f"  Creating Fortran-ordered arrays ({nijt}x{nkt})...")
        
        temperature = FortranArray.prepare_array((nijt, nkt))
        pressure = FortranArray.prepare_array((nijt, nkt))
        
        # Fill with test data
        temperature[:] = 273.15 + np.random.rand(nijt, nkt) * 20  # 273-293 K
        pressure[:] = 101325.0 + np.random.rand(nijt, nkt) * 10000  # ~1000 hPa
        
        logger.info(f"  ✓ Temperature: shape={temperature.shape}, "
                   f"fortran={temperature.flags['F_CONTIGUOUS']}")
        logger.info(f"  ✓ Pressure: shape={pressure.shape}, "
                   f"fortran={pressure.flags['F_CONTIGUOUS']}")
        
        # Note: Actual ctypes calls require proper function signatures
        # due to Fortran's complex derived types
        logger.info("  Note: Full ctypes binding requires handling Fortran derived types")
        logger.info("        Recommend using fmodpy for complete functionality")
        
    except Exception as e:
        logger.error(f"✗ Error: {e}")


def example_check_compiled_library():
    """Check if the Fortran library is compiled and available."""
    print("\n" + "="*60)
    print("Checking Compiled Library Status")
    print("="*60)
    
    build_dir = Path("build_fortran")
    lib_file = build_dir / "libice_adjust_phyex.so"
    
    if lib_file.exists():
        size_mb = lib_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Library found: {lib_file}")
        logger.info(f"  Size: {size_mb:.2f} MB")
        logger.info(f"  Ready for binding!")
    else:
        logger.warning(f"✗ Library not found: {lib_file}")
        logger.info("  Compile it first:")
        logger.info("    cd build_fortran && cmake .. && make")


def example_array_conversion():
    """Demonstrate Fortran array conventions."""
    print("\n" + "="*60)
    print("Example 3: Fortran Array Conventions")
    print("="*60)
    
    # Create C-ordered array (Python default - row-major)
    c_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    logger.info(f"C-ordered array (Python default):")
    logger.info(f"  Shape: {c_array.shape}")
    logger.info(f"  F_CONTIGUOUS: {c_array.flags['F_CONTIGUOUS']}")
    logger.info(f"  C_CONTIGUOUS: {c_array.flags['C_CONTIGUOUS']}")
    
    # Convert to Fortran-ordered (column-major)
    f_array = np.asfortranarray(c_array)
    logger.info(f"\nFortran-ordered array (required for Fortran):")
    logger.info(f"  Shape: {f_array.shape}")
    logger.info(f"  F_CONTIGUOUS: {f_array.flags['F_CONTIGUOUS']}")
    logger.info(f"  C_CONTIGUOUS: {f_array.flags['C_CONTIGUOUS']}")
    
    # Create directly as Fortran-ordered
    f_direct = np.zeros((3, 4), dtype=np.float64, order='F')
    logger.info(f"\nDirect Fortran-ordered creation:")
    logger.info(f"  Shape: {f_direct.shape}")
    logger.info(f"  F_CONTIGUOUS: {f_direct.flags['F_CONTIGUOUS']}")
    
    logger.info("\n✓ Always use Fortran-ordered arrays when calling Fortran!")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" Fortran-Python Binding Examples for dwarf-p-ice3")
    print("="*70)
    
    # Check library status
    example_check_compiled_library()
    
    # Show array conventions
    example_array_conversion()
    
    # Try ctypes approach
    example_ctypes_binding()
    
    # Try fmodpy approach
    example_fmodpy_binding()
    
    print("\n" + "="*70)
    print("Summary:")
    print("  - fmodpy: Recommended for most use cases")
    print("  - ctypes: Good for simple direct library access")
    print("  - Cython: Use for performance-critical custom code")
    print("\nSee docs/fortran_python_bindings.md for detailed guide")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
