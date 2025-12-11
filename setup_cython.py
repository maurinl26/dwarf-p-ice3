"""
Setup script for building Cython extensions for dwarf-p-ice3.

Usage:
    python setup_cython.py build_ext --inplace

This will compile the Cython extensions and place them in the source tree.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from pathlib import Path

# Get paths
root_dir = Path(__file__).parent
build_fortran_dir = root_dir / "build_fortran"
src_dir = root_dir / "src"

# Define the extensions
extensions = [
    Extension(
        name="ice3.cython_bindings.condensation_wrapper",
        sources=["src/ice3/cython_bindings/condensation_wrapper.pyx"],
        include_dirs=[
            np.get_include(),
            str(build_fortran_dir),
        ],
        library_dirs=[str(build_fortran_dir)],
        libraries=["ice_adjust_phyex"],  # Link against the Fortran library
        extra_compile_args=[
            "-O3",  # Optimization
            "-fPIC",  # Position independent code
        ],
        extra_link_args=[
            f"-Wl,-rpath,{build_fortran_dir}",  # Runtime library path
        ],
        language="c",
    ),
    Extension(
        name="ice3.cython_bindings.ice_adjust_wrapper",
        sources=["src/ice3/cython_bindings/ice_adjust_wrapper.pyx"],
        include_dirs=[
            np.get_include(),
            str(build_fortran_dir),
        ],
        library_dirs=[str(build_fortran_dir)],
        libraries=["ice_adjust_phyex"],  # Link against the Fortran library
        extra_compile_args=[
            "-O3",  # Optimization
            "-fPIC",  # Position independent code
        ],
        extra_link_args=[
            f"-Wl,-rpath,{build_fortran_dir}",  # Runtime library path
        ],
        language="c",
    ),
]

# Compiler directives for optimization
compiler_directives = {
    "language_level": "3",
    "boundscheck": False,  # Disable bounds checking for performance
    "wraparound": False,  # Disable negative indexing
    "cdivision": True,  # Use C division semantics
    "embedsignature": True,  # Embed function signatures in docstrings
    "initializedcheck": False,  # Disable memoryview initialization checks
}

# Build configuration
setup(
    name="dwarf-p-ice3-cython",
    version="0.1.0",
    description="Cython bindings for PHYEX Fortran library",
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=True,  # Generate HTML annotation files
        nthreads=4,  # Parallel compilation
    ),
    zip_safe=False,
)

print("\n" + "="*70)
print("Cython Extension Build Complete!")
print("="*70)
print("\nBuilt extensions:")
for ext in extensions:
    print(f"  - {ext.name}")
print("\nTo use:")
print("  from ice3.cython_bindings.condensation_wrapper import call_condensation_cython")
print("\nAnnotated HTML files (showing C code) are in the build directory.")
print("="*70 + "\n")
