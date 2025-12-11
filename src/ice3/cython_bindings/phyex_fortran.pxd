# cython: language_level=3
"""
Cython declarations for PHYEX Fortran library interfaces.

This file declares the C/Fortran interfaces that will be wrapped.
"""

# Standard C types
from libc.stdint cimport int32_t, int64_t
cimport numpy as cnp

# Fortran uses column-major (Fortran-contiguous) arrays
# We need to declare this properly for Cython

cdef extern from *:
    """
    // Fortran name mangling: add trailing underscore and use lowercase
    // Note: These are simplified signatures. Full implementation would need
    // to handle Fortran derived types properly.
    
    // Simple test function (if available in library)
    extern void simple_test_() nogil;
    """
    
    void simple_test_() nogil


# Declare a simplified interface
# In practice, you'd need one declaration per Fortran subroutine you want to wrap
cdef extern from *:
    """
    // Simplified condensation interface
    // Real implementation would declare all the TYPE structures
    """
    pass
