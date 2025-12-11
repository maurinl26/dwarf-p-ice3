# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython wrapper for PHYEX condensation subroutine.

This demonstrates low-level Cython binding to Fortran code.
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport exp, log, sqrt, fabs

# Initialize NumPy C API
cnp.import_array()

# Note: External Fortran function declarations would go here
# For this demonstration, we use pure Cython implementations


# More realistic condensation wrapper (structure only, not functional)
# This shows the pattern for wrapping complex Fortran subroutines

cdef class CondensationParams:
    """
    Python wrapper for Fortran derived types.
    
    This demonstrates how you'd wrap Fortran TYPE structures in Cython.
    In production, each field would correspond to Fortran TYPE members.
    """
    cdef public int nijt, nkt
    cdef public double[:, ::1] ppabs, pzz, prhodref, pt
    
    def __init__(self, int nijt, int nkt):
        self.nijt = nijt
        self.nkt = nkt
        # In real implementation: allocate all arrays


def call_condensation_cython(
    cnp.ndarray[double, ndim=2, mode='fortran'] ppabs,
    cnp.ndarray[double, ndim=2, mode='fortran'] pzz,
    cnp.ndarray[double, ndim=2, mode='fortran'] prhodref,
    cnp.ndarray[double, ndim=2, mode='fortran'] pt,
    cnp.ndarray[double, ndim=2, mode='fortran'] prv_in,
    cnp.ndarray[double, ndim=2, mode='fortran'] prc_in,
    cnp.ndarray[double, ndim=2, mode='fortran'] pri_in,
):
    """
    Cython wrapper for CONDENSATION Fortran subroutine.
    
    This demonstrates the structure of a full wrapper. The actual
    implementation would need to:
    1. Pack parameters into Fortran derived types
    2. Call the Fortran subroutine
    3. Unpack results
    
    Parameters
    ----------
    ppabs : ndarray (nijt, nkt)
        Absolute pressure [Pa]
    pzz : ndarray (nijt, nkt)
        Height [m]
    prhodref : ndarray (nijt, nkt)
        Reference density [kg/m³]
    pt : ndarray (nijt, nkt)
        Temperature [K]
    prv_in : ndarray (nijt, nkt)
        Water vapor mixing ratio [kg/kg]
    prc_in : ndarray (nijt, nkt)
        Cloud water mixing ratio [kg/kg]
    pri_in : ndarray (nijt, nkt)
        Cloud ice mixing ratio [kg/kg]
    
    Returns
    -------
    dict
        Dictionary with output arrays:
        - prv_out: Adjusted water vapor
        - prc_out: Adjusted cloud water
        - pri_out: Adjusted cloud ice
        - pcldfr: Cloud fraction
    
    Notes
    -----
    This is a demonstration structure. Full implementation requires:
    - Handling all Fortran TYPE structures (DIMPHYEX_t, CST_t, etc.)
    - Proper memory management for derived types
    - Correct passing of optional parameters
    """
    # Check array shapes
    cdef int nijt = ppabs.shape[0]
    cdef int nkt = ppabs.shape[1]
    
    # Allocate output arrays (Fortran-contiguous)
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] prv_out = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] prc_out = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] pri_out = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] pcldfr = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    
    # In a real implementation, you would:
    # 1. Create and populate Fortran derived type structures
    # 2. Call the actual Fortran subroutine
    # 3. Extract results from Fortran structures
    
    # For now, just copy inputs to outputs as placeholder
    prv_out[:, :] = prv_in[:, :]
    prc_out[:, :] = prc_in[:, :]
    pri_out[:, :] = pri_in[:, :]
    pcldfr[:, :] = 0.0
    
    return {
        'prv_out': np.asarray(prv_out),
        'prc_out': np.asarray(prc_out),
        'pri_out': np.asarray(pri_out),
        'pcldfr': np.asarray(pcldfr),
    }


# Performance-critical inner loop example
cdef inline double compute_saturation_vapor_pressure(double T) nogil:
    """
    Fast inline computation (demonstrates Cython performance).
    
    This is the kind of function where Cython shines - small,
    frequently-called functions that can be inlined and run
    without Python overhead.
    """
    cdef double TT = 273.15
    cdef double ALVTT = 2.8345e6
    cdef double RV = 461.5
    return 611.2 * exp(ALVTT / RV * (1.0/TT - 1.0/T))


def vectorized_saturation_pressure(cnp.ndarray[double, ndim=2, mode='c'] temperature):
    """
    Example of vectorized operation using Cython.
    
    This demonstrates Cython's performance benefits for
    numerical operations.
    
    Parameters
    ----------
    temperature : ndarray
        Temperature array [K]
    
    Returns
    -------
    es : ndarray
        Saturation vapor pressure [Pa]
    """
    cdef int i, j
    cdef int ni = temperature.shape[0]
    cdef int nj = temperature.shape[1]
    cdef cnp.ndarray[double, ndim=2, mode='c'] es = np.zeros((ni, nj), dtype=np.float64)
    
    # Release GIL for parallel execution potential
    with nogil:
        for i in range(ni):
            for j in range(nj):
                es[i, j] = compute_saturation_vapor_pressure(temperature[i, j])
    
    return es


# Utility functions
def check_fortran_array(arr, str name="array"):
    """
    Check if array is Fortran-contiguous and raise helpful error if not.
    
    Parameters
    ----------
    arr : ndarray
        Array to check
    name : str
        Name of array for error message
    """
    if not arr.flags['F_CONTIGUOUS']:
        raise ValueError(
            f"{name} must be Fortran-contiguous. "
            f"Use np.asfortranarray() or create with order='F'"
        )


def prepare_fortran_array(shape, dtype=np.float64):
    """
    Create a Fortran-contiguous array.
    
    Parameters
    ----------
    shape : tuple
        Array shape
    dtype : dtype
        Data type (default: float64)
    
    Returns
    -------
    ndarray
        Fortran-contiguous array
    """
    return np.zeros(shape, dtype=dtype, order='F')


# Module initialization
def get_cython_info():
    """
    Get information about this Cython module.
    
    Returns
    -------
    dict
        Module information
    """
    return {
        'compiled': True,
        'language_level': 3,
        'numpy_included': True,
        'nogil_support': True,
        'description': 'Cython wrapper for PHYEX Fortran library',
    }
