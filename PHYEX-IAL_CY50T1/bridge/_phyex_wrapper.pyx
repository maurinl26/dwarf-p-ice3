# distutils: language = c
# cython: language_level = 3
import numpy as np
cimport numpy as np

# Declare memory view data types for double precision (JPRB/float64)
ctypedef np.float64_t DTYPE_t

# 1. External C Declaration
# This matches the signature defined in the Fortran bridge:
# SUBROUTINE c_ice_adjust_wrap(nlon, nlev, ptr_pt, ptr_pq, ptr_pp) BIND(C, name="c_ice_adjust")
cdef extern:
    void c_ice_adjust(
        int nlon, 
        int nlev, 
        double *ptr_pt, 
        double *ptr_pq, 
        double *ptr_pp
    )

# 2. Python-Callable Wrapper Function
def ice_adjust(
    # Use C-contiguous memory view for the fastest possible access.
    # We must ensure the input NumPy arrays are F-contiguous (Fortran order)
    # as Fortran uses column-major order. Cython can handle the translation
    # to C pointers easily.
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] pt, 
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] pq, 
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] pp
):
    """
    Cython wrapper for the PHYEX ICE_ADJUST routine.
    Input arrays must be 2D, Fortran-contiguous (KLON, KLEV), and float64.
    PT and PQ are modified in-place.
    """
    
    # 3. Get the dimensions
    cdef int nlon = pt.shape[0]
    cdef int nlev = pt.shape[1]

    # Type check for safety (handled by Cython but good practice)
    if (pq.shape[0] != nlon or pq.shape[1] != nlev or 
        pp.shape[0] != nlon or pp.shape[1] != nlev):
        raise ValueError("All input arrays must have the same (KLON, KLEV) shape.")

    # 4. Call the Fortran function using C pointers
    # We access the raw data buffer (address of the first element)
    c_ice_adjust(
        nlon, 
        nlev, 
        &pt[0, 0],  # Pointer to the start of pt array
        &pq[0, 0],  # Pointer to the start of pq array
        &pp[0, 0]   # Pointer to the start of pp array
    )

    # Note: Cython modified the arrays IN-PLACE, so we don't need to return anything new.
    # We return the inputs for convenience, but they are the same objects passed in.
    return pt, pq