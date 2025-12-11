# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython wrapper for ICE_ADJUST Fortran subroutine.

This wrapper provides a high-performance interface to the PHYEX ICE_ADJUST
routine, using fmodpy for the Fortran interface and Cython for optimal
array handling and performance.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log, sqrt
import sys
from pathlib import Path

# Initialize NumPy C API
cnp.import_array()

# Import the Fortran module via fmodpy
_ice_adjust_fortran = None

def _load_fortran_module():
    """Lazy load the Fortran ICE_ADJUST module."""
    global _ice_adjust_fortran
    if _ice_adjust_fortran is None:
        try:
            # Add src to path if needed
            src_path = Path(__file__).parent.parent.parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            from ice3.utils.compile_fortran import compile_fortran_stencil
            _ice_adjust_fortran = compile_fortran_stencil(
                fortran_script="../PHYEX-IAL_CY50T1/micro/ice_adjust.F90",
                fortran_module="ice_adjust",
                fortran_stencil="ice_adjust"
            )
        except Exception as e:
            raise ImportError(f"Could not load ICE_ADJUST Fortran module: {e}")
    return _ice_adjust_fortran


cdef class IceAdjustState:
    """
    Cython class to hold ICE_ADJUST state arrays.
    
    This provides fast access to arrays without Python overhead.
    """
    cdef public int nijt, nkt
    cdef public double[:, ::1] prhodj, pexnref, prhodref
    cdef public double[:, ::1] ppabst, pzz, pexn
    cdef public double[:, ::1] prv, prc, pri, pth
    cdef public double[:, ::1] prvs, prcs, pris, pths
    cdef public double[:, ::1] pcldfr, picldfr, pwcldfr
    
    def __init__(self, int nijt, int nkt):
        """Initialize arrays with given dimensions."""
        self.nijt = nijt
        self.nkt = nkt
        
        # Allocate all arrays as Fortran-contiguous
        self.prhodj = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.pexnref = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.prhodref = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.ppabst = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.pzz = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.pexn = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.prv = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.prc = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.pri = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.pth = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.prvs = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.prcs = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.pris = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.pths = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.pcldfr = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.picldfr = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        self.pwcldfr = np.zeros((nijt, nkt), dtype=np.float64, order='F')


def call_ice_adjust(
    cnp.ndarray[double, ndim=2, mode='fortran'] prhodj,
    cnp.ndarray[double, ndim=2, mode='fortran'] pexnref,
    cnp.ndarray[double, ndim=2, mode='fortran'] prhodref,
    cnp.ndarray[double, ndim=2, mode='fortran'] ppabst,
    cnp.ndarray[double, ndim=2, mode='fortran'] pzz,
    cnp.ndarray[double, ndim=2, mode='fortran'] pexn,
    cnp.ndarray[double, ndim=2, mode='fortran'] prv,
    cnp.ndarray[double, ndim=2, mode='fortran'] prc,
    cnp.ndarray[double, ndim=2, mode='fortran'] pri,
    cnp.ndarray[double, ndim=2, mode='fortran'] pth,
    cnp.ndarray[double, ndim=2, mode='fortran'] prvs,
    cnp.ndarray[double, ndim=2, mode='fortran'] prcs,
    cnp.ndarray[double, ndim=2, mode='fortran'] pris,
    cnp.ndarray[double, ndim=2, mode='fortran'] pths,
    double ptstep = 1.0,
    dict params = None,
):
    """
    Call ICE_ADJUST Fortran subroutine with Cython performance.
    
    This function provides a high-performance interface to ICE_ADJUST,
    with type checking and array validation done at C-level.
    
    Parameters
    ----------
    prhodj : ndarray (nijt, nkt), Fortran-contiguous
        Dry density * Jacobian [kg/m³]
    pexnref : ndarray (nijt, nkt), Fortran-contiguous
        Reference Exner function
    prhodref : ndarray (nijt, nkt), Fortran-contiguous
        Reference density [kg/m³]
    ppabst : ndarray (nijt, nkt), Fortran-contiguous
        Absolute pressure [Pa]
    pzz : ndarray (nijt, nkt), Fortran-contiguous
        Height [m]
    pexn : ndarray (nijt, nkt), Fortran-contiguous
        Exner function
    prv : ndarray (nijt, nkt), Fortran-contiguous
        Water vapor mixing ratio [kg/kg]
    prc : ndarray (nijt, nkt), Fortran-contiguous
        Cloud water mixing ratio [kg/kg]
    pri : ndarray (nijt, nkt), Fortran-contiguous
        Cloud ice mixing ratio [kg/kg]
    pth : ndarray (nijt, nkt), Fortran-contiguous
        Potential temperature [K]
    prvs : ndarray (nijt, nkt), Fortran-contiguous
        Water vapor source (modified in-place)
    prcs : ndarray (nijt, nkt), Fortran-contiguous
        Cloud water source (modified in-place)
    pris : ndarray (nijt, nkt), Fortran-contiguous
        Cloud ice source (modified in-place)
    pths : ndarray (nijt, nkt), Fortran-contiguous
        Temperature source (modified in-place)
    ptstep : float, optional
        Time step [s], default 1.0
    params : dict, optional
        Additional parameters for ICE_ADJUST
    
    Returns
    -------
    dict
        Dictionary containing:
        - pcldfr: Cloud fraction
        - picldfr: Ice cloud fraction
        - pwcldfr: Water cloud fraction
        - prvs, prcs, pris, pths: Updated sources
    
    Notes
    -----
    All input arrays must be Fortran-contiguous (order='F').
    Use np.asfortranarray() to convert if needed.
    
    This wrapper uses fmodpy for the actual Fortran call but provides
    Cython-level performance for array handling and validation.
    """
    # Get dimensions
    cdef int nijt = prhodj.shape[0]
    cdef int nkt = prhodj.shape[1]
    
    # Validate array shapes at C-level (fast)
    cdef list arrays = [
        prhodj, pexnref, prhodref, ppabst, pzz, pexn,
        prv, prc, pri, pth, prvs, prcs, pris, pths
    ]
    cdef cnp.ndarray arr
    cdef int i
    
    for i, arr in enumerate(arrays):
        if arr.shape[0] != nijt or arr.shape[1] != nkt:
            raise ValueError(f"Array {i} has wrong shape: {arr.shape} != ({nijt}, {nkt})")
        if not arr.flags['F_CONTIGUOUS']:
            raise ValueError(f"Array {i} must be Fortran-contiguous")
    
    # Allocate output arrays
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] pcldfr = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] picldfr = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] pwcldfr = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] pssio = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] pssiu = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] pifr = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    
    # Create dummy arrays for parameters not used in this simplified interface
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] psigs = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] pmfconv = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] pcf_mf = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] prc_mf = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] pri_mf = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] pweight_mf_cloud = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=1, mode='fortran'] psigqsat = np.ones(nijt, dtype=np.float64, order='F')
    
    # Additional hydrometeor arrays
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] prr = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] prs = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] prg = np.zeros((nijt, nkt), dtype=np.float64, order='F')
    
    # Load Fortran module
    ice_adjust_f = _load_fortran_module()
    
    # Prepare parameters (simplified - would need full implementation)
    # For now, use minimal parameters
    if params is None:
        params = {}
    
    # Call Fortran subroutine (via fmodpy)
    # Note: This is a simplified call. Full implementation would need all parameters.
    try:
        # The actual call would look like this with proper setup:
        # result = ice_adjust_f(...)
        
        # For demonstration, we'll call with basic parameters
        # In production, you'd set up all the derived types properly
        raise NotImplementedError(
            "Full ICE_ADJUST call requires proper setup of Fortran derived types. "
            "Use the fmodpy version directly via: "
            "from ice3.utils.compile_fortran import compile_fortran_stencil"
        )
        
    except Exception as e:
        raise RuntimeError(f"Error calling ICE_ADJUST: {e}")
    
    return {
        'pcldfr': np.asarray(pcldfr),
        'picldfr': np.asarray(picldfr),
        'pwcldfr': np.asarray(pwcldfr),
        'pssio': np.asarray(pssio),
        'pssiu': np.asarray(pssiu),
        'pifr': np.asarray(pifr),
        'prvs': np.asarray(prvs),
        'prcs': np.asarray(prcs),
        'pris': np.asarray(pris),
        'pths': np.asarray(pths),
    }


def call_ice_adjust_simple(
    cnp.ndarray[double, ndim=2, mode='fortran'] temperature,
    cnp.ndarray[double, ndim=2, mode='fortran'] pressure,
    cnp.ndarray[double, ndim=2, mode='fortran'] rv,
    cnp.ndarray[double, ndim=2, mode='fortran'] rc,
    cnp.ndarray[double, ndim=2, mode='fortran'] ri,
    double dt = 1.0,
):
    """
    Simplified ICE_ADJUST interface for easy comparison with gt4py.
    
    This function provides a simplified interface that's easier to compare
    with gt4py implementations.
    
    Parameters
    ----------
    temperature : ndarray (nijt, nkt)
        Temperature [K]
    pressure : ndarray (nijt, nkt)
        Pressure [Pa]
    rv : ndarray (nijt, nkt)
        Water vapor mixing ratio [kg/kg]
    rc : ndarray (nijt, nkt)
        Cloud water mixing ratio [kg/kg]
    ri : ndarray (nijt, nkt)
        Cloud ice mixing ratio [kg/kg]
    dt : float
        Time step [s]
    
    Returns
    -------
    dict
        Updated fields: temperature, rv, rc, ri, cloud_fraction
    """
    cdef int nijt = temperature.shape[0]
    cdef int nkt = temperature.shape[1]
    
    # For now, use fmodpy directly
    # This would be the place to add the full Cython implementation
    from ice3.utils.compile_fortran import compile_fortran_stencil
    
    raise NotImplementedError(
        "Use the fmodpy interface for now. "
        "To enable full Cython ICE_ADJUST: "
        "1. Create simplified Fortran wrapper "
        "2. Bind with Cython "
        "3. Test against gt4py"
    )


# Performance utilities
cdef inline void copy_array_2d(double[:, ::1] src, double[:, ::1] dst) nogil:
    """Fast 2D array copy at C-level."""
    cdef int i, j
    cdef int ni = src.shape[0]
    cdef int nj = src.shape[1]
    
    for i in range(ni):
        for j in range(nj):
            dst[i, j] = src[i, j]


def prepare_ice_adjust_arrays(int nijt, int nkt):
    """
    Prepare all arrays needed for ICE_ADJUST call.
    
    Returns IceAdjustState object with pre-allocated Fortran arrays.
    """
    return IceAdjustState(nijt, nkt)


def get_module_info():
    """Get information about this module."""
    return {
        'name': 'ice_adjust_wrapper',
        'description': 'Cython wrapper for ICE_ADJUST',
        'status': 'framework_ready',
        'note': 'Full implementation requires Fortran wrapper subroutine',
        'recommendation': 'Use fmodpy for production, Cython for performance-critical custom code',
    }
