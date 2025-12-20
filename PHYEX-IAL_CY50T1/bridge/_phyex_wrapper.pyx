# distutils: language = c
# cython: language_level = 3
"""
Cython wrapper for PHYEX ICE_ADJUST routine.

This module provides a Python interface to the Fortran ICE_ADJUST subroutine
through a C bridge defined in phyex_bridge.F90.
"""
import numpy as np
cimport numpy as np

# Declare memory view data types for single precision (JPRB/float32)
ctypedef np.float32_t DTYPE_t

# External C Declaration matching phyex_bridge.F90
cdef extern:
    void c_ice_adjust(
        int nlon,
        int nlev, 
        int krr,
        float timestep,
        float *ptr_sigqsat,
        float *ptr_pabs,
        float *ptr_sigs,
        float *ptr_th,
        float *ptr_exn,
        float *ptr_exn_ref,
        float *ptr_rho_dry_ref,
        float *ptr_rv,
        float *ptr_rc,
        float *ptr_ri,
        float *ptr_rr,
        float *ptr_rs,
        float *ptr_rg,
        float *ptr_cf_mf,
        float *ptr_rc_mf,
        float *ptr_ri_mf,
        float *ptr_rvs,
        float *ptr_rcs,
        float *ptr_ris,
        float *ptr_ths,
        float *ptr_cldfr,
        float *ptr_icldfr,
        float *ptr_wcldfr
    )

# Python-Callable Wrapper Function
def ice_adjust(
    # Scalar parameters
    float timestep,
    int krr,
    # 1D arrays (Fortran-contiguous)
    np.ndarray[DTYPE_t, ndim=1, mode="fortran"] sigqsat,
    # 2D arrays (Fortran-contiguous, shape: nlon × nlev)
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] pabs,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] sigs,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] th,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] exn,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] exn_ref,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rho_dry_ref,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rv,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rc,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] ri,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rr,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rs,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rg,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] cf_mf,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rc_mf,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] ri_mf,
    # Tendency arrays (input/output, modified in-place)
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rvs,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rcs,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] ris,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] ths,
    # Output arrays (modified in-place)
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] cldfr,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] icldfr,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] wcldfr
):
    """
    Cython wrapper for the PHYEX ICE_ADJUST routine.
    
    Performs saturation adjustment for mixed-phase clouds, computing
    condensation/deposition and updating mixing ratios and temperature.
    
    Parameters
    ----------
    timestep : float
        Time step (seconds)
    krr : int
        Number of moist variables (typically 6 or 7)
    sigqsat : ndarray (nlon,)
        Standard deviation of saturation mixing ratio
    pabs : ndarray (nlon, nlev)
        Absolute pressure (Pa)
    sigs : ndarray (nlon, nlev)
        Sigma_s for subgrid turbulent mixing
    th : ndarray (nlon, nlev)
        Potential temperature (K)
    exn : ndarray (nlon, nlev)
        Exner function
    exn_ref : ndarray (nlon, nlev)
        Reference Exner function
    rho_dry_ref : ndarray (nlon, nlev)
        Reference dry air density (kg/m³)
    rv, rc, ri, rr, rs, rg : ndarray (nlon, nlev)
        Mixing ratios for vapor, cloud liquid, cloud ice, rain, snow, graupel (kg/kg)
    cf_mf, rc_mf, ri_mf : ndarray (nlon, nlev)
        Mass flux cloud fraction and mixing ratios
    rvs, rcs, ris, ths : ndarray (nlon, nlev)
        Tendency fields (modified in-place)
    cldfr, icldfr, wcldfr : ndarray (nlon, nlev)
        Output cloud fractions (modified in-place)
    
    Returns
    -------
    None
        All output arrays are modified in-place.
    
    Notes
    -----
    - All 2D arrays must be Fortran-contiguous with shape (nlon, nlev)
    - All arrays must be float32 (np.float32) for single precision
    - Tendency arrays (rvs, rcs, ris, ths) are both input and output
    - Cloud fraction arrays (cldfr, icldfr, wcldfr) are output only
    
    Examples
    --------
    >>> import numpy as np
    >>> from _phyex_wrapper import ice_adjust
    >>> 
    >>> # Initialize arrays (Fortran order, single precision)
    >>> nlon, nlev = 10, 20
    >>> sigqsat = np.ones(nlon, dtype=np.float32, order='F') * 0.01
    >>> pabs = np.ones((nlon, nlev), dtype=np.float32, order='F') * 85000.0
    >>> # ... initialize other arrays ...
    >>> 
    >>> # Call ice_adjust
    >>> ice_adjust(
    ...     timestep=60.0, krr=6,
    ...     sigqsat=sigqsat, pabs=pabs, sigs=sigs, th=th,
    ...     exn=exn, exn_ref=exn_ref, rho_dry_ref=rho_dry_ref,
    ...     rv=rv, rc=rc, ri=ri, rr=rr, rs=rs, rg=rg,
    ...     cf_mf=cf_mf, rc_mf=rc_mf, ri_mf=ri_mf,
    ...     rvs=rvs, rcs=rcs, ris=ris, ths=ths,
    ...     cldfr=cldfr, icldfr=icldfr, wcldfr=wcldfr
    ... )
    """
    
    # Get dimensions
    cdef int nlon = pabs.shape[0]
    cdef int nlev = pabs.shape[1]
    
    # Validate 1D array shapes
    if sigqsat.shape[0] != nlon:
        raise ValueError("sigqsat shape mismatch: expected ({},), got ({},)".format(
            nlon, sigqsat.shape[0]))
    
    # Validate 2D array shapes
    cdef list arrays_2d = [
        ('pabs', pabs), ('sigs', sigs), ('th', th), ('exn', exn),
        ('exn_ref', exn_ref), ('rho_dry_ref', rho_dry_ref),
        ('rv', rv), ('rc', rc), ('ri', ri), ('rr', rr), ('rs', rs), ('rg', rg),
        ('cf_mf', cf_mf), ('rc_mf', rc_mf), ('ri_mf', ri_mf),
        ('rvs', rvs), ('rcs', rcs), ('ris', ris), ('ths', ths),
        ('cldfr', cldfr), ('icldfr', icldfr), ('wcldfr', wcldfr)
    ]
    
    for name, arr in arrays_2d:
        if arr.shape[0] != nlon or arr.shape[1] != nlev:
            raise ValueError(
                "{} shape mismatch: expected ({}, {}), got ({}, {})".format(
                    name, nlon, nlev, arr.shape[0], arr.shape[1])
            )
    
    # Call the Fortran function through C bridge
    c_ice_adjust(
        nlon, nlev, krr, timestep,
        &sigqsat[0],
        &pabs[0, 0], &sigs[0, 0], &th[0, 0], &exn[0, 0], &exn_ref[0, 0],
        &rho_dry_ref[0, 0], &rv[0, 0], &rc[0, 0], &ri[0, 0],
        &rr[0, 0], &rs[0, 0], &rg[0, 0],
        &cf_mf[0, 0], &rc_mf[0, 0], &ri_mf[0, 0],
        &rvs[0, 0], &rcs[0, 0], &ris[0, 0], &ths[0, 0],
        &cldfr[0, 0], &icldfr[0, 0], &wcldfr[0, 0]
    )
    
    # Arrays are modified in-place, no return needed