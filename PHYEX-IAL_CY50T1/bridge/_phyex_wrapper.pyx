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
    
    void c_rain_ice(
        int nlon,
        int nlev,
        int krr,
        float timestep,
        # 2D input arrays
        float *ptr_exn,
        float *ptr_dzz,
        float *ptr_rhodj,
        float *ptr_rhodref,
        float *ptr_exnref,
        float *ptr_pabs,
        float *ptr_cldfr,
        float *ptr_icldfr,
        float *ptr_ssio,
        float *ptr_ssiu,
        float *ptr_ifr,
        float *ptr_tht,
        float *ptr_rvt,
        float *ptr_rct,
        float *ptr_rrt,
        float *ptr_rit,
        float *ptr_rst,
        float *ptr_rgt,
        float *ptr_sigs,
        # 2D input/output arrays
        float *ptr_cit,
        float *ptr_hlc_hrc,
        float *ptr_hlc_hcf,
        float *ptr_hli_hri,
        float *ptr_hli_hcf,
        float *ptr_ths,
        float *ptr_rvs,
        float *ptr_rcs,
        float *ptr_rrs,
        float *ptr_ris,
        float *ptr_rss,
        float *ptr_rgs,
        # 2D output arrays
        float *ptr_evap3d,
        float *ptr_rainfr,
        # 1D output arrays
        float *ptr_inprc,
        float *ptr_inprr,
        float *ptr_inprs,
        float *ptr_inprg,
        float *ptr_indep
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


# Python-Callable Wrapper for RAIN_ICE
def rain_ice(
    # Scalar parameters
    float timestep,
    int krr,
    # 2D input arrays (Fortran-contiguous)
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] exn,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] dzz,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rhodj,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rhodref,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] exnref,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] pabs,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] cldfr,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] icldfr,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] ssio,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] ssiu,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] ifr,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] tht,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rvt,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rct,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rrt,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rit,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rst,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rgt,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] sigs,
    # Input/output arrays
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] cit,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] hlc_hrc,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] hlc_hcf,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] hli_hri,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] hli_hcf,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] ths,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rvs,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rcs,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rrs,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] ris,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rss,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rgs,
    # Output arrays
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] evap3d,
    np.ndarray[DTYPE_t, ndim=2, mode="fortran"] rainfr,
    np.ndarray[DTYPE_t, ndim=1, mode="fortran"] inprc,
    np.ndarray[DTYPE_t, ndim=1, mode="fortran"] inprr,
    np.ndarray[DTYPE_t, ndim=1, mode="fortran"] inprs,
    np.ndarray[DTYPE_t, ndim=1, mode="fortran"] inprg,
    np.ndarray[DTYPE_t, ndim=1, mode="fortran"] indep
):
    """
    Call the PHYEX RAIN_ICE microphysics routine.
    
    Computes explicit microphysical sources for mixed-phase cloud microphysics.
    
    Parameters
    ----------
    timestep : float32
        Time step (seconds)
    krr : int
        Number of moist variables (typically 6 for vapor+cloud+rain+ice+snow+graupel)
    
    2D Input Arrays (all float32, shape: nlon × nlev, Fortran-contiguous):
    exn, dzz, rhodj, rhodref, exnref, pabs : atmospheric state
    cldfr, icldfr, ssio, ssiu, ifr : cloud fractions and supersaturation
    tht, rvt, rct, rrt, rit, rst, rgt : mixing ratios at time t
    sigs : subgrid turbulence parameter
    
    2D Input/Output Arrays (float32, shape: nlon × nlev):
    cit : pristine ice number concentration
    hlc_hrc, hlc_hcf, hli_hri, hli_hcf : homogeneous/heterogeneous cloud parameters
    ths, rvs, rcs, rrs, ris, rss, rgs : tendency arrays
    
    2D Output Arrays (float32, shape: nlon × nlev):
    evap3d : rain evaporation profile
    rainfr : precipitation fraction
    
    1D Output Arrays (float32, shape: nlon):
    inprc, inprr, inprs, inprg, indep : instantaneous precipitation rates
    
    Notes
    -----
    - All arrays must be float32 (np.float32) for single precision
    - All arrays must be Fortran-contiguous (order='F')
    - Tendency and output arrays are modified in-place
    """
    
    # Get dimensions
    cdef int nlon = exn.shape[0]
    cdef int nlev = exn.shape[1]
    
    # Validate 2D array shapes
    cdef list arrays_2d = [
        ('exn', exn), ('dzz', dzz), ('rhodj', rhodj), ('rhodref', rhodref),
        ('exnref', exnref), ('pabs', pabs), ('cldfr', cldfr), ('icldfr', icldfr),
        ('ssio', ssio), ('ssiu', ssiu), ('ifr', ifr),
        ('tht', tht), ('rvt', rvt), ('rct', rct), ('rrt', rrt),
        ('rit', rit), ('rst', rst), ('rgt', rgt), ('sigs', sigs),
        ('cit', cit), ('hlc_hrc', hlc_hrc), ('hlc_hcf', hlc_hcf),
        ('hli_hri', hli_hri), ('hli_hcf', hli_hcf),
        ('ths', ths), ('rvs', rvs), ('rcs', rcs), ('rrs', rrs),
        ('ris', ris), ('rss', rss), ('rgs', rgs),
        ('evap3d', evap3d), ('rainfr', rainfr)
    ]
    
    for name, arr in arrays_2d:
        if arr.shape[0] != nlon or arr.shape[1] != nlev:
            raise ValueError(
                "{} shape mismatch: expected ({}, {}), got ({}, {})".format(
                    name, nlon, nlev, arr.shape[0], arr.shape[1])
            )
    
    # Validate 1D array shapes
    if inprc.shape[0] != nlon:
        raise ValueError("inprc shape mismatch: expected ({},), got ({},)".format(
            nlon, inprc.shape[0]))
    if inprr.shape[0] != nlon:
        raise ValueError("inprr shape mismatch: expected ({},), got ({},)".format(
            nlon, inprr.shape[0]))
    if inprs.shape[0] != nlon:
        raise ValueError("inprs shape mismatch: expected ({},), got ({},)".format(
            nlon, inprs.shape[0]))
    if inprg.shape[0] != nlon:
        raise ValueError("inprg shape mismatch: expected ({},), got ({},)".format(
            nlon, inprg.shape[0]))
    if indep.shape[0] != nlon:
        raise ValueError("indep shape mismatch: expected ({},), got ({},)".format(
            nlon, indep.shape[0]))
    
    # Call the Fortran function through C bridge
    c_rain_ice(
        nlon, nlev, krr, timestep,
        &exn[0, 0], &dzz[0, 0], &rhodj[0, 0], &rhodref[0, 0],
        &exnref[0, 0], &pabs[0, 0], &cldfr[0, 0], &icldfr[0, 0],
        &ssio[0, 0], &ssiu[0, 0], &ifr[0, 0],
        &tht[0, 0], &rvt[0, 0], &rct[0, 0], &rrt[0, 0],
        &rit[0, 0], &rst[0, 0], &rgt[0, 0], &sigs[0, 0],
        &cit[0, 0],
        &hlc_hrc[0, 0], &hlc_hcf[0, 0], &hli_hri[0, 0], &hli_hcf[0, 0],
        &ths[0, 0], &rvs[0, 0], &rcs[0, 0], &rrs[0, 0],
        &ris[0, 0], &rss[0, 0], &rgs[0, 0],
        &evap3d[0, 0], &rainfr[0, 0],
        &inprc[0], &inprr[0], &inprs[0], &inprg[0], &indep[0]
    )
    
    # Arrays are modified in-place, no return needed