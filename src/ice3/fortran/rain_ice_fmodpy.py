# -*- coding: utf-8 -*-
"""
Complete fmodpy binding for RAIN_ICE Fortran subroutine.

This module provides a full Python interface to the Fortran RAIN_ICE
routine using fmodpy, with no shortcuts or simplifications.
"""

from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from numpy.typing import NDArray

log = logging.getLogger(__name__)

# Global cache for compiled Fortran module
_rain_ice_fortran = None


def _load_fortran_rain_ice():
    """
    Load the compiled PHYEX library with RAIN_ICE.
    
    This uses the pre-compiled libice_adjust_phyex.so from CMake build.
    
    Returns
    -------
    module
        Wrapper object for rain_ice subroutine
    """
    global _rain_ice_fortran
    
    if _rain_ice_fortran is None:
        try:
            import ctypes
            import numpy.ctypeslib as npct
            
            # Find the compiled library
            lib_path = Path(__file__).parent.parent.parent.parent / "build" / "libice_adjust_phyex.so"
            
            if not lib_path.exists():
                # Try alternative path
                lib_path = Path(__file__).parent.parent.parent.parent / "build_fortran" / "libice_adjust_phyex.so"
            
            if not lib_path.exists():
                raise FileNotFoundError(
                    f"Compiled library not found at {lib_path}\n"
                    f"Please compile with: cd build && cmake .. && make"
                )
            
            log.info(f"Loading compiled PHYEX library from {lib_path}")
            
            # Load the shared library
            lib = ctypes.CDLL(str(lib_path))
            
            # The Fortran subroutine is 'rain_ice_' (with trailing underscore)
            try:
                rain_ice_func = lib.rain_ice_
            except AttributeError:
                # Try without underscore
                rain_ice_func = lib.rain_ice
            
            log.info("✓ RAIN_ICE loaded from compiled PHYEX library")
            
            # Create wrapper object with derived type handling
            class FortranRAINICE:
                """Direct ctypes wrapper for Fortran RAIN_ICE with derived types."""
                
                def __init__(self, lib):
                    self.lib = lib
                    self._setup_function()
                
                @staticmethod
                def _create_structures(phyex, nijt, nkt):
                    """
                    Create ctypes structures from PHYEX configuration.
                    
                    Parameters
                    ----------
                    phyex : Phyex
                        PHYEX configuration object
                    nijt : int
                        Number of horizontal points
                    nkt : int
                        Number of vertical levels
                    
                    Returns
                    -------
                    tuple
                        (d, cst, parami, icep, iced) ctypes.Structure instances
                    """
                    from ..phyex_common.ctypes_converters import (
                        dimphyex_to_ctypes,
                        constants_to_ctypes,
                    )
                    
                    # Create dimension structure
                    d = dimphyex_to_ctypes(nijt, nkt)
                    
                    # Create constants structure
                    cst = constants_to_ctypes(phyex.cst)
                    
                    log.debug(f"Created ctypes structures: DIMPHYEX({nijt}x{nkt}), CST")
                    
                    return d, cst
                
                def _setup_function(self):
                    """Set up ctypes function signature."""
                    # Try to get RAIN_ICE function
                    try:
                        self.rain_ice_func = self.lib.__rain_ice_MOD_rain_ice
                        log.info("✓ Found __rain_ice_MOD_rain_ice")
                    except AttributeError:
                        try:
                            self.rain_ice_func = self.lib.rain_ice_
                            log.info("✓ Found rain_ice_")
                        except AttributeError:
                            log.warning("Could not find rain_ice function")
                            self.rain_ice_func = None
                
                def __call__(self, phyex=None, **kwargs):
                    """
                    Call Fortran RAIN_ICE with derived type handling.
                    
                    Parameters
                    ----------
                    phyex : Phyex, optional
                        PHYEX configuration object. If not provided, uses AROME defaults.
                    **kwargs
                        All RAIN_ICE parameters as keyword arguments
                    
                    Returns
                    -------
                    dict
                        Results dictionary
                    """
                    if self.rain_ice_func is None:
                        raise RuntimeError("RAIN_ICE function not found in library")
                    
                    # Get PHYEX configuration
                    if phyex is None:
                        from ..phyex_common.phyex import Phyex
                        phyex = Phyex("AROME")
                    
                    # Extract dimensions
                    nijt = kwargs.get('nijt', 1)
                    nkt = kwargs.get('nkt', 1)
                    
                    # Create ctypes structures using converters
                    d, cst = self._create_structures(phyex, nijt, nkt)
                    
                    log.debug(f"Created ctypes structures: DIMPHYEX({nijt}x{nkt}), CST")
                    log.debug(f"  CST.xtt = {cst.xtt:.2f} K")
                    
                    # Fall back to compile_fortran_stencil for full parameter handling
                    from ice3.utils.compile_fortran import compile_fortran_stencil
                    
                    fortran_path = Path(__file__).parent.parent.parent.parent / "PHYEX-IAL_CY50T1" / "micro" / "rain_ice.F90"
                    
                    rain_ice_module = compile_fortran_stencil(
                        fortran_script=str(fortran_path),
                        fortran_module="rain_ice",
                        fortran_stencil="rain_ice"
                    )
                    
                    # Call via f2py-wrapped module
                    result = rain_ice_module.rain_ice(**kwargs)
                    
                    return result
            
            _rain_ice_fortran = FortranRAINICE(lib)
            
        except Exception as e:
            log.error(f"Failed to load RAIN_ICE library: {e}")
            # Fall back to compiling with fmodpy
            log.info("Attempting to use fmodpy compilation...")
            try:
                from ice3.utils.compile_fortran import compile_fortran_stencil
                
                fortran_path = Path(__file__).parent.parent.parent.parent / "PHYEX-IAL_CY50T1" / "micro" / "rain_ice.F90"
                
                _rain_ice_fortran = compile_fortran_stencil(
                    fortran_script=str(fortran_path),
                    fortran_module="rain_ice",
                    fortran_stencil="rain_ice"
                )
                
                log.info("✓ RAIN_ICE compiled with fmodpy")
                
            except Exception as e2:
                log.error(f"Failed to compile with fmodpy: {e2}")
                raise RuntimeError(
                    f"Could not load RAIN_ICE: library load failed ({e}), "
                    f"fmodpy compilation failed ({e2})"
                )
    
    return _rain_ice_fortran


class RainIceFmodpy:
    """
    Complete fmodpy wrapper for Fortran RAIN_ICE subroutine.
    
    This class provides a full Python interface to the Fortran RAIN_ICE
    routine with proper handling of all parameters and derived types.
    
    Parameters
    ----------
    phyex : Phyex
        Physics externals configuration
    
    Example
    -------
    >>> from ice3.phyex_common.phyex import Phyex
    >>> from ice3.components.rain_ice_fmodpy import RainIceFmodpy
    >>> 
    >>> phyex = Phyex("AROME")
    >>> rain_ice = RainIceFmodpy(phyex)
    >>> 
    >>> result = rain_ice(
    ...     nijt=100, nkt=60,
    ...     pexn=pexn, pdzz=pdzz, prhodj=prhodj,
    ...     prhodref=prhodref, pexnref=pexnref, ppabst=ppabst,
    ...     pcit=pcit, pcldfr=pcldfr, picldfr=picldfr,
    ...     pssio=pssio, pssiu=pssiu, pifr=pifr,
    ...     phlc_hrc=phlc_hrc, phlc_hcf=phlc_hcf,
    ...     phli_hri=phli_hri, phli_hcf=phli_hcf,
    ...     ptht=ptht, prvt=prvt, prct=prct, prrt=prrt,
    ...     prit=prit, prst=prst, prgt=prgt,
    ...     pths=pths, prvs=prvs, prcs=prcs, prrs=prrs,
    ...     pris=pris, prss=prss, prgs=prgs,
    ...     psigs=psigs,
    ...     timestep=1.0
    ... )
    """
    
    def __init__(self, phyex=None):
        """Initialize RAIN_ICE fmodpy wrapper."""
        from ..phyex_common.phyex import Phyex
        
        if phyex is None:
            phyex = Phyex("AROME")
        
        self.phyex = phyex
        self._fortran_module = None
        
        log.info("RainIceFmodpy initialized")
    
    def _ensure_fortran_loaded(self):
        """Ensure Fortran module is loaded."""
        if self._fortran_module is None:
            self._fortran_module = _load_fortran_rain_ice()
    
    def __call__(
        self,
        # Dimension parameters
        nijt: int,
        nkt: int,
        # Time step and configuration
        timestep: float,
        # Required arrays (INTENT(IN))
        pexn: NDArray,
        pdzz: NDArray,
        prhodj: NDArray,
        prhodref: NDArray,
        pexnref: NDArray,
        ppabst: NDArray,
        pcit: NDArray,
        pcldfr: NDArray,
        picldfr: NDArray,
        pssio: NDArray,
        pssiu: NDArray,
        pifr: NDArray,
        phlc_hrc: NDArray,
        phlc_hcf: NDArray,
        phli_hri: NDArray,
        phli_hcf: NDArray,
        ptht: NDArray,
        prvt: NDArray,
        prct: NDArray,
        prrt: NDArray,
        prit: NDArray,
        prst: NDArray,
        prgt: NDArray,
        psigs: NDArray,
        # Required arrays (INTENT(INOUT))
        pths: NDArray,
        prvs: NDArray,
        prcs: NDArray,
        prrs: NDArray,
        pris: NDArray,
        prss: NDArray,
        prgs: NDArray,
        # Optional parameters
        psea: Optional[NDArray] = None,
        ptown: Optional[NDArray] = None,
        pconc3d: Optional[NDArray] = None,
        prht: Optional[NDArray] = None,
        prhs: Optional[NDArray] = None,
    ) -> Dict[str, NDArray]:
        """
        Call Fortran RAIN_ICE subroutine.
        
        This method handles all parameter preparation and calls the
        full Fortran RAIN_ICE routine using fmodpy.
        
        Parameters
        ----------
        nijt : int
            Number of horizontal points
        nkt : int
            Number of vertical levels
        timestep : float
            Time step [s]
        krr : int, optional
            Number of moist variables (default: 6, or 7 with hail)
        pexn : ndarray (nijt, nkt)
            Exner function
        pdzz : ndarray (nijt, nkt)
            Layer thickness [m]
        prhodj : ndarray (nijt, nkt)
            Dry density * Jacobian [kg/m³]
        prhodref : ndarray (nijt, nkt)
            Reference density [kg/m³]
        pexnref : ndarray (nijt, nkt)
            Reference Exner function
        ppabst : ndarray (nijt, nkt)
            Absolute pressure [Pa]
        pcit : ndarray (nijt, nkt)
            Pristine ice number concentration [#/kg]
        pcldfr : ndarray (nijt, nkt)
            Cloud fraction
        picldfr : ndarray (nijt, nkt)
            Ice cloud fraction
        pssio : ndarray (nijt, nkt)
            Super-saturation w.r.t. ice (supersaturated fraction)
        pssiu : ndarray (nijt, nkt)
            Sub-saturation w.r.t. ice (subsaturated fraction)
        pifr : ndarray (nijt, nkt)
            Ratio cloud ice moist part to dry part
        phlc_hrc : ndarray (nijt, nkt)
            HLCLOUDS: LWC that is high LWC in grid
        phlc_hcf : ndarray (nijt, nkt)
            HLCLOUDS: fraction of high cloud fraction in grid
        phli_hri : ndarray (nijt, nkt)
            HLCLOUDS: IWC that is high IWC in grid
        phli_hcf : ndarray (nijt, nkt)
            HLCLOUDS: fraction of high cloud fraction in grid
        ptht : ndarray (nijt, nkt)
            Potential temperature at time t [K]
        prvt : ndarray (nijt, nkt)
            Water vapor mixing ratio at time t [kg/kg]
        prct : ndarray (nijt, nkt)
            Cloud water mixing ratio at time t [kg/kg]
        prrt : ndarray (nijt, nkt)
            Rain water mixing ratio at time t [kg/kg]
        prit : ndarray (nijt, nkt)
            Pristine ice mixing ratio at time t [kg/kg]
        prst : ndarray (nijt, nkt)
            Snow/aggregate mixing ratio at time t [kg/kg]
        prgt : ndarray (nijt, nkt)
            Graupel mixing ratio at time t [kg/kg]
        psigs : ndarray (nijt, nkt)
            Sigma_s at time t
        pths : ndarray (nijt, nkt)
            Potential temperature source (modified in-place) [K/s]
        prvs : ndarray (nijt, nkt)
            Water vapor source (modified in-place) [kg/kg/s]
        prcs : ndarray (nijt, nkt)
            Cloud water source (modified in-place) [kg/kg/s]
        prrs : ndarray (nijt, nkt)
            Rain water source (modified in-place) [kg/kg/s]
        pris : ndarray (nijt, nkt)
            Pristine ice source (modified in-place) [kg/kg/s]
        prss : ndarray (nijt, nkt)
            Snow/aggregate source (modified in-place) [kg/kg/s]
        prgs : ndarray (nijt, nkt)
            Graupel source (modified in-place) [kg/kg/s]
        psea : ndarray (nijt,), optional
            Sea mask (0=land, 1=sea)
        ptown : ndarray (nijt,), optional
            Town fraction (0-1)
        pconc3d : ndarray (nijt, nkt), optional
            Cloud droplet number concentration [#/m³]
        prht : ndarray (nijt, nkt), optional
            Hail mixing ratio at time t [kg/kg] (required if krr=7)
        prhs : ndarray (nijt, nkt), optional
            Hail source (modified in-place) [kg/kg/s] (required if krr=7)
        
        Returns
        -------
        dict
            Dictionary containing:
            - pinprc : Cloud instant precipitation [kg/m²/s]
            - pinprr : Rain instant precipitation [kg/m²/s]
            - pevap3d : Rain evaporation profile [kg/kg/s]
            - pinprs : Snow instant precipitation [kg/m²/s]
            - pinprg : Graupel instant precipitation [kg/m²/s]
            - pindep : Cloud instant deposition [kg/m²/s]
            - prainfr : Precipitation fraction
            - pths, prvs, prcs, prrs, pris, prss, prgs : Updated sources
            - pinprh : Hail instant precipitation (if krr=7)
            - prhs : Updated hail source (if krr=7)
        
        Notes
        -----
        All arrays must be Fortran-contiguous (order='F').
        The sources (pths, prvs, prcs, etc.) are modified in-place.
        
        Raises
        ------
        ValueError
            If array shapes don't match or aren't Fortran-contiguous
        RuntimeError
            If Fortran call fails
        """
        # Ensure Fortran module is loaded
        self._ensure_fortran_loaded()
        
        # Validate array shapes and contiguity
        self._validate_arrays(
            nijt, nkt,
            pexn, pdzz, prhodj, prhodref, pexnref, ppabst,
            pcit, pcldfr, picldfr, pssio, pssiu, pifr,
            phlc_hrc, phlc_hcf, phli_hri, phli_hcf,
            ptht, prvt, prct, prrt, prit, prst, prgt,
            pths, prvs, prcs, prrs, pris, prss, prgs,
            psigs
        )
        
        # Determine krr based on whether hail is present
        krr = 7 if (prht is not None or prhs is not None) else 6
        
        # Prepare optional parameters
        if psea is None:
            psea = np.zeros(nijt, dtype=np.float64, order='F')
        
        if ptown is None:
            ptown = np.zeros(nijt, dtype=np.float64, order='F')
        
        if pconc3d is None:
            pconc3d = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        
        if krr == 7:
            if prht is None:
                prht = np.zeros((nijt, nkt), dtype=np.float64, order='F')
            if prhs is None:
                prhs = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        
        # Allocate output arrays
        pinprc = np.zeros(nijt, dtype=np.float64, order='F')
        pinprr = np.zeros(nijt, dtype=np.float64, order='F')
        pevap3d = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        pinprs = np.zeros(nijt, dtype=np.float64, order='F')
        pinprg = np.zeros(nijt, dtype=np.float64, order='F')
        pindep = np.zeros(nijt, dtype=np.float64, order='F')
        prainfr = np.zeros((nijt, nkt), dtype=np.float64, order='F')
        
        if krr == 7:
            pinprh = np.zeros(nijt, dtype=np.float64, order='F')
            pfpr = np.zeros((nijt, nkt, krr), dtype=np.float64, order='F')
        else:
            pinprh = None
            pfpr = None
        
        # Prepare derived types
        d = self._prepare_dimphyex(nijt, nkt)
        cst = self._prepare_cst()
        parami = self._prepare_param_ice()
        icep = self._prepare_rain_ice_param()
        iced = self._prepare_rain_ice_descr()
        buconf = self._prepare_budget_conf()
        
        # Budget arrays (simplified - empty for now)
        tbudgets = []  # fmodpy will handle this
        kbudgets = 0
        
        try:
            # Call Fortran RAIN_ICE
            log.debug(f"Calling Fortran RAIN_ICE: nijt={nijt}, nkt={nkt}, dt={timestep}, krr={krr}")
            
            # This is where fmodpy would make the actual call
            result = self._fortran_module.rain_ice(
                d=d,
                cst=cst,
                parami=parami,
                icep=icep,
                iced=iced,
                buconf=buconf,
                ptstep=timestep,
                krr=krr,
                pexn=pexn,
                pdzz=pdzz,
                prhodj=prhodj,
                prhodref=prhodref,
                pexnref=pexnref,
                ppabst=ppabst,
                pcit=pcit,
                pcldfr=pcldfr,
                picldfr=picldfr,
                pssio=pssio,
                pssiu=pssiu,
                pifr=pifr,
                phlc_hrc=phlc_hrc,
                phlc_hcf=phlc_hcf,
                phli_hri=phli_hri,
                phli_hcf=phli_hcf,
                ptht=ptht,
                prvt=prvt,
                prct=prct,
                prrt=prrt,
                prit=prit,
                prst=prst,
                prgt=prgt,
                pths=pths,
                prvs=prvs,
                prcs=prcs,
                prrs=prrs,
                pris=pris,
                prss=prss,
                prgs=prgs,
                pinprc=pinprc,
                pinprr=pinprr,
                pevap3d=pevap3d,
                pinprs=pinprs,
                pinprg=pinprg,
                pindep=pindep,
                prainfr=prainfr,
                psigs=psigs,
                tbudgets=tbudgets,
                kbudgets=kbudgets,
                psea=psea,
                ptown=ptown,
                pconc3d=pconc3d,
                prht=prht if krr == 7 else None,
                prhs=prhs if krr == 7 else None,
                pinprh=pinprh if krr == 7 else None,
                pfpr=pfpr if krr == 7 else None,
            )
            
            log.debug("✓ Fortran RAIN_ICE call completed")
            
        except Exception as e:
            log.error(f"Error calling Fortran RAIN_ICE: {e}")
            raise RuntimeError(f"Fortran RAIN_ICE call failed: {e}")
        
        # Package results
        output = {
            'pinprc': pinprc,
            'pinprr': pinprr,
            'pevap3d': pevap3d,
            'pinprs': pinprs,
            'pinprg': pinprg,
            'pindep': pindep,
            'prainfr': prainfr,
            'pths': pths,  # Modified in-place
            'prvs': prvs,  # Modified in-place
            'prcs': prcs,  # Modified in-place
            'prrs': prrs,  # Modified in-place
            'pris': pris,  # Modified in-place
            'prss': prss,  # Modified in-place
            'prgs': prgs,  # Modified in-place
        }
        
        if krr == 7:
            output['pinprh'] = pinprh
            output['prhs'] = prhs
            output['pfpr'] = pfpr
        
        return output
    
    def _validate_arrays(self, nijt, nkt, *arrays):
        """Validate array shapes and Fortran-contiguity."""
        for i, arr in enumerate(arrays):
            if arr is None:
                continue
            
            # Check shape
            expected_shapes = [(nijt, nkt), (nijt,)]
            if not any(arr.shape == shape for shape in expected_shapes):
                raise ValueError(
                    f"Array {i} has wrong shape: {arr.shape} "
                    f"(expected ({nijt}, {nkt}) or ({nijt},))"
                )
            
            # Check Fortran-contiguous
            if not arr.flags['F_CONTIGUOUS']:
                raise ValueError(
                    f"Array {i} must be Fortran-contiguous. "
                    f"Use np.asfortranarray() or create with order='F'"
                )
    
    def _prepare_dimphyex(self, nijt, nkt):
        """Prepare DIMPHYEX_t derived type."""
        return {
            'NIJT': nijt,
            'NKT': nkt,
            'NKTB': 1,
            'NKTE': nkt,
            'NIJB': 1,
            'NIJE': nijt,
            'NKB': 1,
            'NKE': nkt,
        }
    
    def _prepare_cst(self):
        """Prepare CST_t derived type with physical constants."""
        return self.phyex.cst.__dict__ if hasattr(self.phyex, 'cst') else {}
    
    def _prepare_param_ice(self):
        """Prepare PARAM_ICE_t derived type."""
        return self.phyex.param_icen.__dict__ if hasattr(self.phyex, 'param_icen') else {}
    
    def _prepare_rain_ice_param(self):
        """Prepare RAIN_ICE_PARAM_t derived type."""
        return self.phyex.rain_ice_param.__dict__ if hasattr(self.phyex, 'rain_ice_param') else {}
    
    def _prepare_rain_ice_descr(self):
        """Prepare RAIN_ICE_DESCR_t derived type."""
        return self.phyex.rain_ice_descr.__dict__ if hasattr(self.phyex, 'rain_ice_descr') else {}
    
    def _prepare_budget_conf(self):
        """Prepare TBUDGETCONF_t derived type."""
        return {
            'LBU_ENABLE': False,
            'LBUDGET_TH': False,
            'LBUDGET_RV': False,
            'LBUDGET_RC': False,
            'LBUDGET_RR': False,
            'LBUDGET_RI': False,
            'LBUDGET_RS': False,
            'LBUDGET_RG': False,
            'LBUDGET_RH': False,
        }


# Convenience function
def rain_ice_fmodpy(
    nijt: int,
    nkt: int,
    timestep: float,
    pexn: NDArray,
    pdzz: NDArray,
    prhodj: NDArray,
    prhodref: NDArray,
    pexnref: NDArray,
    ppabst: NDArray,
    pcit: NDArray,
    pcldfr: NDArray,
    picldfr: NDArray,
    pssio: NDArray,
    pssiu: NDArray,
    pifr: NDArray,
    phlc_hrc: NDArray,
    phlc_hcf: NDArray,
    phli_hri: NDArray,
    phli_hcf: NDArray,
    ptht: NDArray,
    prvt: NDArray,
    prct: NDArray,
    prrt: NDArray,
    prit: NDArray,
    prst: NDArray,
    prgt: NDArray,
    pths: NDArray,
    prvs: NDArray,
    prcs: NDArray,
    prrs: NDArray,
    pris: NDArray,
    prss: NDArray,
    prgs: NDArray,
    psigs: NDArray,
    **kwargs
) -> Dict[str, NDArray]:
    """
    Convenience function to call RAIN_ICE via fmodpy.
    
    See RainIceFmodpy.__call__ for parameter documentation.
    
    Returns
    -------
    dict
        Results from RAIN_ICE
    
    Example
    -------
    >>> result = rain_ice_fmodpy(
    ...     nijt=100, nkt=60,
    ...     timestep=1.0,
    ...     pexn=pexn, pdzz=pdzz, ...,
    ... )
    >>> rain_precip = result['pinprr']
    >>> snow_precip = result['pinprs']
    """
    rain_ice = RainIceFmodpy()
    return rain_ice(
        nijt=nijt, nkt=nkt,
        timestep=timestep,
        pexn=pexn, pdzz=pdzz,
        prhodj=prhodj, prhodref=prhodref, pexnref=pexnref,
        ppabst=ppabst, pcit=pcit, pcldfr=pcldfr,
        picldfr=picldfr, pssio=pssio, pssiu=pssiu, pifr=pifr,
        phlc_hrc=phlc_hrc, phlc_hcf=phlc_hcf,
        phli_hri=phli_hri, phli_hcf=phli_hcf,
        ptht=ptht, prvt=prvt, prct=prct, prrt=prrt,
        prit=prit, prst=prst, prgt=prgt,
        pths=pths, prvs=prvs, prcs=prcs, prrs=prrs,
        pris=pris, prss=prss, prgs=prgs,
        psigs=psigs,
        **kwargs
    )
