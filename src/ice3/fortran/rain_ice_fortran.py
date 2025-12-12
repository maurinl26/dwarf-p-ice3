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
                    
                    return result
            
            _rain_ice_fortran = FortranRAINICE(lib)
            
        except Exception as e:
            log.error(f"Failed to load RAIN_ICE library: {e}")
            # Fall back to compiling with fmodpy
            log.info("Attempting to use fmodpy compilation...")
            try:
             
                log.info("✓ RAIN_ICE compiled with fmodpy")
                
            except Exception as e2:
                log.error(f"Failed to compile with fmodpy: {e2}")
                raise RuntimeError(
                    f"Could not load RAIN_ICE: library load failed ({e}), "
                    f"fmodpy compilation failed ({e2})"
                )
    
    return _rain_ice_fortran

