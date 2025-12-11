"""
Fortran library bindings using ctypes.

This module provides Python interfaces to the compiled PHYEX Fortran library.
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Locate the compiled Fortran library
def find_fortran_library() -> Path:
    """Find the compiled Fortran shared library."""
    # Try multiple common locations
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "build_fortran" / "libice_adjust_phyex.so",
        Path("/home/maurinl/maurinl26/dwarf-p-ice3/build_fortran/libice_adjust_phyex.so"),
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found Fortran library at: {path}")
            return path
    
    raise FileNotFoundError(
        "Could not find libice_adjust_phyex.so. "
        "Please compile the Fortran library first using cmake."
    )

# Load the library
try:
    _lib_path = find_fortran_library()
    _fortran_lib = ctypes.CDLL(str(_lib_path))
    logger.info("Successfully loaded Fortran library")
except Exception as e:
    logger.warning(f"Could not load Fortran library: {e}")
    _fortran_lib = None


class FortranArray:
    """Helper class to pass NumPy arrays to Fortran."""
    
    @staticmethod
    def to_fortran(arr: np.ndarray, dtype=np.float64) -> ctypes.POINTER(ctypes.c_double):
        """Convert NumPy array to Fortran-compatible pointer."""
        if not arr.flags['F_CONTIGUOUS']:
            arr = np.asfortranarray(arr, dtype=dtype)
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    @staticmethod
    def prepare_array(shape: Tuple[int, ...], dtype=np.float64) -> np.ndarray:
        """Create a Fortran-ordered NumPy array."""
        return np.zeros(shape, dtype=dtype, order='F')


def check_library_loaded():
    """Check if the Fortran library is loaded."""
    if _fortran_lib is None:
        raise RuntimeError(
            "Fortran library not loaded. "
            "Please compile the library first using: cmake and make in build_fortran/"
        )


def call_condensation_simple(
    nijt: int,
    nkt: int,
    ppabs: np.ndarray,
    pzz: np.ndarray,
    prhodref: np.ndarray,
    pt: np.ndarray,
    prv_in: np.ndarray,
    prc_in: np.ndarray,
    pri_in: np.ndarray,
    prr: np.ndarray,
    prs: np.ndarray,
    prg: np.ndarray,
) -> dict:
    """
    Simplified wrapper for the CONDENSATION Fortran subroutine.
    
    This is a basic example showing how to call Fortran from Python.
    For full functionality, use the fmodpy-based bindings in compile_fortran.py
    
    Args:
        nijt: Number of horizontal points
        nkt: Number of vertical levels  
        ppabs: Pressure (Pa) - shape (nijt, nkt)
        pzz: Height (m) - shape (nijt, nkt)
        prhodref: Reference density - shape (nijt, nkt)
        pt: Temperature (K) - shape (nijt, nkt)
        prv_in: Water vapor mixing ratio (kg/kg) - shape (nijt, nkt)
        prc_in: Cloud water mixing ratio (kg/kg) - shape (nijt, nkt)
        pri_in: Cloud ice mixing ratio (kg/kg) - shape (nijt, nkt)
        prr: Rain mixing ratio (kg/kg) - shape (nijt, nkt)
        prs: Snow mixing ratio (kg/kg) - shape (nijt, nkt)
        prg: Graupel mixing ratio (kg/kg) - shape (nijt, nkt)
    
    Returns:
        dict: Dictionary containing output arrays:
            - prv_out: Adjusted water vapor
            - prc_out: Adjusted cloud water
            - pri_out: Adjusted cloud ice
            - pcldfr: Cloud fraction
            - psigrc: Subgrid parameter
    """
    check_library_loaded()
    
    # Prepare output arrays (Fortran-ordered)
    prv_out = FortranArray.prepare_array((nijt, nkt))
    prc_out = FortranArray.prepare_array((nijt, nkt))
    pri_out = FortranArray.prepare_array((nijt, nkt))
    pcldfr = FortranArray.prepare_array((nijt, nkt))
    psigrc = FortranArray.prepare_array((nijt, nkt))
    picldfr = FortranArray.prepare_array((nijt, nkt))
    pwcldfr = FortranArray.prepare_array((nijt, nkt))
    pssio = FortranArray.prepare_array((nijt, nkt))
    pssiu = FortranArray.prepare_array((nijt, nkt))
    pifr = FortranArray.prepare_array((nijt, nkt))
    
    # Note: This is a placeholder showing the structure.
    # Actual implementation requires proper handling of Fortran types,
    # modules, and derived types which is complex with ctypes.
    # For production use, fmodpy (already in the project) is recommended.
    
    logger.warning(
        "Direct ctypes binding to CONDENSATION is complex due to Fortran derived types. "
        "Use the fmodpy-based binding in src/ice3/utils/compile_fortran.py instead."
    )
    
    return {
        'prv_out': prv_out,
        'prc_out': prc_out,
        'pri_out': pri_out,
        'pcldfr': pcldfr,
        'psigrc': psigrc,
        'picldfr': picldfr,
        'pwcldfr': pwcldfr,
        'pssio': pssio,
        'pssiu': pssiu,
        'pifr': pifr,
    }


__all__ = [
    'FortranArray',
    'check_library_loaded',
    'call_condensation_simple',
    'find_fortran_library',
]
