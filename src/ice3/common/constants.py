"""
Physical constants module - GT4Py translation of MODD_CST.

This module defines the physical constants used in the ice adjustment scheme,
translated from the Fortran MODD_CST module.
"""

import numpy as np
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class PhysicalConstants:
    """Physical constants structure equivalent to CST_t in Fortran."""
    
    # Fundamental constants
    XPI: ClassVar[float] = np.pi
    XKARMAN: ClassVar[float] = 0.4
    XLIGHTSPEED: ClassVar[float] = 2.99792458e8
    XPLANCK: ClassVar[float] = 6.6260755e-34
    XBOLTZ: ClassVar[float] = 1.380658e-23
    XAVOGADRO: ClassVar[float] = 6.0221367e23
    
    # Astronomical constants
    XDAY: ClassVar[float] = 86400.0
    XSIYEA: ClassVar[float] = 365.25 * 86400.0
    XSIDAY: ClassVar[float] = 86164.0
    NDAYSEC: ClassVar[int] = 86400
    XOMEGA: ClassVar[float] = 7.292115e-5
    
    # Terrestrial constants
    XRADIUS: ClassVar[float] = 6.371229e6
    XG: ClassVar[float] = 9.80665
    
    # Reference pressure and temperature
    XP00: ClassVar[float] = 1.0e5
    XP00OCEAN: ClassVar[float] = 201325.0
    XRH00OCEAN: ClassVar[float] = 1024.458
    XTH00: ClassVar[float] = 300.0
    XTH00OCEAN: ClassVar[float] = 286.65
    XSA00OCEAN: ClassVar[float] = 34.7
    
    # Radiation constants
    XSTEFAN: ClassVar[float] = 5.6703e-8
    XI0: ClassVar[float] = 1370.0
    
    # Thermodynamic constants
    XMD: ClassVar[float] = 28.9644e-3
    XMV: ClassVar[float] = 18.0153e-3
    XRD: ClassVar[float] = 287.05967
    XRV: ClassVar[float] = 461.524993
    XEPSILO: ClassVar[float] = 0.621981  # XMV/XMD
    XCPD: ClassVar[float] = 1004.708845
    XCPV: ClassVar[float] = 1846.1
    XRHOLW: ClassVar[float] = 1000.0
    XCL: ClassVar[float] = 4218.0
    XCI: ClassVar[float] = 2106.0
    XTT: ClassVar[float] = 273.16
    XLVTT: ClassVar[float] = 2.5008e6
    XLSTT: ClassVar[float] = 2.8345e6
    XLMTT: ClassVar[float] = 0.3337e6
    XESTT: ClassVar[float] = 611.14
    
    # Saturation vapor pressure constants over liquid water
    XALPW: ClassVar[float] = 17.269
    XBETAW: ClassVar[float] = 35.86
    XGAMW: ClassVar[float] = 4.876e-4
    
    # Saturation vapor pressure constants over ice
    XALPI: ClassVar[float] = 21.875
    XBETAI: ClassVar[float] = 7.66
    XGAMI: ClassVar[float] = 4.876e-4
    
    # Other thermodynamic constants
    XCONDI: ClassVar[float] = 2.22
    XALPHAOC: ClassVar[float] = 2.0e-4
    XBETAOC: ClassVar[float] = 7.7e-4
    XROC: ClassVar[float] = 0.69
    XD1: ClassVar[float] = 1.1
    XD2: ClassVar[float] = 23.0
    XRHOLI: ClassVar[float] = 917.0
    
    # Precomputed constants
    RDSRV: ClassVar[float] = XRD / XRV
    RDSCPD: ClassVar[float] = XRD / XCPD
    RINVXP00: ClassVar[float] = 1.0 / XP00
    
    # Machine precision constants (for float64)
    XMNH_TINY: ClassVar[float] = np.finfo(np.float64).tiny
    XMNH_TINY_12: ClassVar[float] = np.sqrt(np.finfo(np.float64).tiny)
    XMNH_EPSILON: ClassVar[float] = np.finfo(np.float64).eps
    XMNH_HUGE: ClassVar[float] = np.finfo(np.float64).max
    XMNH_HUGE_12_LOG: ClassVar[float] = np.log(np.sqrt(np.finfo(np.float64).max))
    XEPS_DT: ClassVar[float] = 1.0e-12
    XRES_FLAT_CART: ClassVar[float] = 1.0e-16
    XRES_OTHER: ClassVar[float] = 1.0e-10
    XRES_PREP: ClassVar[float] = 1.0e-9


# Create a global instance for easy access
CST = PhysicalConstants()
