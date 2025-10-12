"""Shallow convection parameters.

Translation of MODD_CONVPAR_SHAL.F90 and INI_CONVPAR_SHAL.F90 to Python dataclass.
Contains constants used in the shallow convection parameterization.
"""

from dataclasses import dataclass


@dataclass
class CONVPAR_SHAL:
    """Shallow convection parameters.
    
    Attributes:
        XA25: 25 km x 25 km reference grid area (m^2)
        XCRAD: Cloud radius (m)
        XCTIME_SHAL: Convective adjustment time (s)
        XCDEPTH: Minimum necessary cloud depth (m)
        XCDEPTH_D: Maximum allowed cloud thickness (m)
        XDTPERT: Small temperature perturbation factor at LCL
        XATPERT: Parameter for temp perturbation (TKE/Cp coefficient)
        XBTPERT: Parameter for temp perturbation (constant term)
        XENTR: Entrainment constant (m/Pa) = 0.2 (m)
        XZLCL: Maximum allowed height difference between departure level and surface (m)
        XZPBL: Minimum mixed layer depth to sustain convection (Pa)
        XWTRIG: Constant in vertical velocity trigger
        XNHGAM: Accounts for non-hydrostatic pressure in buoyancy term = 2/(1+gamma)
        XTFRZ1: Begin of freezing interval (K)
        XTFRZ2: End of freezing interval (K)
        XSTABT: Stability factor in fractional time integration
        XSTABC: Stability factor in CAPE adjustment
        XAW: Parameter for WLCL = XAW * W + XBW
        XBW: Parameter for WLCL = XAW * W + XBW
        LLSMOOTH: Smoothing flag (default True)
    """
    
    XA25: float = 625.0e6  # 25 km x 25 km reference grid area
    XCRAD: float = 50.0  # cloud radius (m)
    XCTIME_SHAL: float = 10800.0  # convective adjustment time (s)
    XCDEPTH: float = 0.5e3  # minimum necessary shallow cloud depth (m)
    XCDEPTH_D: float = 2.5e3  # maximum allowed shallow cloud depth (m)
    XDTPERT: float = 0.2  # add small Temp perturbation at LCL
    XATPERT: float = 0.0  # 0.=original scheme, recommended = 1000.
    XBTPERT: float = 1.0  # 1.=original scheme, recommended = 0.
    XENTR: float = 0.02  # entrainment constant (m/Pa) = 0.2 (m)
    XZLCL: float = 0.5e3  # maximum allowed height diff between DPL and surface (m)
    XZPBL: float = 40.0e2  # minimum mixed layer depth to sustain convection (Pa)
    XWTRIG: float = 0.0  # constant in vertical velocity trigger (not initialized in ini_convpar_shal)
    XNHGAM: float = 1.3333  # non-hydrostatic pressure term = 2/(1+gamma)
    XTFRZ1: float = 268.16  # begin of freezing interval (K)
    XTFRZ2: float = 248.16  # end of freezing interval (K)
    XSTABT: float = 0.75  # stability factor in fractional time integration
    XSTABC: float = 0.95  # stability factor in CAPE adjustment
    XAW: float = 0.0  # 0.=original scheme, 1=recommended
    XBW: float = 1.0  # 1.=original scheme, 0=recommended
    LLSMOOTH: bool = True  # smoothing flag


def init_convpar_shal() -> CONVPAR_SHAL:
    """Initialize shallow convection parameters with default values.
    
    Returns:
        CONVPAR_SHAL instance with default parameter values from INI_CONVPAR_SHAL.F90
    """
    return CONVPAR_SHAL()
