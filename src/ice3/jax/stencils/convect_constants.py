"""
Physical constants for convection schemes.

Translated from: PHYEX/conv/modd_convpar_shal.F90 and modd_convpar.F90

These constants are specific to shallow and deep convection parameterizations
and extend the base PHYEX constants.
"""

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class ConvectionConstants:
    """
    Constants for shallow and deep convection parameterizations.

    These values are primarily from the AROME operational configuration
    and represent tuning parameters for the convection scheme.
    """

    # Shallow convection parameters (CONVPAR_SHAL)
    XDTPERT: float = 1.0e-4  # Temperature perturbation for trigger (K)
    XNHGAM: float = 1.3333   # Gamma exponent for buoyancy
    XTFRZ1: float = 268.15   # Begin of freezing interval (K)
    XTFRZ2: float = 248.15   # End of freezing interval (K)
    XSTABT: float = 0.90     # Stability threshold for trigger
    XSTABC: float = 0.95     # Cloud stability threshold

    # General convection parameters (CONVPAR)
    XCDEPTH: float = 0.0      # Cloud depth for KE dissipation (m)
    XENTR: float = 0.03       # Entrainment constant (1/m)
    XZLCL: float = 3000.0     # Maximum LCL height (m)
    XZPBL: float = 60.0       # PBL reference height (m)
    XWTRIG: float = 6.00      # Vertical velocity trigger threshold (m/s)
    XCDEPTH_D: float = 4000.0 # Deep convection cloud depth (m)
    XENTR_DRY: float = 0.03   # Dry entrainment rate (1/m)
    XRHDBC: float = 0.9       # Relative humidity at cloud base

    # Microphysics constants
    XMELDPTH: float = 100.0   # Melting depth (m)
    XKCRIT: float = 1.0e-4    # Critical kinetic energy (m²/s²)
    XCDTMAX: float = 900.0    # Maximum convective timestep (s)

    # Shallow convection closure
    XDTADJ: float = 3600.0    # Default adjustment timescale (s)
    XDCAPE: float = 0.1       # Fraction of CAPE removed per timestep

    # Reference values
    XA25: float = 0.25        # Reference grid area (m²)
    XCRAD: float = 1500.0     # Cloud radius (m)

    # Minimum thresholds
    RTMIN: float = 1.0e-15    # Minimum mixing ratio (kg/kg)
    PTMIN: float = 1.0e-10    # Minimum precipitation rate (kg/m²/s)

    # Bolton (1980) constants for saturation vapor pressure
    # These are already in base constants but repeated here for reference
    # e_s = exp(ALPW - BETAW/T - GAMW*log(T))
    # Used in convect_satmixratio


def get_convection_constants() -> dict:
    """
    Get convection constants as a dictionary compatible with JAX stencils.

    Returns
    -------
    dict
        Dictionary mapping constant names to values
    """
    const = ConvectionConstants()
    return {
        # Shallow convection
        "xdtpert": const.XDTPERT,
        "xnhgam": const.XNHGAM,
        "xtfrz1": const.XTFRZ1,
        "xtfrz2": const.XTFRZ2,
        "xstabt": const.XSTABT,
        "xstabc": const.XSTABC,

        # General convection
        "xcdepth": const.XCDEPTH,
        "xentr": const.XENTR,
        "xzlcl": const.XZLCL,
        "xzpbl": const.XZPBL,
        "xwtrig": const.XWTRIG,
        "xcdepth_d": const.XCDEPTH_D,
        "xentr_dry": const.XENTR_DRY,
        "xrhdbc": const.XRHDBC,

        # Microphysics
        "xmeldpth": const.XMELDPTH,
        "xkcrit": const.XKCRIT,
        "xcdtmax": const.XCDTMAX,

        # Closure
        "xdtadj": const.XDTADJ,
        "xdcape": const.XDCAPE,

        # Reference
        "xa25": const.XA25,
        "xcrad": const.XCRAD,

        # Thresholds
        "rtmin": const.RTMIN,
        "ptmin": const.PTMIN,
    }
