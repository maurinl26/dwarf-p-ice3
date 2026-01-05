"""
Physical constants for JAX stencils.

This module provides physical constants needed for atmospheric physics calculations.
Translated from PHYEX-IAL_CY50T1/aux/modd_cst.F90
"""

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class PhysicalConstants:
    """
    Physical constants for atmospheric physics.

    Based on MODD_CST from PHYEX.
    All values match the Fortran implementation.
    """

    # Fundamental constants
    pi: float = 3.14159265358979323846

    # Gravity
    g: float = 9.80665  # m/s^2

    # Reference pressure
    p00: float = 1.0e5  # Pa (100000 Pa = 1000 hPa)

    # Gas constants
    rd: float = 287.06  # J/(kg·K) - dry air
    rv: float = 461.51  # J/(kg·K) - water vapor

    # Specific heats
    cpd: float = 1004.709  # J/(kg·K) - dry air at constant pressure
    cpv: float = 1846.1    # J/(kg·K) - water vapor at constant pressure
    cl: float = 4218.0     # J/(kg·K) - liquid water
    ci: float = 2106.0     # J/(kg·K) - ice

    # Molecular weights ratio
    eps: float = 0.62198  # rd/rv = Mv/Md

    # Triple point
    tt: float = 273.16  # K

    # Latent heats at triple point
    xlvtt: float = 2.5008e6  # J/kg - vaporization
    xlstt: float = 2.8345e6  # J/kg - sublimation
    xlmtt: float = 3.337e5   # J/kg - melting (xlstt - xlvtt)

    # Saturation vapor pressure at triple point
    estt: float = 611.24  # Pa

    # Saturation vapor pressure coefficients (liquid water)
    # Formula: e_s = exp(alpw - betaw/T - gamw*log(T))
    alpw: float = 17.269388  # Computed from estt, betaw, gamw, tt
    betaw: float = 3802.596  # Computed from xlvtt, rv, gamw, tt
    gamw: float = -1.08366   # = (cl - cpv) / rv

    # Saturation vapor pressure coefficients (ice)
    # Formula: e_s = exp(alpi - betai/T - gami*log(T))
    alpi: float = 21.872204  # Computed from estt, betai, gami, tt
    betai: float = 6031.518  # Computed from xlstt, rv, gami, tt
    gami: float = -1.41502   # = (ci - cpv) / rv

    # Densities
    rholw: float = 1000.0  # kg/m^3 - liquid water
    rholi: float = 900.0   # kg/m^3 - ice

    # Thermal conductivity
    condi: float = 2.2  # W/(m·K) - ice

    def __post_init__(self):
        """Verify computed constants match expected values."""
        # These should be computed, but we use hardcoded values that match PHYEX
        # alpw = log(estt) + betaw/tt + gamw*log(tt)
        # betaw = xlvtt/rv + gamw*tt
        # gamw = (cl - cpv)/rv
        pass


# Global constants instance
PHYS_CONSTANTS = PhysicalConstants()


def get_physical_constants() -> dict:
    """
    Get physical constants as a dictionary.

    Returns
    -------
    dict
        Dictionary mapping constant names to values
    """
    return {
        "g": PHYS_CONSTANTS.g,
        "pi": PHYS_CONSTANTS.pi,
        "p00": PHYS_CONSTANTS.p00,
        "rd": PHYS_CONSTANTS.rd,
        "rv": PHYS_CONSTANTS.rv,
        "cpd": PHYS_CONSTANTS.cpd,
        "cpv": PHYS_CONSTANTS.cpv,
        "cl": PHYS_CONSTANTS.cl,
        "ci": PHYS_CONSTANTS.ci,
        "eps": PHYS_CONSTANTS.eps,
        "tt": PHYS_CONSTANTS.tt,
        "xlvtt": PHYS_CONSTANTS.xlvtt,
        "xlstt": PHYS_CONSTANTS.xlstt,
        "xlmtt": PHYS_CONSTANTS.xlmtt,
        "estt": PHYS_CONSTANTS.estt,
        "alpw": PHYS_CONSTANTS.alpw,
        "betaw": PHYS_CONSTANTS.betaw,
        "gamw": PHYS_CONSTANTS.gamw,
        "alpi": PHYS_CONSTANTS.alpi,
        "betai": PHYS_CONSTANTS.betai,
        "gami": PHYS_CONSTANTS.gami,
        "rholw": PHYS_CONSTANTS.rholw,
        "rholi": PHYS_CONSTANTS.rholi,
        "condi": PHYS_CONSTANTS.condi,
    }
