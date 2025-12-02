"""Ice3 Components Module

This module provides high-level components for ICE3/ICE4 microphysics schemes.

Available components:
- IceAdjustModular: GT4Py-based modular ice adjustment
- IceAdjustModularDaCe: DaCe-based modular ice adjustment (GPU-accelerated)
- Ice4TendenciesDaCe: DaCe-based ICE4 tendency calculations (GPU-accelerated)
"""

from .ice_adjust_modular import IceAdjustModular
from .ice_adjust_modular_dace import IceAdjustModularDaCe
from .ice4_tendencies_dace import Ice4TendenciesDaCe

__all__ = [
    "IceAdjustModular",
    "IceAdjustModularDaCe",
    "Ice4TendenciesDaCe",
]
