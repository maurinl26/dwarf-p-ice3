"""
Convection closure adjustment for shallow convection.

Adjusts mass flux and entrainment/detrainment rates using closure factor.
Translated from: PHYEX-IAL_CY50T1/conv/convect_closure_adjust_shal.F90
"""

import jax.numpy as jnp
from jax import Array
from typing import Tuple


def convect_closure_adjust_shal(
    padj: Array,
    zumf: Array,
    zuer: Array,
    zudr: Array,
) -> Tuple[Array, Array, Array]:
    """
    Adjust mass flux using closure adjustment factor.

    This function applies the mass flux adjustment factor computed in the
    closure scheme to scale the updraft mass flux, entrainment, and
    detrainment rates.

    Parameters
    ----------
    padj : Array
        Mass adjustment factor (dimensionless), shape (nit,)
        Typically in range [0, 2], computed from CAPE closure
    zumf : Array
        Initial updraft mass flux (kg/s m²), shape (nit, nkt)
    zuer : Array
        Initial updraft entrainment rate (kg/s m²), shape (nit, nkt)
    zudr : Array
        Initial updraft detrainment rate (kg/s m²), shape (nit, nkt)

    Returns
    -------
    pumf : Array
        Adjusted updraft mass flux (kg/s m²), shape (nit, nkt)
    puer : Array
        Adjusted updraft entrainment rate (kg/s m²), shape (nit, nkt)
    pudr : Array
        Adjusted updraft detrainment rate (kg/s m²), shape (nit, nkt)

    Notes
    -----
    The adjustment is simply a multiplication:

    .. math::
        M_{adj} = M_{init} \\times f_{adj}

    where :math:`f_{adj}` is the closure adjustment factor computed to
    remove a specified fraction of CAPE (typically 80-90%).

    This routine is called after the closure scheme computes the adjustment
    factor needed to stabilize the atmosphere to the desired degree.

    The adjustment preserves the vertical profile shape while scaling the
    overall intensity of convection.
    """

    # Expand padj to broadcast over vertical dimension
    # padj is (nit,), arrays are (nit, nkt)
    padj_3d = padj[:, jnp.newaxis]

    # Apply adjustment factor to all fields
    pumf = zumf * padj_3d
    puer = zuer * padj_3d
    pudr = zudr * padj_3d

    return pumf, puer, pudr
