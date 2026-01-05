"""
Convective mixing function for entrainment/detrainment.

Determine entrainment and detrainment rates from distribution function.
Translated from: PHYEX-IAL_CY50T1/conv/convect_mixing_funct.F90
"""

import jax.numpy as jnp
from jax import Array
from typing import Tuple


def convect_mixing_funct(
    mixc: Array,
    kmf: int = 1,
) -> Tuple[Array, Array]:
    """
    Compute normalized entrainment and detrainment rates.

    This function determines the entrainment and detrainment rates by
    evaluating the area under a distribution function. The integration
    interval is limited by the critical mixed fraction.

    Parameters
    ----------
    mixc : Array
        Critical mixed fraction (dimensionless), shape (...,)
        Typically in range [0, 1]
    kmf : int, optional
        Switch for distribution function:
        - 1: Gaussian distribution (default, implemented)
        - 2: Triangular distribution (not yet implemented)

    Returns
    -------
    er : Array
        Normalized entrainment rate (dimensionless)
    dr : Array
        Normalized detrainment rate (dimensionless)

    Notes
    -----
    The Gaussian distribution function uses:

    .. math::
        f(\\chi) = \\frac{1}{\\sqrt{2\\pi}\\sigma} \\exp\\left(-\\frac{(\\chi - 0.5)^2}{2\\sigma^2}\\right)

    where :math:`\\sigma = 1/6` and :math:`\\chi` is the mixing fraction.

    The entrainment and detrainment rates are computed by integrating
    different regions under this distribution, weighted by mixing fraction.

    Uses error function approximation from Abramowitz & Stegun (1968).

    References
    ----------
    - Abramowitz and Stegun (1968), Handbook of Mathematical Functions
    - Bechtold et al. convection scheme documentation
    """

    if kmf != 1:
        raise NotImplementedError("Only Gaussian distribution (kmf=1) is implemented")

    # Constants for Gaussian distribution
    sigma = 0.166666667  # Standard deviation (1/6)
    fe = 4.931813949     # Integral normalization
    sqrtp = 2.506628     # sqrt(2*pi)
    p = 0.33267          # Error function approximation constant

    # Error function polynomial coefficients
    a1 = 0.4361836
    a2 = -0.1201676
    a3 = 0.9372980
    t1 = 0.500498

    e45 = 0.01111  # Constant for boundary correction

    # Transform mixing fraction to standardized variable
    # zx = (mixc - 0.5) / sigma = 6 * mixc - 3
    zx = 6.0 * mixc - 3.0

    # Error function approximation
    # Using Abramowitz & Stegun polynomial approximation
    zw1 = 1.0 / (1.0 + p * jnp.abs(zx))
    zy = jnp.exp(-0.5 * zx * zx)  # Gaussian kernel
    zw2 = a1 * zw1 + a2 * zw1**2 + a3 * zw1**3

    # Constant term for normalization
    zw11 = a1 * t1 + a2 * t1**2 + a3 * t1**3

    # Compute entrainment and detrainment based on sign of zx
    # These formulas integrate the distribution function weighted by
    # mixing fraction over different regions

    # For zx >= 0 (mixc >= 0.5)
    er_pos = (
        sigma * (0.5 * (sqrtp - e45 * zw11 - zy * zw2) + sigma * (e45 - zy))
        - 0.5 * e45 * mixc * mixc
    )
    dr_pos = (
        sigma * (0.5 * (zy * zw2 - e45 * zw11) + sigma * (e45 - zy))
        - e45 * (0.5 + 0.5 * mixc * mixc - mixc)
    )

    # For zx < 0 (mixc < 0.5)
    er_neg = (
        sigma * (0.5 * (zy * zw2 - e45 * zw11) + sigma * (e45 - zy))
        - 0.5 * e45 * mixc * mixc
    )
    dr_neg = (
        sigma * (0.5 * (sqrtp - e45 * zw11 - zy * zw2) + sigma * (e45 - zy))
        - e45 * (0.5 + 0.5 * mixc * mixc - mixc)
    )

    # Select based on sign of zx
    er = jnp.where(zx >= 0.0, er_pos, er_neg)
    dr = jnp.where(zx >= 0.0, dr_pos, dr_neg)

    # Apply normalization factor
    er = er * fe
    dr = dr * fe

    return er, dr
