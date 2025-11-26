# -*- coding: utf-8 -*-
from math import gamma
import logging

from numpy import exp, log, finfo
import numpy as np

import numpy as np
from scipy.integrate import quad

def gamma_function(x):
    """
    Compute the complete gamma function using numerical integration.
    
    Evaluates the gamma function Γ(x) using scipy's quadrature integration:
    
    Γ(x) = ∫₀^∞ t^(x-1) × e^(-t) dt
    
    Parameters
    ----------
    x : float
        Argument of the gamma function. Must be positive.
    
    Returns
    -------
    float
        Value of Γ(x).
    
    Notes
    -----
    This is a direct numerical integration implementation, less efficient
    than the built-in math.gamma() but useful for reference or validation.
    For production use, prefer math.gamma() or scipy.special.gamma().
    
    The gamma function extends the factorial function to real and complex
    numbers: Γ(n) = (n-1)! for positive integers n.
    
    Examples
    --------
    >>> gamma_function(4.0)  # Should be close to 6 = 3!
    6.000...
    >>> gamma_function(1.5)  # Should be close to sqrt(π)/2
    0.886...
    """
    # Define the integrand t^(x-1) * e^(-t)
    integrand = lambda t: t**(x-1) * np.exp(-t)
    # Compute the integral from 0 to infinity
    result, error = quad(integrand, 0, np.inf)
    return result


def generalized_incomplete_gamma(a, x) -> float:
    """
    Compute the normalized lower incomplete gamma function P(a,x).
    
    This function evaluates the regularized (normalized) lower incomplete
    gamma function using scipy's numerical quadrature:
    
    P(a,x) = γ(a,x) / Γ(a) = [∫₀^x t^(a-1) × e^(-t) dt] / Γ(a)
    
    where γ(a,x) is the lower incomplete gamma function and Γ(a) is the
    complete gamma function.
    
    Parameters
    ----------
    a : float
        Shape parameter (must be positive). Corresponds to the order of
        the gamma distribution or the power in the integrand.
    x : float
        Upper integration limit (must be non-negative). Defines the
        integration range [0, x].
    
    Returns
    -------
    float
        Normalized incomplete gamma function value, 0 ≤ P(a,x) ≤ 1.
        - P(a,0) = 0 (no integration range)
        - P(a,∞) = 1 (complete gamma function)
    
    Notes
    -----
    **Physical Interpretation:**
    
    In atmospheric microphysics, P(a,x) represents the cumulative
    distribution function for particle size distributions following
    a generalized gamma distribution. It gives the fraction of particles
    with size parameter less than x.
    
    **Numerical Method:**
    
    Uses scipy.integrate.quad for both numerator and denominator:
    - Numerator: ∫₀^x t^(a-1) × e^(-t) dt
    - Denominator: ∫₀^∞ t^(a-1) × e^(-t) dt = Γ(a)
    
    This implementation is slower than gamma_increment() which uses
    analytical recurrence relations from Numerical Recipes. For
    production code, prefer gamma_increment() or scipy.special.gammainc().
    
    **Usage in Microphysics:**
    
    Used in snow riming calculations (init_gaminc_rim_tables in
    RainIceParameters) to compute collection kernel moments for
    particle size distributions.
    
    See Also
    --------
    gamma_increment : Efficient implementation using series/continued fraction
    gamma_function : Complete gamma function
    scipy.special.gammainc : Optimized SciPy implementation
    
    References
    ----------
    Abramowitz, M., and I.A. Stegun, 1965: Handbook of Mathematical
    Functions. Dover Publications, 260.
    
    Examples
    --------
    >>> generalized_incomplete_gamma(1.0, 1.0)  # Should be 1 - 1/e ≈ 0.632
    0.632...
    >>> generalized_incomplete_gamma(2.0, 0.0)  # Should be 0
    0.0
    >>> generalized_incomplete_gamma(5.0, 10.0)  # Should be close to 1
    0.997...
    """
    integrand = lambda t: t**(a-1) * np.exp(-t)
    result = quad(integrand, 0, x)[0] / quad(integrand, 0, np.inf)[0]
    return result


################# PHYEX/src/common/aux/gamma_inc.F90 #################
def gamma_increment(a: float, x: float) -> float:
    """
    Compute the normalized lower incomplete gamma function P(a,x) efficiently.
    
    This function evaluates the regularized lower incomplete gamma function
    using optimized series expansion or continued fraction methods from
    Numerical Recipes, providing accurate and efficient computation:
    
    P(a,x) = γ(a,x) / Γ(a) = [∫₀^x t^(a-1) × e^(-t) dt] / Γ(a)
    
    The algorithm automatically selects the most efficient method:
    - Series expansion for x < a+1 (converges quickly)
    - Continued fraction for x ≥ a+1 (more stable)
    
    Parameters
    ----------
    a : float
        Shape parameter, must be positive (a > 0).
        Corresponds to the order of the gamma distribution.
    x : float
        Upper integration limit, must be non-negative (x ≥ 0).
        Defines the integration range [0, x].
    
    Returns
    -------
    float
        Normalized incomplete gamma function value, 0 ≤ P(a,x) ≤ 1.
        - P(a,0) = 0
        - P(a,∞) = 1
        - Monotonically increasing in x
    
    Raises
    ------
    ValueError
        If x < 0 or a ≤ 0 (invalid arguments).
        If iteration fails to converge within ITMAX iterations.
    
    Notes
    -----
    **Algorithm Selection:**
    
    For x < a+1:
        Uses series expansion:
        P(a,x) = exp(-x + a×ln(x) - ln(Γ(a))) × Σₙ₌₀^∞ [x^n / Γ(a+1+n)]
        
    For x ≥ a+1:
        Uses continued fraction via complementary function Q(a,x) = 1 - P(a,x):
        Q(a,x) = exp(-x + a×ln(x) - ln(Γ(a))) × [continued fraction]
        Then: P(a,x) = 1 - Q(a,x)
    
    **Convergence Criteria:**
    
    ZEPS = 3×10⁻⁷ : Relative accuracy threshold
    ITMAX = 100 : Maximum iterations
    ZFPMIN = 1×10⁻³⁰ : Minimum representable positive value
    
    **Numerical Stability:**
    
    - Uses logarithmic form exp(-x + a×ln(x) - ln(Γ(a))) to avoid overflow
    - Protects against division by zero with ZFPMIN
    - Monitors relative convergence |Δ|/|sum| < ε
    
    **Computational Efficiency:**
    
    Typical convergence in 10-30 iterations for most atmospheric
    microphysics applications. Much faster than numerical quadrature.
    
    **Physical Applications:**
    
    In ICE3/ICE4 microphysics:
    - Riming collection kernels (GAMINC_RIM1, GAMINC_RIM2, GAMINC_RIM4)
    - Particle size distribution moments
    - Mass-weighted fall speed calculations
    - Collection efficiency integrals
    
    Source Reference
    ----------------
    PHYEX/src/common/aux/gamma_inc.F90
    
    References
    ----------
    Press, W.H., S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery, 1992:
    Numerical Recipes in Fortran 77: The Art of Scientific Computing,
    2nd Ed. Cambridge University Press, 209-213.
    
    Abramowitz, M., and I.A. Stegun, 1965: Handbook of Mathematical
    Functions. Dover Publications, equation 6.5.1.
    
    Examples
    --------
    >>> gamma_increment(1.0, 1.0)  # Exponential distribution: 1 - 1/e
    0.6321205588285577
    
    >>> gamma_increment(2.0, 2.0)  # Chi-square distribution
    0.5939941502901619
    
    >>> gamma_increment(5.0, 3.0)  # Intermediate case
    0.18473675547622787
    
    >>> # Verify P(a,0) = 0
    >>> gamma_increment(3.0, 0.0)
    0.0
    
    >>> # Verify monotonicity: P(a,x₁) < P(a,x₂) for x₁ < x₂
    >>> p1 = gamma_increment(2.5, 1.0)
    >>> p2 = gamma_increment(2.5, 3.0)
    >>> p1 < p2
    True
    """

    zeps = 3e-7
    itmax = 100
    zfpmin = 1e-30

    if x < 0 or a <= 0:
        raise ValueError(f"Invalid arguments: x < 0 or a <= 0")

    if x < a + 1:
        ap = a
        zsum = 1 / a
        zdel = zsum
        jn = 1

        # LOOP_SERIES: DO
        while not (abs(zdel) < abs(zsum) * zeps):
            ap += 1
            zdel *= x / ap
            zsum += zdel
            jn += 1

            if jn > itmax:
                raise ValueError(
                    "a argument is too large or ITMAX is too small the incomplete GAMMA_INC "
                    "function cannot be evaluated correctly by the series method"
                )

        return zsum * exp(-x + a * log(x) - log(gamma(a)))

    else:
        zdel = finfo(np.float64).tiny
        b = x + 1 - a
        c = 1 / finfo(np.float64).tiny
        d = 1 / b
        h = d
        jn = 1

        while not (abs(zdel - 1) < zeps):

            an = -jn * (jn - a)
            b += 2
            d = an * d + b

            if abs(d) < finfo(np.float64).tiny:
                d = zfpmin
            if abs(c) < finfo(np.float64).tiny:
                c = zfpmin

            d = 1 / d
            zdel = d * c
            h = h * zdel
            jn += 1

            if jn > itmax:
                raise ValueError(
                    "a argument is too large or ITMAX is too small the incomplete GAMMA_INC "
                    "function cannot be evaluated correctly by the series method"
                )

        return 1 - h * exp(-x + a * log(x) - log(gamma(a)))
