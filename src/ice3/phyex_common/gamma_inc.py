# -*- coding: utf-8 -*-
from math import gamma

from numpy import exp, log, finfo

import numpy as np
from scipy.integrate import quad

def gamma_function(x):
    # Define the integrand t^(x-1) * e^(-t)
    integrand = lambda t: t**(x-1) * np.exp(-t)
    # Compute the integral from 0 to infinity
    result, error = quad(integrand, 0, np.inf)
    return result


def generalized_incomplete_gamma(a,x) -> float:
    """Compute the generalized incomplete gamma function using quadrature from scipy (quad)
    
    The purpose of this function is to compute the generalized
    incomplete Gamma function of its argument.

                              /X
                        1     |
     GAMMA_INC(A,X)= -------- | Z**(A-1) EXP(-Z) dZ
                     GAMMA(A) |
                              /0

    Args:
        a (float): factor for gamma function
        x (float): upper bound for the incomplete gamma function

    Returns:
        float: incomplete gamma function
    """
    integrand = lambda t: t**(a-1) * np.exp(-t)
    result = quad(integrand, 0, x)[0] / quad(integrand, 0, np.inf)[0]
    return result


def gamma_inc(a: float, x: float) -> float:
    """Compute the genernalized gamma function 

    The purpose of this function is to compute the generalized
    incomplete Gamma function of its argument.

                              /X
                        1     |
     GAMMA_INC(A,X)= -------- | Z**(A-1) EXP(-Z) dZ
                     GAMMA(A) |
                              /0

     Reference : Press, Teukolsky, Vetterling and Flannery: Numerical Recipes, 209-213

     Args:
         a (float): _description_
         x (float): _description_
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
