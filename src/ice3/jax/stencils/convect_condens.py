"""
Convective condensation calculations.

Compute temperature, cloud water, and ice content from enthalpy and total water.
Translated from: PHYEX-IAL_CY50T1/conv/convect_condens.F90
"""

import jax.numpy as jnp
from jax import Array
from typing import Tuple

from .constants import PHYS_CONSTANTS


def convect_condens(
    ice_flag: float,
    eps0: float,
    pres: Array,
    thl: Array,
    rw: Array,
    rco: Array,
    rio: Array,
    z: Array,
    tfrz1: float = 268.15,
    tfrz2: float = 248.15,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:
    """
    Compute temperature and cloud condensate from enthalpy and total water.

    This function iteratively extracts cloud condensate (liquid and ice)
    from enthalpy and total water mixing ratio. It uses a 6-iteration
    Newton-Raphson-like scheme with damping for stability.

    Parameters
    ----------
    ice_flag : float
        Flag for ice (1.0 = yes, 0.0 = no ice)
    eps0 : float
        Ratio rd/rv (typically 0.62198)
    pres : Array
        Pressure (Pa), shape (...,)
    thl : Array
        Enthalpy (J/kg), shape (...,)
    rw : Array
        Total water mixing ratio (kg/kg), shape (...,)
    rco : Array
        Cloud water estimate from lower level (kg/kg), shape (...,)
    rio : Array
        Cloud ice estimate from lower level (kg/kg), shape (...,)
    z : Array
        Level height (m), shape (...,)
    tfrz1 : float, optional
        Begin of freezing interval (K), default 268.15
    tfrz2 : float, optional
        End of freezing interval (K), default 248.15

    Returns
    -------
    t : Array
        Temperature (K)
    ew : Array
        Water saturation mixing ratio (kg/kg)
    rc : Array
        Cloud water mixing ratio (kg/kg)
    ri : Array
        Cloud ice mixing ratio (kg/kg)
    lv : Array
        Latent heat of vaporization (J/kg)
    ls : Array
        Latent heat of sublimation (J/kg)
    cph : Array
        Specific heat (J/kg/K)

    Notes
    -----
    The iteration scheme solves for temperature such that:

    .. math::
        T = (\\theta_l + r_c L_v + r_i L_s - g z (1 + r_w)) / C_{ph}

    where :math:`r_c` and :math:`r_i` depend on T through saturation.

    Freezing fraction is computed as:

    .. math::
        f_{ice} = \\max(0, \\min(1, (T_{frz1} - T) / (T_{frz1} - T_{frz2}))) \\times ice\\_flag

    The iteration uses 0.4 damping for stability.
    """

    # Constants
    g = PHYS_CONSTANTS.g
    cpd = PHYS_CONSTANTS.cpd
    cpv = PHYS_CONSTANTS.cpv
    cl = PHYS_CONSTANTS.cl
    ci = PHYS_CONSTANTS.ci
    tt = PHYS_CONSTANTS.tt
    xlvtt = PHYS_CONSTANTS.xlvtt
    xlstt = PHYS_CONSTANTS.xlstt
    alpw = PHYS_CONSTANTS.alpw
    betaw = PHYS_CONSTANTS.betaw
    gamw = PHYS_CONSTANTS.gamw
    alpi = PHYS_CONSTANTS.alpi
    betai = PHYS_CONSTANTS.betai
    gami = PHYS_CONSTANTS.gami

    # Initialize specific heat (moisture dependent)
    # Note: definition differs from CONVECT_SATMIXRATIO
    cph = cpd + cpv * rw

    # Gravitational potential energy term
    work1 = (1.0 + rw) * g * z

    # Initial temperature estimate from enthalpy
    # Using condensate estimates from lower level
    t = (thl + rco * xlvtt + rio * xlstt - work1) / cph
    t = jnp.clip(t, 180.0, 330.0)  # Overflow bounds

    # Iterative refinement (6 iterations)
    def iter_step(t_prev, _):
        """Single iteration step."""
        # Saturation mixing ratio over liquid water
        ew = jnp.exp(alpw - betaw / t_prev - gamw * jnp.log(t_prev))
        ew = eps0 * ew / (pres - ew)

        # Saturation mixing ratio over ice
        ei = jnp.exp(alpi - betai / t_prev - gami * jnp.log(t_prev))
        ei = eps0 * ei / (pres - ei)

        # Freezing fraction (linear transition between tfrz1 and tfrz2)
        fice = jnp.clip((tfrz1 - t_prev) / (tfrz1 - tfrz2), 0.0, 1.0) * ice_flag

        # Mixed saturation (liquid + ice weighted by freezing fraction)
        esat_mix = (1.0 - fice) * ew + fice * ei

        # Extract condensate
        rc = jnp.maximum(0.0, (1.0 - fice) * (rw - esat_mix))
        ri = jnp.maximum(0.0, fice * (rw - esat_mix))

        # Latent heats (temperature dependent)
        lv = xlvtt + (cpv - cl) * (t_prev - tt)
        ls = xlstt + (cpv - ci) * (t_prev - tt)

        # Solve for temperature from enthalpy balance
        t_new = (thl + rc * lv + ri * ls - work1) / cph

        # Damped update for convergence (factor 0.4)
        t_updated = t_prev + (t_new - t_prev) * 0.4
        t_updated = jnp.clip(t_updated, 175.0, 330.0)

        return t_updated, None

    # Run iterations using lax.scan
    from jax import lax
    t_final, _ = lax.scan(iter_step, t, jnp.arange(6))

    # Compute final outputs with converged temperature
    ew = jnp.exp(alpw - betaw / t_final - gamw * jnp.log(t_final))
    ew = eps0 * ew / (pres - ew)

    ei = jnp.exp(alpi - betai / t_final - gami * jnp.log(t_final))
    ei = eps0 * ei / (pres - ei)

    fice = jnp.clip((tfrz1 - t_final) / (tfrz1 - tfrz2), 0.0, 1.0) * ice_flag
    esat_mix = (1.0 - fice) * ew + fice * ei

    rc = jnp.maximum(0.0, (1.0 - fice) * (rw - esat_mix))
    ri = jnp.maximum(0.0, fice * (rw - esat_mix))

    lv = xlvtt + (cpv - cl) * (t_final - tt)
    ls = xlstt + (cpv - ci) * (t_final - tt)

    return t_final, ew, rc, ri, lv, ls, cph
