"""
Compute LCL properties from adjusted environment.

Determines thermodynamic properties at the new lifting condensation level (LCL)
for use in closure calculations.

Translated from: PHYEX-IAL_CY50T1/conv/convect_closure_thrvlcl.F90
"""

import jax.numpy as jnp
from jax import Array, lax
from typing import Tuple, NamedTuple

from .constants import PHYS_CONSTANTS
from .satmixratio import convect_satmixratio


class ClosureLCLOutputs(NamedTuple):
    """Output structure for convect_closure_thrvlcl."""
    pthlcl: Array  # Theta at LCL (K)
    prvlcl: Array  # Vapor mixing ratio at LCL (kg/kg)
    pzlcl: Array  # Height at LCL (m)
    ptlcl: Array  # Temperature at LCL (K)
    ptelcl: Array  # Environment temperature at LCL (K)
    klcl: Array  # Vertical index of LCL (int)


def convect_closure_thrvlcl(
    ppres: Array,
    pth: Array,
    prv: Array,
    pz: Array,
    owork1: Array,
    kdpl: Array,
    kpbl: Array,
    jcvexb: int = 0,
    jcvext: int = 0,
) -> ClosureLCLOutputs:
    """
    Determine thermodynamic properties at new LCL for closure.

    This routine computes LCL properties from an already-adjusted environment
    by constructing a mixed layer from DPL to PBL, then determining LCL
    properties using the Bolton (1980) empirical formula.

    The routine performs the following steps:
    1. Construct a pressure-weighted mixed layer between DPL and PBL
    2. Compute LCL temperature and pressure using Bolton (1980) formula
    3. Apply corrections to be consistent with MNH saturation formula
    4. Handle oversaturated cases
    5. Determine LCL level index and interpolated height

    Parameters
    ----------
    ppres : Array
        Pressure (Pa), shape (nit, nkt)
    pth : Array
        Potential temperature theta (K), shape (nit, nkt)
    prv : Array
        Water vapor mixing ratio (kg/kg), shape (nit, nkt)
    pz : Array
        Height of grid points (m), shape (nit, nkt)
    owork1 : Array
        Logical mask for active grid points (bool), shape (nit,)
    kdpl : Array
        Vertical index of departure level (DPL), shape (nit,)
    kpbl : Array
        Vertical index of PBL top, shape (nit,)
    jcvexb : int, optional
        Extra vertical levels at bottom (default 0)
    jcvext : int, optional
        Extra vertical levels at top (default 0)

    Returns
    -------
    ClosureLCLOutputs
        Named tuple containing:
        - pthlcl: Theta at LCL (K), shape (nit,)
        - prvlcl: Vapor mixing ratio at LCL (kg/kg), shape (nit,)
        - pzlcl: Height at LCL (m), shape (nit,)
        - ptlcl: Temperature at LCL (K), shape (nit,)
        - ptelcl: Environment temperature at LCL (K), shape (nit,)
        - klcl: Vertical index of LCL, shape (nit,)

    Notes
    -----
    The routine uses Bolton (1980) empirical formula to determine LCL:

    First, the dewpoint temperature is computed:

    .. math::
        T_d = \\frac{4780.8 - 32.19 \\cdot \\ln(e/613.3)}{17.502 - \\ln(e/613.3)}

    Then the adiabatic saturation temperature (LCL temperature):

    .. math::
        T_{LCL} = T_d - (0.212 + 1.571 \\times 10^{-3}(T_d - T_t) -
                         4.36 \\times 10^{-4}(T_{mix} - T_t))(T_{mix} - T_d)

    The LCL pressure is then computed from the Poisson equation:

    .. math::
        p_{LCL} = p_{00} \\left(\\frac{T_{LCL}}{\\theta_{LCL}}\\right)^{c_{pd}/R_d}

    Corrections are applied to ensure consistency with the MNH saturation
    formula, and oversaturated cases are handled by setting LCL properties
    to mixed layer values.

    References
    ----------
    - Bolton (1980), Mon. Wea. Rev., Vol. 108, 1046-1053
    - Fritsch and Chappell (1980), J. Atm. Sci., Vol. 37, 1722-1761
    - PHYEX documentation Book 2
    """

    # Constants
    cst = PHYS_CONSTANTS
    zeps = cst.rd / cst.rv  # ~0.622
    zcpord = cst.cpd / cst.rd  # ~3.5
    zrdocp = cst.rd / cst.cpd  # ~0.286

    # Array dimensions
    nit, nkt = ppres.shape
    ikb = jcvexb
    ike = nkt - jcvext - 1

    # Initialize output arrays
    pthlcl = jnp.full(nit, 300.0)
    prvlcl = jnp.zeros(nit)
    pzlcl = pz[:, ikb]
    ptlcl = jnp.full(nit, 300.0)
    ptelcl = jnp.full(nit, 300.0)
    klcl = jnp.full(nit, ikb + 1, dtype=jnp.int32)

    # Working arrays
    zdpthmix = jnp.zeros(nit)
    zpresmix = jnp.zeros(nit)
    ztmix = jnp.full(nit, 230.0)
    zplcl = jnp.full(nit, 1e4)

    # ===== 1. Construct mixed layer between DPL and PBL =====
    # Loop over vertical levels from ikb+1 to ike
    def accumulate_mixed_layer(carry, jk):
        """Accumulate pressure-weighted properties in mixed layer."""
        zdpthmix_a, zpresmix_a, pthlcl_a, prvlcl_a = carry

        # Index for next level
        jkm = jk + 1

        # Clip indices to avoid out-of-bounds
        jk_clip = jnp.clip(jk, 0, nkt - 1)
        jkm_clip = jnp.clip(jkm, 0, nkt - 1)

        # Create mask for points in mixed layer (kdpl <= jk <= kpbl)
        idx_i = jnp.arange(nit)
        in_layer = (jk >= kdpl) & (jk <= kpbl)

        # Compute pressure difference
        zwork1 = ppres[idx_i, jk_clip] - ppres[idx_i, jkm_clip]

        # Accumulate weighted sums
        zdpthmix_new = jnp.where(in_layer, zdpthmix_a + zwork1, zdpthmix_a)
        zpresmix_new = jnp.where(in_layer, zpresmix_a + ppres[idx_i, jk_clip] * zwork1, zpresmix_a)
        pthlcl_new = jnp.where(in_layer, pthlcl_a + pth[idx_i, jk_clip] * zwork1, pthlcl_a)
        prvlcl_new = jnp.where(in_layer, prvlcl_a + prv[idx_i, jk_clip] * zwork1, prvlcl_a)

        return (zdpthmix_new, zpresmix_new, pthlcl_new, prvlcl_new), None

    # Scan over vertical levels
    (zdpthmix, zpresmix, pthlcl, prvlcl), _ = lax.scan(
        accumulate_mixed_layer,
        (zdpthmix, zpresmix, pthlcl, prvlcl),
        jnp.arange(ikb + 1, ike)
    )

    # Normalize by total pressure depth (only where owork1 is True)
    zdpthmix_safe = jnp.where(zdpthmix > 0, zdpthmix, 1.0)
    zpresmix = jnp.where(owork1, zpresmix / zdpthmix_safe, zpresmix)
    pthlcl = jnp.where(owork1, pthlcl / zdpthmix_safe, pthlcl)
    prvlcl = jnp.where(owork1, prvlcl / zdpthmix_safe, prvlcl)

    # ===== 2. Compute LCL using Bolton (1980) empirical formula =====

    # Mixed layer temperature from potential temperature
    ztmix = pthlcl * (zpresmix / cst.p00) ** zrdocp

    # Vapor pressure in mixed layer
    zevmix = prvlcl * zpresmix / (prvlcl + zeps)
    zevmix = jnp.maximum(1e-8, zevmix)

    # Dewpoint temperature
    zwork1 = jnp.log(zevmix / 613.3)
    zwork1 = (4780.8 - 32.19 * zwork1) / (17.502 - zwork1)

    # Adiabatic saturation temperature (LCL temperature)
    # T_lcl = T_d - correction * (T_mix - T_d)
    correction = 0.212 + 1.571e-3 * (zwork1 - cst.tt) - 4.36e-4 * (ztmix - cst.tt)
    ptlcl = zwork1 - correction * (ztmix - zwork1)
    ptlcl = jnp.minimum(ptlcl, ztmix)

    # LCL pressure from Poisson equation
    zplcl = cst.p00 * (ptlcl / pthlcl) ** zcpord

    # Bound pressure to avoid overflow in later calculations
    zplcl = jnp.clip(zplcl, 10.0, 2e5)

    # ===== 3. Correct TLCL for consistency with MNH saturation formula =====

    # Get saturation properties at initial LCL
    zwork1_sat, zlv, zls, zcph = convect_satmixratio(zplcl, ptlcl)

    # Compute derivative of saturation mixing ratio wrt temperature
    # dr_sat/dT = r_sat/T * (betaw/T - gamw)
    zwork2 = zwork1_sat / ptlcl * (cst.betaw / ptlcl - cst.gamw)

    # Correction term: (r_sat - r_mix) / (1 + Lv/Cph * dr_sat/dT)
    zwork2 = (zwork1_sat - prvlcl) / (1.0 + zlv / zcph * zwork2)

    # Apply temperature correction: T_lcl = T_lcl - Lv/Cph * correction
    ptlcl = jnp.where(owork1, ptlcl - zlv / zcph * zwork2, ptlcl)

    # ===== 4. Handle oversaturated cases =====

    # Check if mixed layer is oversaturated
    zwork1_mix, zlv_mix, zls_mix, zcph_mix = convect_satmixratio(zpresmix, ztmix)

    # Oversaturation condition
    is_oversat = owork1 & (prvlcl > zwork1_mix)

    # For oversaturated points, adjust to saturation
    zwork2_mix = zwork1_mix / ztmix * (cst.betaw / ztmix - cst.gamw)
    zwork2_mix = (zwork1_mix - prvlcl) / (1.0 + zlv_mix / zcph_mix * zwork2_mix)

    # Update LCL properties for oversaturated cases
    ptlcl = jnp.where(is_oversat, ztmix + zlv_mix / zcph_mix * zwork2_mix, ptlcl)
    prvlcl = jnp.where(is_oversat, prvlcl - zwork2_mix, prvlcl)
    zplcl = jnp.where(is_oversat, zpresmix, zplcl)
    pthlcl = jnp.where(is_oversat, ptlcl * (cst.p00 / zplcl) ** zrdocp, pthlcl)

    # ===== 5. Determine vertical index at LCL =====

    # Find level where zplcl <= ppres (LCL is between klcl-1 and klcl)
    def find_lcl_level(carry, jk):
        """Find vertical level at LCL."""
        klcl_a = carry

        jk_clip = jnp.clip(jk, 0, nkt - 1)
        idx_i = jnp.arange(nit)

        # Update klcl where zplcl <= ppres and point is active
        mask = (zplcl <= ppres[idx_i, jk_clip]) & owork1
        klcl_new = jnp.where(mask, jk + 1, klcl_a)
        pzlcl_new = jnp.where(mask, pz[idx_i, jk + 1], klcl_a)

        return klcl_new, pzlcl_new

    # Initial value for pzlcl in the loop
    pzlcl_temp = pzlcl.copy()

    # Scan from ikb to ike-1
    klcl, pzlcl_scan = lax.scan(
        find_lcl_level,
        klcl,
        jnp.arange(ikb, ike)
    )

    # Update pzlcl with the last valid value from scan
    # We need to extract the final pzlcl from the scan outputs
    # Since scan returns pzlcl at each step, we take the last one
    pzlcl = pzlcl_scan[-1] if pzlcl_scan.size > 0 else pzlcl

    # ===== 6. Estimate precise height and environment temperature at LCL =====

    # Safe indexing
    idx_i = jnp.arange(nit)
    jkm = jnp.clip(klcl - 1, ikb, nkt - 2)
    klcl_clip = jnp.clip(klcl, ikb + 1, nkt - 1)

    # Gather values at klcl-1 and klcl for each grid point
    ppres_jkm = ppres[idx_i, jkm]
    ppres_klcl = ppres[idx_i, klcl_clip]
    pth_jkm = pth[idx_i, jkm]
    pth_klcl = pth[idx_i, klcl_clip]
    pz_jkm = pz[idx_i, jkm]
    pz_klcl = pz[idx_i, klcl_clip]

    # Logarithmic interpolation factor
    # zdp = log(p_lcl/p[k-1]) / log(p[k]/p[k-1])
    zdp = jnp.log(zplcl / (ppres_jkm + 1e-20)) / \
          jnp.log(ppres_klcl / (ppres_jkm + 1e-20) + 1e-20)
    zdp = jnp.clip(zdp, 0.0, 1.0)

    # Interpolate temperature to LCL
    # Convert theta to temperature at each level
    zwork1 = pth_klcl * (ppres_klcl / cst.p00) ** zrdocp
    zwork2 = pth_jkm * (ppres_jkm / cst.p00) ** zrdocp
    zwork1 = zwork2 + (zwork1 - zwork2) * zdp

    # Interpolate height to LCL
    zwork2 = pz_jkm + (pz_klcl - pz_jkm) * zdp

    # Update outputs where owork1 is True
    ptelcl = jnp.where(owork1, zwork1, ptelcl)
    pzlcl = jnp.where(owork1, zwork2, pzlcl)

    return ClosureLCLOutputs(
        pthlcl=pthlcl,
        prvlcl=prvlcl,
        pzlcl=pzlcl,
        ptlcl=ptlcl,
        ptelcl=ptelcl,
        klcl=klcl,
    )
