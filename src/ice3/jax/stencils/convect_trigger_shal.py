"""
Shallow convection trigger function.

Determines convective columns and cloudy values of theta, qv at LCL.
Translated from: PHYEX-IAL_CY50T1/conv/convect_trigger_shal.F90
"""

import jax.numpy as jnp
from jax import Array
from jax import lax
from typing import Tuple, NamedTuple, Optional

from .constants import PHYS_CONSTANTS
from .satmixratio import convect_satmixratio


class ConvectionParameters:
    """
    Parameters for shallow convection scheme.

    Based on MODD_CONVPAR_SHAL from PHYEX.
    """

    # Reference grid area
    xa25: float = 625.0e6  # m^2 (25 km x 25 km)

    # Cloud and convection parameters
    xcrad: float = 1500.0  # cloud radius (m)
    xctime_shal: float = 10800.0  # convective adjustment time (s)
    xcdepth: float = 0.0  # minimum necessary cloud depth (m)
    xcdepth_d: float = 3000.0  # maximum allowed cloud thickness (m)

    # Temperature perturbation parameters
    xdtpert: float = 0.2  # temperature perturbation (K)
    xatpert: float = 1.0  # TKE coefficient for perturbation
    xbtpert: float = 0.0  # constant term for perturbation

    # Entrainment
    xentr: float = 0.03  # entrainment constant (m/Pa or m for shallow)

    # Triggering parameters
    xzlcl: float = 3500.0  # max height difference between surface and DPL (m)
    xzpbl: float = 40.0  # minimum mixed layer depth to sustain convection (hPa)
    xwtrig: float = 6.00  # constant in vertical velocity trigger

    # Non-hydrostatic pressure coefficient
    xnhgam: float = 0.33  # = 2 / (1 + gamma)

    # Freezing interval (used in other routines)
    xtfrz1: float = 268.15  # begin of freezing interval (K)
    xtfrz2: float = 248.15  # end of freezing interval (K)

    # Stability factors
    xstabt: float = 0.95  # stability factor for time integration
    xstabc: float = 0.95  # stability factor for CAPE adjustment

    # Vertical velocity at LCL parameters
    xaw: float = 1.0  # coefficient for W(surface)
    xbw: float = 0.4  # constant term (m/s)

    # Smoothing option
    llsmooth: bool = False  # apply smoothing at cloud top


class TriggerOutputs(NamedTuple):
    """Output structure for convect_trigger_shal."""
    otrig: Array  # Trigger mask (bool)
    pthlcl: Array  # Theta at LCL (K)
    ptlcl: Array  # Temperature at LCL (K)
    prvlcl: Array  # Vapor mixing ratio at LCL (kg/kg)
    pwlcl: Array  # Parcel velocity at LCL (m/s)
    pzlcl: Array  # Height at LCL (m)
    pthvelcl: Array  # Environment theta_v at LCL (K)
    klcl: Array  # Vertical index of LCL (int)
    kdpl: Array  # Vertical index of DPL (int)
    kpbl: Array  # Vertical index of PBL top (int)


def convect_trigger_shal(
    ppres: Array,
    pth: Array,
    pthv: Array,
    pthes: Array,
    prv: Array,
    pw: Array,
    pz: Array,
    ptkecls: Array,
    jcvexb: int = 0,
    jcvext: int = 0,
    params: Optional[ConvectionParameters] = None,
) -> TriggerOutputs:
    """
    Determine convective columns and LCL properties for shallow convection.

    This routine identifies grid points where shallow convection should be
    triggered by:
    1. Constructing a mixed layer of at least XZPBL depth
    2. Computing LCL properties using Bolton (1980) formula
    3. Estimating cloud top via CAPE calculation
    4. Checking if cloud depth exceeds XCDEPTH threshold

    Parameters
    ----------
    ppres : Array
        Pressure (Pa), shape (nit, nkt)
    pth : Array
        Potential temperature theta (K), shape (nit, nkt)
    pthv : Array
        Virtual potential temperature theta_v (K), shape (nit, nkt)
    pthes : Array
        Saturation equivalent potential temperature (K), shape (nit, nkt)
    prv : Array
        Water vapor mixing ratio (kg/kg), shape (nit, nkt)
    pw : Array
        Vertical velocity (m/s), shape (nit, nkt)
    pz : Array
        Height of grid points (m), shape (nit, nkt)
    ptkecls : Array
        Turbulent kinetic energy at surface (m^2/s^2), shape (nit,)
    jcvexb : int, optional
        Extra vertical levels at bottom (default 0)
    jcvext : int, optional
        Extra vertical levels at top (default 0)
    params : ConvectionParameters, optional
        Convection parameters (default uses standard values)

    Returns
    -------
    TriggerOutputs
        Named tuple containing:
        - otrig: Trigger mask (bool), shape (nit,)
        - pthlcl: Theta at LCL (K), shape (nit,)
        - ptlcl: Temperature at LCL (K), shape (nit,)
        - prvlcl: Vapor mixing ratio at LCL (kg/kg), shape (nit,)
        - pwlcl: Parcel velocity at LCL (m/s), shape (nit,)
        - pzlcl: Height at LCL (m), shape (nit,)
        - pthvelcl: Environment theta_v at LCL (K), shape (nit,)
        - klcl: Vertical index of LCL, shape (nit,)
        - kdpl: Vertical index of DPL, shape (nit,)
        - kpbl: Vertical index of PBL top, shape (nit,)

    Notes
    -----
    The routine uses Bolton (1980) formula to determine LCL temperature
    and pressure:

    .. math::
        T_{LCL} = \\frac{4780.8 - 32.19 \\cdot \\ln(e/613.3)}{17.502 - \\ln(e/613.3)}

    where e is the vapor pressure.

    Cloud top is estimated using a CAPE-based criterion, where the updraft
    kinetic energy becomes negative.

    The trigger condition requires:
    - Cloud depth >= XCDEPTH (typically 0 m for shallow)
    - CAPE > 10 m^2/s^2
    - Mixed layer center < XZLCL above surface (typically 3500 m)

    References
    ----------
    - Bolton (1980), Mon. Wea. Rev., Vol. 108, 1046-1053
    - Fritsch and Chappell (1980), J. Atm. Sci., Vol. 37, 1722-1761
    - Bechtold et al. shallow convection scheme documentation
    """

    if params is None:
        params = ConvectionParameters()

    # Constants
    cst = PHYS_CONSTANTS
    zeps = cst.rd / cst.rv  # ~0.622
    zepsa = cst.rv / cst.rd  # ~1.608
    zcpord = cst.cpd / cst.rd  # ~3.5
    zrdocp = cst.rd / cst.cpd  # ~0.286

    # Array dimensions
    nit, nkt = ppres.shape
    ikb = jcvexb
    ike = nkt - jcvext - 1
    jt = ike - 2

    # Initialize outputs
    otrig = jnp.zeros(nit, dtype=bool)
    pthlcl = jnp.ones(nit)
    ptlcl = jnp.ones(nit)
    prvlcl = jnp.zeros(nit)
    pwlcl = jnp.zeros(nit)
    pzlcl = pz[:, ikb]
    pthvelcl = jnp.ones(nit)
    klcl = jnp.full(nit, ikb, dtype=jnp.int32)
    kdpl = jnp.full(nit, ikb, dtype=jnp.int32)
    kpbl = jnp.full(nit, ikb, dtype=jnp.int32)

    # Working arrays
    zzdpl = pz[:, ikb].copy()
    gtrig2 = jnp.ones(nit, dtype=bool)

    # Precompute auxiliary arrays for efficiency
    # These are pressure differences and weighted values
    def compute_aux_arrays():
        """Compute auxiliary arrays once."""
        zzzx1 = ppres[:, ikb+1:ike] - ppres[:, ikb+2:ike+1]  # dp between levels
        zzppres = ppres[:, ikb+1:ike] * zzzx1
        zzpth = pth[:, ikb+1:ike] * zzzx1
        zzprv = jnp.maximum(0.0, prv[:, ikb+1:ike]) * zzzx1
        return zzzx1, zzppres, zzpth, zzprv

    zzzx1, zzppres, zzpth, zzprv = compute_aux_arrays()

    # Main loop over departure levels (JKK)
    def loop_over_jkk(carry, jkk):
        """Loop body for each departure level."""
        (otrig_c, pthlcl_c, ptlcl_c, prvlcl_c, pwlcl_c, pzlcl_c,
         pthvelcl_c, klcl_c, kdpl_c, kpbl_c, zzdpl_c, gtrig2_c) = carry

        # Check if we should continue processing
        # Exit when center of mixed layer is > XZLCL above surface
        gwork1 = (zzdpl_c - pz[:, ikb]) < params.xzlcl

        # Update DPL position for this iteration
        zzdpl_new = jnp.where(gwork1, pz[:, jkk], zzdpl_c)
        idpl = jnp.where(gwork1, jkk, kdpl_c)

        # Initialize mixed layer accumulation
        zdpthmix = jnp.zeros(nit)
        zpresmix = jnp.zeros(nit)
        zthlcl = jnp.zeros(nit)
        zrvlcl = jnp.zeros(nit)
        ipbl = kpbl_c.copy()

        # Construct mixed layer of at least XZPBL depth (in pressure)
        # Loop from ikb+1 to ike-1 with conditional based on jkk
        def accumulate_layer(carry, jk):
            """Accumulate properties in mixed layer."""
            zdpthmix_a, zpresmix_a, zthlcl_a, zrvlcl_a, ipbl_a = carry

            # Index into auxiliary arrays (offset by ikb+1)
            jk_aux = jk - (ikb + 1)
            # JAX FIX: Add condition to only process when jk >= jkk
            # This replaces the dynamic range jnp.arange(jkk, ike) with a fixed range + mask
            mask = (gwork1 & (zdpthmix_a < params.xzpbl) & (jk >= jkk) &
                   (jk >= ikb + 1) & (jk < ike))

            # Accumulate weighted values (only when mask is True)
            zdpthmix_new = jnp.where(mask, zdpthmix_a + zzzx1[:, jk_aux], zdpthmix_a)
            zpresmix_new = jnp.where(mask, zpresmix_a + zzppres[:, jk_aux], zpresmix_a)
            zthlcl_new = jnp.where(mask, zthlcl_a + zzpth[:, jk_aux], zthlcl_a)
            zrvlcl_new = jnp.where(mask, zrvlcl_a + zzprv[:, jk_aux], zrvlcl_a)
            ipbl_new = jnp.where(mask, jk, ipbl_a)

            return (zdpthmix_new, zpresmix_new, zthlcl_new, zrvlcl_new, ipbl_new), None

        # JAX FIX: Use fixed range (ikb+1 to ike) instead of dynamic range (jkk to ike)
        # The jk >= jkk condition in the mask handles the dynamic start
        (zdpthmix, zpresmix, zthlcl, zrvlcl, ipbl), _ = lax.scan(
            accumulate_layer,
            (zdpthmix, zpresmix, zthlcl, zrvlcl, ipbl),
            jnp.arange(ikb + 1, ike)  # Fixed range
        )

        # Compute mixed layer mean values
        zdpthmix_safe = jnp.where(zdpthmix > 0, zdpthmix, 1.0)
        zpresmix = zpresmix / zdpthmix_safe

        # Add temperature perturbation based on TKE
        temp_pert = (params.xatpert * jnp.minimum(3.0, ptkecls) / cst.cpd +
                     params.xbtpert) * params.xdtpert
        zthlcl = zthlcl / zdpthmix_safe + temp_pert
        zrvlcl = zrvlcl / zdpthmix_safe

        # Virtual potential temperature of mixed layer
        zthvlcl = zthlcl * (1.0 + zepsa * zrvlcl) / (1.0 + zrvlcl)

        # ===== Compute LCL using Bolton (1980) empirical formula =====

        # Temperature and vapor pressure of mixed layer
        ztmix = zthlcl * (zpresmix / cst.p00) ** zrdocp
        zevmix = zrvlcl * zpresmix / (zrvlcl + zeps)
        zevmix = jnp.maximum(1e-8, zevmix)

        # Dewpoint temperature
        zx1 = jnp.log(zevmix / 613.3)
        zx1 = (4780.8 - 32.19 * zx1) / (17.502 - zx1)

        # Adiabatic saturation temperature (LCL temperature)
        ztlcl = zx1 - (0.212 + 1.571e-3 * (zx1 - cst.tt) -
                       4.36e-4 * (ztmix - cst.tt)) * (ztmix - zx1)
        ztlcl = jnp.minimum(ztlcl, ztmix)

        # LCL pressure
        zplcl = cst.p00 * (ztlcl / zthlcl) ** zcpord

        # ===== Call saturation mixing ratio at LCL and mixed layer =====
        zewa, zlva, zlsa, zcpha = convect_satmixratio(zplcl, ztlcl)
        zewb, zlvb, zlsb, zcphb = convect_satmixratio(zpresmix, ztmix)

        # ===== Correct TLCL to be consistent with saturation formula =====
        zlsa_deriv = zewa / ztlcl * (cst.betaw / ztlcl - cst.gamw)  # dr_sat/dT
        zlsa_corr = (zewa - zrvlcl) / (1.0 + zlva / zcpha * zlsa_deriv)
        ztlcl = ztlcl - zlva / zcpha * zlsa_corr

        # ===== Handle oversaturated case =====
        is_oversat = zrvlcl > zewb

        zlsb_deriv = zewb / ztmix * (cst.betaw / ztmix - cst.gamw)
        zlsb_corr = (zewb - zrvlcl) / (1.0 + zlvb / zcphb * zlsb_deriv)

        ztlcl = jnp.where(is_oversat, ztmix - zlvb / zcphb * zlsb_corr, ztlcl)
        zrvlcl = jnp.where(is_oversat, zrvlcl - zlsb_corr, zrvlcl)
        zplcl = jnp.where(is_oversat, zpresmix, zplcl)
        zthlcl = jnp.where(is_oversat, ztlcl * (cst.p00 / zplcl) ** zrdocp, zthlcl)
        zthvlcl = jnp.where(is_oversat,
                           zthlcl * (1.0 + zepsa * zrvlcl) / (1.0 + zrvlcl),
                           zthvlcl)

        # ===== Determine vertical index at LCL =====
        # Find level where ZPLCL <= PPRES (LCL is between ilcl-1 and ilcl)
        def find_lcl_level(carry, jk):
            """Find LCL level."""
            ilcl_a = carry
            # JAX FIX: Add condition to only process when jk >= jkk
            mask = gwork1 & (jk >= jkk) & (jk <= jt) & (zplcl <= ppres[:, jk])
            ilcl_new = jnp.where(mask, jk + 1, ilcl_a)
            return ilcl_new, None

        # JAX FIX: Use fixed range instead of dynamic range
        ilcl, _ = lax.scan(find_lcl_level, klcl_c, jnp.arange(ikb, jt + 1))

        # ===== Interpolate to get precise LCL height and theta_v =====
        # Safe indexing with bounds checking
        jkm = jnp.clip(ilcl - 1, ikb, nkt - 2)
        ilcl_clipped = jnp.clip(ilcl, ikb + 1, nkt - 1)

        # Gather values at jkm and ilcl for each grid point
        idx_i = jnp.arange(nit)
        ppres_jkm = ppres[idx_i, jkm]
        ppres_ilcl = ppres[idx_i, ilcl_clipped]
        pthv_jkm = pthv[idx_i, jkm]
        pthv_ilcl = pthv[idx_i, ilcl_clipped]
        pz_jkm = pz[idx_i, jkm]
        pz_ilcl = pz[idx_i, ilcl_clipped]

        # Logarithmic interpolation factor
        zdp = jnp.log(zplcl / (ppres_jkm + 1e-20)) / \
              jnp.log(ppres_ilcl / (ppres_jkm + 1e-20) + 1e-20)
        zdp = jnp.clip(zdp, 0.0, 1.0)

        zthvelcl = pthv_jkm + (pthv_ilcl - pthv_jkm) * zdp
        zzlcl = pz_jkm + (pz_ilcl - pz_jkm) * zdp

        # ===== Compute parcel vertical velocity at LCL =====
        zwlcl = params.xaw * jnp.maximum(0.0, pw[:, ikb]) + params.xbw
        zwlclsqrent = 1.05 * zwlcl * zwlcl  # 1.05 accounts for entrainment

        # ===== Compute updraft equivalent potential temperature =====
        ztheul = ztlcl * (zthlcl / (ztlcl + 1e-10)) ** (1.0 - 0.28 * zrvlcl) * \
                 jnp.exp((3374.6525 / (ztlcl + 1e-10) - 2.5403) * zrvlcl * (1.0 + 0.81 * zrvlcl))

        # ===== Estimate cloud top via CAPE calculation =====
        # Start from minimum LCL level
        jlclmin = jnp.min(jnp.where(gwork1, ilcl, ike))
        jlclmin = jnp.maximum(ikb, jlclmin - 1)

        # JAX FIX: zcape should be per-gridpoint array, not scalar
        # Using minimum LCL across all points for computational stability
        zcape = jnp.ones(nit) * jnp.maximum(jlclmin - ikb, 0) * cst.g
        zcap = jnp.zeros(nit)
        ztop = jnp.zeros(nit)
        zwork3 = jnp.zeros(nit)

        def cape_loop(acc, jl):
            """Compute CAPE contribution from layer."""
            zcape_a, zcap_a, ztop_a, zwork3_a = acc
            jk = jl + 1

            # Skip if jk is out of bounds
            jk_safe = jnp.clip(jk, 0, nkt - 1)

            # JAX FIX: Add mask to only process when jl >= jlclmin
            active_mask = gwork1 & (jl >= jlclmin) & (jl <= jt)

            # Buoyancy contribution (set to 0 below LCL)
            zx1 = (2.0 * ztheul / (pthes[:, jk_safe] + pthes[:, jl] + 1e-10) - 1.0) * \
                  (pz[:, jk_safe] - pz[:, jl])
            zx1 = jnp.where(jl < ilcl, 0.0, zx1)
            zx1 = jnp.where(active_mask, zx1, 0.0)  # Only apply when active

            zcape_new = jnp.where(active_mask, zcape_a + cst.g * jnp.maximum(1.0, zx1), zcape_a)
            zcap_new = jnp.where(active_mask, zcap_a + zx1, zcap_a)

            # Check if updraft kinetic energy becomes negative
            zx2 = jnp.sign(params.xnhgam * cst.g * zcap_new + zwlclsqrent)
            zwork3_new = jnp.where(active_mask,
                                  jnp.maximum(-1.0, zwork3_a + jnp.minimum(0.0, zx2)),
                                  zwork3_a)

            # Extract cloud top (where criterion first fulfilled)
            ztop_new = pz[:, jl] * 0.5 * (1.0 + zx2) * (1.0 + zwork3_new) + \
                      ztop_a * 0.5 * (1.0 - zx2)
            ztop_new = jnp.where(active_mask, jnp.maximum(ztop_new, ztop_a), ztop_a)

            return (zcape_new, zcap_new, ztop_new, zwork3_new), None

        # JAX FIX: Run CAPE loop with fixed range from ikb to jt
        # The active_mask handles the dynamic start at jlclmin
        (zcape, zcap, ztop, zwork3), _ = lax.scan(
            cape_loop,
            (zcape, zcap, ztop, zwork3),
            jnp.arange(ikb, jt + 1)  # Fixed range
        )

        # ===== Check trigger condition =====
        # Sufficient cloud depth, positive CAPE, and active points
        trigger_cond = gwork1 & gtrig2_c & ((ztop - zzlcl) >= params.xcdepth) & (zcape > 10.0)

        # Update outputs where trigger condition is met
        otrig_new = otrig_c | trigger_cond
        pthlcl_new = jnp.where(trigger_cond, zthlcl, pthlcl_c)
        ptlcl_new = jnp.where(trigger_cond, ztlcl, ptlcl_c)
        prvlcl_new = jnp.where(trigger_cond, zrvlcl, prvlcl_c)
        pwlcl_new = jnp.where(trigger_cond, zwlcl, pwlcl_c)
        pzlcl_new = jnp.where(trigger_cond, zzlcl, pzlcl_c)
        pthvelcl_new = jnp.where(trigger_cond, zthvelcl, pthvelcl_c)
        klcl_new = jnp.where(trigger_cond, ilcl, klcl_c)
        kdpl_new = jnp.where(trigger_cond, idpl, kdpl_c)
        kpbl_new = jnp.where(trigger_cond, ipbl, kpbl_c)
        gtrig2_new = jnp.where(trigger_cond, False, gtrig2_c)

        return ((otrig_new, pthlcl_new, ptlcl_new, prvlcl_new, pwlcl_new, pzlcl_new,
                pthvelcl_new, klcl_new, kdpl_new, kpbl_new, zzdpl_new, gtrig2_new), None)

    # Execute main loop over departure levels
    initial_carry = (otrig, pthlcl, ptlcl, prvlcl, pwlcl, pzlcl, pthvelcl,
                     klcl, kdpl, kpbl, zzdpl, gtrig2)

    (otrig, pthlcl, ptlcl, prvlcl, pwlcl, pzlcl, pthvelcl,
     klcl, kdpl, kpbl, _, _), _ = lax.scan(
        loop_over_jkk,
        initial_carry,
        jnp.arange(ikb + 1, ike - 1)
    )

    return TriggerOutputs(
        otrig=otrig,
        pthlcl=pthlcl,
        ptlcl=ptlcl,
        prvlcl=prvlcl,
        pwlcl=pwlcl,
        pzlcl=pzlcl,
        pthvelcl=pthvelcl,
        klcl=klcl,
        kdpl=kdpl,
        kpbl=kpbl,
    )
