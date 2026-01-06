"""
Fritsch-Chappell closure scheme for shallow convection.

This module implements the modified Fritsch-Chappell closure scheme that
determines final adjusted environmental values and convective tendencies
by removing CAPE through an iterative adjustment procedure.

Translated from: PHYEX-IAL_CY50T1/conv/convect_closure_shal.F90

References
----------
- Fritsch and Chappell (1980), J. Atmos. Sci.
- Kain and Fritsch (1993), Meteor. Monographs
- PHYEX Book 1 & 2 documentation
"""

import jax
import jax.numpy as jnp
from jax import Array, lax
from typing import Tuple, NamedTuple

from .constants import PHYS_CONSTANTS
from .convect_constants import ConvectionConstants
from .satmixratio import convect_satmixratio
from .convect_closure_thrvlcl import convect_closure_thrvlcl
from .convect_closure_adjust_shal import convect_closure_adjust_shal


class ClosureOutputs(NamedTuple):
    """Output structure for convect_closure_shal."""
    pthc: Array      # Convectively adjusted grid scale theta (K), shape (nit, nkt)
    prwc: Array      # Convectively adjusted grid scale r_w (kg/kg), shape (nit, nkt)
    prcc: Array      # Convectively adjusted grid scale r_c (kg/kg), shape (nit, nkt)
    pric: Array      # Convectively adjusted grid scale r_i (kg/kg), shape (nit, nkt)
    pwsub: Array     # Environmental compensating subsidence (Pa/s), shape (nit, nkt)
    pumf: Array      # Updated updraft mass flux (kg/s), shape (nit, nkt)
    puer: Array      # Updated updraft entrainment (kg/s), shape (nit, nkt)
    pudr: Array      # Updated updraft detrainment (kg/s), shape (nit, nkt)
    ptimec: Array    # Convection time step (s), shape (nit,)
    kftsteps: int    # Maximum fractional time steps


def convect_closure_shal(
    ppres: Array,
    pdpres: Array,
    pz: Array,
    plmass: Array,
    pthl: Array,
    pth: Array,
    prw: Array,
    prc: Array,
    pri: Array,
    otrig1: Array,
    klcl: Array,
    kdpl: Array,
    kpbl: Array,
    kctl: Array,
    pumf: Array,
    puer: Array,
    pudr: Array,
    puthl: Array,
    purw: Array,
    purc: Array,
    puri: Array,
    pcape: Array,
    ptimec: Array,
    jcvexb: int = 0,
    jcvext: int = 0,
    xstabt: float = 0.90,
    xstabc: float = 0.95,
    xa25: float = 0.25,
) -> ClosureOutputs:
    """
    Fritsch-Chappell closure scheme for shallow convection.

    This routine determines the final adjusted environmental values of theta_l,
    r_w, r_c, and r_i over a time step PTIMEC. The adjustment is performed
    iteratively (4 iterations) to remove CAPE while maintaining mass continuity.

    The main steps are:
    1. Compute mass flux adjustment limits from mass conservation
    2. For 4 iterations:
        a. Compute environmental subsidence (omega) from mass continuity
        b. Determine fractional time steps for numerical stability
        c. Update environment using mass flux convergence
        d. Compute adjusted CAPE at new LCL
        e. Adjust mass flux to remove desired fraction of CAPE
    3. Convert adjusted enthalpy to final theta

    Parameters
    ----------
    ppres : Array
        Pressure (Pa), shape (nit, nkt)
    pdpres : Array
        Pressure difference between layer bottom and top (Pa), shape (nit, nkt)
    pz : Array
        Height of model layers (m), shape (nit, nkt)
    plmass : Array
        Mass of model layers (kg), shape (nit, nkt)
    pthl : Array
        Grid scale enthalpy (J/kg), shape (nit, nkt)
    pth : Array
        Grid scale theta (K), shape (nit, nkt)
    prw : Array
        Grid scale total water mixing ratio (kg/kg), shape (nit, nkt)
    prc : Array
        Grid scale cloud water mixing ratio (kg/kg), shape (nit, nkt)
    pri : Array
        Grid scale cloud ice mixing ratio (kg/kg), shape (nit, nkt)
    otrig1 : Array
        Logical mask for triggered convection (bool), shape (nit,)
    klcl : Array
        Index of lifting condensation level, shape (nit,)
    kdpl : Array
        Index of departure level, shape (nit,)
    kpbl : Array
        Index of PBL top, shape (nit,)
    kctl : Array
        Index of cloud top level, shape (nit,)
    pumf : Array
        Updraft mass flux (kg/s), shape (nit, nkt)
    puer : Array
        Updraft entrainment (kg/s), shape (nit, nkt)
    pudr : Array
        Updraft detrainment (kg/s), shape (nit, nkt)
    puthl : Array
        Updraft enthalpy (J/kg), shape (nit, nkt)
    purw : Array
        Updraft total water (kg/kg), shape (nit, nkt)
    purc : Array
        Updraft cloud water (kg/kg), shape (nit, nkt)
    puri : Array
        Updraft cloud ice (kg/kg), shape (nit, nkt)
    pcape : Array
        Available potential energy (J/kg), shape (nit,)
    ptimec : Array
        Convection time step (s), shape (nit,)
    jcvexb : int, optional
        Extra vertical levels at bottom (default 0)
    jcvext : int, optional
        Extra vertical levels at top (default 0)
    xstabt : float, optional
        Stability factor in time integration (default 0.90)
    xstabc : float, optional
        Stability factor in CAPE adjustment (default 0.95)
    xa25 : float, optional
        Reference grid area (default 0.25)

    Returns
    -------
    ClosureOutputs
        Named tuple containing:
        - pthc: Adjusted grid scale theta (K)
        - prwc: Adjusted water vapor (kg/kg)
        - prcc: Adjusted cloud water (kg/kg)
        - pric: Adjusted cloud ice (kg/kg)
        - pwsub: Environmental subsidence (Pa/s)
        - pumf: Adjusted updraft mass flux (kg/s)
        - puer: Adjusted updraft entrainment (kg/s)
        - pudr: Adjusted updraft detrainment (kg/s)
        - ptimec: Convection time step (s)
        - kftsteps: Maximum fractional time steps

    Notes
    -----
    The closure scheme removes CAPE through an iterative adjustment:

    1. Mass continuity determines environmental subsidence:

       .. math::
           \\frac{\\partial \\omega}{\\partial p} = E - D

    2. Environmental adjustment equations (key physics):

       .. math::
           \\frac{\\partial \\theta_l}{\\partial t} =
               \\frac{1}{m}[M_{in}\\theta_{l,in} - M_{out}\\theta_{l,out} +
               D\\theta_{u} - E\\theta_e]

       .. math::
           \\frac{\\partial r_w}{\\partial t} =
               \\frac{1}{m}[M_{in}r_{w,in} - M_{out}r_{w,out} +
               Dr_{w,u} - Er_{w,e}]

    3. Mass flux adjustment to remove CAPE:

       .. math::
           M_{new} = M_{old} \\times f_{adj}

       where :math:`f_{adj}` is computed to remove 80-95% of CAPE.

    The scheme uses fractional time stepping to ensure numerical stability
    when the CFL condition for vertical advection is violated.
    """

    cst = PHYS_CONSTANTS

    # Constants
    zeps = cst.rd / cst.rv
    zcpord = cst.cpd / cst.rd
    zrdocp = cst.rd / cst.cpd

    # Array dimensions
    nit, nkt = ppres.shape
    ikb = jcvexb
    ike = nkt - jcvext - 1

    # Initialize outputs
    zthlc = pthl.copy()
    prwc = prw.copy()
    prcc = jnp.maximum(0.0, prc)
    pric = jnp.maximum(0.0, pri)
    pthc = pth.copy()
    pwsub = jnp.zeros_like(ppres)

    # Save initial mass flux values
    zumf = pumf.copy()
    zuer = puer.copy()
    zudr = pudr.copy()

    # Working arrays
    zadj = jnp.ones(nit)
    zwork5 = jnp.where(otrig1, 1.0, 0.0)
    ilcl = klcl.copy()

    # Compute adjustment limits from mass conservation
    # The inflow can't exceed the mass in the layer
    zadjmax = jnp.full(nit, 1000.0)

    # Find min/max levels for optimization
    jctlmax = jnp.max(jnp.where(otrig1, kctl, ikb))
    jdplmin = jnp.min(jnp.where(otrig1, kdpl, ike))
    jlclmax = jnp.max(jnp.where(otrig1, ilcl, ikb))

    jctlmax = jnp.minimum(jctlmax, ike)

    # Compute ZADJMAX: limit adjustment to prevent mass violation
    def compute_zadjmax(carry, jk):
        """Compute maximum adjustment factor from mass conservation."""
        zadjmax_c = carry

        # Mask for points in layer between DPL and LCL
        idx_i = jnp.arange(nit)
        jk_clip = jnp.clip(jk, 0, nkt - 1)

        in_layer = (jk > kdpl) & (jk <= ilcl)

        # Compute limit from entrainment and layer mass
        zwork1 = plmass[idx_i, jk_clip] / ((puer[idx_i, jk_clip] + 1e-5) * ptimec + 1e-20)

        # Update minimum
        zadjmax_new = jnp.where(in_layer, jnp.minimum(zadjmax_c, zwork1), zadjmax_c)

        return zadjmax_new, None

    # JAX FIX: Use fixed range instead of dynamic range based on jdplmin/jlclmax
    # Add condition in compute_zadjmax to only process when in valid range
    zadjmax, _ = lax.scan(
        compute_zadjmax,
        zadjmax,
        jnp.arange(ikb, ike + 1)  # Fixed range
    )

    # Precompute pi values (Exner function)
    zpiaux = (cst.p00 / ppres) ** zrdocp

    # Logical mask for adjustment
    gwork1 = otrig1.copy()

    # ===== Main adjustment iteration loop (4 iterations) =====
    def adjustment_iteration(carry, jiter):
        """Single iteration of CAPE removal adjustment."""
        zthlc_i, prwc_i, prcc_i, pric_i, pthc_i, pwsub_i, pumf_i, puer_i, pudr_i, zadj_i, gwork1_i, ptimec_i = carry

        ztimec = ptimec_i.copy()
        pwsub_iter = jnp.zeros_like(ppres)
        zomg = jnp.zeros_like(ppres)

        # Reinitialize adjusted values
        zthlc_iter = jnp.where(gwork1_i[:, None], pthl, zthlc_i)
        prwc_iter = jnp.where(gwork1_i[:, None], prw, prwc_i)
        prcc_iter = jnp.where(gwork1_i[:, None], jnp.maximum(0.0, prc), prcc_i)
        pric_iter = jnp.where(gwork1_i[:, None], jnp.maximum(0.0, pri), pric_i)
        pthc_iter = jnp.where(gwork1_i[:, None], pth, pthc_i)

        # ===== Compute environmental subsidence (omega) from mass continuity =====
        def compute_omega(carry, jk):
            """Compute vertical velocity from mass continuity."""
            pwsub_c, ztimec_c, zomg_c = carry

            jkp = jnp.maximum(ikb + 1, jk - 1)
            jkp_clip = jnp.clip(jkp, 0, nkt - 1)
            jk_clip = jnp.clip(jk, 0, nkt - 1)
            jkm_clip = jnp.clip(jk - 1, 0, nkt - 1)

            idx_i = jnp.arange(nit)
            in_cloud = gwork1_i & (jk <= kctl)

            # Mass continuity: d(omega)/dp = -(E - D)
            zwork1 = -(puer_i[idx_i, jkp_clip] - pudr_i[idx_i, jkp_clip]) / (plmass[idx_i, jkp_clip] + 1e-20)

            # Integrate from bottom
            pwsub_new = pwsub_c[idx_i, jkp_clip] - pdpres[idx_i, jkm_clip] * zwork1
            pwsub_c = pwsub_c.at[idx_i, jk_clip].set(
                jnp.where(in_cloud, pwsub_new, pwsub_c[idx_i, jk_clip])
            )

            # Fractional time step for stability (CFL condition)
            zwork1_cfl = xstabt * pdpres[idx_i, jkp_clip] / (jnp.abs(pwsub_new) + 1e-10)
            ztimec_new = jnp.where(in_cloud, jnp.minimum(ztimec_c, zwork1_cfl), ztimec_c)

            # Convert omega to mass flux units
            zomg_new = pwsub_new * xa25 / cst.g
            zomg_c = zomg_c.at[idx_i, jk_clip].set(
                jnp.where(in_cloud, zomg_new, zomg_c[idx_i, jk_clip])
            )

            return (pwsub_c, ztimec_new, zomg_c), None

        # JAX FIX: Use fixed range instead of dynamic jctlmax
        # The compute_omega function already has (jk <= kctl) condition
        (pwsub_iter, ztimec, zomg), _ = lax.scan(
            compute_omega,
            (pwsub_iter, ztimec, zomg),
            jnp.arange(ikb + 1, ike + 1)  # Fixed range
        )

        # ===== Mass conservation check =====
        idx_i = jnp.arange(nit)
        kctl_clip = jnp.clip(kctl, 0, nkt - 1)

        zwork1_mass = (pudr_i[idx_i, kctl_clip] * pdpres[idx_i, kctl_clip] /
                       (plmass[idx_i, kctl_clip] + 0.1) - pwsub_iter[idx_i, kctl_clip])

        # Mask out points with poor mass conservation
        mass_ok = ~(gwork1_i & (jnp.abs(zwork1_mass) > 0.01))
        gwork1_new = gwork1_i & mass_ok
        ptimec_new = jnp.where(~mass_ok & gwork1_i, 0.1, ptimec_i)
        zwork5_new = jnp.where(~mass_ok & gwork1_i, 0.0, 1.0)

        pwsub_iter = pwsub_iter * zwork5_new[:, None]

        # ===== Fractional time stepping =====
        itstep = (ptimec_new / (ztimec + 1e-20)).astype(jnp.int32) + 1
        ztimec = ptimec_new / itstep.astype(jnp.float32)
        ztimc = jnp.broadcast_to(ztimec[:, None], (nit, nkt))

        kftsteps = jnp.max(itstep)

        # JAX FIX: Use fixed max fractional time steps for JAX compatibility
        # The fractional_step function has (itstep >= jstep + 1) condition
        # Typical values are < 100, but we use 100 as conservative upper bound
        max_ftsteps = 100

        # ===== Fractional time step loop =====
        def fractional_step(carry_f, jstep):
            """Single fractional time step."""
            zthlc_f, prwc_f, prcc_f, pric_f = carry_f

            gwork3 = (itstep >= jstep + 1) & gwork1_new

            # ===== Compute mass flux convergence =====
            # This is the vectorized version for computational efficiency

            zthmfin = jnp.zeros((nit, nkt))
            zthmfout = jnp.zeros((nit, nkt))
            zrwmfin = jnp.zeros((nit, nkt))
            zrwmfout = jnp.zeros((nit, nkt))
            zrcmfin = jnp.zeros((nit, nkt))
            zrcmfout = jnp.zeros((nit, nkt))
            zrimfin = jnp.zeros((nit, nkt))
            zrimfout = jnp.zeros((nit, nkt))

            def compute_mass_flux(carry_mf, jk):
                """Compute mass flux at level interfaces."""
                mfin, mfout = carry_mf

                jkp = jnp.maximum(ikb + 1, jk - 1)
                jk_clip = jnp.clip(jk, 0, nkt - 1)
                jkp_clip = jnp.clip(jkp, 0, nkt - 1)

                idx_i = jnp.arange(nit)
                gwork4 = gwork3 & (jk <= kctl)

                # Upwind scheme based on sign of omega
                # If omega > 0 (upward), use values from below (jk)
                # If omega < 0 (downward), use values from above (jkp)

                omega_jk = zomg[idx_i, jk_clip]

                # Inflow (from above when omega < 0)
                mfin_thl = -jnp.minimum(omega_jk, 0.0) * zthlc_f[idx_i, jkp_clip]
                mfin_rw = -jnp.minimum(omega_jk, 0.0) * prwc_f[idx_i, jkp_clip]
                mfin_rc = -jnp.minimum(omega_jk, 0.0) * prcc_f[idx_i, jkp_clip]
                mfin_ri = -jnp.minimum(omega_jk, 0.0) * pric_f[idx_i, jkp_clip]

                # Outflow (upward when omega > 0)
                mfout_thl = jnp.maximum(omega_jk, 0.0) * zthlc_f[idx_i, jk_clip]
                mfout_rw = jnp.maximum(omega_jk, 0.0) * prwc_f[idx_i, jk_clip]
                mfout_rc = jnp.maximum(omega_jk, 0.0) * prcc_f[idx_i, jk_clip]
                mfout_ri = jnp.maximum(omega_jk, 0.0) * pric_f[idx_i, jk_clip]

                # Store at current level
                mfin_new = {
                    'thl': mfin['thl'].at[idx_i, jk_clip].set(jnp.where(gwork4, mfin_thl, mfin['thl'][idx_i, jk_clip])),
                    'rw': mfin['rw'].at[idx_i, jk_clip].set(jnp.where(gwork4, mfin_rw, mfin['rw'][idx_i, jk_clip])),
                    'rc': mfin['rc'].at[idx_i, jk_clip].set(jnp.where(gwork4, mfin_rc, mfin['rc'][idx_i, jk_clip])),
                    'ri': mfin['ri'].at[idx_i, jk_clip].set(jnp.where(gwork4, mfin_ri, mfin['ri'][idx_i, jk_clip])),
                }

                mfout_new = {
                    'thl': mfout['thl'].at[idx_i, jk_clip].set(jnp.where(gwork4, mfout_thl, mfout['thl'][idx_i, jk_clip])),
                    'rw': mfout['rw'].at[idx_i, jk_clip].set(jnp.where(gwork4, mfout_rw, mfout['rw'][idx_i, jk_clip])),
                    'rc': mfout['rc'].at[idx_i, jk_clip].set(jnp.where(gwork4, mfout_rc, mfout['rc'][idx_i, jk_clip])),
                    'ri': mfout['ri'].at[idx_i, jk_clip].set(jnp.where(gwork4, mfout_ri, mfout['ri'][idx_i, jk_clip])),
                }

                return (mfin_new, mfout_new), None

            mfin_init = {'thl': zthmfin, 'rw': zrwmfin, 'rc': zrcmfin, 'ri': zrimfin}
            mfout_init = {'thl': zthmfout, 'rw': zrwmfout, 'rc': zrcmfout, 'ri': zrimfout}

            # JAX FIX: Use fixed range instead of dynamic jctlmax
            # The compute_mass_flux function already has (jk <= kctl) condition
            (mfin_final, mfout_final), _ = lax.scan(
                compute_mass_flux,
                (mfin_init, mfout_init),
                jnp.arange(ikb + 1, ike + 1)  # Fixed range
            )

            zthmfin = mfin_final['thl']
            zthmfout = mfout_final['thl']
            zrwmfin = mfin_final['rw']
            zrwmfout = mfout_final['rw']
            zrcmfin = mfin_final['rc']
            zrcmfout = mfout_final['rc']
            zrimfin = mfin_final['ri']
            zrimfout = mfout_final['ri']

            # ===== Update environment (MAIN EQUATIONS) =====
            idx_i = jnp.arange(nit)

            def update_environment(jk):
                """Update environmental values at one level."""
                jk_clip = jnp.clip(jk, 0, nkt - 1)
                gwork4 = gwork3 & (jk <= kctl)

                # Time step / mass
                dt_over_m = ztimc[idx_i, jk_clip] / (plmass[idx_i, jk_clip] + 1e-20)

                # Enthalpy tendency
                dthl = (zthmfin[idx_i, jk_clip] + pudr_i[idx_i, jk_clip] * puthl[idx_i, jk_clip] -
                        zthmfout[idx_i, jk_clip] - puer_i[idx_i, jk_clip] * pthl[idx_i, jk_clip])

                # Water vapor tendency
                drw = (zrwmfin[idx_i, jk_clip] + pudr_i[idx_i, jk_clip] * purw[idx_i, jk_clip] -
                       zrwmfout[idx_i, jk_clip] - puer_i[idx_i, jk_clip] * prw[idx_i, jk_clip])

                # Cloud water tendency
                drc = (zrcmfin[idx_i, jk_clip] + pudr_i[idx_i, jk_clip] * purc[idx_i, jk_clip] -
                       zrcmfout[idx_i, jk_clip] - puer_i[idx_i, jk_clip] * jnp.maximum(0.0, prc[idx_i, jk_clip]))

                # Cloud ice tendency
                dri = (zrimfin[idx_i, jk_clip] + pudr_i[idx_i, jk_clip] * puri[idx_i, jk_clip] -
                       zrimfout[idx_i, jk_clip] - puer_i[idx_i, jk_clip] * jnp.maximum(0.0, pri[idx_i, jk_clip]))

                # Apply updates
                zthlc_new = zthlc_f[idx_i, jk_clip] + dt_over_m * dthl
                prwc_new = prwc_f[idx_i, jk_clip] + dt_over_m * drw
                prcc_new = prcc_f[idx_i, jk_clip] + dt_over_m * drc
                pric_new = pric_f[idx_i, jk_clip] + dt_over_m * dri

                return (
                    jnp.where(gwork4, zthlc_new, zthlc_f[idx_i, jk_clip]),
                    jnp.where(gwork4, prwc_new, prwc_f[idx_i, jk_clip]),
                    jnp.where(gwork4, prcc_new, prcc_f[idx_i, jk_clip]),
                    jnp.where(gwork4, pric_new, pric_f[idx_i, jk_clip]),
                )

            # JAX FIX: Use fixed range instead of dynamic jctlmax
            # The update_environment function already has (jk <= kctl) condition
            # Vectorized update over vertical levels
            updates = jax.vmap(update_environment)(jnp.arange(ikb, ike + 1))

            # Reconstruct full arrays
            zthlc_new = zthlc_f.at[idx_i[:, None], jnp.arange(ikb, ike + 1)[None, :]].set(updates[0].T)
            prwc_new = prwc_f.at[idx_i[:, None], jnp.arange(ikb, ike + 1)[None, :]].set(updates[1].T)
            prcc_new = prcc_f.at[idx_i[:, None], jnp.arange(ikb, ike + 1)[None, :]].set(updates[2].T)
            pric_new = pric_f.at[idx_i[:, None], jnp.arange(ikb, ike + 1)[None, :]].set(updates[3].T)

            return (zthlc_new, prwc_new, prcc_new, pric_new), None

        # JAX FIX: Use fixed range for fractional time steps
        # Execute fractional time steps
        (zthlc_iter, prwc_iter, prcc_iter, pric_iter), _ = lax.scan(
            fractional_step,
            (zthlc_iter, prwc_iter, prcc_iter, pric_iter),
            jnp.arange(max_ftsteps)  # Fixed upper bound
        )

        # ===== Convert enthalpy to theta =====
        def compute_theta(jk):
            """Compute final theta from adjusted enthalpy."""
            jk_clip = jnp.clip(jk, 0, nkt - 1)
            idx_i = jnp.arange(nit)
            gwork4 = gwork1_new & (jk <= kctl)

            zcph = cst.cpd + prwc_iter[idx_i, jk_clip] * cst.cpv

            # First temperature estimate
            zwork2 = pth[idx_i, jk_clip] / zpiaux[idx_i, jk_clip]
            zlv = cst.xlvtt + (cst.cpv - cst.cl) * (zwork2 - cst.tt)
            zls = cst.xlstt + (cst.cpv - cst.ci) * (zwork2 - cst.tt)

            # Linearized temperature
            zwork2_new = (zthlc_iter[idx_i, jk_clip] +
                          zlv * prcc_iter[idx_i, jk_clip] +
                          zls * pric_iter[idx_i, jk_clip] -
                          (1.0 + prwc_iter[idx_i, jk_clip]) * cst.g * pz[idx_i, jk_clip]) / zcph

            zwork2_new = jnp.clip(zwork2_new, 180.0, 340.0)
            pthc_new = zwork2_new * zpiaux[idx_i, jk_clip]

            return jnp.where(gwork4, pthc_new, pthc_iter[idx_i, jk_clip])

        # JAX FIX: Use fixed range instead of dynamic jctlmax
        # The compute_theta function already has (jk <= kctl) condition
        # Vectorized theta computation
        pthc_new_vals = jax.vmap(compute_theta)(jnp.arange(ikb + 1, ike + 1))
        pthc_iter = pthc_iter.at[idx_i[:, None], jnp.arange(ikb + 1, ike + 1)[None, :]].set(pthc_new_vals.T)

        # ===== Compute new LCL properties =====
        lcl_outputs = convect_closure_thrvlcl(
            ppres, pthc_iter, prwc_iter, pz, gwork1_new, kdpl, kpbl, jcvexb, jcvext
        )

        zthlcl = lcl_outputs.pthlcl
        zrvlcl = lcl_outputs.prvlcl
        zzlcl = lcl_outputs.pzlcl
        ztlcl = lcl_outputs.ptlcl
        ztelcl = lcl_outputs.ptelcl
        ilcl = lcl_outputs.klcl

        # Bound LCL values
        ztlcl = jnp.clip(ztlcl, 230.0, 335.0)
        ztelcl = jnp.clip(ztelcl, 230.0, 335.0)
        zthlcl = jnp.clip(zthlcl, 230.0, 345.0)
        zrvlcl = jnp.clip(zrvlcl, 0.0, 1.0)

        # ===== Compute adjusted CAPE =====
        zcape = jnp.zeros(nit)
        zpi = jnp.clip(zthlcl / ztlcl, 0.95, 1.5)
        zwork1 = cst.p00 / zpi ** zcpord

        # Saturation at LCL
        zwork3, zlv, zls, zcph = convect_satmixratio(zwork1, ztelcl)
        zwork3 = jnp.clip(zwork3, 0.0, 0.1)

        # Theta_e for undilute updraft
        ztheul = (ztlcl * zpi ** (1.0 - 0.28 * zrvlcl) *
                  jnp.exp((3374.6525 / ztlcl - 2.5403) * zrvlcl * (1.0 + 0.81 * zrvlcl)))

        # Theta_e saturated at LCL
        zthes1 = (ztelcl * zpi ** (1.0 - 0.28 * zwork3) *
                  jnp.exp((3374.6525 / ztelcl - 2.5403) * zwork3 * (1.0 + 0.81 * zwork3)))

        # Integrate CAPE from LCL to CTL
        jlclmin = jnp.maximum(jnp.min(jnp.where(gwork1_new, ilcl, ike)), ikb)

        def integrate_cape(carry, jk):
            """Integrate CAPE over vertical column."""
            zcape_c, zthes1_c = carry

            jkp = jk - 1
            jk_clip = jnp.clip(jk, 0, nkt - 1)
            jkp_clip = jnp.clip(jkp, 0, nkt - 1)
            idx_i = jnp.arange(nit)

            gwork3 = (jk >= ilcl) & (jk <= kctl) & gwork1_new

            # Temperature at level
            zwork2 = pthc_iter[idx_i, jk_clip] / zpiaux[idx_i, jk_clip]

            # Saturation mixing ratio
            zwork3_sat, zlv_jk, zls_jk, zcph_jk = convect_satmixratio(ppres[idx_i, jk_clip], zwork2)

            # Theta_e saturated
            zthes2 = (zwork2 * zpiaux[idx_i, jk_clip] ** (1.0 - 0.28 * zwork3_sat) *
                      jnp.exp((3374.6525 / zwork2 - 2.5403) * zwork3_sat * (1.0 + 0.81 * zwork3_sat)))

            # Layer thickness
            is_lcl = jk == ilcl
            zwork3_dz = jnp.where(is_lcl,
                                  pz[idx_i, jk_clip] - zzlcl,
                                  pz[idx_i, jk_clip] - pz[idx_i, jkp_clip])

            # Buoyancy
            zwork1_buoy = 2.0 * ztheul / (zthes1_c + zthes2) - 1.0

            # Accumulate CAPE
            zcape_new = zcape_c + cst.g * zwork3_dz * jnp.maximum(0.0, zwork1_buoy)

            zcape_c = jnp.where(gwork3, zcape_new, zcape_c)
            zthes1_c = jnp.where(gwork3, zthes2, zthes1_c)

            return (zcape_c, zthes1_c), None

        # JAX FIX: Use fixed range instead of dynamic jlclmin and jctlmax
        # The integrate_cape function already has (jk >= ilcl) & (jk <= kctl) condition
        (zcape, _), _ = lax.scan(
            integrate_cape,
            (zcape, zthes1),
            jnp.arange(ikb, ike + 1)  # Fixed range
        )

        # ===== Determine mass adjustment factor =====
        zwork1 = jnp.maximum(pcape - zcape, 0.2 * pcape)
        zwork2 = zcape / (pcape + 1e-8)

        # Continue adjusting if CAPE removal is insufficient
        gwork1_cont = gwork1_new & ((zwork2 > 0.2) | (zcape == 0.0))

        # Adjust factor based on CAPE removal
        zadj_new = jnp.where(
            (zcape == 0.0) & gwork1_cont,
            zadj_i * 0.5,
            jnp.where(
                (zcape != 0.0) & gwork1_cont,
                zadj_i * xstabc * pcape / (zwork1 + 1e-8),
                zadj_i
            )
        )

        zadj_new = jnp.minimum(zadj_new, zadjmax)

        # ===== Apply adjustment to mass flux =====
        pumf_new, puer_new, pudr_new = convect_closure_adjust_shal(
            zadj_new, zumf, zuer, zudr
        )

        return (
            zthlc_iter, prwc_iter, prcc_iter, pric_iter, pthc_iter,
            pwsub_iter, pumf_new, puer_new, pudr_new, zadj_new,
            gwork1_cont, ptimec_new
        ), kftsteps

    # Execute 4 adjustment iterations
    carry_init = (zthlc, prwc, prcc, pric, pthc, pwsub, pumf, puer, pudr, zadj, gwork1, ptimec)
    carry_final, kftsteps_all = lax.scan(
        adjustment_iteration,
        carry_init,
        jnp.arange(4)
    )

    (zthlc, prwc, prcc, pric, pthc, pwsub, pumf, puer, pudr, zadj, gwork1, ptimec) = carry_final
    kftsteps = jnp.max(kftsteps_all)

    # ===== Final adjustment: convert total water to vapor =====
    prwc = jnp.maximum(0.0, prwc - prcc - pric)

    return ClosureOutputs(
        pthc=pthc,
        prwc=prwc,
        prcc=prcc,
        pric=pric,
        pwsub=pwsub,
        pumf=pumf,
        puer=puer,
        pudr=pudr,
        ptimec=ptimec,
        kftsteps=int(kftsteps),
    )
