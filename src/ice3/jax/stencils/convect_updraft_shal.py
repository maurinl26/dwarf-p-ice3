"""
Shallow convection updraft calculations.

Compute updraft properties from DPL to CTL for shallow convection.
Translated from: PHYEX-IAL_CY50T1/conv/convect_updraft_shal.F90
"""

import jax.numpy as jnp
from jax import Array, lax
from typing import Tuple, NamedTuple

from .constants import PHYS_CONSTANTS
from .convect_condens import convect_condens
from .convect_mixing_funct import convect_mixing_funct


class UpdraftOutputs(NamedTuple):
    """Output structure for convect_updraft_shal."""
    pumf: Array  # Updraft mass flux (kg/s), shape (nit, nkt)
    puer: Array  # Updraft entrainment (kg/s), shape (nit, nkt)
    pudr: Array  # Updraft detrainment (kg/s), shape (nit, nkt)
    puthl: Array  # Updraft enthalpy (J/kg), shape (nit, nkt)
    puthv: Array  # Updraft theta_v (K), shape (nit, nkt)
    purw: Array  # Updraft total water (kg/kg), shape (nit, nkt)
    purc: Array  # Updraft cloud water (kg/kg), shape (nit, nkt)
    puri: Array  # Updraft cloud ice (kg/kg), shape (nit, nkt)
    pcape: Array  # Available potential energy (J/kg), shape (nit,)
    kctl: Array  # Vertical index of CTL, shape (nit,)
    ketl: Array  # Vertical index of ETL, shape (nit,)
    otrig: Array  # Trigger mask (bool), shape (nit,)


def convect_updraft_shal(
    ppres: Array,
    pdpres: Array,
    pz: Array,
    pthl: Array,
    pthv: Array,
    pthes: Array,
    prw: Array,
    pthlcl: Array,
    ptlcl: Array,
    prvlcl: Array,
    pwlcl: Array,
    pzlcl: Array,
    pthvelcl: Array,
    pmflcl: float,
    otrig: Array,
    klcl: Array,
    kdpl: Array,
    kpbl: Array,
    gtrig1: Array,
    kice: int = 1,
    jcvexb: int = 0,
    jcvext: int = 0,
    xentr: float = 0.03,
    xcrad: float = 1500.0,
    xcdepth: float = 0.0,
    xcdepth_d: float = 3000.0,
    xnhgam: float = 0.33,
    xtfrz1: float = 268.15,
    xtfrz2: float = 248.15,
) -> UpdraftOutputs:
    """
    Compute updraft properties from DPL to CTL for shallow convection.

    This routine determines updraft properties (mass flux, thermodynamics,
    condensate) by computing at every model level starting from bottom.
    The computation includes entrainment/detrainment, condensation, and
    CAPE accumulation.

    Parameters
    ----------
    ppres : Array
        Pressure (Pa), shape (nit, nkt)
    pdpres : Array
        Pressure difference between bottom and top of layer (Pa), shape (nit, nkt)
    pz : Array
        Height of model layer (m), shape (nit, nkt)
    pthl : Array
        Grid scale enthalpy (J/kg), shape (nit, nkt)
    pthv : Array
        Grid scale theta_v (K), shape (nit, nkt)
    pthes : Array
        Grid scale saturated theta_e (K), shape (nit, nkt)
    prw : Array
        Grid scale total water mixing ratio (kg/kg), shape (nit, nkt)
    pthlcl : Array
        Theta at LCL (K), shape (nit,)
    ptlcl : Array
        Temperature at LCL (K), shape (nit,)
    prvlcl : Array
        Vapor mixing ratio at LCL (kg/kg), shape (nit,)
    pwlcl : Array
        Parcel velocity at LCL (m/s), shape (nit,)
    pzlcl : Array
        Height at LCL (m), shape (nit,)
    pthvelcl : Array
        Environment theta_v at LCL (K), shape (nit,)
    pmflcl : float
        Cloud base unit mass flux (kg/s)
    otrig : Array
        Logical mask for convection (bool), shape (nit,)
    klcl : Array
        Vertical index of LCL (int), shape (nit,)
    kdpl : Array
        Vertical index of DPL (int), shape (nit,)
    kpbl : Array
        Vertical index of source layer top (int), shape (nit,)
    gtrig1 : Array
        Logical mask for convection (bool), shape (nit,)
    kice : int, optional
        Flag for ice (1 = yes, 0 = no ice), default 1
    jcvexb : int, optional
        Extra vertical levels at bottom, default 0
    jcvext : int, optional
        Extra vertical levels at top, default 0
    xentr : float, optional
        Entrainment constant, default 0.03
    xcrad : float, optional
        Cloud radius (m), default 1500.0
    xcdepth : float, optional
        Minimum necessary cloud depth (m), default 0.0
    xcdepth_d : float, optional
        Maximum allowed cloud thickness (m), default 3000.0
    xnhgam : float, optional
        Coefficient for buoyancy term in w equation, default 0.33
    xtfrz1 : float, optional
        Begin of freezing interval (K), default 268.15
    xtfrz2 : float, optional
        End of freezing interval (K), default 248.15

    Returns
    -------
    UpdraftOutputs
        Named tuple containing updraft properties and indices

    Notes
    -----
    The routine performs the following steps:

    1. Initialize updraft properties between DPL and LCL
    2. Loop upward from LCL to find CTL:
       - Compute condensate, temperature, theta_v at level k+1
       - Update vertical velocity using buoyancy and entrainment
       - Compute entrainment/detrainment from mixing function
       - Accumulate CAPE using undilute theta_e
       - Update mass flux and thermodynamic properties
       - Check for CTL (negative w^2 or excessive detrainment)
    3. Adjust profiles above ETL for linear decrease to CTL
    4. Set mass flux in source layer (linear increase from DPL to PBL)
    5. Check trigger conditions (cloud depth, CAPE)

    The undilute theta_e is computed using Bolton (1980):

    .. math::
        \\theta_e = T_{LCL} \\left(\\frac{\\theta_{LCL}}{T_{LCL}}\\right)^{1-0.28r_v}
        \\exp\\left(\\left(\\frac{3374.6525}{T_{LCL}} - 2.5403\\right) r_v (1 + 0.81 r_v)\\right)

    CAPE is computed using theta_e instead of theta_v:

    .. math::
        \\text{CAPE} = \\int g \\left(\\frac{2\\theta_e^{parcel}}{\\theta_e^{env,k} + \\theta_e^{env,k+1}} - 1\\right) dz

    References
    ----------
    - Kain and Fritsch (1990), J. Atmos. Sci.
    - Kain and Fritsch (1993), Meteor. Monographs
    - Bolton (1980), Mon. Wea. Rev., Vol. 108, 1046-1053
    """

    # Constants
    cst = PHYS_CONSTANTS
    zepsa = cst.rv / cst.rd  # ~1.608
    zrdocp = cst.rd / cst.cpd  # ~0.286
    zice = float(kice)
    zeps0 = cst.rd / cst.rv  # ~0.622

    # Array dimensions
    nit, nkt = ppres.shape
    ikb = jcvexb
    ike = nkt - jcvext - 1

    # Initialize outputs
    pumf = jnp.zeros((nit, nkt))
    puer = jnp.zeros((nit, nkt))
    pudr = jnp.zeros((nit, nkt))
    puthl = jnp.zeros((nit, nkt))
    puthv = jnp.zeros((nit, nkt))
    purw = jnp.zeros((nit, nkt))
    purc = jnp.zeros((nit, nkt))
    puri = jnp.zeros((nit, nkt))
    pcape = jnp.zeros(nit)
    kctl = jnp.full(nit, ikb, dtype=jnp.int32)
    ketl = klcl.copy()

    # Initialize working arrays
    zuw1 = pwlcl * pwlcl  # Square of vertical velocity at level k
    ze1 = jnp.zeros(nit)  # Fractional entrainment at level k
    zd1 = jnp.zeros(nit)  # Fractional detrainment at level k
    gwork2 = jnp.ones(nit, dtype=bool)  # Mask for active updrafts

    # Compute undilute updraft theta_e for CAPE (Bolton 1980)
    ztheul = ptlcl * (pthlcl / ptlcl) ** (1.0 - 0.28 * prvlcl) * \
             jnp.exp((3374.6525 / ptlcl - 2.5403) * prvlcl * (1.0 + 0.81 * prvlcl))

    # Define accurate enthalpy for updraft
    zwork1 = (cst.cpd + prvlcl * cst.cpv) * ptlcl + (1.0 + prvlcl) * cst.g * pzlcl

    # ===== Set updraft properties between DPL and LCL =====
    def init_dpl_to_lcl(jk):
        """Initialize properties from DPL to LCL at level jk."""
        mask = (jk >= kdpl) & (jk < klcl)
        mask_2d = mask[:, None]  # Broadcast to (nit, 1)

        pumf_jk = jnp.where(mask, pmflcl, 0.0)
        puthl_jk = jnp.where(mask, zwork1, 0.0)
        puthv_jk = jnp.where(mask, pthlcl * (1.0 + zepsa * prvlcl) / (1.0 + prvlcl), 0.0)
        purw_jk = jnp.where(mask, prvlcl, 0.0)

        return pumf_jk, puthl_jk, puthv_jk, purw_jk

    # Vectorized initialization over vertical levels
    def set_level(carry, jk):
        """Set properties at level jk."""
        pumf_c, puthl_c, puthv_c, purw_c = carry
        pumf_jk, puthl_jk, puthv_jk, purw_jk = init_dpl_to_lcl(jk)

        pumf_c = pumf_c.at[:, jk].set(pumf_jk)
        puthl_c = puthl_c.at[:, jk].set(puthl_jk)
        puthv_c = puthv_c.at[:, jk].set(puthv_jk)
        purw_c = purw_c.at[:, jk].set(purw_jk)

        return (pumf_c, puthl_c, puthv_c, purw_c), None

    (pumf, puthl, puthv, purw), _ = lax.scan(
        set_level,
        (pumf, puthl, puthv, purw),
        jnp.arange(ikb, ike + 1)
    )

    # ===== Main updraft loop from LCL to CTL =====
    def updraft_loop_body(carry, jk):
        """Updraft loop body for level jk -> jk+1."""
        (pumf_c, puer_c, pudr_c, puthl_c, puthv_c, purw_c, purc_c, puri_c,
         pcape_c, kctl_c, ketl_c, zuw1_c, ze1_c, zd1_c, gwork2_c) = carry

        jkp = jk + 1

        # Mask for active updraft computations
        gwork4 = jk >= (klcl - 1)
        gwork1 = gwork4 & gwork2_c

        # Factor for buoyancy at first level above LCL
        zwork6 = jnp.where(jk == (klcl - 1), 0.0, 1.0)

        # ===== Estimate condensate at level k+1 =====
        zut, zurv, purc_new, puri_new, zlv, zls, zcph = convect_condens(
            zice, zeps0,
            ppres[:, jkp],
            puthl_c[:, jk],
            purw_c[:, jk],
            purc_c[:, jk],
            puri_c[:, jk],
            pz[:, jkp],
            xtfrz1,
            xtfrz2,
        )

        purc_c = purc_c.at[:, jkp].set(purc_new)
        puri_c = puri_c.at[:, jkp].set(puri_new)

        # Pi and theta_v at k+1
        zpi = (cst.p00 / ppres[:, jkp]) ** zrdocp
        puthv_new = zpi * zut * (1.0 + zepsa * zurv) / (1.0 + purw_c[:, jk])
        puthv_c = puthv_c.at[:, jkp].set(jnp.where(gwork1, puthv_new, puthv_c[:, jkp]))

        # ===== Compute vertical velocity =====
        zwork3 = pz[:, jkp] - pz[:, jk] * zwork6 - (1.0 - zwork6) * pzlcl
        zwork4 = pthv[:, jk] * zwork6 + (1.0 - zwork6) * pthvelcl
        zwork5 = 2.0 * zuw1_c * puer_c[:, jk] / jnp.maximum(0.1, pumf_c[:, jk])

        buoyancy = ((puthv_c[:, jk] + puthv_c[:, jkp]) / (zwork4 + pthv[:, jkp]) - 1.0)
        zuw2 = zuw1_c + zwork3 * xnhgam * cst.g * buoyancy - zwork5

        # ===== Update properties (no precipitation) =====
        purw_c = purw_c.at[:, jkp].set(jnp.where(gwork1, purw_c[:, jk], purw_c[:, jkp]))
        puthl_c = puthl_c.at[:, jkp].set(jnp.where(gwork1, puthl_c[:, jk], puthl_c[:, jkp]))
        zuw1_new = jnp.where(gwork1, zuw2, zuw1_c)

        # ===== Compute entrainment/detrainment =====
        # Critical mixed fraction
        zmixf = jnp.full(nit, 0.1)
        zwork1_mix = zmixf * pthl[:, jkp] + (1.0 - zmixf) * puthl_c[:, jkp]
        zwork2_mix = zmixf * prw[:, jkp] + (1.0 - zmixf) * purw_c[:, jkp]

        zut_mix, zwork3_rv, zwork4_rc, zwork5_ri, _, _, _ = convect_condens(
            zice, zeps0, ppres[:, jkp], zwork1_mix, zwork2_mix,
            purc_c[:, jkp], puri_c[:, jkp], pz[:, jkp], xtfrz1, xtfrz2,
        )

        zwork3_thv = (zut_mix * zpi * (1.0 + zepsa * (zwork2_mix - zwork4_rc - zwork5_ri)) /
                     (1.0 + zwork2_mix))

        zmixf = jnp.maximum(0.0, puthv_c[:, jkp] - pthv[:, jkp]) * zmixf / \
                jnp.maximum(1e-10, puthv_c[:, jkp] - zwork3_thv)
        zmixf = jnp.clip(zmixf, 0.0, 1.0)

        ze2, zd2 = convect_mixing_funct(zmixf, kmf=1)
        ze2 = jnp.minimum(zd2, jnp.maximum(0.3, ze2))

        zwork1_rate = xentr * cst.g / xcrad * pumf_c[:, jk] * (pz[:, jkp] - pz[:, jk])
        zwork2_mask = jnp.where(gwork1, 1.0, 0.0)

        puer_new = jnp.where(
            puthv_c[:, jkp] > pthv[:, jkp],
            0.5 * zwork1_rate * (ze1_c + ze2) * zwork2_mask,
            0.0
        )
        pudr_new = jnp.where(
            puthv_c[:, jkp] > pthv[:, jkp],
            0.5 * zwork1_rate * (zd1_c + zd2) * zwork2_mask,
            zwork1_rate * zwork2_mask
        )

        puer_c = puer_c.at[:, jkp].set(puer_new)
        pudr_c = pudr_c.at[:, jkp].set(pudr_new)

        # ETL determination
        ketl_new = jnp.where(
            (puthv_c[:, jkp] > pthv[:, jkp]) & (jk > klcl + 1) & gwork1,
            jkp, ketl_c
        )

        # CTL determination
        gwork2_new = gwork2_c & (pumf_c[:, jk] - pudr_c[:, jkp] > 10.0) & (zuw2 > 0.0)
        kctl_new = jnp.where(gwork2_new, jkp, kctl_c)
        gwork1 = gwork2_new & gwork4

        # ===== CAPE computation =====
        zwork3_cape = pz[:, jkp] - pz[:, jk] * zwork6 - (1.0 - zwork6) * pzlcl
        zwork2_thes = (pthes[:, jk] + (1.0 - zwork6) *
                      (pthes[:, jkp] - pthes[:, jk]) /
                      jnp.maximum(1e-10, pz[:, jkp] - pz[:, jk]) * (pzlcl - pz[:, jk]))

        zwork1_cape = 2.0 * ztheul / (zwork2_thes + pthes[:, jkp] + 1e-10) - 1.0
        pcape_new = pcape_c + cst.g * zwork3_cape * jnp.maximum(0.0, zwork1_cape)
        pcape_c = jnp.where(gwork1, pcape_new, pcape_c)

        # ===== Update mass flux and thermodynamics =====
        pumf_new = jnp.maximum(0.1, pumf_c[:, jk] - pudr_c[:, jkp] + puer_c[:, jkp])
        pumf_c = pumf_c.at[:, jkp].set(jnp.where(gwork1, pumf_new, pumf_c[:, jkp]))

        puthl_new = ((pumf_c[:, jk] * puthl_c[:, jk] + puer_c[:, jkp] * pthl[:, jk] -
                     pudr_c[:, jkp] * puthl_c[:, jk]) / pumf_c[:, jkp])
        puthl_c = puthl_c.at[:, jkp].set(jnp.where(gwork1, puthl_new, puthl_c[:, jkp]))

        purw_new = ((pumf_c[:, jk] * purw_c[:, jk] + puer_c[:, jkp] * prw[:, jk] -
                    pudr_c[:, jkp] * purw_c[:, jk]) / pumf_c[:, jkp])
        purw_c = purw_c.at[:, jkp].set(jnp.where(gwork1, purw_new, purw_c[:, jkp]))

        ze1_new = jnp.where(gwork1, ze2, ze1_c)
        zd1_new = jnp.where(gwork1, zd2, zd1_c)

        return ((pumf_c, puer_c, pudr_c, puthl_c, puthv_c, purw_c, purc_c, puri_c,
                pcape_c, kctl_new, ketl_new, zuw1_new, ze1_new, zd1_new, gwork2_new), None)

    # Execute main updraft loop
    initial_carry = (pumf, puer, pudr, puthl, puthv, purw, purc, puri,
                    pcape, kctl, ketl, zuw1, ze1, zd1, gwork2)

    (pumf, puer, pudr, puthl, puthv, purw, purc, puri,
     pcape, kctl, ketl, _, _, _, _), _ = lax.scan(
        updraft_loop_body,
        initial_carry,
        jnp.arange(ikb + 1, ike)
    )

    # ===== Post-processing =====

    # Check trigger condition
    idx_i = jnp.arange(nit)
    kctl_safe = jnp.clip(kctl, 0, nkt - 1)
    zwork1_depth = pz[idx_i, kctl_safe] - pzlcl
    otrig_check = (zwork1_depth >= xcdepth) & (zwork1_depth < xcdepth_d) & (pcape > 1.0)
    otrig_out = otrig & otrig_check
    kctl = jnp.where(~otrig_out, ikb, kctl)

    # Ensure ETL bounds
    ketl = jnp.maximum(ketl, klcl + 2)
    ketl = jnp.minimum(ketl, kctl)

    # ===== Adjust when ETL == CTL =====
    zwork1_etl = jnp.where(ketl == kctl, 1.0, 0.0)

    # Vectorized adjustment at ETL
    ketl_safe = jnp.clip(ketl, 0, nkt - 1)
    kctl_safe_p1 = jnp.clip(kctl + 1, 0, nkt - 1)

    pudr_etl = pudr[idx_i, ketl_safe] + (pumf[idx_i, ketl_safe] - puer[idx_i, ketl_safe]) * zwork1_etl
    puer_etl = puer[idx_i, ketl_safe] * (1.0 - zwork1_etl)
    pumf_etl = pumf[idx_i, ketl_safe] * (1.0 - zwork1_etl)

    pudr = pudr.at[idx_i, ketl_safe].set(pudr_etl)
    puer = puer.at[idx_i, ketl_safe].set(puer_etl)
    pumf = pumf.at[idx_i, ketl_safe].set(pumf_etl)

    # Zero at CTL+1
    puer = puer.at[idx_i, kctl_safe_p1].set(0.0)
    pudr = pudr.at[idx_i, kctl_safe_p1].set(0.0)
    purw = purw.at[idx_i, kctl_safe_p1].set(0.0)
    purc = purc.at[idx_i, kctl_safe_p1].set(0.0)
    puri = puri.at[idx_i, kctl_safe_p1].set(0.0)
    puthl = puthl.at[idx_i, kctl_safe_p1].set(0.0)

    kctl_safe_p2 = jnp.clip(kctl + 2, 0, nkt - 1)
    purc = purc.at[idx_i, kctl_safe_p2].set(0.0)
    puri = puri.at[idx_i, kctl_safe_p2].set(0.0)

    # ===== Linear decrease between ETL and CTL =====
    def compute_linear_decrease(carry, jk):
        """Compute pressure depth between ETL and CTL."""
        zwork1_sum = carry
        mask = (jk > ketl) & (jk <= kctl)
        zwork1_new = jnp.where(mask, zwork1_sum + pdpres[:, jk], zwork1_sum)
        return zwork1_new, None

    zwork1_pdepth, _ = lax.scan(compute_linear_decrease, jnp.zeros(nit), jnp.arange(ikb, ike + 1))

    # Detrainment rate per unit pressure
    zwork1_rate = pumf[idx_i, ketl_safe] / jnp.maximum(1.0, zwork1_pdepth)

    def apply_linear_decrease(carry, jk):
        """Apply linear detrainment between ETL and CTL."""
        pudr_c, pumf_c = carry
        jkp = jk - 1
        jkp_safe = jnp.clip(jkp, 0, nkt - 1)
        mask = (jk > ketl) & (jk <= kctl)

        pudr_jk = jnp.where(mask, pdpres[:, jk] * zwork1_rate, pudr_c[:, jk])
        pumf_jk = jnp.where(mask, pumf_c[idx_i, jkp_safe] - pudr_jk, pumf_c[:, jk])

        pudr_c = pudr_c.at[:, jk].set(pudr_jk)
        pumf_c = pumf_c.at[:, jk].set(pumf_jk)

        return (pudr_c, pumf_c), None

    (pudr, pumf), _ = lax.scan(apply_linear_decrease, (pudr, pumf), jnp.arange(ikb + 1, ike + 1))

    # ===== Source layer mass flux =====
    iwork = kpbl
    iwork_safe = jnp.clip(iwork, 0, nkt - 1)
    kdpl_safe = jnp.clip(kdpl, 0, nkt - 1)

    zwork2_depth = ppres[idx_i, kdpl_safe] - ppres[idx_i, iwork_safe] + pdpres[idx_i, kdpl_safe]

    def set_source_layer(carry, jk):
        """Set mass flux in source layer."""
        puer_c, pumf_c = carry
        jkm = jk - 1
        jkm_safe = jnp.clip(jkm, 0, nkt - 1)
        mask = (jk >= kdpl) & (jk <= iwork) & gtrig1

        puer_add = pmflcl * pdpres[:, jk] / (zwork2_depth + 0.1)
        puer_jk = jnp.where(mask, puer_c[:, jk] + puer_add, puer_c[:, jk])
        pumf_jk = jnp.where(mask & (jk > ikb), pumf_c[idx_i, jkm_safe] + puer_jk, pumf_c[:, jk])

        puer_c = puer_c.at[:, jk].set(puer_jk)
        pumf_c = pumf_c.at[:, jk].set(pumf_jk)

        return (puer_c, pumf_c), None

    (puer, pumf), _ = lax.scan(set_source_layer, (puer, pumf), jnp.arange(ikb, ike + 1))

    # ===== Zero out if not triggered =====
    otrig_2d = otrig_out[:, None]  # Broadcast to (nit, 1)

    pumf = jnp.where(otrig_2d, pumf, 0.0)
    pudr = jnp.where(otrig_2d, pudr, 0.0)
    puer = jnp.where(otrig_2d, puer, 0.0)
    puthl = jnp.where(otrig_2d, puthl, pthl)
    purw = jnp.where(otrig_2d, purw, prw)
    purc = jnp.where(otrig_2d, purc, 0.0)
    puri = jnp.where(otrig_2d, puri, 0.0)

    return UpdraftOutputs(
        pumf=pumf,
        puer=puer,
        pudr=pudr,
        puthl=puthl,
        puthv=puthv,
        purw=purw,
        purc=purc,
        puri=puri,
        pcape=pcape,
        kctl=kctl,
        ketl=ketl,
        otrig=otrig_out,
    )
