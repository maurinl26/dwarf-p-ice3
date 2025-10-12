"""Shallow convection updraft computations.

Translation of convect_updraft_shal.F90 to JAX.
Computes updraft properties from departure level (DPL) to cloud top level (CTL).
"""

import jax.numpy as jnp
from jax import Array

from ice3.phyex_common.constants import Constants
from ice3.phyex_common.phyex import PHYEX
from .convpar_shal import CONVPAR_SHAL
from .condens import convect_condens
from .mixing_funct import convect_mixing_funct


def convect_updraft_shal(
    cvp_shal: CONVPAR_SHAL,
    cst: Constants,
    phyex: PHYEX,
    kice: int,
    ppres: Array,  # (nit, nkt) - pressure
    pdpres: Array,  # (nit, nkt) - pressure difference
    pz: Array,  # (nit, nkt) - height
    pthl: Array,  # (nit, nkt) - grid scale enthalpy
    pthv: Array,  # (nit, nkt) - grid scale theta_v
    pthes: Array,  # (nit, nkt) - grid scale saturated theta_e
    prw: Array,  # (nit, nkt) - grid scale total water
    pthlcl: Array,  # (nit,) - theta at LCL
    ptlcl: Array,  # (nit,) - temp at LCL
    prvlcl: Array,  # (nit,) - vapor mixing ratio at LCL
    pwlcl: Array,  # (nit,) - parcel velocity at LCL
    pzlcl: Array,  # (nit,) - height at LCL
    pthvelcl: Array,  # (nit,) - environm. theta_v at LCL
    pmflcl: float,  # cloud base unit mass flux
    klcl: Array,  # (nit,) - LCL index
    kdpl: Array,  # (nit,) - DPL index
    kpbl: Array,  # (nit,) - PBL top index
    gtrig1: Array,  # (nit,) - logical mask for convection
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Compute updraft properties from DPL to CTL.
    
    This is a simplified JAX-compatible version of the Fortran routine.
    Some optimizations from the original (like loop tiling) are simplified.
    
    Args:
        cvp_shal: Shallow convection parameters
        cst: Physical constants
        phyex: PHYEX dimensions
        kice: Flag for ice (1=yes, 0=no)
        ppres: Pressure at model levels (Pa), shape (nit, nkt)
        pdpres: Pressure difference between layers (Pa), shape (nit, nkt)
        pz: Height of model levels (m), shape (nit, nkt)
        pthl: Grid scale enthalpy (J/kg), shape (nit, nkt)
        pthv: Grid scale theta_v (K), shape (nit, nkt)
        pthes: Grid scale saturated theta_e (K), shape (nit, nkt)
        prw: Grid scale total water mixing ratio (kg/kg), shape (nit, nkt)
        pthlcl: Theta at LCL (K), shape (nit,)
        ptlcl: Temperature at LCL (K), shape (nit,)
        prvlcl: Vapor mixing ratio at LCL (kg/kg), shape (nit,)
        pwlcl: Parcel velocity at LCL (m/s), shape (nit,)
        pzlcl: Height at LCL (m), shape (nit,)
        pthvelcl: Environmental theta_v at LCL (K), shape (nit,)
        pmflcl: Cloud base unit mass flux (kg/s)
        klcl: LCL indices, shape (nit,)
        kdpl: DPL indices, shape (nit,)
        kpbl: PBL top indices, shape (nit,)
        gtrig1: Logical mask for convection, shape (nit,)
        
    Returns:
        Tuple of (pumf, puer, pudr, puthl, puthv, purw, purc, puri, pcape, kctl, ketl, otrig):
            pumf: Updraft mass flux (kg/s), shape (nit, nkt)
            puer: Updraft entrainment (kg/s), shape (nit, nkt)
            pudr: Updraft detrainment (kg/s), shape (nit, nkt)
            puthl: Updraft enthalpy (J/kg), shape (nit, nkt)
            puthv: Updraft theta_v (K), shape (nit, nkt)
            purw: Updraft total water (kg/kg), shape (nit, nkt)
            purc: Updraft cloud water (kg/kg), shape (nit, nkt)
            puri: Updraft cloud ice (kg/kg), shape (nit, nkt)
            pcape: Available potential energy (J/kg), shape (nit,)
            kctl: Cloud top level indices, shape (nit,)
            ketl: Equilibrium level indices, shape (nit,)
            otrig: Trigger mask for convection, shape (nit,)
            
    Reference:
        Book1,2 of PHYEX documentation (routine CONVECT_UPDRAFT)
        Kain and Fritsch, 1990, J. Atmos. Sci.
        Original: P. BECHTOLD, Laboratoire d'Aerologie, 07/11/95
    """
    nit, nkt = ppres.shape
    ikb = phyex.kib
    ike = nkt - phyex.kte
    
    # Constants
    zepsa = cst.XRV / cst.XRD
    zrdocp = cst.XRD / cst.XCPD
    zice = float(kice)
    zeps0 = cst.XRD / cst.XRV
    
    # Initialize output arrays
    pumf = jnp.zeros((nit, nkt))
    puer = jnp.zeros((nit, nkt))
    pudr = jnp.zeros((nit, nkt))
    puthl = jnp.zeros((nit, nkt))
    puthv = jnp.zeros((nit, nkt))
    purw = jnp.zeros((nit, nkt))
    purc = jnp.zeros((nit, nkt))
    puri = jnp.zeros((nit, nkt))
    pcape = jnp.zeros(nit)
    kctl = jnp.ones(nit, dtype=jnp.int32) * ikb
    ketl = klcl.copy()
    
    # Work arrays
    zuw1 = pwlcl * pwlcl
    zuw2 = jnp.zeros(nit)
    ze1 = jnp.zeros(nit)
    zd1 = jnp.zeros(nit)
    gwork2 = jnp.ones(nit, dtype=bool)
    
    # Compute undilute updraft theta_e for CAPE (Bolton 1980)
    ztheul = (ptlcl * (pthlcl / ptlcl) ** (1.0 - 0.28 * prvlcl) *
              jnp.exp((3374.6525 / ptlcl - 2.5403) * prvlcl * (1.0 + 0.81 * prvlcl)))
    
    # Define accurate enthalpy for updraft
    zwork1 = ((cst.XCPD + prvlcl * cst.XCPV) * ptlcl +
              (1.0 + prvlcl) * cst.XG * pzlcl)
    
    # Set updraft properties between DPL and LCL
    for jk in range(ikb, ike):
        mask = (jk >= kdpl) & (jk < klcl)
        pumf = pumf.at[:, jk].set(jnp.where(mask, pmflcl, pumf[:, jk]))
        puthl = puthl.at[:, jk].set(jnp.where(mask, zwork1, puthl[:, jk]))
        puthv_val = pthlcl * (1.0 + zepsa * prvlcl) / (1.0 + prvlcl)
        puthv = puthv.at[:, jk].set(jnp.where(mask, puthv_val, puthv[:, jk]))
        purw = purw.at[:, jk].set(jnp.where(mask, prvlcl, purw[:, jk]))
    
    # Main updraft loop
    for jk in range(ikb + 1, ike - 1):
        jkp = jk + 1
        
        # Determine which points to process
        gwork4 = jk >= (klcl - 1)
        gwork1 = gwork4 & gwork2
        
        # Factor for first level above LCL
        zwork6 = jnp.where(jk == (klcl - 1), 0.0, 1.0)
        
        # Estimate condensate at level k+1
        pt, pew, prc_new, pri_new, plv, pls, pcph = convect_condens(
            cst, zice, zeps0,
            ppres[:, jkp], puthl[:, jk], purw[:, jk],
            purc[:, jk], puri[:, jk], pz[:, jkp],
            cvp_shal.XTFRZ1, cvp_shal.XTFRZ2
        )
        
        purc = purc.at[:, jkp].set(prc_new)
        puri = puri.at[:, jkp].set(pri_new)
        
        # Compute updraft theta_v
        zpi = (cst.XP00 / ppres[:, jkp]) ** zrdocp
        zurv = pew  # Using saturation value from condens
        puthv_new = zpi * pt * (1.0 + zepsa * zurv) / (1.0 + purw[:, jk])
        puthv = puthv.at[:, jkp].set(jnp.where(gwork1, puthv_new, puthv[:, jkp]))
        
        # Compute vertical velocity squared
        zwork3 = pz[:, jkp] - pz[:, jk] * zwork6 - (1.0 - zwork6) * pzlcl
        zwork4 = pthv[:, jk] * zwork6 + (1.0 - zwork6) * pthvelcl
        zwork5 = 2.0 * zuw1 * puer[:, jk] / jnp.maximum(0.1, pumf[:, jk])
        
        buoyancy = ((puthv[:, jk] + puthv[:, jkp]) / (zwork4 + pthv[:, jkp]) - 1.0)
        zuw2_new = (zuw1 + zwork3 * cvp_shal.XNHGAM * cst.XG * buoyancy - zwork5)
        zuw2 = jnp.where(gwork1, zuw2_new, zuw2)
        
        # Update r_c, r_i, enthalpy, r_w
        purw = purw.at[:, jkp].set(jnp.where(gwork1, purw[:, jk], purw[:, jkp]))
        puthl = puthl.at[:, jkp].set(jnp.where(gwork1, puthl[:, jk], puthl[:, jkp]))
        
        # Update vertical velocity
        zuw1 = jnp.where(gwork1, zuw2, zuw1)
        
        # Compute critical mixed fraction for entrainment/detrainment
        zmixf = jnp.ones(nit) * 0.1
        zwork1_mix = zmixf * pthl[:, jkp] + (1.0 - zmixf) * puthl[:, jkp]
        zwork2_mix = zmixf * prw[:, jkp] + (1.0 - zmixf) * purw[:, jkp]
        
        # Compute theta_v of mixture
        pt_mix, _, prc_mix, pri_mix, plv_mix, pls_mix, pcph_mix = convect_condens(
            cst, zice, zeps0,
            ppres[:, jkp], zwork1_mix, zwork2_mix,
            purc[:, jkp], puri[:, jkp], pz[:, jkp],
            cvp_shal.XTFRZ1, cvp_shal.XTFRZ2
        )
        
        zwork3_mix = (pt_mix * zpi * (1.0 + zepsa * (zwork2_mix - prc_mix - pri_mix)) /
                      (1.0 + zwork2_mix))
        
        # Compute final critical mixed fraction
        zmixf = (jnp.maximum(0.0, puthv[:, jkp] - pthv[:, jkp]) * zmixf /
                 (puthv[:, jkp] - zwork3_mix + 1.0e-10))
        zmixf = jnp.maximum(0.0, jnp.minimum(1.0, zmixf))
        
        # Compute entrainment and detrainment
        ze2, zd2 = convect_mixing_funct(zmixf, kmf=1)
        ze2 = jnp.minimum(zd2, jnp.maximum(0.3, ze2))
        
        # Rate of environmental inflow
        zwork1_entr = cvp_shal.XENTR * cst.XG / cvp_shal.XCRAD * pumf[:, jk] * (pz[:, jkp] - pz[:, jk])
        
        # Apply entrainment/detrainment based on buoyancy
        is_buoyant = puthv[:, jkp] > pthv[:, jkp]
        puer_new = jnp.where(is_buoyant & gwork1,
                             0.5 * zwork1_entr * (ze1 + ze2),
                             0.0)
        pudr_new = jnp.where(is_buoyant & gwork1,
                             0.5 * zwork1_entr * (zd1 + zd2),
                             jnp.where(gwork1, zwork1_entr, 0.0))
        
        puer = puer.at[:, jkp].set(puer_new)
        pudr = pudr.at[:, jkp].set(pudr_new)
        
        # Update equilibrium temperature level
        is_etl = (puthv[:, jkp] > pthv[:, jkp]) & (jk > klcl + 1) & gwork1
        ketl = jnp.where(is_etl, jkp, ketl)
        
        # Check if CTL is reached
        gwork2_new = (pumf[:, jk] - pudr[:, jkp] > 10.0) & (zuw2 > 0.0)
        gwork2 = jnp.where(gwork1, gwork2_new, gwork2)
        kctl = jnp.where(gwork2, jkp, kctl)
        
        # Compute CAPE
        zwork2_cape = pthes[:, jk] + (1.0 - zwork6) * (pthes[:, jkp] - pthes[:, jk]) / \
                      (pz[:, jkp] - pz[:, jk]) * (pzlcl - pz[:, jk])
        zwork1_cape = (2.0 * ztheul) / (zwork2_cape + pthes[:, jkp]) - 1.0
        pcape_add = cst.XG * zwork3 * jnp.maximum(0.0, zwork1_cape)
        pcape = jnp.where(gwork1, pcape + pcape_add, pcape)
        
        # Update mass flux
        pumf_new = pumf[:, jk] - pudr[:, jkp] + puer[:, jkp]
        pumf_new = jnp.maximum(pumf_new, 0.1)
        pumf = pumf.at[:, jkp].set(jnp.where(gwork1, pumf_new, pumf[:, jkp]))
        
        # Update enthalpy and total water
        puthl_new = ((pumf[:, jk] * puthl[:, jk] + puer[:, jkp] * pthl[:, jk] -
                      pudr[:, jkp] * puthl[:, jk]) / pumf[:, jkp])
        puthl = puthl.at[:, jkp].set(jnp.where(gwork1, puthl_new, puthl[:, jkp]))
        
        purw_new = ((pumf[:, jk] * purw[:, jk] + puer[:, jkp] * prw[:, jk] -
                     pudr[:, jkp] * purw[:, jk]) / pumf[:, jkp])
        purw = purw.at[:, jkp].set(jnp.where(gwork1, purw_new, purw[:, jkp]))
        
        # Update fractional entrainment/detrainment
        ze1 = jnp.where(gwork1, ze2, ze1)
        zd1 = jnp.where(gwork1, zd2, zd1)
    
    # Check cloud thickness and CAPE criteria
    jk_ctl = kctl
    zwork1_thick = pz[jnp.arange(nit), jk_ctl] - pzlcl
    otrig = ((zwork1_thick >= cvp_shal.XCDEPTH) &
             (zwork1_thick < cvp_shal.XCDEPTH_D) &
             (pcape > 1.0))
    
    # Reset outputs where convection is not triggered
    kctl = jnp.where(~otrig, ikb, kctl)
    ketl = jnp.maximum(ketl, klcl + 2)
    ketl = jnp.minimum(ketl, kctl)
    
    # Handle case where ETL == CTL
    is_same = ketl == kctl
    jk_etl = ketl
    pudr_add = jnp.where(is_same, pumf[jnp.arange(nit), jk_etl] - puer[jnp.arange(nit), jk_etl], 0.0)
    for jk in range(ikb, ike):
        mask = (jk == ketl) & is_same
        pudr = pudr.at[:, jk].add(jnp.where(mask, pudr_add, 0.0))
        puer = puer.at[:, jk].set(jnp.where(mask, 0.0, puer[:, jk]))
        pumf = pumf.at[:, jk].set(jnp.where(mask, 0.0, pumf[:, jk]))
    
    # Set mass flux in source layer (linear increase)
    iwork = kpbl
    zwork2 = ppres[jnp.arange(nit), kdpl] - ppres[jnp.arange(nit), iwork] + pdpres[jnp.arange(nit), kdpl]
    
    for jk in range(ikb, ike):
        mask = (jk >= kdpl) & (jk <= iwork) & gtrig1
        puer_add = pmflcl * pdpres[:, jk] / (zwork2 + 0.1)
        puer = puer.at[:, jk].add(jnp.where(mask, puer_add, 0.0))
        if jk > ikb:
            pumf = pumf.at[:, jk].set(jnp.where(mask, pumf[:, jk-1] + puer[:, jk], pumf[:, jk]))
    
    # Zero out where convection not triggered
    for jk in range(nkt):
        pumf = pumf.at[:, jk].set(jnp.where(otrig, pumf[:, jk], 0.0))
        puer = puer.at[:, jk].set(jnp.where(otrig, puer[:, jk], 0.0))
        pudr = pudr.at[:, jk].set(jnp.where(otrig, pudr[:, jk], 0.0))
        puthl = puthl.at[:, jk].set(jnp.where(otrig, puthl[:, jk], pthl[:, jk]))
        purw = purw.at[:, jk].set(jnp.where(otrig, purw[:, jk], prw[:, jk]))
        purc = purc.at[:, jk].set(jnp.where(otrig, purc[:, jk], 0.0))
        puri = puri.at[:, jk].set(jnp.where(otrig, puri[:, jk], 0.0))
    
    return pumf, puer, pudr, puthl, puthv, purw, purc, puri, pcape, kctl, ketl, otrig
