"""Shallow convection trigger function.

Translation of convect_trigger_shal.F90 to JAX.
Determines convective columns and cloudy values at the lifting condensation level (LCL).
"""

import jax.numpy as jnp
from jax import Array

from ice3.phyex_common.constants import Constants
from ice3.phyex_common.phyex import PHYEX
from .convpar_shal import CONVPAR_SHAL
from .satmixratio import convect_satmixratio


def convect_trigger_shal(
    cvp_shal: CONVPAR_SHAL,
    cst: Constants,
    phyex: PHYEX,
    ppres: Array,  # (nit, nkt) - pressure at model levels
    pth: Array,  # (nit, nkt) - potential temperature
    pthv: Array,  # (nit, nkt) - virtual potential temperature
    pthes: Array,  # (nit, nkt) - saturated equivalent potential temperature
    prv: Array,  # (nit, nkt) - water vapor mixing ratio
    pw: Array,  # (nit, nkt) - vertical velocity
    pz: Array,  # (nit, nkt) - height of model levels
    ptkecls: Array,  # (nit,) - TKE in the cloud layer scheme
    kdpl: Array,  # (nit,) - departure level index
    kpbl: Array,  # (nit,) - PBL top index
    klcl: Array,  # (nit,) - LCL index
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Determine convective columns and properties at LCL.
    
    This routine determines which grid points have active shallow convection
    by computing mixed layer properties and checking stability criteria.
    
    Args:
        cvp_shal: Shallow convection parameters
        cst: Physical constants
        phyex: PHYEX dimensions and configuration
        ppres: Pressure at model levels (Pa), shape (nit, nkt)
        pth: Potential temperature (K), shape (nit, nkt)
        pthv: Virtual potential temperature (K), shape (nit, nkt)
        pthes: Saturated equivalent potential temperature (K), shape (nit, nkt)
        prv: Water vapor mixing ratio (kg/kg), shape (nit, nkt)
        pw: Vertical velocity (m/s), shape (nit, nkt)
        pz: Height of model levels (m), shape (nit, nkt)
        ptkecls: TKE in cloud layer scheme (m^2/s^2), shape (nit,)
        kdpl: Departure level indices, shape (nit,)
        kpbl: PBL top indices, shape (nit,)
        klcl: LCL indices, shape (nit,)
        
    Returns:
        Tuple of (pthlcl, ptlcl, prvlcl, pwlcl, pzlcl, pthvelcl, klcl_out, kdpl_out, kpbl_out, otrig):
            pthlcl: Potential temperature at LCL (K), shape (nit,)
            ptlcl: Temperature at LCL (K), shape (nit,)
            prvlcl: Vapor mixing ratio at LCL (kg/kg), shape (nit,)
            pwlcl: Parcel velocity at LCL (m/s), shape (nit,)
            pzlcl: Height at LCL (m), shape (nit,)
            pthvelcl: Environmental virtual potential temperature at LCL (K), shape (nit,)
            klcl_out: Updated LCL indices, shape (nit,)
            kdpl_out: Updated departure level indices, shape (nit,)
            kpbl_out: Updated PBL top indices, shape (nit,)
            otrig: Trigger mask for convection (bool), shape (nit,)
            
    Reference:
        Book2 of PHYEX documentation (routine TRIGGER_FUNCT)
        Fritsch and Chappell (1980), J. Atm. Sci., Vol. 37, 1722-1761.
        Original: P. BECHTOLD, Laboratoire d'Aerologie, 07/11/95
    """
    nit, nkt = ppres.shape
    ikb = phyex.kib
    ike = nkt - phyex.kte
    
    # Constants
    zeps = cst.XRD / cst.XRV
    zepsa = cst.XRV / cst.XRD
    zcpord = cst.XCPD / cst.XRD
    zrdocp = cst.XRD / cst.XCPD
    
    # Initialize output arrays
    pthlcl = jnp.ones(nit)
    ptlcl = jnp.ones(nit)
    prvlcl = jnp.zeros(nit)
    pwlcl = jnp.zeros(nit)
    pzlcl = pz[:, ikb]
    pthvelcl = jnp.ones(nit)
    otrig = jnp.zeros(nit, dtype=bool)
    
    # Working arrays
    idpl = kdpl.copy()
    ipbl = kpbl.copy()
    ilcl = klcl.copy()
    zzdpl = pz[:, ikb]
    gtrig2 = jnp.ones(nit, dtype=bool)
    
    # Auxiliary arrays for efficiency
    zzzx1 = jnp.zeros_like(ppres)
    zzzx1 = zzzx1.at[:, ikb+1:ike-1].set(ppres[:, ikb+1:ike-1] - ppres[:, ikb+2:ike])
    zzppres = ppres * zzzx1
    zzpth = pth * zzzx1
    zzprv = jnp.maximum(0.0, prv) * zzzx1
    
    # Initialize mixed layer variables
    zthlcl = jnp.zeros(nit)
    zrvlcl = jnp.zeros(nit)
    ztlcl = jnp.zeros(nit)
    zzlcl = jnp.zeros(nit)
    zthvelcl = jnp.zeros(nit)
    zdpthmix = jnp.zeros(nit)
    zpresmix = jnp.zeros(nit)
    zthvlcl = jnp.zeros(nit)
    zwlcl = jnp.zeros(nit)
    
    # Main loop over vertical levels for trigger test
    # Note: This is a simplified JAX-compatible version of the Fortran nested loops
    # For full JIT compilation, this would need to be refactored using lax.scan
    for jkk in range(ikb + 1, ike - 2):
        # Check if we should continue processing this level
        gwork1 = (zzdpl - pz[:, ikb]) < cvp_shal.XZLCL
        
        # Reset mixed layer variables where we start new processing
        zdpthmix = jnp.where(gwork1, 0.0, zdpthmix)
        zpresmix = jnp.where(gwork1, 0.0, zpresmix)
        zthlcl = jnp.where(gwork1, 0.0, zthlcl)
        zrvlcl = jnp.where(gwork1, 0.0, zrvlcl)
        zzdpl = jnp.where(gwork1, pz[:, jkk], zzdpl)
        idpl = jnp.where(gwork1, jkk, idpl)
        
        # Construct mixed layer of at least XZPBL (50 hPa depth)
        for jk in range(jkk, ike - 1):
            gwork_pbl = gwork1 & (zdpthmix < cvp_shal.XZPBL)
            ipbl = jnp.where(gwork_pbl, jk, ipbl)
            zdpthmix = jnp.where(gwork_pbl, zdpthmix + zzzx1[:, jk], zdpthmix)
            zpresmix = jnp.where(gwork_pbl, zpresmix + zzppres[:, jk], zpresmix)
            zthlcl = jnp.where(gwork_pbl, zthlcl + zzpth[:, jk], zthlcl)
            zrvlcl = jnp.where(gwork_pbl, zrvlcl + zzprv[:, jk], zrvlcl)
        
        # Normalize mixed layer values and add temperature perturbation
        zpresmix = jnp.where(gwork1, zpresmix / zdpthmix, zpresmix)
        temp_pert = (cvp_shal.XATPERT * jnp.minimum(3.0, ptkecls) / cst.XCPD + 
                     cvp_shal.XBTPERT) * cvp_shal.XDTPERT
        zthlcl = jnp.where(gwork1, zthlcl / zdpthmix + temp_pert, zthlcl)
        zrvlcl = jnp.where(gwork1, zrvlcl / zdpthmix, zrvlcl)
        zthvlcl = jnp.where(gwork1, 
                           zthlcl * (1.0 + zepsa * zrvlcl) / (1.0 + zrvlcl),
                           zthvlcl)
        
        # Determine temperature and pressure at LCL using Bolton formula
        ztmix = zthlcl * (zpresmix / cst.XP00) ** zrdocp
        zevmix = zrvlcl * zpresmix / (zrvlcl + zeps)
        zevmix = jnp.maximum(1.0e-8, zevmix)
        
        # Compute dewpoint temperature
        zx1 = jnp.log(zevmix / 613.3)
        zx1 = (4780.8 - 32.19 * zx1) / (17.502 - zx1)
        
        # Compute adiabatic saturation temperature (LCL temperature)
        ztlcl_new = zx1 - (0.212 + 1.571e-3 * (zx1 - cst.XTT) -
                           4.36e-4 * (ztmix - cst.XTT)) * (ztmix - zx1)
        ztlcl = jnp.where(gwork1, jnp.minimum(ztlcl_new, ztmix), ztlcl)
        
        # Compute pressure at LCL
        zplcl = cst.XP00 * (ztlcl / zthlcl) ** zcpord
        
        # Compute saturation mixing ratio and related quantities
        zewa, zlva, zlsa, zcpha = convect_satmixratio(zplcl, ztlcl, cst)
        zewb, zlvb, zlsb, zcphb = convect_satmixratio(zpresmix, ztmix, cst)
        
        # Correct ZTLCL to be consistent with MNH saturation formula
        zlsa_temp = zewa / ztlcl * (cst.XBETAW / ztlcl - cst.XGAMW)
        zlsa_corr = (zewa - zrvlcl) / (1.0 + zlva / zcpha * zlsa_temp)
        ztlcl = jnp.where(gwork1, ztlcl - zlva / zcpha * zlsa_corr, ztlcl)
        
        # Handle oversaturated case
        is_oversat = gwork1 & (zrvlcl > zewb)
        zlsb_temp = zewb / ztmix * (cst.XBETAW / ztmix - cst.XGAMW)
        zlsb_corr = (zewb - zrvlcl) / (1.0 + zlvb / zcphb * zlsb_temp)
        ztlcl = jnp.where(is_oversat, ztmix - zlvb / zcphb * zlsb_corr, ztlcl)
        zrvlcl = jnp.where(is_oversat, zrvlcl - zlsb_corr, zrvlcl)
        zplcl = jnp.where(is_oversat, zpresmix, zplcl)
        zthlcl = jnp.where(is_oversat, ztlcl * (cst.XP00 / zplcl) ** zrdocp, zthlcl)
        zthvlcl = jnp.where(is_oversat,
                           zthlcl * (1.0 + zepsa * zrvlcl) / (1.0 + zrvlcl),
                           zthvlcl)
        
        # Determine LCL level index
        for jk in range(jkk, ike - 1):
            condition = (zplcl <= ppres[:, jk]) & gwork1
            ilcl = jnp.where(condition, jk + 1, ilcl)
        
        # Compute height and environmental theta_v at LCL
        jk_lcl = ilcl
        jkm_lcl = ilcl - 1
        
        # Linear interpolation for LCL properties
        zdp = jnp.log(zplcl / ppres[jnp.arange(nit), jkm_lcl]) / \
              jnp.log(ppres[jnp.arange(nit), jk_lcl] / ppres[jnp.arange(nit), jkm_lcl])
        
        zthvelcl_new = (pthv[jnp.arange(nit), jkm_lcl] + 
                       (pthv[jnp.arange(nit), jk_lcl] - pthv[jnp.arange(nit), jkm_lcl]) * zdp)
        zthvelcl = jnp.where(gwork1, zthvelcl_new, zthvelcl)
        
        zzlcl_new = (pz[jnp.arange(nit), jkm_lcl] + 
                    (pz[jnp.arange(nit), jk_lcl] - pz[jnp.arange(nit), jkm_lcl]) * zdp)
        zzlcl = jnp.where(gwork1, zzlcl_new, zzlcl)
        
        # Compute parcel vertical velocity at LCL
        zwlcl_new = cvp_shal.XAW * jnp.maximum(0.0, pw[:, ikb]) + cvp_shal.XBW
        zwlcl = jnp.where(gwork1, zwlcl_new, zwlcl)
        zwlclsqrent = 1.05 * zwlcl * zwlcl
        
        # Compute equivalent potential temperature
        ztheul = (ztlcl * (zthlcl / ztlcl) ** (1.0 - 0.28 * zrvlcl) *
                 jnp.exp((3374.6525 / ztlcl - 2.5403) * zrvlcl * (1.0 + 0.81 * zrvlcl)))
        ztheul = jnp.where(gwork1, ztheul, 0.0)
        
        # Simplified CAPE computation
        # Note: Full Fortran implementation has complex nested loops for CAPE calculation
        # This is a simplified version for core functionality
        zcape = jnp.zeros(nit)
        zcap = jnp.zeros(nit)
        ztop = jnp.zeros(nit)
        zwork3 = jnp.zeros(nit)
        
        # Compute CAPE from LCL to cloud top
        jlclmin = jnp.min(jnp.where(gwork1, ilcl, ike))
        for jl in range(max(ikb, jlclmin - 1), ike - 2):
            jk = jl + 1
            
            # Compute buoyancy term
            zx1 = (2.0 * ztheul / (pthes[:, jk] + pthes[:, jl]) - 1.0) * (pz[:, jk] - pz[:, jl])
            zx1 = jnp.where(jl < ilcl, 0.0, zx1)
            
            zcape = zcape + cst.XG * jnp.maximum(1.0, zx1)
            zcap = zcap + zx1
            
            # Check for cloud top (where buoyancy becomes negative)
            zx2 = jnp.sign(cvp_shal.XNHGAM * cst.XG * zcap + zwlclsqrent)
            zwork3 = jnp.maximum(-1.0, zwork3 + jnp.minimum(0.0, zx2))
            
            # Update cloud top height
            ztopp = ztop
            ztop = pz[:, jl] * 0.5 * (1.0 + zx2) * (1.0 + zwork3) + \
                   ztop * 0.5 * (1.0 - zx2)
            ztop = jnp.maximum(ztop, ztopp)
        
        # Check for sufficient cloud depth and trigger convection
        sufficient_depth = ((ztop - zzlcl) >= cvp_shal.XCDEPTH) & gtrig2 & (zcape > 10.0)
        
        # Update outputs where convection is triggered
        otrig = jnp.where(sufficient_depth, True, otrig)
        gtrig2 = jnp.where(sufficient_depth, False, gtrig2)
        pthlcl = jnp.where(sufficient_depth, zthlcl, pthlcl)
        prvlcl = jnp.where(sufficient_depth, zrvlcl, prvlcl)
        ptlcl = jnp.where(sufficient_depth, ztlcl, ptlcl)
        pwlcl = jnp.where(sufficient_depth, zwlcl, pwlcl)
        pzlcl = jnp.where(sufficient_depth, zzlcl, pzlcl)
        pthvelcl = jnp.where(sufficient_depth, zthvelcl, pthvelcl)
    
    return pthlcl, ptlcl, prvlcl, pwlcl, pzlcl, pthvelcl, ilcl, idpl, ipbl, otrig
