# -*- coding: utf-8 -*-
"""
JAX implementation of ice adjustment stencil for saturation adjustment in mixed-phase clouds.

This module implements the saturation adjustment scheme that maintains
thermodynamic equilibrium between water vapor, liquid cloud droplets, and
ice crystals. It accounts for subgrid-scale variability and includes
autoconversion processes for both liquid and ice condensate.

Translated from GT4Py implementation in src/ice3/stencils/ice_adjust.py
Source: PHYEX/src/common/micro/ice_adjust.F90
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from typing import Dict, Any, Tuple

from ..functions.ice_adjust import (
    sublimation_latent_heat,
    vaporisation_latent_heat,
)
from ..functions.tiwmx import e_sat_i, e_sat_w


def ice_adjust(
    sigqsat: Array,
    pabs: Array,
    sigs: Array,
    th: Array,
    exn: Array,
    exn_ref: Array,
    rho_dry_ref: Array,
    rv: Array,
    ri: Array,
    rc: Array,
    rr: Array,
    rs: Array,
    rg: Array,
    cf_mf: Array,
    rc_mf: Array,
    ri_mf: Array,
    rvs: Array,
    rcs: Array,
    ris: Array,
    ths: Array,
    dt: float,
    constants: Dict[str, Any],
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """
    Perform saturation adjustment for mixed-phase microphysics using JAX.
    
    This function adjusts water vapor, cloud liquid, and cloud ice mixing
    ratios to maintain thermodynamic equilibrium, accounting for subgrid-scale
    variability in relative humidity. It computes cloud fraction and implements
    autoconversion processes for both liquid droplets and ice crystals.
    
    Parameters
    ----------
    sigqsat : Array
        Standard deviation of saturation mixing ratio for subgrid variability.
    pabs : Array
        Absolute pressure (Pa).
    sigs : Array
        Sigma_s for subgrid-scale turbulent mixing.
    th : Array
        Potential temperature (K).
    exn : Array
        Exner function (dimensionless).
    exn_ref : Array
        Reference Exner function for tendency computation.
    rho_dry_ref : Array
        Reference dry air density (kg/m³).
    rv, ri, rc, rr, rs, rg : Array
        Mixing ratios for vapor, ice, cloud, rain, snow, graupel (kg/kg).
    cf_mf : Array
        Cloud fraction from mass flux scheme.
    rc_mf, ri_mf : Array
        Liquid/ice mixing ratios from mass flux scheme (kg/kg).
    rvs, rcs, ris, ths : Array
        Tendency fields for vapor, cloud, ice, and theta (per timestep).
    dt : float
        Time step (s).
    constants : Dict[str, Any]
        Dictionary containing all physical constants and parameters.
        
    Returns
    -------
    Tuple of Arrays:
        t : Array
            Updated temperature (K).
        rv_out : Array
            Output adjusted vapor mixing ratio (kg/kg).
        rc_out : Array
            Output adjusted cloud liquid mixing ratio (kg/kg).
        ri_out : Array
            Output adjusted cloud ice mixing ratio (kg/kg).
        cldfr : Array
            Output cloud fraction (0-1).
        hlc_hrc, hlc_hcf : Array
            Subgrid autoconversion diagnostics for liquid.
        hli_hri, hli_hcf : Array
            Subgrid autoconversion diagnostics for ice.
        cph : Array
            Specific heat capacity of moist air (J/(kg·K)).
        lv, ls : Array
            Latent heats of vaporization and sublimation (J/kg).
        
    Notes
    -----
    Algorithm Steps:
    1. Compute temperature and latent heats
    2. Calculate specific heat for moist air
    3. Apply subgrid condensation scheme (CB02 method)
    4. Compute supersaturation coefficients and cloud fraction
    5. Partition condensate between liquid and ice based on temperature
    6. Update tendencies with energy conservation
    7. Handle subgrid autoconversion for droplets and ice crystals
    
    The scheme uses:
    - CB02 subgrid condensation (Chaboureau and Bechtold, 2002)
    - Statistical cloud scheme with assumed PDF of relative humidity
    - Temperature-dependent ice fraction (FRAC_ICE_ADJUST modes)
    - Subgrid autoconversion with PDF assumptions (None or Triangle)
    
    References
    ----------
    Chaboureau, J.-P., and P. Bechtold, 2002: A simple cloud
    parameterization derived from cloud resolving model data. 
    J. Atmos. Sci., 59, 2362-2372.
    """
    # Extract constants
    NRR = int(constants["NRR"])
    CPD = constants["CPD"]
    CPV = constants["CPV"]
    CL = constants["CL"]
    CI = constants["CI"]
    RD = constants["RD"]
    RV = constants["RV"]
    TT = constants["TT"]
    TMAXMIX = constants["TMAXMIX"]
    TMINMIX = constants["TMINMIX"]
    CRIAUTC = constants["CRIAUTC"]
    CRIAUTI = constants["CRIAUTI"]
    ACRIAUTI = constants["ACRIAUTI"]
    BCRIAUTI = constants["BCRIAUTI"]
    
    # Configuration flags
    LSUBG_COND = constants.get("LSUBG_COND", True)
    LSIGMAS = constants.get("LSIGMAS", True)
    LSTATNW = constants.get("LSTATNW", False)
    FRAC_ICE_ADJUST = constants.get("FRAC_ICE_ADJUST", 0)
    CONDENS = constants.get("CONDENS", 0)  # 0 = CB02
    SUBG_MF_PDF = constants.get("SUBG_MF_PDF", 0)  # 0 = None, 1 = Triangle
    OCND2 = constants.get("OCND2", False)
    
    # 2.3 Compute temperature and latent heats
    t = th * exn
    lv = vaporisation_latent_heat(t, constants)
    ls = sublimation_latent_heat(t, constants)
    
    # 2.4 Specific heat for moist air
    if NRR == 6:
        cph = CPD + CPV * rv + CL * (rc + rr) + CI * (ri + rs + rg)
    elif NRR == 5:
        cph = CPD + CPV * rv + CL * (rc + rr) + CI * (ri + rs)
    elif NRR == 4:
        cph = CPD + CPV * rv + CL * (rc + rr)
    elif NRR == 2:
        cph = CPD + CPV * rv + CL * rc + CI * ri
    else:
        cph = CPD + CPV * rv + CL * (rc + rr) + CI * (ri + rs + rg)
    
    # Initialize outputs
    cldfr = jnp.zeros_like(rv)
    rv_out = jnp.zeros_like(rv)
    rc_out = jnp.zeros_like(rv)
    ri_out = jnp.zeros_like(rv)
    hlc_hrc = jnp.zeros_like(rv)
    hlc_hcf = jnp.zeros_like(rv)
    hli_hri = jnp.zeros_like(rv)
    hli_hcf = jnp.zeros_like(rv)
    
    # 3. Subgrid condensation scheme
    if LSUBG_COND:
        # PRIFACT for OCND2 = False (AROME default)
        prifact = 1.0
        
        # Store total water mixing ratio
        rt = rv + rc + ri * prifact
        
        # Compute saturation vapor pressures
        if not OCND2:
            pv = jnp.minimum(e_sat_w(t, constants), 0.99 * pabs)
            piv = jnp.minimum(e_sat_i(t, constants), 0.99 * pabs)
        
        # Compute ice fraction
        frac_tmp = jnp.where(
            rc + ri > 1e-20,
            rc / (rc + ri),
            0.0
        )
        
        if FRAC_ICE_ADJUST == 3:
            # Default Mode (S)
            frac_tmp = jnp.clip(frac_tmp, 0.0, 1.0)
        elif FRAC_ICE_ADJUST == 0:
            # AROME mode - temperature-based
            frac_tmp = jnp.clip((TMAXMIX - t) / (TMAXMIX - TMINMIX), 0.0, 1.0)
        
        # Supersaturation coefficients
        qsl = RD / RV * pv / (pabs - pv)
        qsi = RD / RV * piv / (pabs - piv)
        
        # Interpolate between liquid and solid as a function of temperature
        qsl = (1.0 - frac_tmp) * qsl + frac_tmp * qsi
        lvs = (1.0 - frac_tmp) * lv + frac_tmp * ls
        
        # Coefficients a and b
        ah = lvs * qsl / (RV * t**2) * (1.0 + RV * qsl / RD)
        a = 1.0 / (1.0 + lvs / cph * ah)
        b = ah * a
        sbar = a * (rt - qsl + ah * lvs * (rc + ri * prifact) / cph)
        
        # Compute sigma
        if LSIGMAS and not LSTATNW:
            sigma = jnp.where(
                sigqsat != 0,
                jnp.sqrt((2.0 * sigs)**2 + (sigqsat * qsl * a)**2),
                2.0 * sigs
            )
        else:
            sigma = 2.0 * sigs
        
        sigma = jnp.maximum(1e-10, sigma)
        q1 = sbar / sigma
        
        # CB02 Fractional cloudiness and cloud condensate
        if CONDENS == 0:
            # Condensation amount
            cond_tmp = jnp.where(
                q1 > 0.0,
                jnp.where(
                    q1 <= 2.0,
                    jnp.minimum(jnp.exp(-1.0) + 0.66 * q1 + 0.086 * q1**2, 2.0),
                    q1
                ),
                jnp.exp(1.2 * q1 - 1.0)
            ) * sigma
            
            # Cloud fraction
            cldfr = jnp.where(
                cond_tmp >= 1e-12,
                jnp.clip(0.5 + 0.36 * jnp.arctan(1.55 * q1), 0.0, 1.0),
                0.0
            )
            
            # Set condensation to zero if cloud fraction is zero
            cond_tmp = jnp.where(cldfr == 0, 0.0, cond_tmp)
            
            # Partition condensate
            if not OCND2:
                rc_out = (1.0 - frac_tmp) * cond_tmp  # liquid
                ri_out = frac_tmp * cond_tmp  # solid
                t = t + ((rc_out - rc) * lv + (ri_out - ri) * ls) / cph
                rv_out = rt - rc_out - ri_out * prifact
    
    # 5.0 Compute the variation of mixing ratios
    w1 = (rc_out - rc) / dt
    w2 = (ri_out - ri) / dt
    
    # 5.1 Compute the sources (apply limits)
    w1 = jnp.where(w1 < 0.0, jnp.maximum(w1, -rcs), jnp.minimum(w1, rvs))
    rvs = rvs - w1
    rcs = rcs + w1
    ths = ths + w1 * lv / (cph * exn_ref)
    
    w2 = jnp.where(w2 < 0.0, jnp.maximum(w2, -ris), jnp.minimum(w2, rvs))
    rvs = rvs - w2
    ris = ris + w2
    ths = ths + w2 * ls / (cph * exn_ref)
    
    # Cloud fraction computation
    if not LSUBG_COND:
        cldfr = jnp.where((rcs + ris) * dt > 1e-12, 1.0, 0.0)
    else:
        # Add mass flux contributions
        w1_mf = rc_mf / dt
        w2_mf = ri_mf / dt
        
        # Limit by available water vapor
        total_mf = w1_mf + w2_mf
        scale = jnp.where(
            total_mf > rvs,
            rvs / total_mf,
            1.0
        )
        w1_mf = w1_mf * scale
        w2_mf = w2_mf * scale
        
        cldfr = jnp.minimum(1.0, cldfr + cf_mf)
        rcs = rcs + w1_mf
        ris = ris + w2_mf
        rvs = rvs - w1_mf - w2_mf
        ths = ths + (w1_mf * lv + w2_mf * ls) / (cph * exn_ref)
        
        # Droplets subgrid autoconversion
        criaut = CRIAUTC / rho_dry_ref
        
        if SUBG_MF_PDF == 0:  # None
            # Simple threshold approach
            mask = w1_mf * dt > cf_mf * criaut
            hlc_hrc = jnp.where(mask, hlc_hrc + w1_mf * dt, hlc_hrc)
            hlc_hcf = jnp.where(mask, jnp.minimum(1.0, hlc_hcf + cf_mf), hlc_hcf)
            
        elif SUBG_MF_PDF == 1:  # Triangle PDF
            w1_dt = w1_mf * dt
            crit = cf_mf * criaut
            
            # Case 1: w1*dt > cf_mf * criaut
            mask1 = w1_dt > crit
            hcf1 = 1.0 - 0.5 * (crit * cf_mf / jnp.maximum(1e-20, w1_dt))**2
            hr1 = w1_dt - (crit * cf_mf)**3 / (3.0 * jnp.maximum(1e-20, w1_dt)**2)
            
            # Case 2: 2*w1*dt <= cf_mf * criaut
            mask2 = (2.0 * w1_dt <= crit) & ~mask1
            hcf2 = 0.0
            hr2 = 0.0
            
            # Case 3: else
            mask3 = ~mask1 & ~mask2
            hcf3 = (2.0 * w1_dt - crit)**2 / (2.0 * jnp.maximum(1e-20, w1_dt)**2)
            hr3 = (4.0 * w1_dt**3 - 3.0 * w1_dt * crit**2 + crit**3) / (
                3.0 * jnp.maximum(1e-20, w1_dt)**2
            )
            
            hcf = jnp.where(mask1, hcf1, jnp.where(mask2, hcf2, hcf3)) * cf_mf
            hr = jnp.where(mask1, hr1, jnp.where(mask2, hr2, hr3))
            
            hlc_hcf = jnp.minimum(1.0, hlc_hcf + hcf)
            hlc_hrc = hlc_hrc + hr
        
        # Ice subgrid autoconversion
        criaut_ice = jnp.minimum(
            CRIAUTI,
            10.0 ** (ACRIAUTI * (t - TT) + BCRIAUTI)
        )
        
        if SUBG_MF_PDF == 0:  # None
            mask = w2_mf * dt > cf_mf * criaut_ice
            hli_hri = jnp.where(mask, hli_hri + w2_mf * dt, hli_hri)
            hli_hcf = jnp.where(mask, jnp.minimum(1.0, hli_hcf + cf_mf), hli_hcf)
            
        elif SUBG_MF_PDF == 1:  # Triangle PDF
            w2_dt = w2_mf * dt
            crit = cf_mf * criaut_ice
            
            # Case 1: w2*dt > cf_mf * criaut
            mask1 = w2_dt > crit
            hcf1 = 1.0 - 0.5 * (crit * cf_mf / jnp.maximum(1e-20, w2_dt))**2
            hri1 = w2_dt - (crit * cf_mf)**3 / (3.0 * jnp.maximum(1e-20, w2_dt)**2)
            
            # Case 2: 2*w2*dt <= cf_mf * criaut
            mask2 = (2.0 * w2_dt <= crit) & ~mask1
            hcf2 = 0.0
            hri2 = 0.0
            
            # Case 3: else
            mask3 = ~mask1 & ~mask2
            hcf3 = (2.0 * w2_dt - crit)**2 / (2.0 * jnp.maximum(1e-20, w2_dt)**2)
            hri3 = (4.0 * w2_dt**3 - 3.0 * w2_dt * crit**2 + crit**3) / (
                3.0 * jnp.maximum(1e-20, w2_dt)**2
            )
            
            hcf = jnp.where(mask1, hcf1, jnp.where(mask2, hcf2, hcf3)) * cf_mf
            hri = jnp.where(mask1, hri1, jnp.where(mask2, hri2, hri3))
            
            hli_hcf = jnp.minimum(1.0, hli_hcf + hcf)
            hli_hri = hli_hri + hri
    
    return (
        t,
        rv_out,
        rc_out,
        ri_out,
        cldfr,
        hlc_hrc,
        hlc_hcf,
        hli_hri,
        hli_hcf,
        cph,
        lv,
        ls,
        rvs,
        rcs,
        ris,
        ths,
    )
