# -*- coding: utf-8 -*-
"""
JAX implementation of rain-ice scheme orchestration stencils.

This module contains utility stencils that support the main rain-ice
microphysics scheme. These functions handle tendency aggregation,
thermodynamic computations, masking, and auxiliary processes.

Translated from GT4Py implementation in src/ice3/stencils/rain_ice.py
Source: PHYEX/src/common/micro/rain_ice.F90
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from typing import Dict, Any, Tuple


def rain_ice_total_tendencies(
    wr_th: Array,
    wr_v: Array,
    wr_c: Array,
    wr_r: Array,
    wr_i: Array,
    wr_s: Array,
    wr_g: Array,
    ls_fact: Array,
    lv_fact: Array,
    exnref: Array,
    ths: Array,
    rvs: Array,
    rcs: Array,
    rrs: Array,
    ris: Array,
    rss: Array,
    rgs: Array,
    rvheni: Array,
    rv_t: Array,
    rc_t: Array,
    rr_t: Array,
    ri_t: Array,
    rs_t: Array,
    rg_t: Array,
    constants: Dict[str, Any],
) -> Tuple[Array, ...]:
    """
    Compute total tendencies and update source terms.
    
    Parameters
    ----------
    wr_th, wr_v, wr_c, wr_r, wr_i, wr_s, wr_g : Array
        Initial values for potential temperature and mixing ratios (kg/kg).
    ls_fact, lv_fact : Array
        Latent heat factors L_s/c_p and L_v/c_p (K).
    exnref : Array
        Reference Exner function (dimensionless).
    ths, rvs, rcs, rrs, ris, rss, rgs : Array
        Source terms (tendencies) to be updated (units/s).
    rvheni : Array
        Vapor consumed by heterogeneous nucleation (kg/kg/s).
    rv_t, rc_t, rr_t, ri_t, rs_t, rg_t : Array
        Current mixing ratios at time t (kg/kg).
    constants : Dict[str, Any]
        Physical constants including INV_TSTEP.
        
    Returns
    -------
    Tuple of updated arrays
    """
    INV_TSTEP = constants["INV_TSTEP"]
    
    # Hydrometeor tendencies
    wr_v = (wr_v - rv_t) * INV_TSTEP
    wr_c = (wr_c - rc_t) * INV_TSTEP
    wr_r = (wr_r - rr_t) * INV_TSTEP
    wr_i = (wr_i - ri_t) * INV_TSTEP
    wr_s = (wr_s - rs_t) * INV_TSTEP
    wr_g = (wr_g - rg_t) * INV_TSTEP
    
    # Theta tendency
    wr_th = (wr_c + wr_r) * lv_fact + (wr_i + wr_s + wr_g) * ls_fact
    
    # Update sources with nucleation
    ths = ths + wr_th + rvheni * ls_fact
    rvs = rvs + wr_v - rvheni
    rcs = rcs + wr_c
    rrs = rrs + wr_r
    ris = ris + wr_i + rvheni
    rss = rss + wr_s
    rgs = rgs + wr_g
    
    return wr_th, wr_v, wr_c, wr_r, wr_i, wr_s, wr_g, ths, rvs, rcs, rrs, ris, rss, rgs


def rain_ice_thermo(
    exn: Array,
    th_t: Array,
    rv_t: Array,
    rc_t: Array,
    rr_t: Array,
    ri_t: Array,
    rs_t: Array,
    rg_t: Array,
    constants: Dict[str, Any],
) -> Tuple[Array, Array]:
    """
    Compute thermodynamic functions for the microphysics scheme.
    
    Parameters
    ----------
    exn : Array
        Exner function (dimensionless).
    th_t : Array
        Potential temperature at time t (K).
    rv_t, rc_t, rr_t, ri_t, rs_t, rg_t : Array
        Mixing ratios at time t (kg/kg).
    constants : Dict[str, Any]
        Physical constants.
        
    Returns
    -------
    ls_fact, lv_fact : Tuple[Array, Array]
        Latent heat factors L_s/c_p and L_v/c_p (K).
    """
    CPD = constants["CPD"]
    CPV = constants["CPV"]
    CL = constants["CL"]
    CI = constants["CI"]
    LSTT = constants["LSTT"]
    LVTT = constants["LVTT"]
    TT = constants["TT"]
    
    divider = CPD + CPV * rv_t + CL * (rc_t + rr_t) + CI * (ri_t + rs_t + rg_t)
    t = th_t * exn
    ls_fact = (LSTT + (CPV - CI) * (t - TT)) / divider
    lv_fact = (LVTT + (CPV - CL) * (t - TT)) / divider
    
    return ls_fact, lv_fact


def rain_ice_mask(
    rc_t: Array,
    rr_t: Array,
    ri_t: Array,
    rs_t: Array,
    rg_t: Array,
    constants: Dict[str, Any],
) -> Array:
    """
    Determine which grid points require microphysical computations.
    
    Parameters
    ----------
    rc_t, rr_t, ri_t, rs_t, rg_t : Array
        Mixing ratios for hydrometeors (kg/kg).
    constants : Dict[str, Any]
        Physical constants including minimum thresholds.
        
    Returns
    -------
    ldmicro : Array (bool)
        Mask indicating grid points needing microphysics.
    """
    C_RTMIN = constants["C_RTMIN"]
    R_RTMIN = constants["R_RTMIN"]
    I_RTMIN = constants["I_RTMIN"]
    S_RTMIN = constants["S_RTMIN"]
    G_RTMIN = constants["G_RTMIN"]
    
    ldmicro = (
        (rc_t > C_RTMIN) |
        (rr_t > R_RTMIN) |
        (ri_t > I_RTMIN) |
        (rs_t > S_RTMIN) |
        (rg_t > G_RTMIN)
    )
    
    return ldmicro


def initial_values_saving(
    th_t: Array,
    rv_t: Array,
    rc_t: Array,
    rr_t: Array,
    ri_t: Array,
    rs_t: Array,
    rg_t: Array,
    constants: Dict[str, Any],
) -> Tuple[Array, ...]:
    """
    Save initial values before microphysical processing.
    
    Parameters
    ----------
    th_t, rv_t, rc_t, rr_t, ri_t, rs_t, rg_t : Array
        Current values to be saved.
    constants : Dict[str, Any]
        Physical constants including LWARM flag.
        
    Returns
    -------
    Tuple of saved values and initialized diagnostics
    """
    LWARM = constants.get("LWARM", True)
    
    wr_th = th_t.copy()
    wr_v = rv_t.copy()
    wr_c = rc_t.copy()
    wr_r = rr_t.copy()
    wr_i = ri_t.copy()
    wr_s = rs_t.copy()
    wr_g = rg_t.copy()
    
    # Initialize diagnostics
    evap3d = jnp.zeros_like(th_t) if LWARM else jnp.zeros_like(th_t)
    rainfr = jnp.zeros_like(th_t)
    
    return wr_th, wr_v, wr_c, wr_r, wr_i, wr_s, wr_g, evap3d, rainfr


def ice4_precipitation_fraction_sigma(
    sigs: Array,
) -> Array:
    """
    Compute cloud water variance for subgrid precipitation scheme.
    
    Parameters
    ----------
    sigs : Array
        Standard deviation of saturation (dimensionless).
        
    Returns
    -------
    sigma_rc : Array
        Variance of cloud water mixing ratio (dimensionless).
    """
    return sigs ** 2


def rain_fraction_sedimentation(
    rrs: Array,
    rss: Array,
    rgs: Array,
    constants: Dict[str, Any],
) -> Tuple[Array, Array, Array]:
    """
    Initialize precipitation mixing ratios for sedimentation.
    
    Parameters
    ----------
    rrs, rss, rgs : Array
        Source terms (tendencies) for rain, snow, graupel (kg/kg/s).
    constants : Dict[str, Any]
        Physical constants including TSTEP.
        
    Returns
    -------
    wr_r, wr_s, wr_g : Tuple[Array, Array, Array]
        Precipitation at top boundary (kg/kg).
    """
    TSTEP = constants["TSTEP"]
    
    # Apply only at top level (k=0)
    wr_r = rrs * TSTEP
    wr_s = rss * TSTEP
    wr_g = rgs * TSTEP
    
    return wr_r, wr_s, wr_g


def ice4_rainfr_vert(
    prfr: Array,
    rr: Array,
    rs: Array,
    rg: Array,
    constants: Dict[str, Any],
) -> Array:
    """
    Compute vertical rain fraction for diagnostics (backward sweep).
    
    Parameters
    ----------
    prfr : Array
        Rain fraction (0-1).
    rr, rs, rg : Array
        Rain, snow, graupel mixing ratios (kg/kg).
    constants : Dict[str, Any]
        Physical constants including minimum thresholds.
        
    Returns
    -------
    prfr : Array
        Updated rain fraction.
    """
    R_RTMIN = constants["R_RTMIN"]
    S_RTMIN = constants["S_RTMIN"]
    G_RTMIN = constants["G_RTMIN"]
    
    # Backward sweep along z-axis
    nz = prfr.shape[2]
    for k in range(nz - 2, -1, -1):
        has_precip = (
            (rr[:, :, k] > R_RTMIN) |
            (rs[:, :, k] > S_RTMIN) |
            (rg[:, :, k] > G_RTMIN)
        )
        
        prfr_k = jnp.where(
            has_precip,
            jnp.maximum(prfr[:, :, k], prfr[:, :, k + 1]),
            0.0
        )
        
        # Set to 1 if has precip but no rain fraction yet
        prfr_k = jnp.where(has_precip & (prfr_k == 0), 1.0, prfr_k)
        
        prfr = prfr.at[:, :, k].set(prfr_k)
    
    return prfr


def fog_deposition(
    rcs: Array,
    rc_t: Array,
    rhodref: Array,
    dzz: Array,
    inprc: Array,
    constants: Dict[str, Any],
) -> Tuple[Array, Array]:
    """
    Compute fog deposition on vegetation (surface process).
    
    Parameters
    ----------
    rcs : Array
        Cloud droplet source term (kg/kg/s).
    rc_t : Array
        Cloud droplet mixing ratio (kg/kg).
    rhodref : Array
        Reference air density (kg/m³).
    dzz : Array
        Vertical grid spacing (m).
    inprc : Array
        Accumulated deposition on surface (m).
    constants : Dict[str, Any]
        Physical constants including V_DEPOSC, RHOLW.
        
    Returns
    -------
    rcs, inprc : Tuple[Array, Array]
        Updated cloud source and deposition.
    """
    VDEPOSC = constants["VDEPOSC"]
    RHOLW = constants["RHOLW"]
    
    # Apply only at surface level (k=0)
    rcs_0 = rcs[:, :, 0] - VDEPOSC * rc_t[:, :, 0] / dzz[:, :, 0]
    rcs = rcs.at[:, :, 0].set(rcs_0)
    
    inprc = inprc + VDEPOSC * rc_t[:, :, 0] * rhodref[:, :, 0] / RHOLW
    
    return rcs, inprc
