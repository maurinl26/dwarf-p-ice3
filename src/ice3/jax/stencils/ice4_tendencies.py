# -*- coding: utf-8 -*-
"""
JAX implementation of ICE4 tendency management and aggregation stencils.

This module handles initialization, update, and post-processing of
microphysical tendencies in the ICE4 scheme. It aggregates contributions
from different microphysical processes with proper thermodynamic consistency.

Translated from GT4Py implementation in src/ice3/stencils/ice4_tendencies.py
Source: PHYEX/src/common/micro/mode_ice4_tendencies.F90
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from typing import Dict, Any, Tuple


def ice4_nucleation_post_processing(
    t: Array,
    exn: Array,
    ls_fact: Array,
    th_t: Array,
    rv_t: Array,
    ri_t: Array,
    rvheni_mr: Array,
) -> Tuple[Array, Array, Array, Array]:
    """
    Apply heterogeneous ice nucleation changes to prognostic variables.
    
    Parameters
    ----------
    t, exn, ls_fact, th_t, rv_t, ri_t : Array
        Thermodynamic state variables
    rvheni_mr : Array
        Vapor consumed by heterogeneous nucleation (kg/kg)
        
    Returns
    -------
    Tuple of updated fields
    """
    th_t = th_t + rvheni_mr * ls_fact
    t = th_t * exn
    rv_t = rv_t - rvheni_mr
    ri_t = ri_t + rvheni_mr
    
    return t, th_t, rv_t, ri_t


def ice4_rrhong_post_processing(
    t: Array,
    exn: Array,
    ls_fact: Array,
    lv_fact: Array,
    th_t: Array,
    rr_t: Array,
    rg_t: Array,
    rrhong_mr: Array,
) -> Tuple[Array, Array, Array, Array]:
    """
    Apply homogeneous freezing of rain to prognostic variables.
    
    Parameters
    ----------
    t, exn, ls_fact, lv_fact, th_t, rr_t, rg_t : Array
        Thermodynamic state variables
    rrhong_mr : Array
        Rain frozen by homogeneous nucleation (kg/kg)
        
    Returns
    -------
    Tuple of updated fields
    """
    th_t = th_t + rrhong_mr * (ls_fact - lv_fact)
    t = th_t * exn
    rr_t = rr_t - rrhong_mr
    rg_t = rg_t + rrhong_mr
    
    return t, th_t, rr_t, rg_t


def ice4_rimltc_post_processing(
    t: Array,
    exn: Array,
    ls_fact: Array,
    lv_fact: Array,
    rimltc_mr: Array,
    th_t: Array,
    rc_t: Array,
    ri_t: Array,
) -> Tuple[Array, Array, Array, Array]:
    """
    Apply ice crystal melting to prognostic variables.
    
    Parameters
    ----------
    t, exn, ls_fact, lv_fact, rimltc_mr, th_t, rc_t, ri_t : Array
        Thermodynamic state variables
        
    Returns
    -------
    Tuple of updated fields
    """
    th_t = th_t - rimltc_mr * (ls_fact - lv_fact)
    t = th_t * exn
    rc_t = rc_t + rimltc_mr
    ri_t = ri_t - rimltc_mr
    
    return t, th_t, rc_t, ri_t


def ice4_fast_rg_pre_post_processing(
    rvdepg: Array,
    rsmltg: Array,
    rraccsg: Array,
    rsaccrg: Array,
    rcrimsg: Array,
    rsrimcg: Array,
    rrhong_mr: Array,
    rsrimcg_mr: Array,
) -> Tuple[Array, Array]:
    """
    Aggregate graupel source terms from various processes.
    
    Parameters
    ----------
    rvdepg, rsmltg, rraccsg, rsaccrg, rcrimsg, rsrimcg : Array
        Process rates
    rrhong_mr, rsrimcg_mr : Array
        Mixing ratio conversions
        
    Returns
    -------
    rgsi, rgsi_mr : Tuple[Array, Array]
        Aggregated graupel sources
    """
    rgsi = rvdepg + rsmltg + rraccsg + rsaccrg + rcrimsg + rsrimcg
    rgsi_mr = rrhong_mr + rsrimcg_mr
    
    return rgsi, rgsi_mr


def ice4_increment_update(
    ls_fact: Array,
    lv_fact: Array,
    theta_increment: Array,
    rv_increment: Array,
    rc_increment: Array,
    rr_increment: Array,
    ri_increment: Array,
    rs_increment: Array,
    rg_increment: Array,
    rvheni_mr: Array,
    rimltc_mr: Array,
    rrhong_mr: Array,
    rsrimcg_mr: Array,
) -> Tuple[Array, ...]:
    """
    Update increment fields with nucleation and phase change processes.
    
    Parameters
    ----------
    ls_fact, lv_fact : Array
        Latent heat factors
    theta_increment, rv_increment, etc. : Array
        Increment fields
    rvheni_mr, rimltc_mr, rrhong_mr, rsrimcg_mr : Array
        Process mixing ratios
        
    Returns
    -------
    Tuple of updated increments
    """
    theta_increment = theta_increment + (
        rvheni_mr * ls_fact
        + rrhong_mr * (ls_fact - lv_fact)
        - rimltc_mr * (ls_fact - lv_fact)
    )
    
    rv_increment = rv_increment - rvheni_mr
    rc_increment = rc_increment + rimltc_mr
    rr_increment = rr_increment - rrhong_mr
    ri_increment = ri_increment + rvheni_mr - rimltc_mr
    rs_increment = rs_increment - rsrimcg_mr
    rg_increment = rg_increment + rrhong_mr + rsrimcg_mr
    
    return (theta_increment, rv_increment, rc_increment, rr_increment,
            ri_increment, rs_increment, rg_increment)


def ice4_derived_fields(
    t: Array,
    rhodref: Array,
    pres: Array,
    rv_t: Array,
    constants: Dict[str, Any],
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Compute derived microphysical fields for process calculations.
    
    Parameters
    ----------
    t, rhodref, pres, rv_t : Array
        State variables
    constants : Dict[str, Any]
        Physical constants
        
    Returns
    -------
    ssi, ka, dv, ai, cj : Tuple of Arrays
        Derived fields
    """
    ALPI = constants["ALPI"]
    BETAI = constants["BETAI"]
    GAMI = constants["GAMI"]
    CI = constants["CI"]
    CPV = constants["CPV"]
    EPSILO = constants["EPSILO"]
    LSTT = constants["LSTT"]
    P00 = constants["P00"]
    RV = constants["RV"]
    SCFAC = constants["SCFAC"]
    TT = constants["TT"]
    
    # Saturation vapor pressure over ice
    zw = jnp.exp(ALPI - BETAI / t - GAMI * jnp.log(t))
    
    # Supersaturation over ice
    ssi = rv_t * (pres - zw) / (EPSILO * zw) - 1.0
    
    # Thermal conductivity of air
    ka = 2.38e-2 + 7.1e-5 * (t - TT)
    
    # Diffusivity of water vapor
    dv = 2.11e-5 * (t / TT) ** 1.94 * (P00 / pres)
    
    # Thermodynamic function for deposition
    ai = (LSTT + (CPV - CI) * (t - TT)) ** 2 / (ka * RV * t**2) + (
        (RV * t) / (dv * zw)
    )
    
    # Ventilation coefficient
    cj = SCFAC * rhodref**0.3 / jnp.sqrt(1.718e-5 + 4.9e-8 * (t - TT))
    
    return ssi, ka, dv, ai, cj


def ice4_slope_parameters(
    rhodref: Array,
    t: Array,
    rr_t: Array,
    rs_t: Array,
    rg_t: Array,
    constants: Dict[str, Any],
) -> Tuple[Array, Array, Array, Array]:
    """
    Compute slope parameters for hydrometeor size distributions.
    
    Parameters
    ----------
    rhodref, t, rr_t, rs_t, rg_t : Array
        State variables
    constants : Dict[str, Any]
        Physical constants and flags
        
    Returns
    -------
    lbdar, lbdar_rf, lbdas, lbdag : Tuple of Arrays
        Slope parameters (m^-1)
    """
    LBR = constants["LBR"]
    LBEXR = constants["LBEXR"]
    LBS = constants["LBS"]
    LBEXS = constants["LBEXS"]
    LBG = constants["LBG"]
    LBEXG = constants["LBEXG"]
    R_RTMIN = constants["R_RTMIN"]
    S_RTMIN = constants["S_RTMIN"]
    G_RTMIN = constants["G_RTMIN"]
    LSNOW_T = constants.get("LSNOW_T", False)
    LBDAS_MIN = constants.get("LBDAS_MIN", 1.0e3)
    LBDAS_MAX = constants.get("LBDAS_MAX", 1.0e7)
    TRANS_MP_GAMMAS = constants.get("TRANS_MP_GAMMAS", 1.0)
    
    # Rain slope
    lbdar = jnp.where(
        rr_t > 0,
        LBR * (rhodref * jnp.maximum(rr_t, R_RTMIN)) ** LBEXR,
        0.0
    )
    lbdar_rf = lbdar
    
    # Snow slope
    if LSNOW_T:
        # Temperature-dependent formulation
        lbdas_warm = (
            jnp.clip(10.0 ** (14.554 - 0.0423 * t), LBDAS_MIN, LBDAS_MAX)
            * TRANS_MP_GAMMAS
        )
        lbdas_cold = (
            jnp.clip(10.0 ** (6.226 - 0.0106 * t), LBDAS_MIN, LBDAS_MAX)
            * TRANS_MP_GAMMAS
        )
        lbdas = jnp.where(
            rs_t > 0,
            jnp.where(t > 263.15, lbdas_warm, lbdas_cold),
            0.0
        )
    else:
        # Standard formulation
        lbdas = jnp.where(
            rs_t > 0,
            jnp.minimum(
                LBDAS_MAX,
                LBS * (rhodref * jnp.maximum(rs_t, S_RTMIN)) ** LBEXS
            ),
            0.0
        )
    
    # Graupel slope
    lbdag = jnp.where(
        rg_t > 0,
        LBG * (rhodref * jnp.maximum(rg_t, G_RTMIN)) ** LBEXG,
        0.0
    )
    
    return lbdar, lbdar_rf, lbdas, lbdag


def ice4_total_tendencies_update(
    ls_fact: Array,
    lv_fact: Array,
    th_tnd: Array,
    rv_tnd: Array,
    rc_tnd: Array,
    rr_tnd: Array,
    ri_tnd: Array,
    rs_tnd: Array,
    rg_tnd: Array,
    # Process rates (30+ parameters)
    rchoni: Array,
    rvdeps: Array,
    riaggs: Array,
    riauts: Array,
    rvdepg: Array,
    rcautr: Array,
    rcaccr: Array,
    rrevav: Array,
    rcberi: Array,
    rsmltg: Array,
    rcmltsr: Array,
    rraccss: Array,
    rraccsg: Array,
    rsaccrg: Array,
    rcrimss: Array,
    rcrimsg: Array,
    rsrimcg: Array,
    ricfrrg: Array,
    rrcfrig: Array,
    ricfrr: Array,
    rcwetg: Array,
    riwetg: Array,
    rrwetg: Array,
    rswetg: Array,
    rcdryg: Array,
    ridryg: Array,
    rrdryg: Array,
    rsdryg: Array,
    rgmltr: Array,
    rwetgh: float = 0.0,
) -> Tuple[Array, ...]:
    """
    Aggregate all microphysical process contributions to total tendencies.
    
    Parameters
    ----------
    ls_fact, lv_fact : Array
        Latent heat factors (K)
    th_tnd, rv_tnd, rc_tnd, rr_tnd, ri_tnd, rs_tnd, rg_tnd : Array
        Total tendency fields
    rchoni, rvdeps, ... : Array
        Individual process rates (kg/kg/s)
    rwetgh : float
        Wet growth to hail (not used in 6-category scheme)
        
    Returns
    -------
    Tuple of updated tendencies
    """
    # Potential temperature tendency
    th_tnd = th_tnd + (
        rvdepg * ls_fact
        + rchoni * (ls_fact - lv_fact)
        + rvdeps * ls_fact
        - rrevav * lv_fact
        + rcrimss * (ls_fact - lv_fact)
        + rcrimsg * (ls_fact - lv_fact)
        + rraccss * (ls_fact - lv_fact)
        + rraccsg * (ls_fact - lv_fact)
        + (rrcfrig - ricfrr) * (ls_fact - lv_fact)
        + (rcwetg + rrwetg) * (ls_fact - lv_fact)
        + (rcdryg + rrdryg) * (ls_fact - lv_fact)
        - rgmltr * (ls_fact - lv_fact)
        + rcberi * (ls_fact - lv_fact)
    )
    
    # Vapor tendency
    rv_tnd = rv_tnd - rvdepg - rvdeps + rrevav
    
    # Cloud tendency
    rc_tnd = rc_tnd + (
        -rchoni - rcautr - rcaccr - rcrimss - rcrimsg
        - rcmltsr - rcwetg - rcdryg - rcberi
    )
    
    # Rain tendency
    rr_tnd = rr_tnd + (
        rcautr + rcaccr - rrevav - rraccss - rraccsg
        + rcmltsr - rrcfrig + ricfrr - rrwetg - rrdryg + rgmltr
    )
    
    # Ice tendency
    ri_tnd = ri_tnd + (
        rchoni - riaggs - riauts - ricfrrg - ricfrr
        - riwetg - ridryg + rcberi
    )
    
    # Snow tendency
    rs_tnd = rs_tnd + (
        rvdeps + riaggs + riauts + rcrimss - rsrimcg
        + rraccss - rsaccrg - rsmltg - rswetg - rsdryg
    )
    
    # Graupel tendency
    rg_tnd = rg_tnd + (
        rvdepg + rcrimsg + rsrimcg + rraccsg + rsaccrg + rsmltg
        + ricfrrg + rrcfrig + rcwetg + riwetg + rswetg + rrwetg
        + rcdryg + ridryg + rsdryg + rrdryg - rgmltr - rwetgh
    )
    
    return th_tnd, rv_tnd, rc_tnd, rr_tnd, ri_tnd, rs_tnd, rg_tnd
