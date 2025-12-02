# -*- coding: utf-8 -*-
"""
Condensation - DaCe Implementation

This module implements the CB02 statistical cloud condensation scheme
for the ICE3/ICE4 microphysics, translated from GT4Py to DaCe.

Stencils implemented:
- condensation: Subgrid condensation using CB02 statistical scheme
- sigrc_computation: Compute subgrid cloud condensate variance

Reference:
    PHYEX/src/common/micro/condensation.F90
    
Author:
    Translated to Python/DaCe from GT4Py by Cline AI Assistant
"""

import dace
import math

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")


def e_sat_w(t: dace.float32) -> dace.float32:
    """Saturation vapor pressure over liquid water (Pa)"""
    # Constants from PHYEX
    XALPW = 17.5327  # Clausius-Clapeyron constant
    XBETAW = 32.49   # Clausius-Clapeyron constant
    XGAMW = 0.0	     # Clausius-Clapeyron constant
    XTT = 273.16     # Triple point temperature (K)
    XESTT = 611.14   # Saturation vapor pressure at triple point (Pa)
    
    return XESTT * exp(XALPW - XBETAW / t - XGAMW * log(t))


def e_sat_i(t: dace.float32) -> dace.float32:
    """Saturation vapor pressure over ice (Pa)"""
    # Constants from PHYEX
    XALPI = 22.0756  # Clausius-Clapeyron constant
    XBETAI = 3.1068  # Clausius-Clapeyron constant
    XGAMI = 6.6088   # Clausius-Clapeyron constant
    XESTT = 611.14   # Saturation vapor pressure at triple point (Pa)
    
    return XESTT * exp(XALPI - XBETAI / t - XGAMI * log(t))


@dace.program
def condensation(
    sigqsat: dace.float32[I, J, K],
    pabs: dace.float32[I, J, K],
    sigs: dace.float32[I, J, K],
    t: dace.float32[I, J, K],
    rv: dace.float32[I, J, K],
    ri: dace.float32[I, J, K],
    rc: dace.float32[I, J, K],
    rv_out: dace.float32[I, J, K],
    rc_out: dace.float32[I, J, K],
    ri_out: dace.float32[I, J, K],
    cldfr: dace.float32[I, J, K],
    cph: dace.float32[I, J, K],
    lv: dace.float32[I, J, K],
    ls: dace.float32[I, J, K],
    q1: dace.float32[I, J, K],
    pv_out: dace.float32[I, J, K],
    piv_out: dace.float32[I, J, K],
    frac_out: dace.float32[I, J, K],
    qsl_out: dace.float32[I, J, K],
    qsi_out: dace.float32[I, J, K],
    sigma_out: dace.float32[I, J, K],
    cond_out: dace.float32[I, J, K],
    a_out: dace.float32[I, J, K],
    b_out: dace.float32[I, J, K],
    sbar_out: dace.float32[I, J, K],
    CONDENS: dace.int32,
    FRAC_ICE_ADJUST: dace.int32,
    LSIGMAS: dace.bool,
    LSTATNW: dace.bool,
    OCND2: dace.bool,
    RD: dace.float32,
    RV: dace.float32,
    TMAXMIX: dace.float32,
    TMINMIX: dace.float32,
):
    """
    Compute subgrid condensation using CB02 statistical cloud scheme.
    
    Parameters
    ----------
    sigqsat : dace.float32[I, J, K]
        Saturation mixing ratio variance coefficient
    pabs : dace.float32[I, J, K]
        Absolute pressure (Pa)
    sigs : dace.float32[I, J, K]
        Subgrid standard deviation
    t : dace.float32[I, J, K]
        Temperature (K), modified by latent heating
    rv, rc, ri : dace.float32[I, J, K]
        Mixing ratios for vapor, cloud, ice (kg/kg)
    rv_out, rc_out, ri_out : dace.float32[I, J, K]
        Output mixing ratios
    cldfr : dace.float32[I, J, K]
        Cloud fraction (0-1)
    cph, lv, ls : dace.float32[I, J, K]
        Thermodynamic fields
    q1 : dace.float32[I, J, K]
        Normalized supersaturation
    pv_out, piv_out, frac_out, qsl_out, qsi_out : dace.float32[I, J, K]
        Diagnostic outputs
    sigma_out, cond_out, a_out, b_out, sbar_out : dace.float32[I, J, K]
        Diagnostic outputs
    CONDENS : dace.int32
        Condensation scheme (0=CB02)
    FRAC_ICE_ADJUST : dace.int32
        Ice fraction method (0=temperature, 3=statistical)
    LSIGMAS, LSTATNW, OCND2 : dace.bool
        Configuration flags
    RD, RV : dace.float32
        Gas constants
    TMAXMIX, TMINMIX : dace.float32
        Mixed-phase temperature bounds (K)
    """
    @dace.map
    def compute_condensation(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Initialize outputs
        cldfr[i, j, k] = 0.0
        rv_out[i, j, k] = 0.0
        rc_out[i, j, k] = 0.0
        ri_out[i, j, k] = 0.0
        
        # Local variables
        prifact = 1.0 if not OCND2 else 1.0
        
        # Store total water mixing ratio
        rt = rv[i, j, k] + rc[i, j, k] + ri[i, j, k] * prifact
        
        # Compute saturation vapor pressures
        if not OCND2:
            pv = min(e_sat_w(t[i, j, k]), 0.99 * pabs[i, j, k])
            piv = min(e_sat_i(t[i, j, k]), 0.99 * pabs[i, j, k])
        else:
            pv = e_sat_w(t[i, j, k])
            piv = e_sat_i(t[i, j, k])
        
        # Compute ice fraction
        if not OCND2:
            if rc[i, j, k] + ri[i, j, k] > 1e-20:
                frac_tmp = rc[i, j, k] / (rc[i, j, k] + ri[i, j, k])
            else:
                frac_tmp = 0.0
            
            if FRAC_ICE_ADJUST == 3:
                # Statistical mode
                frac_tmp = max(0.0, min(1.0, frac_tmp))
            elif FRAC_ICE_ADJUST == 0:
                # AROME mode (temperature-based)
                frac_tmp = max(0.0, min(1.0, (TMAXMIX - t[i, j, k]) / (TMAXMIX - TMINMIX)))
        else:
            frac_tmp = 0.0
        
        # Compute saturation mixing ratios
        qsl = RD / RV * pv / (pabs[i, j, k] - pv)
        qsi = RD / RV * piv / (pabs[i, j, k] - piv)
        
        # Store diagnostics
        pv_out[i, j, k] = pv
        piv_out[i, j, k] = piv
        qsl_out[i, j, k] = qsl
        qsi_out[i, j, k] = qsi
        frac_out[i, j, k] = frac_tmp
        
        # Interpolate between liquid and solid
        qsl_mixed = (1.0 - frac_tmp) * qsl + frac_tmp * qsi
        lvs = (1.0 - frac_tmp) * lv[i, j, k] + frac_tmp * ls[i, j, k]
        
        # Supersaturation coefficients
        ah = lvs * qsl_mixed / (RV * t[i, j, k] ** 2) * (1.0 + RV * qsl_mixed / RD)
        a = 1.0 / (1.0 + lvs / cph[i, j, k] * ah)
        b = ah * a
        sbar = a * (rt - qsl_mixed + ah * lvs * (rc[i, j, k] + ri[i, j, k] * prifact) / cph[i, j, k])
        
        # Store coefficients
        a_out[i, j, k] = a
        b_out[i, j, k] = b
        sbar_out[i, j, k] = sbar
        
        # Compute subgrid variance
        if LSIGMAS and not LSTATNW:
            if sigqsat[i, j, k] != 0.0:
                sigma = sqrt((2.0 * sigs[i, j, k]) ** 2 + (sigqsat[i, j, k] * qsl_mixed * a) ** 2)
            else:
                sigma = 2.0 * sigs[i, j, k]
        else:
            sigma = 2.0 * sigs[i, j, k]
        
        sigma = max(1e-10, sigma)
        q1[i, j, k] = sbar / sigma
        sigma_out[i, j, k] = sigma
        
        # CB02 condensation scheme
        if CONDENS == 0:
            if q1[i, j, k] > 0.0:
                if q1[i, j, k] <= 2.0:
                    cond_tmp = min(exp(-1.0) + 0.66 * q1[i, j, k] + 0.086 * q1[i, j, k] ** 2, 2.0)
                else:
                    cond_tmp = q1[i, j, k]
            else:
                cond_tmp = exp(1.2 * q1[i, j, k] - 1.0)
            
            cond_tmp *= sigma
            
            # Cloud fraction
            if cond_tmp >= 1e-12:
                cldfr[i, j, k] = max(0.0, min(1.0, 0.5 + 0.36 * atan(1.55 * q1[i, j, k])))
            else:
                cldfr[i, j, k] = 0.0
            
            if cldfr[i, j, k] == 0.0:
                cond_tmp = 0.0
            
            cond_out[i, j, k] = cond_tmp
            
            # Split into liquid and ice
            if not OCND2:
                rc_out[i, j, k] = (1.0 - frac_tmp) * cond_tmp
                ri_out[i, j, k] = frac_tmp * cond_tmp
                t[i, j, k] += ((rc_out[i, j, k] - rc[i, j, k]) * lv[i, j, k] + 
                              (ri_out[i, j, k] - ri[i, j, k]) * ls[i, j, k]) / cph[i, j, k]
                rv_out[i, j, k] = rt - rc_out[i, j, k] - ri_out[i, j, k] * prifact


@dace.program
def sigrc_computation(
    q1: dace.float32[I, J, K],
    sigrc: dace.float32[I, J, K],
    inq2: dace.int32,
    src_1d: dace.float32[34],
):
    """
    Compute subgrid cloud condensate variance using lookup table.
    
    Parameters
    ----------
    q1 : dace.float32[I, J, K]
        Normalized supersaturation
    sigrc : dace.float32[I, J, K]
        Subgrid cloud condensate variance (output)
    inq2 : dace.int32
        Lookup table index (unused)
    src_1d : dace.float32[34]
        Precomputed variance lookup table
    """
    @dace.map
    def compute_variance(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Compute lookup table index
        q1_val = min(100.0, max(-100.0, 2.0 * q1[i, j, k]))
        inq1 = floor(q1_val)
        inq2_local = min(max(-22, inq1), 10)
        
        # Linear interpolation
        inc = 2.0 * q1[i, j, k] - floor(2.0 * q1[i, j, k])
        idx1 = int(inq2_local + 22)
        idx2 = int(inq2_local + 23)
        
        # Ensure indices are within table bounds
        idx1 = max(0, min(33, idx1))
        idx2 = max(0, min(33, idx2))
        
        sigrc[i, j, k] = min(1.0, (1.0 - inc) * src_1d[idx1] + inc * src_1d[idx2])
