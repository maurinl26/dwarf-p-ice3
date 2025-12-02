# -*- coding: utf-8 -*-
"""
ICE4 Fast RS (Snow/Aggregate) Processes - DaCe Implementation

This module implements the fast growth processes for snow/aggregates in the ICE4
microphysics scheme, translated from the Fortran reference in mode_ice4_fast_rs.F90.

Processes implemented:
- Cloud droplet riming of aggregates (RCRIMSS, RCRIMSG, RSRIMCG)
- Rain accretion onto aggregates (RRACCSS, RRACCSG, RSACCRG)  
- Conversion-melting of aggregates (RSMLTG, RCMLTSR)

Reference:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_processes.F90
    
Author:
    Translated to Python/DaCe from Fortran by Cline AI Assistant
"""

import dace
from typing import Tuple

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")
F = dace.symbol("F")


def index_micro1d_rim(
    lambda_s: dace.float32,
    RIMINTP1: dace.float32,
    RIMINTP2: dace.float32,
    NGAMINC: dace.int32,
) -> Tuple[dace.int32, dace.float32]:
    """Compute index in logspace for 1D interpolation table (riming)

    Args:
        lambda_s: Snow slope parameter
        RIMINTP1: Scaling factor 1
        RIMINTP2: Scaling factor 2
        NGAMINC: Number of points in lookup table

    Returns:
        Tuple of (floor index, fractional part)
    """
    index = max(1.00001, min(NGAMINC - 0.00001, RIMINTP1 * log(lambda_s) + RIMINTP2))
    return floor(index), index - floor(index)


# Helper functions for 2D interpolation (for accretion processes)
def index_micro2d_acc_s(
    lambda_s: dace.float32,
    ACCINTP1S: dace.float32,
    ACCINTP2S: dace.float32,
    NACCLBDAS: dace.int32,
) -> Tuple[dace.int32, dace.float32]:
    """Compute index in logspace for 2D interpolation table (snow dimension)

    Args:
        lambda_s: Snow slope parameter
        ACCINTP1S: Scaling factor 1 for snow
        ACCINTP2S: Scaling factor 2 for snow
        NACCLBDAS: Number of points in lookup table for snow

    Returns:
        Tuple of (floor index, fractional part)
    """
    index = max(1.00001, min(NACCLBDAS - 0.00001, ACCINTP1S * log(lambda_s) + ACCINTP2S))
    return floor(index), index - floor(index)


def index_micro2d_acc_r(
    lambda_r: dace.float32,
    ACCINTP1R: dace.float32,
    ACCINTP2R: dace.float32,
    NACCLBDAR: dace.int32,
) -> Tuple[dace.int32, dace.float32]:
    """Compute index in logspace for 2D interpolation table (rain dimension)

    Args:
        lambda_r: Rain slope parameter
        ACCINTP1R: Scaling factor 1 for rain
        ACCINTP2R: Scaling factor 2 for rain
        NACCLBDAR: Number of points in lookup table for rain

    Returns:
        Tuple of (floor index, fractional part)
    """
    index = max(1.00001, min(NACCLBDAR - 0.00001, ACCINTP1R * log(lambda_r) + ACCINTP2R))
    return floor(index), index - floor(index)


@dace.program
def compute_freezing_rate(
    prhodref: dace.float32[I, J, K],
    ppres: dace.float32[I, J, K],
    pdv: dace.float32[I, J, K],
    pka: dace.float32[I, J, K],
    pcj: dace.float32[I, J, K],
    plbdas: dace.float32[I, J, K],
    pt: dace.float32[I, J, K],
    prvt: dace.float32[I, J, K],
    prst: dace.float32[I, J, K],
    priaggs: dace.float32[I, J, K],
    ldcompute: dace.bool[I, J, K],
    ldsoft: dace.bool,
    levlimit: dace.bool,
    zfreez_rate: dace.float32[I, J, K],
    freez1_tend: dace.float32[I, J, K],
    freez2_tend: dace.float32[I, J, K],
    S_RTMIN: dace.float32,
    XEPSILO: dace.float32,
    XALPI: dace.float32,
    XBETAI: dace.float32,
    XGAMI: dace.float32,
    XTT: dace.float32,
    XLVTT: dace.float32,
    XCPV: dace.float32,
    XCL: dace.float32,
    XCI: dace.float32,
    XLMTT: dace.float32,
    XESTT: dace.float32,
    XRV: dace.float32,
    X0DEPS: dace.float32,
    X1DEPS: dace.float32,
    XEX0DEPS: dace.float32,
    XEX1DEPS: dace.float32,
):
    """Compute maximum freezing rate for snow processes"""
    @dace.map
    def compute_freezing(i: _[0:I], j: _[0:J], k: _[0:K]):
        if prst[i, j, k] > S_RTMIN and ldcompute[i, j, k]:
            if not ldsoft:
                # Compute vapor pressure
                prs_ev = prvt[i, j, k] * ppres[i, j, k] / (XEPSILO + prvt[i, j, k])
                
                # Apply saturation limit if requested
                if levlimit:
                    prs_ev = min(prs_ev, exp(XALPI - XBETAI / pt[i, j, k] - XGAMI * log(pt[i, j, k])))
                
                # Compute first freezing term
                freez1_tend[i, j, k] = pka[i, j, k] * (XTT - pt[i, j, k]) + \
                    (pdv[i, j, k] * (XLVTT + (XCPV - XCL) * (pt[i, j, k] - XTT)) * \
                     (XESTT - prs_ev) / (XRV * pt[i, j, k]))
                
                freez1_tend[i, j, k] = freez1_tend[i, j, k] * \
                    (X0DEPS * plbdas[i, j, k] ** XEX0DEPS + \
                     X1DEPS * pcj[i, j, k] * plbdas[i, j, k] ** XEX1DEPS) / \
                    (prhodref[i, j, k] * (XLMTT - XCL * (XTT - pt[i, j, k])))
                
                # Compute second freezing term
                freez2_tend[i, j, k] = (prhodref[i, j, k] * (XLMTT + (XCI - XCL) * (XTT - pt[i, j, k]))) / \
                    (prhodref[i, j, k] * (XLMTT - XCL * (XTT - pt[i, j, k])))
            
            # Compute total freezing rate
            zfreez_rate[i, j, k] = max(0.0, max(0.0, freez1_tend[i, j, k] + \
                                    freez2_tend[i, j, k] * priaggs[i, j, k]) - priaggs[i, j, k])
        else:
            freez1_tend[i, j, k] = 0.0
            freez2_tend[i, j, k] = 0.0
            zfreez_rate[i, j, k] = 0.0


@dace.program
def cloud_droplet_riming_snow(
    prhodref: dace.float32[I, J, K],
    plbdas: dace.float32[I, J, K],
    pt: dace.float32[I, J, K],
    prct: dace.float32[I, J, K],
    prst: dace.float32[I, J, K],
    ldcompute: dace.bool[I, J, K],
    ldsoft: dace.bool,
    csnowriming: str,
    grim: dace.bool[I, J, K],
    zfreez_rate: dace.float32[I, J, K],
    prcrimss: dace.float32[I, J, K],
    prcrimsg: dace.float32[I, J, K],
    prsrimcg: dace.float32[I, J, K],
    zzw1: dace.float32[I, J, K],
    zzw2: dace.float32[I, J, K],
    zzw3: dace.float32[I, J, K],
    rcrims_tend: dace.float32[I, J, K],
    rcrimss_tend: dace.float32[I, J, K],
    rsrimcg_tend: dace.float32[I, J, K],
    ker_gaminc_rim1: dace.float32[F],
    ker_gaminc_rim2: dace.float32[F],
    ker_gaminc_rim4: dace.float32[F],
    C_RTMIN: dace.float32,
    S_RTMIN: dace.float32,
    XTT: dace.float32,
    XCRIMSS: dace.float32,
    XEXCRIMSS: dace.float32,
    XCRIMSG: dace.float32,
    XEXCRIMSG: dace.float32,
    XCEXVT: dace.float32,
    XSRIMCG: dace.float32,
    XEXSRIMCG: dace.float32,
    XSRIMCG2: dace.float32,
    XSRIMCG3: dace.float32,
    XEXSRIMCG2: dace.float32,
    RIMINTP1: dace.float32,
    RIMINTP2: dace.float32,
    NGAMINC: dace.int32,
):
    """Compute cloud droplet riming of aggregates"""
    
    # Initialize masks and tendencies
    @dace.map
    def init_riming(i: _[0:I], j: _[0:J], k: _[0:K]):
        if prct[i, j, k] > C_RTMIN and prst[i, j, k] > S_RTMIN and ldcompute[i, j, k]:
            grim[i, j, k] = True
        else:
            grim[i, j, k] = False
            rcrims_tend[i, j, k] = 0.0
            rcrimss_tend[i, j, k] = 0.0
            rsrimcg_tend[i, j, k] = 0.0
    
    # Interpolate and compute riming rates
    @dace.map
    def compute_riming_interp(i: _[0:I], j: _[0:J], k: _[0:K]):
        if (not ldsoft) and grim[i, j, k]:
            # Compute interpolation indices
            index_floor, index_frac = index_micro1d_rim(
                plbdas[i, j, k], RIMINTP1, RIMINTP2, NGAMINC
            )
            
            # Interpolate from lookup tables
            zzw1[i, j, k] = ker_gaminc_rim1[index_floor + 1] * index_frac - \
                          ker_gaminc_rim1[index_floor] * (index_frac - 1.0)
            
            zzw2[i, j, k] = ker_gaminc_rim2[index_floor + 1] * index_frac - \
                          ker_gaminc_rim2[index_floor] * (index_frac - 1.0)
            
            zzw3[i, j, k] = ker_gaminc_rim4[index_floor + 1] * index_frac - \
                          ker_gaminc_rim4[index_floor] * (index_frac - 1.0)
            
            # Riming of small sized aggregates
            rcrimss_tend[i, j, k] = XCRIMSS * zzw1[i, j, k] * prct[i, j, k] * \
                                   plbdas[i, j, k] ** XEXCRIMSS * \
                                   prhodref[i, j, k] ** (-XCEXVT)
            
            # Riming-conversion of large sized aggregates
            rcrims_tend[i, j, k] = XCRIMSG * prct[i, j, k] * \
                                  plbdas[i, j, k] ** XEXCRIMSG * \
                                  prhodref[i, j, k] ** (-XCEXVT)
            
            # Conversion to graupel (Murakami 1990)
            if csnowriming == 'M90 ':
                zzw_tmp = rcrims_tend[i, j, k] - rcrimss_tend[i, j, k]
                term_conversion = XSRIMCG * plbdas[i, j, k] ** XEXSRIMCG * (1.0 - zzw2[i, j, k])
                
                rsrimcg_tend[i, j, k] = zzw_tmp * term_conversion / \
                    max(1.0e-20, XSRIMCG3 * XSRIMCG2 * plbdas[i, j, k] ** XEXSRIMCG2 * \
                        (1.0 - zzw3[i, j, k]) - XSRIMCG3 * term_conversion)
            else:
                rsrimcg_tend[i, j, k] = 0.0
    
    # Apply freezing rate limitations and temperature conditions
    @dace.map
    def apply_freezing_limits(i: _[0:I], j: _[0:J], k: _[0:K]):
        if grim[i, j, k] and pt[i, j, k] < XTT:
            # Apply freezing rate limits
            prcrimss[i, j, k] = min(zfreez_rate[i, j, k], rcrimss_tend[i, j, k])
            zfreez_remaining = max(0.0, zfreez_rate[i, j, k] - prcrimss[i, j, k])
            
            # Proportion we can freeze
            zzw_prop = min(1.0, zfreez_remaining / max(1.0e-20, rcrims_tend[i, j, k] - prcrimss[i, j, k]))
            
            prcrimsg[i, j, k] = zzw_prop * max(0.0, rcrims_tend[i, j, k] - prcrimss[i, j, k])
            zfreez_remaining = max(0.0, zfreez_remaining - prcrimsg[i, j, k])
            
            prsrimcg[i, j, k] = zzw_prop * rsrimcg_tend[i, j, k]
            
            # Ensure positive values
            prsrimcg[i, j, k] = prsrimcg[i, j, k] * max(0.0, -sign(1.0, -prcrimsg[i, j, k]))
            prcrimsg[i, j, k] = max(0.0, prcrimsg[i, j, k])
        else:
            prcrimss[i, j, k] = 0.0
            prcrimsg[i, j, k] = 0.0
            prsrimcg[i, j, k] = 0.0


@dace.program
def rain_accretion_snow(
    prhodref: dace.float32[I, J, K],
    plbdas: dace.float32[I, J, K],
    plbdar: dace.float32[I, J, K],
    pt: dace.float32[I, J, K],
    prrt: dace.float32[I, J, K],
    prst: dace.float32[I, J, K],
    ldcompute: dace.bool[I, J, K],
    ldsoft: dace.bool,
    gacc: dace.bool[I, J, K],
    zfreez_rate: dace.float32[I, J, K],
    prraccss: dace.float32[I, J, K],
    prraccsg: dace.float32[I, J, K],
    prsaccrg: dace.float32[I, J, K],
    zzw1: dace.float32[I, J, K],
    zzw2: dace.float32[I, J, K],
    zzw3: dace.float32[I, J, K],
    zzw_coef: dace.float32[I, J, K],
    rraccs_tend: dace.float32[I, J, K],
    rraccss_tend: dace.float32[I, J, K],
    rsaccrg_tend: dace.float32[I, J, K],
    ker_raccss: dace.float32[F, F],
    ker_raccs: dace.float32[F, F],
    ker_saccrg: dace.float32[F, F],
    R_RTMIN: dace.float32,
    S_RTMIN: dace.float32,
    XTT: dace.float32,
    XFRACCSS: dace.float32,
    XCXS: dace.float32,
    XBS: dace.float32,
    XCEXVT: dace.float32,
    XLBRACCS1: dace.float32,
    XLBRACCS2: dace.float32,
    XLBRACCS3: dace.float32,
    XFSACCRG: dace.float32,
    XLBSACCR1: dace.float32,
    XLBSACCR2: dace.float32,
    XLBSACCR3: dace.float32,
    ACCINTP1S: dace.float32,
    ACCINTP2S: dace.float32,
    NACCLBDAS: dace.int32,
    ACCINTP1R: dace.float32,
    ACCINTP2R: dace.float32,
    NACCLBDAR: dace.int32,
):
    """Compute rain accretion onto aggregates"""
    
    # Initialize masks and tendencies
    @dace.map
    def init_accretion(i: _[0:I], j: _[0:J], k: _[0:K]):
        if prrt[i, j, k] > R_RTMIN and prst[i, j, k] > S_RTMIN and ldcompute[i, j, k]:
            gacc[i, j, k] = True
        else:
            gacc[i, j, k] = False
            rraccs_tend[i, j, k] = 0.0
            rraccss_tend[i, j, k] = 0.0
            rsaccrg_tend[i, j, k] = 0.0
    
    # Interpolate and compute accretion rates
    @dace.map
    def compute_accretion_interp(i: _[0:I], j: _[0:J], k: _[0:K]):
        if (not ldsoft) and gacc[i, j, k]:
            # Compute 2D interpolation indices
            index_floor_s, index_frac_s = index_micro2d_acc_s(
                plbdas[i, j, k], ACCINTP1S, ACCINTP2S, NACCLBDAS
            )
            index_floor_r, index_frac_r = index_micro2d_acc_r(
                plbdar[i, j, k], ACCINTP1R, ACCINTP2R, NACCLBDAR
            )
            
            # Bilinear interpolation for RACCSS kernel
            zzw1[i, j, k] = (ker_raccss[index_floor_s + 1, index_floor_r + 1] * index_frac_r - \
                           ker_raccss[index_floor_s + 1, index_floor_r] * (index_frac_r - 1.0)) * index_frac_s - \
                          (ker_raccss[index_floor_s, index_floor_r + 1] * index_frac_r - \
                           ker_raccss[index_floor_s, index_floor_r] * (index_frac_r - 1.0)) * (index_frac_s - 1.0)
            
            # Bilinear interpolation for RACCS kernel
            zzw2[i, j, k] = (ker_raccs[index_floor_s + 1, index_floor_r + 1] * index_frac_r - \
                           ker_raccs[index_floor_s + 1, index_floor_r] * (index_frac_r - 1.0)) * index_frac_s - \
                          (ker_raccs[index_floor_s, index_floor_r + 1] * index_frac_r - \
                           ker_raccs[index_floor_s, index_floor_r] * (index_frac_r - 1.0)) * (index_frac_s - 1.0)
            
            # Bilinear interpolation for SACCRG kernel
            zzw3[i, j, k] = (ker_saccrg[index_floor_s + 1, index_floor_r + 1] * index_frac_r - \
                           ker_saccrg[index_floor_s + 1, index_floor_r] * (index_frac_r - 1.0)) * index_frac_s - \
                          (ker_saccrg[index_floor_s, index_floor_r + 1] * index_frac_r - \
                           ker_saccrg[index_floor_s, index_floor_r] * (index_frac_r - 1.0)) * (index_frac_s - 1.0)
            
            # Coefficient for RRACCS
            zzw_coef[i, j, k] = XFRACCSS * (plbdas[i, j, k] ** XCXS) * \
                               (prhodref[i, j, k] ** (-XCEXVT - 1.0)) * \
                               (XLBRACCS1 / (plbdas[i, j, k] ** 2) + \
                                XLBRACCS2 / (plbdas[i, j, k] * plbdar[i, j, k]) + \
                                XLBRACCS3 / (plbdar[i, j, k] ** 2)) / (plbdar[i, j, k] ** 4)
            
            # Raindrop accretion on small sized aggregates
            rraccss_tend[i, j, k] = zzw1[i, j, k] * zzw_coef[i, j, k]
            
            # Raindrop accretion on aggregates
            rraccs_tend[i, j, k] = zzw2[i, j, k] * zzw_coef[i, j, k]
            
            # Raindrop accretion-conversion to graupel
            rsaccrg_tend[i, j, k] = XFSACCRG * zzw3[i, j, k] * \
                                   (plbdas[i, j, k] ** (XCXS - XBS)) * \
                                   (prhodref[i, j, k] ** (-XCEXVT - 1.0)) * \
                                   (XLBSACCR1 / (plbdar[i, j, k] ** 2) + \
                                    XLBSACCR2 / (plbdar[i, j, k] * plbdas[i, j, k]) + \
                                    XLBSACCR3 / (plbdas[i, j, k] ** 2)) / plbdar[i, j, k]
    
    # Apply freezing rate limitations and temperature conditions
    @dace.map
    def apply_freezing_limits(i: _[0:I], j: _[0:J], k: _[0:K]):
        if gacc[i, j, k] and pt[i, j, k] < XTT:
            # Apply freezing rate limits
            prraccss[i, j, k] = min(zfreez_rate[i, j, k], rraccss_tend[i, j, k])
            zfreez_remaining = max(0.0, zfreez_rate[i, j, k] - prraccss[i, j, k])
            
            # Proportion we can freeze
            zzw_prop = min(1.0, zfreez_remaining / max(1.0e-20, rraccs_tend[i, j, k] - prraccss[i, j, k]))
            
            prraccsg[i, j, k] = zzw_prop * max(0.0, rraccs_tend[i, j, k] - prraccss[i, j, k])
            zfreez_remaining = max(0.0, zfreez_remaining - prraccsg[i, j, k])
            
            prsaccrg[i, j, k] = zzw_prop * rsaccrg_tend[i, j, k]
            
            # Ensure positive values
            prsaccrg[i, j, k] = prsaccrg[i, j, k] * max(0.0, -sign(1.0, -prraccsg[i, j, k]))
            prraccsg[i, j, k] = max(0.0, prraccsg[i, j, k])
        else:
            prraccss[i, j, k] = 0.0
            prraccsg[i, j, k] = 0.0
            prsaccrg[i, j, k] = 0.0


@dace.program
def conversion_melting_snow(
    prhodref: dace.float32[I, J, K],
    ppres: dace.float32[I, J, K],
    pdv: dace.float32[I, J, K],
    pka: dace.float32[I, J, K],
    pcj: dace.float32[I, J, K],
    plbdas: dace.float32[I, J, K],
    pt: dace.float32[I, J, K],
    prvt: dace.float32[I, J, K],
    prst: dace.float32[I, J, K],
    ldcompute: dace.bool[I, J, K],
    ldsoft: dace.bool,
    levlimit: dace.bool,
    prsmltg: dace.float32[I, J, K],
    prcmltsr: dace.float32[I, J, K],
    rcrims_tend: dace.float32[I, J, K],
    rraccs_tend: dace.float32[I, J, K],
    S_RTMIN: dace.float32,
    XEPSILO: dace.float32,
    XALPW: dace.float32,
    XBETAW: dace.float32,
    XGAMW: dace.float32,
    XTT: dace.float32,
    XLVTT: dace.float32,
    XCPV: dace.float32,
    XCL: dace.float32,
    XLMTT: dace.float32,
    XESTT: dace.float32,
    XRV: dace.float32,
    X0DEPS: dace.float32,
    X1DEPS: dace.float32,
    XEX0DEPS: dace.float32,
    XEX1DEPS: dace.float32,
    XFSCVMG: dace.float32,
):
    """Compute conversion-melting of aggregates"""
    @dace.map
    def compute_melting(i: _[0:I], j: _[0:J], k: _[0:K]):
        if prst[i, j, k] > S_RTMIN and pt[i, j, k] > XTT and ldcompute[i, j, k]:
            if not ldsoft:
                # Compute vapor pressure
                prs_ev = prvt[i, j, k] * ppres[i, j, k] / (XEPSILO + prvt[i, j, k])
                
                # Apply saturation limit if requested
                if levlimit:
                    prs_ev = min(prs_ev, exp(XALPW - XBETAW / pt[i, j, k] - XGAMW * log(pt[i, j, k])))
                
                # Compute melting term
                prsmltg[i, j, k] = pka[i, j, k] * (XTT - pt[i, j, k]) + \
                    (pdv[i, j, k] * (XLVTT + (XCPV - XCL) * (pt[i, j, k] - XTT)) * \
                     (XESTT - prs_ev) / (XRV * pt[i, j, k]))
                
                # Compute RSMLT
                prsmltg[i, j, k] = XFSCVMG * max(0.0, (-prsmltg[i, j, k] * \
                    (X0DEPS * plbdas[i, j, k] ** XEX0DEPS + \
                     X1DEPS * pcj[i, j, k] * plbdas[i, j, k] ** XEX1DEPS) - \
                    (rcrims_tend[i, j, k] + rraccs_tend[i, j, k]) * \
                    (prhodref[i, j, k] * XCL * (XTT - pt[i, j, k]))) / \
                    (prhodref[i, j, k] * XLMTT))
                
                # Collection rate (both species liquid, no heat exchange)
                prcmltsr[i, j, k] = rcrims_tend[i, j, k]
        else:
            prsmltg[i, j, k] = 0.0
            prcmltsr[i, j, k] = 0.0
