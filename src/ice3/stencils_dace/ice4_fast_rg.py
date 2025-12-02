# -*- coding: utf-8 -*-
"""
ICE4 Fast RG (Graupel) Processes - DaCe Implementation

This module implements the fast growth processes for graupel in the ICE4
microphysics scheme, translated from the Fortran reference in mode_ice4_fast_rg.F90.

Processes implemented:
- Rain contact freezing (RICFRRG, RRCFRIG, PRICFRR)
- Wet and dry collection of cloud droplets and pristine ice on graupel
- Collection of snow on graupel (wet and dry growth)
- Collection of rain on graupel (dry growth)
- Graupel growth mode determination (wet vs dry)
- Conversion to hail (if KRR=7)
- Graupel melting

Key Features:
- Bilinear interpolation from 2D lookup tables for collection kernels
- Heat balance constraints for contact freezing
- Temperature-dependent collection efficiencies
- Wet/dry growth mode transition based on heat budget

Reference:
    PHYEX-IAL_CY50T1/common/micro/mode_ice4_fast_rg.F90
    
Author:
    Translated to Python/DaCe from Fortran by Cline AI Assistant
"""

import dace
from typing import Tuple

I = dace.symbol("I")
J = dace.symbol("J")
K = dace.symbol("K")
F = dace.symbol("F")


def index_micro2d_dry_g(
    lambda_g: dace.float32,
    DRYINTP1G: dace.float32,
    DRYINTP2G: dace.float32,
    NDRYLBDAG: dace.int32,
) -> Tuple[dace.int32, dace.float32]:
    """Compute index in logspace for table (graupel dimension)

    Args:
        lambda_g: Graupel slope parameter
        DRYINTP1G: Scaling factor 1 for graupel
        DRYINTP2G: Scaling factor 2 for graupel
        NDRYLBDAG: Number of points in lookup table for graupel

    Returns:
        Tuple of (floor index, fractional part)
    """
    index = max(1.00001, min(NDRYLBDAG - 0.00001, DRYINTP1G * log(lambda_g) + DRYINTP2G))
    return floor(index), index - floor(index)


def index_micro2d_dry_r(
    lambda_r: dace.float32,
    DRYINTP1R: dace.float32,
    DRYINTP2R: dace.float32,
    NDRYLBDAR: dace.int32,
) -> Tuple[dace.int32, dace.float32]:
    """Compute index in logspace for table (rain dimension)

    Args:
        lambda_r: Rain slope parameter
        DRYINTP1R: Scaling factor 1 for rain
        DRYINTP2R: Scaling factor 2 for rain
        NDRYLBDAR: Number of points in lookup table for rain

    Returns:
        Tuple of (floor index, fractional part)
    """
    index = max(1.00001, min(NDRYLBDAR - 0.00001, DRYINTP1R * log(lambda_r) + DRYINTP2R))
    return floor(index), index - floor(index)


def index_micro2d_dry_s(
    lambda_s: dace.float32,
    DRYINTP1S: dace.float32,
    DRYINTP2S: dace.float32,
    NDRYLBDAS: dace.int32,
) -> Tuple[dace.int32, dace.float32]:
    """Compute index in logspace for table (snow dimension)

    Args:
        lambda_s: Snow slope parameter
        DRYINTP1S: Scaling factor 1 for snow
        DRYINTP2S: Scaling factor 2 for snow
        NDRYLBDAS: Number of points in lookup table for snow

    Returns:
        Tuple of (floor index, fractional part)
    """
    index = max(1.00001, min(NDRYLBDAS - 0.00001, DRYINTP1S * log(lambda_s) + DRYINTP2S))
    return floor(index), index - floor(index)


@dace.program
def rain_contact_freezing(
    prhodref: dace.float32[I, J, K],
    plbdar: dace.float32[I, J, K],
    pt: dace.float32[I, J, K],
    prit: dace.float32[I, J, K],
    prrt: dace.float32[I, J, K],
    pcit: dace.float32[I, J, K],
    ldcompute: dace.bool[I, J, K],
    ldsoft: dace.bool,
    lcrflimit: dace.bool,
    pricfrrg: dace.float32[I, J, K],
    prrcfrig: dace.float32[I, J, K],
    pricfrr: dace.float32[I, J, K],
    I_RTMIN: dace.float32,
    R_RTMIN: dace.float32,
    XICFRR: dace.float32,
    XEXICFRR: dace.float32,
    XCEXVT: dace.float32,
    XRCFRI: dace.float32,
    XEXRCFRI: dace.float32,
    XTT: dace.float32,
    XCI: dace.float32,
    XCL: dace.float32,
    XLVTT: dace.float32,
):
    """Compute rain contact freezing"""
    @dace.map
    def compute_contact_freezing(i: _[0:I], j: _[0:J], k: _[0:K]):
        if prit[i, j, k] > I_RTMIN and prrt[i, j, k] > R_RTMIN and ldcompute[i, j, k]:
            if not ldsoft:
                # RICFRRG - pristine ice collection by rain leading to graupel
                pricfrrg[i, j, k] = XICFRR * prit[i, j, k] * \
                    plbdar[i, j, k] ** XEXICFRR * \
                    prhodref[i, j, k] ** (-XCEXVT)
                
                # RRCFRIG - rain freezing by contact with pristine ice
                prrcfrig[i, j, k] = XRCFRI * pcit[i, j, k] * \
                    plbdar[i, j, k] ** XEXRCFRI * \
                    prhodref[i, j, k] ** (-XCEXVT - 1.0)
                
                if lcrflimit:
                    # Limit based on heat balance
                    # Proportion of process that can take place
                    zzw_prop = max(0.0, min(1.0, 
                        (pricfrrg[i, j, k] * XCI + prrcfrig[i, j, k] * XCL) * (XTT - pt[i, j, k]) / \
                        max(1.0e-20, XLVTT * prrcfrig[i, j, k])))
                    
                    prrcfrig[i, j, k] = zzw_prop * prrcfrig[i, j, k]
                    pricfrr[i, j, k] = (1.0 - zzw_prop) * pricfrrg[i, j, k]
                    pricfrrg[i, j, k] = zzw_prop * pricfrrg[i, j, k]
                else:
                    pricfrr[i, j, k] = 0.0
        else:
            pricfrrg[i, j, k] = 0.0
            prrcfrig[i, j, k] = 0.0
            pricfrr[i, j, k] = 0.0


@dace.program
def cloud_pristine_collection_graupel(
    prhodref: dace.float32[I, J, K],
    plbdag: dace.float32[I, J, K],
    pt: dace.float32[I, J, K],
    prct: dace.float32[I, J, K],
    prit: dace.float32[I, J, K],
    prgt: dace.float32[I, J, K],
    ldcompute: dace.bool[I, J, K],
    ldsoft: dace.bool,
    rcdryg_tend: dace.float32[I, J, K],
    ridryg_tend: dace.float32[I, J, K],
    riwetg_tend: dace.float32[I, J, K],
    C_RTMIN: dace.float32,
    I_RTMIN: dace.float32,
    G_RTMIN: dace.float32,
    XTT: dace.float32,
    XFCDRYG: dace.float32,
    XFIDRYG: dace.float32,
    XCOLIG: dace.float32,
    XCOLEXIG: dace.float32,
    XCXG: dace.float32,
    XDG: dace.float32,
    XCEXVT: dace.float32,
):
    """Compute wet and dry collection of cloud and pristine ice on graupel"""
    @dace.map
    def compute_collection(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Cloud droplet collection
        if prgt[i, j, k] > G_RTMIN and prct[i, j, k] > C_RTMIN and ldcompute[i, j, k]:
            if not ldsoft:
                rcdryg_tend[i, j, k] = plbdag[i, j, k] ** (XCXG - XDG - 2.0) * \
                    prhodref[i, j, k] ** (-XCEXVT)
                rcdryg_tend[i, j, k] = XFCDRYG * prct[i, j, k] * rcdryg_tend[i, j, k]
        else:
            rcdryg_tend[i, j, k] = 0.0
        
        # Pristine ice collection
        if prgt[i, j, k] > G_RTMIN and prit[i, j, k] > I_RTMIN and ldcompute[i, j, k]:
            if not ldsoft:
                base_tend = plbdag[i, j, k] ** (XCXG - XDG - 2.0) * \
                    prhodref[i, j, k] ** (-XCEXVT)
                
                ridryg_tend[i, j, k] = XFIDRYG * exp(XCOLEXIG * (pt[i, j, k] - XTT)) * \
                    prit[i, j, k] * base_tend
                
                riwetg_tend[i, j, k] = ridryg_tend[i, j, k] / \
                    (XCOLIG * exp(XCOLEXIG * (pt[i, j, k] - XTT)))
        else:
            ridryg_tend[i, j, k] = 0.0
            riwetg_tend[i, j, k] = 0.0


@dace.program
def snow_collection_on_graupel(
    prhodref: dace.float32[I, J, K],
    plbdas: dace.float32[I, J, K],
    plbdag: dace.float32[I, J, K],
    pt: dace.float32[I, J, K],
    prst: dace.float32[I, J, K],
    prgt: dace.float32[I, J, K],
    ldcompute: dace.bool[I, J, K],
    ldsoft: dace.bool,
    gdry: dace.bool[I, J, K],
    zzw: dace.float32[I, J, K],
    rswetg_tend: dace.float32[I, J, K],
    rsdryg_tend: dace.float32[I, J, K],
    ker_sdryg: dace.float32[F, F],
    S_RTMIN: dace.float32,
    G_RTMIN: dace.float32,
    XTT: dace.float32,
    XFSDRYG: dace.float32,
    XCOLSG: dace.float32,
    XCOLEXSG: dace.float32,
    XCXS: dace.float32,
    XBS: dace.float32,
    XCXG: dace.float32,
    XCEXVT: dace.float32,
    XLBSDRYG1: dace.float32,
    XLBSDRYG2: dace.float32,
    XLBSDRYG3: dace.float32,
    DRYINTP1G: dace.float32,
    DRYINTP2G: dace.float32,
    NDRYLBDAG: dace.int32,
    DRYINTP1S: dace.float32,
    DRYINTP2S: dace.float32,
    NDRYLBDAS: dace.int32,
):
    """Compute wet and dry collection of snow on graupel"""
    
    # Initialize masks
    @dace.map
    def init_snow_collection(i: _[0:I], j: _[0:J], k: _[0:K]):
        if prst[i, j, k] > S_RTMIN and prgt[i, j, k] > G_RTMIN and ldcompute[i, j, k]:
            gdry[i, j, k] = True
        else:
            gdry[i, j, k] = False
            rsdryg_tend[i, j, k] = 0.0
            rswetg_tend[i, j, k] = 0.0
    
    # Interpolate and compute collection rates
    @dace.map
    def compute_snow_interp(i: _[0:I], j: _[0:J], k: _[0:K]):
        if (not ldsoft) and gdry[i, j, k]:
            # Compute 2D interpolation indices
            index_floor_g, index_frac_g = index_micro2d_dry_g(
                plbdag[i, j, k], DRYINTP1G, DRYINTP2G, NDRYLBDAG
            )
            index_floor_s, index_frac_s = index_micro2d_dry_s(
                plbdas[i, j, k], DRYINTP1S, DRYINTP2S, NDRYLBDAS
            )
            
            # Bilinear interpolation
            zzw[i, j, k] = (ker_sdryg[index_floor_g + 1, index_floor_s + 1] * index_frac_s - \
                          ker_sdryg[index_floor_g + 1, index_floor_s] * (index_frac_s - 1.0)) * index_frac_g - \
                         (ker_sdryg[index_floor_g, index_floor_s + 1] * index_frac_s - \
                          ker_sdryg[index_floor_g, index_floor_s] * (index_frac_s - 1.0)) * (index_frac_g - 1.0)
            
            # Compute wet growth rate
            rswetg_tend[i, j, k] = XFSDRYG * zzw[i, j, k] / XCOLSG * \
                (plbdas[i, j, k] ** (XCXS - XBS)) * (plbdag[i, j, k] ** XCXG) * \
                (prhodref[i, j, k] ** (-XCEXVT - 1.0)) * \
                (XLBSDRYG1 / (plbdag[i, j, k] ** 2) + \
                 XLBSDRYG2 / (plbdag[i, j, k] * plbdas[i, j, k]) + \
                 XLBSDRYG3 / (plbdas[i, j, k] ** 2))
            
            # Compute dry growth rate
            rsdryg_tend[i, j, k] = rswetg_tend[i, j, k] * XCOLSG * \
                exp(XCOLEXSG * (pt[i, j, k] - XTT))


@dace.program
def rain_accretion_on_graupel(
    prhodref: dace.float32[I, J, K],
    plbdar: dace.float32[I, J, K],
    plbdag: dace.float32[I, J, K],
    prrt: dace.float32[I, J, K],
    prgt: dace.float32[I, J, K],
    ldcompute: dace.bool[I, J, K],
    ldsoft: dace.bool,
    gdry: dace.bool[I, J, K],
    zzw: dace.float32[I, J, K],
    rrdryg_tend: dace.float32[I, J, K],
    ker_rdryg: dace.float32[F, F],
    R_RTMIN: dace.float32,
    G_RTMIN: dace.float32,
    XFRDRYG: dace.float32,
    XCXG: dace.float32,
    XCEXVT: dace.float32,
    XLBRDRYG1: dace.float32,
    XLBRDRYG2: dace.float32,
    XLBRDRYG3: dace.float32,
    DRYINTP1G: dace.float32,
    DRYINTP2G: dace.float32,
    NDRYLBDAG: dace.int32,
    DRYINTP1R: dace.float32,
    DRYINTP2R: dace.float32,
    NDRYLBDAR: dace.int32,
):
    """Compute accretion of raindrops on graupel"""
    
    # Initialize masks
    @dace.map
    def init_rain_accretion(i: _[0:I], j: _[0:J], k: _[0:K]):
        if prrt[i, j, k] > R_RTMIN and prgt[i, j, k] > G_RTMIN and ldcompute[i, j, k]:
            gdry[i, j, k] = True
        else:
            gdry[i, j, k] = False
            rrdryg_tend[i, j, k] = 0.0
    
    # Interpolate and compute accretion rates
    @dace.map
    def compute_rain_interp(i: _[0:I], j: _[0:J], k: _[0:K]):
        if (not ldsoft) and gdry[i, j, k]:
            # Compute 2D interpolation indices
            index_floor_g, index_frac_g = index_micro2d_dry_g(
                plbdag[i, j, k], DRYINTP1G, DRYINTP2G, NDRYLBDAG
            )
            index_floor_r, index_frac_r = index_micro2d_dry_r(
                plbdar[i, j, k], DRYINTP1R, DRYINTP2R, NDRYLBDAR
            )
            
            # Bilinear interpolation
            zzw[i, j, k] = (ker_rdryg[index_floor_g + 1, index_floor_r + 1] * index_frac_r - \
                          ker_rdryg[index_floor_g + 1, index_floor_r] * (index_frac_r - 1.0)) * index_frac_g - \
                         (ker_rdryg[index_floor_g, index_floor_r + 1] * index_frac_r - \
                          ker_rdryg[index_floor_g, index_floor_r] * (index_frac_r - 1.0)) * (index_frac_g - 1.0)
            
            # Compute dry growth rate
            rrdryg_tend[i, j, k] = XFRDRYG * zzw[i, j, k] * \
                (plbdar[i, j, k] ** (-4.0)) * (plbdag[i, j, k] ** XCXG) * \
                (prhodref[i, j, k] ** (-XCEXVT - 1.0)) * \
                (XLBRDRYG1 / (plbdag[i, j, k] ** 2) + \
                 XLBRDRYG2 / (plbdag[i, j, k] * plbdar[i, j, k]) + \
                 XLBRDRYG3 / (plbdar[i, j, k] ** 2))


@dace.program
def compute_graupel_growth_mode(
    prhodref: dace.float32[I, J, K],
    ppres: dace.float32[I, J, K],
    pdv: dace.float32[I, J, K],
    pka: dace.float32[I, J, K],
    pcj: dace.float32[I, J, K],
    plbdag: dace.float32[I, J, K],
    pt: dace.float32[I, J, K],
    prvt: dace.float32[I, J, K],
    prgt: dace.float32[I, J, K],
    prgsi: dace.float32[I, J, K],
    prgsi_mr: dace.float32[I, J, K],
    pricfrrg: dace.float32[I, J, K],
    prrcfrig: dace.float32[I, J, K],
    ldcompute: dace.bool[I, J, K],
    ldsoft: dace.bool,
    levlimit: dace.bool,
    lnullwetg: dace.bool,
    lwetgpost: dace.bool,
    krr: dace.int32,
    ldwetg: dace.bool[I, J, K],
    lldryg: dace.bool[I, J, K],
    zrdryg_init: dace.float32[I, J, K],
    zrwetg_init: dace.float32[I, J, K],
    prwetgh: dace.float32[I, J, K],
    prwetgh_mr: dace.float32[I, J, K],
    prcwetg: dace.float32[I, J, K],
    priwetg: dace.float32[I, J, K],
    prrwetg: dace.float32[I, J, K],
    prswetg: dace.float32[I, J, K],
    prcdryg: dace.float32[I, J, K],
    pridryg: dace.float32[I, J, K],
    prrdryg: dace.float32[I, J, K],
    prsdryg: dace.float32[I, J, K],
    rcdryg_tend: dace.float32[I, J, K],
    ridryg_tend: dace.float32[I, J, K],
    riwetg_tend: dace.float32[I, J, K],
    rsdryg_tend: dace.float32[I, J, K],
    rswetg_tend: dace.float32[I, J, K],
    rrdryg_tend: dace.float32[I, J, K],
    freez1_tend: dace.float32[I, J, K],
    freez2_tend: dace.float32[I, J, K],
    G_RTMIN: dace.float32,
    XTT: dace.float32,
    XEPSILO: dace.float32,
    XALPI: dace.float32,
    XBETAI: dace.float32,
    XGAMI: dace.float32,
    XLVTT: dace.float32,
    XCPV: dace.float32,
    XCL: dace.float32,
    XCI: dace.float32,
    XESTT: dace.float32,
    XRV: dace.float32,
    XLMTT: dace.float32,
    X0DEPG: dace.float32,
    X1DEPG: dace.float32,
    XEX0DEPG: dace.float32,
    XEX1DEPG: dace.float32,
):
    """Determine graupel growth mode (wet vs dry) and compute final tendencies"""
    
    @dace.map
    def compute_growth_mode(i: _[0:I], j: _[0:J], k: _[0:K]):
        if prgt[i, j, k] > G_RTMIN and ldcompute[i, j, k]:
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
                    (X0DEPG * plbdag[i, j, k] ** XEX0DEPG + \
                     X1DEPG * pcj[i, j, k] * plbdag[i, j, k] ** XEX1DEPG) / \
                    (prhodref[i, j, k] * (XLMTT - XCL * (XTT - pt[i, j, k])))
                
                # Compute second freezing term
                freez2_tend[i, j, k] = (prhodref[i, j, k] * (XLMTT + (XCI - XCL) * (XTT - pt[i, j, k]))) / \
                    (prhodref[i, j, k] * (XLMTT - XCL * (XTT - pt[i, j, k])))
            
            # Initial dry growth rate
            zrdryg_init[i, j, k] = rcdryg_tend[i, j, k] + ridryg_tend[i, j, k] + \
                rsdryg_tend[i, j, k] + rrdryg_tend[i, j, k]
            
            # Initial wet growth rate
            zrwetg_init[i, j, k] = max(riwetg_tend[i, j, k] + rswetg_tend[i, j, k],
                max(0.0, freez1_tend[i, j, k] + \
                    freez2_tend[i, j, k] * (riwetg_tend[i, j, k] + rswetg_tend[i, j, k])))
            
            # Determine if wet growth
            ldwetg[i, j, k] = max(0.0, zrwetg_init[i, j, k] - riwetg_tend[i, j, k] - rswetg_tend[i, j, k]) <= \
                max(0.0, zrdryg_init[i, j, k] - ridryg_tend[i, j, k] - rsdryg_tend[i, j, k])
            
            if lnullwetg:
                ldwetg[i, j, k] = ldwetg[i, j, k] and (zrdryg_init[i, j, k] > 0.0)
            else:
                ldwetg[i, j, k] = ldwetg[i, j, k] and (zrwetg_init[i, j, k] > 0.0)
            
            if not lwetgpost:
                ldwetg[i, j, k] = ldwetg[i, j, k] and (pt[i, j, k] < XTT)
            
            # Determine if limited dry growth
            lldryg[i, j, k] = (pt[i, j, k] < XTT) and (zrdryg_init[i, j, k] > 1.0e-20) and \
                (max(0.0, zrwetg_init[i, j, k] - riwetg_tend[i, j, k] - rswetg_tend[i, j, k]) > \
                 max(0.0, zrdryg_init[i, j, k] - ridryg_tend[i, j, k] - rsdryg_tend[i, j, k]))
        else:
            freez1_tend[i, j, k] = 0.0
            freez2_tend[i, j, k] = 0.0
            zrdryg_init[i, j, k] = 0.0
            zrwetg_init[i, j, k] = 0.0
            ldwetg[i, j, k] = False
            lldryg[i, j, k] = False
    
    # Compute conversion to hail (if KRR == 7)
    @dace.map
    def compute_hail_conversion(i: _[0:I], j: _[0:J], k: _[0:K]):
        if krr == 7:
            if ldwetg[i, j, k]:
                prwetgh[i, j, k] = (max(0.0, prgsi[i, j, k] + pricfrrg[i, j, k] + prrcfrig[i, j, k]) + \
                    zrwetg_init[i, j, k]) * zrdryg_init[i, j, k] / (zrwetg_init[i, j, k] + zrdryg_init[i, j, k])
                prwetgh_mr[i, j, k] = max(0.0, prgsi_mr[i, j, k]) * zrdryg_init[i, j, k] / \
                    (zrwetg_init[i, j, k] + zrdryg_init[i, j, k])
            else:
                prwetgh[i, j, k] = 0.0
                prwetgh_mr[i, j, k] = 0.0
        else:
            prwetgh[i, j, k] = 0.0
            prwetgh_mr[i, j, k] = 0.0
    
    # Compute final wet and dry growth tendencies
    @dace.map
    def compute_final_tendencies(i: _[0:I], j: _[0:J], k: _[0:K]):
        # Wet growth (aggregated minus collected)
        if ldwetg[i, j, k]:
            prrwetg[i, j, k] = -(riwetg_tend[i, j, k] + rswetg_tend[i, j, k] + \
                rcdryg_tend[i, j, k] - zrwetg_init[i, j, k])
            prcwetg[i, j, k] = rcdryg_tend[i, j, k]
            priwetg[i, j, k] = riwetg_tend[i, j, k]
            prswetg[i, j, k] = rswetg_tend[i, j, k]
        else:
            prrwetg[i, j, k] = 0.0
            prcwetg[i, j, k] = 0.0
            priwetg[i, j, k] = 0.0
            prswetg[i, j, k] = 0.0
        
        # Dry growth (limited)
        if lldryg[i, j, k]:
            prcdryg[i, j, k] = rcdryg_tend[i, j, k]
            prrdryg[i, j, k] = rrdryg_tend[i, j, k]
            pridryg[i, j, k] = ridryg_tend[i, j, k]
            prsdryg[i, j, k] = rsdryg_tend[i, j, k]
        else:
            prcdryg[i, j, k] = 0.0
            prrdryg[i, j, k] = 0.0
            pridryg[i, j, k] = 0.0
            prsdryg[i, j, k] = 0.0


@dace.program
def graupel_melting(
    prhodref: dace.float32[I, J, K],
    ppres: dace.float32[I, J, K],
    pdv: dace.float32[I, J, K],
    pka: dace.float32[I, J, K],
    pcj: dace.float32[I, J, K],
    plbdag: dace.float32[I, J, K],
    pt: dace.float32[I, J, K],
    prvt: dace.float32[I, J, K],
    prgt: dace.float32[I, J, K],
    ldcompute: dace.bool[I, J, K],
    ldsoft: dace.bool,
    levlimit: dace.bool,
    prgmltr: dace.float32[I, J, K],
    rcdryg_tend: dace.float32[I, J, K],
    rrdryg_tend: dace.float32[I, J, K],
    G_RTMIN: dace.float32,
    XTT: dace.float32,
    XEPSILO: dace.float32,
    XALPW: dace.float32,
    XBETAW: dace.float32,
    XGAMW: dace.float32,
    XLVTT: dace.float32,
    XCPV: dace.float32,
    XCL: dace.float32,
    XESTT: dace.float32,
    XRV: dace.float32,
    XLMTT: dace.float32,
    X0DEPG: dace.float32,
    X1DEPG: dace.float32,
    XEX0DEPG: dace.float32,
    XEX1DEPG: dace.float32,
):
    """Compute melting of graupel"""
    @dace.map
    def compute_melting(i: _[0:I], j: _[0:J], k: _[0:K]):
        if prgt[i, j, k] > G_RTMIN and pt[i, j, k] > XTT and ldcompute[i, j, k]:
            if not ldsoft:
                # Compute vapor pressure
                prs_ev = prvt[i, j, k] * ppres[i, j, k] / (XEPSILO + prvt[i, j, k])
                
                # Apply saturation limit if requested
                if levlimit:
                    prs_ev = min(prs_ev, exp(XALPW - XBETAW / pt[i, j, k] - XGAMW * log(pt[i, j, k])))
                
                # Compute melting rate
                prgmltr[i, j, k] = pka[i, j, k] * (XTT - pt[i, j, k]) + \
                    pdv[i, j, k] * (XLVTT + (XCPV - XCL) * (pt[i, j, k] - XTT)) * \
                    (XESTT - prs_ev) / (XRV * pt[i, j, k])
                
                prgmltr[i, j, k] = max(0.0, (-prgmltr[i, j, k] * \
                    (X0DEPG * plbdag[i, j, k] ** XEX0DEPG + \
                     X1DEPG * pcj[i, j, k] * plbdag[i, j, k] ** XEX1DEPG) - \
                    (rcdryg_tend[i, j, k] + rrdryg_tend[i, j, k]) * \
                    (prhodref[i, j, k] * XCL * (XTT - pt[i, j, k]))) / \
                    (prhodref[i, j, k] * XLMTT))
        else:
            prgmltr[i, j, k] = 0.0
