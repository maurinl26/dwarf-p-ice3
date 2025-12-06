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

import numpy as np
from gt4py.cartesian.gtscript import Field


def index_micro1d_rim(
    lambda_s: np.float32,
    RIMINTP1: np.float32,
    RIMINTP2: np.float32,
    NGAMINC: np.int32,
):
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
    lambda_s: np.float32,
    ACCINTP1S: np.float32,
    ACCINTP2S: np.float32,
    NACCLBDAS: np.int32,
):
    """Compute index in logspace for 2D interpolation table (snow dimension)

    Args:
        lambda_s: Snow slope parameter
        ACCINTP1S: Scaling factor 1 for snow
        ACCINTP2S: Scaling factor 2 for snow
        NACCLBDAS: Number of points in lookup table for snow

    Returns:
        Tuple of (floor index, fractional part)
    """
    index = max(
        1.00001, min(NACCLBDAS - 0.00001, ACCINTP1S * log(lambda_s) + ACCINTP2S)
    )
    return floor(index), index - floor(index)


def index_micro2d_acc_r(
    lambda_r: np.float32,
    ACCINTP1R: np.float32,
    ACCINTP2R: np.float32,
    NACCLBDAR: np.int32,
):
    """Compute index in logspace for 2D interpolation table (rain dimension)

    Args:
        lambda_r: Rain slope parameter
        ACCINTP1R: Scaling factor 1 for rain
        ACCINTP2R: Scaling factor 2 for rain
        NACCLBDAR: Number of points in lookup table for rain

    Returns:
        Tuple of (floor index, fractional part)
    """
    index = max(
        1.00001, min(NACCLBDAR - 0.00001, ACCINTP1R * log(lambda_r) + ACCINTP2R)
    )
    return floor(index), index - floor(index)


# GT4Py stencil
def compute_freezing_rate(
    prhodref: Field["float"],
    ppres: Field["float"],
    pdv: Field["float"],
    pka: Field["float"],
    pcj: Field["float"],
    plbdas: Field["float"],
    pt: Field["float"],
    prvt: Field["float"],
    prst: Field["float"],
    priaggs: Field["float"],
    ldcompute: Field["bool"],
    ldsoft: "bool",
    levlimit: "bool",
    zfreez_rate: Field["float"],
    freez1_tend: Field["float"],
    freez2_tend: Field["float"],
):
    """Compute maximum freezing rate for snow processes"""
    from __externals__ import (
        S_RTMIN,
        XEPSILO,
        XALPI,
        XBETAI,
        XGAMI,
        XTT,
        XLVTT,
        XCPV,
        XCL,
        XCI,
        XLMTT,
        XESTT,
        XRV,
        X0DEPS,
        X1DEPS,
        XEX0DEPS,
        XEX1DEPS,
    )

    with computation(PARALLEL), interval(...):
        if prst > S_RTMIN and ldcompute:
            if not ldsoft:
                # Compute vapor pressure
                prs_ev = prvt * ppres / (XEPSILO + prvt)

                # Apply saturation limit if requested
                if levlimit:
                    prs_ev = min(prs_ev, exp(XALPI - XBETAI / pt - XGAMI * log(pt)))

                # Compute first freezing term
                freez1_tend = pka * (XTT - pt) + (
                    pdv
                    * (XLVTT + (XCPV - XCL) * (pt - XTT))
                    * (XESTT - prs_ev)
                    / (XRV * pt)
                )

                freez1_tend = (
                    freez1_tend
                    * (X0DEPS * plbdas**XEX0DEPS + X1DEPS * pcj * plbdas**XEX1DEPS)
                    / (prhodref * (XLMTT - XCL * (XTT - pt)))
                )

                # Compute second freezing term
                freez2_tend = (prhodref * (XLMTT + (XCI - XCL) * (XTT - pt))) / (
                    prhodref * (XLMTT - XCL * (XTT - pt))
                )

            # Compute total freezing rate
            zfreez_rate = max(
                0.0, max(0.0, freez1_tend + freez2_tend * priaggs) - priaggs
            )
        else:
            freez1_tend = 0.0
            freez2_tend = 0.0
            zfreez_rate = 0.0


# Cupy interpolation stencil
def cloud_droplet_riming_snow(
    prhodref: NDArray,
    plbdas: NDArray,
    pt: NDArray,
    prct: NDArray,
    prst: NDArray,
    ldcompute: np.bool[I, J, K],
    ldsoft: np.bool,
    csnowriming: str,
    grim: np.bool[I, J, K],
    zfreez_rate: NDArray,
    prcrimss: NDArray,
    prcrimsg: NDArray,
    prsrimcg: NDArray,
    zzw1: NDArray,
    zzw2: NDArray,
    zzw3: NDArray,
    rcrims_tend: NDArray,
    rcrimss_tend: NDArray,
    rsrimcg_tend: NDArray,
    ker_gaminc_rim1: NDArray[80],
    ker_gaminc_rim2: np.float32[80],
    ker_gaminc_rim4: np.float32[80],
    C_RTMIN: np.float32,
    S_RTMIN: np.float32,
    XTT: np.float32,
    XCRIMSS: np.float32,
    XEXCRIMSS: np.float32,
    XCRIMSG: np.float32,
    XEXCRIMSG: np.float32,
    XCEXVT: np.float32,
    XSRIMCG: np.float32,
    XEXSRIMCG: np.float32,
    XSRIMCG2: np.float32,
    XSRIMCG3: np.float32,
    XEXSRIMCG2: np.float32,
    RIMINTP1: np.float32,
    RIMINTP2: np.float32,
    NGAMINC: np.int32,
):
    """Compute cloud droplet riming of aggregates"""

    # Initialize masks and tendencies
    grim = prct > C_RTMIN and prst > S_RTMIN and ldcompute
    rcrims_tend = np.where(grim, rcrims_tend, 0.0)
    rcrimss_tend = np.where(grim, rcrimss_tend, 0.0)
    rsrimcg_tend = np.where(grim, rsrimcg_tend, 0.0)

    # Interpolate and compute riming rates
    if (not ldsoft) and grim:
            
            index = np.clip(
                RIMINTP1 * np.log(plbdas) + RIMINTP2,
                1.00001,
                NGAMINC - 0.00001
            ) 
            
            # Compute interpolation indices
            idx = np.floor(index).astype(np.int32)
            weight = index - idx
            idx2 = idx + 1

            # Interpolate from lookup tables
            zzw1 = (
                ker_gaminc_rim1.take(idx2) * weight
            - ker_gaminc_rim1.take(idx) * (weight - 1.)
            )

            zzw2 = (
                ker_gaminc_rim2.take(idx2) * weight
            - ker_gaminc_rim2.take(idx) * (weight - 1.)
            )

            zzw3 = (
                ker_gaminc_rim4.take(idx2) * weight
            - ker_gaminc_rim4.take(idx) * (weight - 1.)
            )
        

    # Riming of small sized aggregates
    rcrimss_tend = XCRIMSS * zzw1 * prct * plbdas**XEXCRIMSS * prhodref ** (-XCEXVT)

    # Riming-conversion of large sized aggregates
    rcrims_tend = XCRIMSG * prct * plbdas**XEXCRIMSG * prhodref ** (-XCEXVT)

    # Conversion to graupel (Murakami 1990)
    if csnowriming == "M90":
        zzw_tmp = rcrims_tend - rcrimss_tend
        term_conversion = XSRIMCG * plbdas**XEXSRIMCG * (1.0 - zzw2)

        rsrimcg_tend = (
            zzw_tmp
            * term_conversion
            / max(
                1.0e-20,
                XSRIMCG3 * XSRIMCG2 * plbdas**XEXSRIMCG2 * (1.0 - zzw3)
                - XSRIMCG3 * term_conversion,
            )
        )
    else:
        rsrimcg_tend = 0.0

    # Apply freezing rate limitations and temperature conditions
    if grim and pt < XTT:
        # Apply freezing rate limits
        prcrimss = min(zfreez_rate, rcrimss_tend)
        zfreez_remaining = max(0.0, zfreez_rate - prcrimss)

        # Proportion we can freeze
        zzw_prop = min(1.0, zfreez_remaining / max(1.0e-20, rcrims_tend - prcrimss))

        prcrimsg = zzw_prop * max(0.0, rcrims_tend - prcrimss)
        zfreez_remaining = max(0.0, zfreez_remaining - prcrimsg)

        prsrimcg = zzw_prop * rsrimcg_tend

        # Ensure positive values
        prsrimcg = prsrimcg * max(0.0, -sign(1.0, -prcrimsg))
        prcrimsg = max(0.0, prcrimsg)
    else:
        prcrimss = 0.0
        prcrimsg = 0.0
        prsrimcg = 0.0


# Cupy interpolation stencil
def rain_accretion_snow(
    prhodref: NDArray,
    plbdas: NDArray,
    plbdar: NDArray,
    pt: NDArray,
    prrt: NDArray,
    prst: NDArray,
    ldcompute: bool,
    ldsoft: bool,
    gacc: np.bool[I, J, K],
    zfreez_rate: NDArray,
    prraccss: NDArray,
    prraccsg: NDArray,
    prsaccrg: NDArray,
    zzw1: NDArray,
    zzw2: NDArray,
    zzw3: NDArray,
    zzw_coef: NDArray,
    rraccs_tend: NDArray,
    rraccss_tend: NDArray,
    rsaccrg_tend: NDArray,
    ker_raccss: np.float32[80, 80],
    ker_raccs: np.float32[80, 80],
    ker_saccrg: np.float32[80, 80],
    R_RTMIN: np.float32,
    S_RTMIN: np.float32,
    XTT: np.float32,
    XFRACCSS: np.float32,
    XCXS: np.float32,
    XBS: np.float32,
    XCEXVT: np.float32,
    XLBRACCS1: np.float32,
    XLBRACCS2: np.float32,
    XLBRACCS3: np.float32,
    XFSACCRG: np.float32,
    XLBSACCR1: np.float32,
    XLBSACCR2: np.float32,
    XLBSACCR3: np.float32,
    ACCINTP1S: np.float32,
    ACCINTP2S: np.float32,
    NACCLBDAS: np.int32,
    ACCINTP1R: np.float32,
    ACCINTP2R: np.float32,
    NACCLBDAR: np.int32,
):
    """Compute rain accretion onto aggregates"""

    # Initialize masks and tendencies
    gacc = prrt > R_RTMIN and prst > S_RTMIN and ldcompute
    rraccs_tend = np.where(gacc, rraccs_tend, 0.0)
    rraccss_tend = np.where(gacc, rraccss_tend, 0.0)
    rsaccrg_tend = np.where(gacc, rsaccrg_tend, 0.0)

    # Interpolate and compute accretion rates
    if (not ldsoft) and gacc:
        # Compute 2D interpolation indices

        # Snow
        index = np.clip(
            ACCINTP1S * np.log(plbdas) + ACCINTP2S,
            1.00001,
            NACCLBDAS - 0.00001
        )

        idx_s = np.floor(index).astype(np.int32)
        weight_s = index - idx_s
        idx_s2 = idx_s + 1

        # Rain
        index = np.clip(
            ACCINTP1R * np.log(plbdar) + ACCINTP2R,
            1.00001,
            NACCLBDAR - 0.00001
            )
        idx_r = np.floor(index).astype(np.int32)
        weight_r = idx_r - index
        idx_r2 = idx_r + 1

        # Bilinear interpolation for RACCSS kernel
        zzw1 = (
                ker_raccss.take(idx_s2, axis=0).take(idx_r2) * weight_r
                - ker_raccss.take(idx_s2, axis=0).take(idx_r) * (weight_r - 1.0)
            ) * weight_s - (
                ker_raccss.take(idx_s, axis=0).take(idx_r2) * weight_r
                - ker_raccss.take(idx_s, axis=0).take(idx_r) * (weight_r - 1.0)
            ) * (weight_s - 1.0)

        # Bilinear interpolation for RACCS kernel
        zzw2 = (
                ker_raccs.take(idx_s2, axis=0).take(idx_r2) * weight_r
                - ker_raccs.take(idx_s2, axis=0).take(idx_r) * (weight_r - 1.0)
            ) * weight_s - (
                ker_raccs.take(idx_s, axis=0).take(idx_r2) * weight_r
                - ker_raccs.take(idx_s, axis=0).take(idx_r) * (weight_r - 1.0)
            ) * (weight_s - 1.0)

        # Bilinear interpolation for SACCRG kernel
        zzw3 = (
                ker_saccrg.take(idx_s2, axis=0).take(idx_r2) * weight_r
                - ker_saccrg.take(idx_s2, axis=0).take(idx_r) * (weight_r - 1.0)
            ) * weight_s - (
                ker_saccrg.take(idx_s, axis=0).take(idx_r2) * weight_r
                - ker_saccrg.take(idx_s, idx_r) * (weight_r - 1.0)
            ) * (weight_s - 1.0)

    # Coefficient for RRACCS
    zzw_coef = (
        XFRACCSS
        * (plbdas**XCXS)
        * (prhodref ** (-XCEXVT - 1.0))
        * (
            XLBRACCS1 / (plbdas**2)
            + XLBRACCS2 / (plbdas * plbdar)
            + XLBRACCS3 / (plbdar**2)
        )
        / (plbdar**4)
    )

    # Raindrop accretion on small sized aggregates
    rraccss_tend = zzw1 * zzw_coef

    # Raindrop accretion on aggregates
    rraccs_tend = zzw2 * zzw_coef

    # Raindrop accretion-conversion to graupel
    rsaccrg_tend = (
        XFSACCRG
        * zzw3
        * (plbdas ** (XCXS - XBS))
        * (prhodref ** (-XCEXVT - 1.0))
        * (
            XLBSACCR1 / (plbdar**2)
            + XLBSACCR2 / (plbdar * plbdas)
            + XLBSACCR3 / (plbdas**2)
        )
        / plbdar
    )

    # Apply freezing rate limitations and temperature conditions
    if gacc and pt < XTT:
        # Apply freezing rate limits
        prraccss = min(zfreez_rate, rraccss_tend)
        zfreez_remaining = max(0.0, zfreez_rate - prraccss)

        # Proportion we can freeze
        zzw_prop = min(1.0, zfreez_remaining / max(1.0e-20, rraccs_tend - prraccss))

        prraccsg = zzw_prop * max(0.0, rraccs_tend - prraccss)
        zfreez_remaining = max(0.0, zfreez_remaining - prraccsg)

        prsaccrg = zzw_prop * rsaccrg_tend

        # Ensure positive values
        prsaccrg = prsaccrg * max(0.0, -sign(1.0, -prraccsg))
        prraccsg = max(0.0, prraccsg)
    else:
        prraccss = 0.0
        prraccsg = 0.0
        prsaccrg = 0.0


# GT4Py stencil
def conversion_melting_snow(
    prhodref: Field["float"],
    ppres: Field["float"],
    pdv: Field["float"],
    pka: Field["float"],
    pcj: Field["float"],
    plbdas: Field["float"],
    pt: Field["float"],
    prvt: Field["float"],
    prst: Field["float"],
    ldcompute: Field["bool"],
    ldsoft: "bool",
    levlimit: "bool",
    prsmltg: Field["float"],
    prcmltsr: Field["float"],
    rcrims_tend: Field["float"],
    rraccs_tend: Field["float"],
):
    """Compute conversion-melting of aggregates"""
    from __externals__ import (
        S_RTMIN,
        XEPSILO,
        XALPW,
        XBETAW,
        XGAMW,
        XTT,
        XLVTT,
        XCPV,
        XCL,
        XLMTT,
        XESTT,
        XRV,
        X0DEPS,
        X1DEPS,
        XEX0DEPS,
        XEX1DEPS,
        XFSCVMG,
    )

    with computation(PARALLEL), interval(...):
        if prst > S_RTMIN and pt > XTT and ldcompute:
            if not ldsoft:
                # Compute vapor pressure
                prs_ev = prvt * ppres / (XEPSILO + prvt)

                # Apply saturation limit if requested
                if levlimit:
                    prs_ev = min(prs_ev, exp(XALPW - XBETAW / pt - XGAMW * log(pt)))

                # Compute melting term
                prsmltg = pka * (XTT - pt) + (
                    pdv
                    * (XLVTT + (XCPV - XCL) * (pt - XTT))
                    * (XESTT - prs_ev)
                    / (XRV * pt)
                )

                # Compute RSMLT
                prsmltg = XFSCVMG * max(
                    0.0,
                    (
                        -prsmltg
                        * (X0DEPS * plbdas**XEX0DEPS + X1DEPS * pcj * plbdas**XEX1DEPS)
                        - (rcrims_tend + rraccs_tend) * (prhodref * XCL * (XTT - pt))
                    )
                    / (prhodref * XLMTT),
                )

                # Collection rate (both species liquid, no heat exchange)
                prcmltsr = rcrims_tend
        else:
            prsmltg = 0.0
            prcmltsr = 0.0
