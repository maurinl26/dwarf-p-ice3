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

import cupy as cp
import numpy as np
from gt4py.cartesian import Field
from numpy.typing import NDArray


def index_micro2d_dry_g(
    lambda_g: np.float32,
    DRYINTP1G: np.float32,
    DRYINTP2G: np.float32,
    NDRYLBDAG: np.int32,
):
    """Compute index in logspace for table (graupel dimension)

    Args:
        lambda_g: Graupel slope parameter
        DRYINTP1G: Scaling factor 1 for graupel
        DRYINTP2G: Scaling factor 2 for graupel
        NDRYLBDAG: Number of points in lookup table for graupel

    Returns:
        Tuple of (floor index, fractional part)
    """
    index = max(
        1.00001, min(NDRYLBDAG - 0.00001, DRYINTP1G * log(lambda_g) + DRYINTP2G)
    )
    return floor(index), index - floor(index)


def index_micro2d_dry_r(
    lambda_r: np.float32,
    DRYINTP1R: np.float32,
    DRYINTP2R: np.float32,
    NDRYLBDAR: np.int32,
):
    """Compute index in logspace for table (rain dimension)

    Args:
        lambda_r: Rain slope parameter
        DRYINTP1R: Scaling factor 1 for rain
        DRYINTP2R: Scaling factor 2 for rain
        NDRYLBDAR: Number of points in lookup table for rain

    Returns:
        Tuple of (floor index, fractional part)
    """
    index = max(
        1.00001, min(NDRYLBDAR - 0.00001, DRYINTP1R * log(lambda_r) + DRYINTP2R)
    )
    return floor(index), index - floor(index)


def index_micro2d_dry_s(
    lambda_s: np.float32,
    DRYINTP1S: np.float32,
    DRYINTP2S: np.float32,
    NDRYLBDAS: np.int32,
):
    """Compute index in logspace for table (snow dimension)

    Args:
        lambda_s: Snow slope parameter
        DRYINTP1S: Scaling factor 1 for snow
        DRYINTP2S: Scaling factor 2 for snow
        NDRYLBDAS: Number of points in lookup table for snow

    Returns:
        Tuple of (floor index, fractional part)
    """
    index = max(
        1.00001, min(NDRYLBDAS - 0.00001, DRYINTP1S * log(lambda_s) + DRYINTP2S)
    )
    return floor(index), index - floor(index)


# GT4Py stencil
def rain_contact_freezing(
    prhodref: Field["float"],
    plbdar: Field["float"],
    pt: Field["float"],
    prit: Field["float"],
    prrt: Field["float"],
    pcit: Field["float"],
    ldcompute: Field["bool"],
    ldsoft: "bool",
    lcrflimit: "bool",
    pricfrrg: Field["float"],
    prrcfrig: Field["float"],
    pricfrr: Field["float"],
):
    """Compute rain contact freezing"""
    from __externals__ import (
        I_RTMIN,
        R_RTMIN,
        XICFRR,
        XEXICFRR,
        XCEXVT,
        XRCFRI,
        XEXRCFRI,
        XTT,
        XCI,
        XCL,
        XLVTT,
        EPS,
    )

    with computation(PARALLEL), interval(...):
        if prit > I_RTMIN and prrt > R_RTMIN and ldcompute:
            if not ldsoft:
                # RICFRRG - pristine ice collection by rain leading to graupel
                pricfrrg = XICFRR * prit * plbdar**XEXICFRR * prhodref ** (-XCEXVT)

                # RRCFRIG - rain freezing by contact with pristine ice
                prrcfrig = (
                    XRCFRI * pcit * plbdar**XEXRCFRI * prhodref ** (-XCEXVT - 1.0)
                )

                if lcrflimit:
                    # Limit based on heat balance
                    # Proportion of process that can take place
                    zzw_prop = np.max(
                        0.0,
                        np.min(
                            1.0,
                            (pricfrrg * XCI + prrcfrig * XCL)
                            * (XTT - pt)
                            / np.max(EPS, XLVTT * prrcfrig),
                        ),
                    )

                    prrcfrig = zzw_prop * prrcfrig
                    pricfrr = (1.0 - zzw_prop) * pricfrrg
                    pricfrrg = zzw_prop * pricfrrg
                else:
                    pricfrr = 0.0
        else:
            pricfrrg = 0.0
            prrcfrig = 0.0
            pricfrr = 0.0


# GT4Py stencil
def cloud_pristine_collection_graupel(
    prhodref: Field["float"],
    plbdag: Field["float"],
    pt: Field["float"],
    prct: Field["float"],
    prit: Field["float"],
    prgt: Field["float"],
    ldcompute: Field["bool"],
    ldsoft: "bool",
    rcdryg_tend: Field["float"],
    ridryg_tend: Field["float"],
    riwetg_tend: Field["float"],
):
    """Compute wet and dry collection of cloud and pristine ice on graupel"""
    from __externals__ import (
        C_RTMIN,
        I_RTMIN,
        G_RTMIN,
        XTT,
        XFCDRYG,
        XFIDRYG,
        XCOLIG,
        XCOLEXIG,
        XCXG,
        XDG,
        XCEXVT,
    )

    with computation(PARALLEL), interval(...):
        # Cloud droplet collection
        if prgt > G_RTMIN and prct > C_RTMIN and ldcompute:
            if not ldsoft:
                rcdryg_tend = plbdag ** (XCXG - XDG - 2.0) * prhodref ** (-XCEXVT)
                rcdryg_tend = XFCDRYG * prct * rcdryg_tend
        else:
            rcdryg_tend = 0.0

        # Pristine ice collection
        if prgt > G_RTMIN and prit > I_RTMIN and ldcompute:
            if not ldsoft:
                base_tend = plbdag ** (XCXG - XDG - 2.0) * prhodref ** (-XCEXVT)

                ridryg_tend = XFIDRYG * exp(XCOLEXIG * (pt - XTT)) * prit * base_tend

                riwetg_tend = ridryg_tend / (XCOLIG * exp(XCOLEXIG * (pt - XTT)))
        else:
            ridryg_tend = 0.0
            riwetg_tend = 0.0


# Cupy for interpolation
def snow_collection_on_graupel(
    prhodref: NDArray,
    plbdas: NDArray,
    plbdag: NDArray,
    pt: NDArray,
    prst: NDArray,
    prgt: NDArray,
    ldcompute: NDArray[bool],
    ldsoft: bool,
    gdry: NDArray[bool],
    zzw: NDArray,
    rswetg_tend: NDArray,
    rsdryg_tend: NDArray,
    ker_sdryg: NDArray[80, 80],
    S_RTMIN: np.float32,
    G_RTMIN: np.float32,
    XTT: np.float32,
    XFSDRYG: np.float32,
    XCOLSG: np.float32,
    XCOLEXSG: np.float32,
    XCXS: np.float32,
    XBS: np.float32,
    XCXG: np.float32,
    XCEXVT: np.float32,
    XLBSDRYG1: np.float32,
    XLBSDRYG2: np.float32,
    XLBSDRYG3: np.float32,
    DRYINTP1G: np.float32,
    DRYINTP2G: np.float32,
    NDRYLBDAG: np.int32,
    DRYINTP1S: np.float32,
    DRYINTP2S: np.float32,
    NDRYLBDAS: np.int32,
):
    """Compute wet and dry collection of snow on graupel"""

    # Initialize masks
    gdry = prst > S_RTMIN and prgt > G_RTMIN and ldcompute
    rsdryg_tend = np.where(gdry, rsdryg_tend, 0.0)
    rswetg_tend = np.where(gdry, rswetg_tend, 0.0)

    # Interpolate and compute collection rates
    # TODO : cupy.take
    if (not ldsoft) and gdry:
            # Compute 2D interpolation indices
            idx_g, weight_g = index_micro2d_dry_g(
                plbdag, DRYINTP1G, DRYINTP2G, NDRYLBDAG
            )
            idx_g2 = idx_g + 1

            idx_s, weight_s = index_micro2d_dry_s(
                plbdas, DRYINTP1S, DRYINTP2S, NDRYLBDAS
            )
            idx_s2 = idx_s + 1

            # Bilinear interpolation
            zzw = (
                ker_sdryg.take(idx_g2, axis=0).take(idx_s2)* weight_s
                - ker_sdryg.take(idx_g2, axis=0).take(idx_s) * (weight_s - 1.0)
            ) * weight_g - (
                ker_sdryg.take(idx_g, axis=0).take(idx_s2) * weight_s
                - ker_sdryg.take(idx_g, axis=0).take(idx_s) * (weight_s - 1.0)
            ) * (weight_g - 1.0)

    # Compute wet growth rate
    rswetg_tend = (
        XFSDRYG
        * zzw
        / XCOLSG
        * (plbdas ** (XCXS - XBS))
        * (plbdag**XCXG)
        * (prhodref ** (-XCEXVT - 1.0))
        * (
            XLBSDRYG1 / (plbdag**2)
            + XLBSDRYG2 / (plbdag * plbdas)
            + XLBSDRYG3 / (plbdas**2)
        )
    )

    # Compute dry growth rate
    rsdryg_tend = rswetg_tend * XCOLSG * np.exp(XCOLEXSG * (pt - XTT))

    return (
        rswetg_tend,
        rsdryg_tend
    )


# Cupy for interpolation
def rain_accretion_on_graupel(
    prhodref: NDArray,
    plbdar: NDArray,
    plbdag: NDArray,
    prrt: NDArray,
    prgt: NDArray,
    ldcompute: NDArray,
    ldsoft: np.bool,
    zzw: NDArray,
    rrdryg_tend: NDArray,
    ker_rdryg: NDArray,
    R_RTMIN: np.float32,
    G_RTMIN: np.float32,
    XFRDRYG: np.float32,
    XCXG: np.float32,
    XCEXVT: np.float32,
    XLBRDRYG1: np.float32,
    XLBRDRYG2: np.float32,
    XLBRDRYG3: np.float32,
    DRYINTP1G: np.float32,
    DRYINTP2G: np.float32,
    NDRYLBDAG: np.int32,
    DRYINTP1R: np.float32,
    DRYINTP2R: np.float32,
    NDRYLBDAR: np.int32,
):
    """Compute accretion of raindrops on graupel"""

    # Initialize masks
    gdry = prrt > R_RTMIN and prgt > G_RTMIN and ldcompute
    rrdryg_tend = np.where(gdry, rrdryg_tend, 0)

    # Interpolate and compute accretion rates
    # TODO : move to cupy.take
    if (not ldsoft) and gdry:
        
        # Compute 2D interpolation indices
        idx_g, weight_g = index_micro2d_dry_g(
            plbdag, DRYINTP1G, DRYINTP2G, NDRYLBDAG
            )
        idx_g2 = idx_g + 1
        
        idx_r, weight_r = index_micro2d_dry_r(
            plbdar, DRYINTP1R, DRYINTP2R, NDRYLBDAR
            )
        idx_r2 = idx_r + 1

        # Bilinear interpolation
        zzw = (
                ker_rdryg.take(idx_g2, axis=0).take(idx_r2) * weight_r,
                - ker_rdryg.take(idx_g2, axis=0).take(idx_r) * (weight_r - 1.)
            ) * weight_g - (
                ker_rdryg.take(idx_g, axis=0).take(idx_r) * weight_r
                - ker_rdryg.take(idx_g, axis=0).take(idx_r) * (weight_r - 1.0)
            ) * (weight_g - 1.0)
        
    return zzw

    # Compute dry growth rate
    rrdryg_tend = (
        XFRDRYG
        * zzw
        * (plbdar ** (-4.0))
        * (plbdag**XCXG)
        * (prhodref ** (-XCEXVT - 1.0))
        * (
            XLBRDRYG1 / (plbdag**2)
            + XLBRDRYG2 / (plbdag * plbdar)
            + XLBRDRYG3 / (plbdar**2)
        )
    )


# GT4Py stencil
def compute_graupel_growth_mode(
    prhodref: Field["float"],
    ppres: Field["float"],
    pdv: Field["float"],
    pka: Field["float"],
    pcj: Field["float"],
    plbdag: Field["float"],
    pt: Field["float"],
    prvt: Field["float"],
    prgt: Field["float"],
    prgsi: Field["float"],
    prgsi_mr: Field["float"],
    pricfrrg: Field["float"],
    prrcfrig: Field["float"],
    ldcompute: Field["float"],
    ldsoft: "bool",
    levlimit: "bool",
    lnullwetg: "bool",
    lwetgpost: "bool",
    krr: "int",
    ldwetg: Field["bool"],
    lldryg: Field["bool"],
    zrdryg_init: Field["float"],
    zrwetg_init: Field["float"],
    prwetgh: Field["float"],
    prwetgh_mr: Field["float"],
    prcwetg: Field["float"],
    priwetg: Field["float"],
    prrwetg: Field["float"],
    prswetg: Field["float"],
    prcdryg: Field["float"],
    pridryg: Field["float"],
    prrdryg: Field["float"],
    prsdryg: Field["float"],
    rcdryg_tend: Field["float"],
    ridryg_tend: Field["float"],
    riwetg_tend: Field["float"],
    rsdryg_tend: Field["float"],
    rswetg_tend: Field["float"],
    rrdryg_tend: Field["float"],
    freez1_tend: Field["float"],
    freez2_tend: Field["float"],
):
    """Determine graupel growth mode (wet vs dry) and compute final tendencies"""

    from __externals__ import (
        G_RTMIN,
        XTT,
        XEPSILO,
        XALPI,
        XBETAI,
        XGAMI,
        XLVTT,
        XCPV,
        XCL,
        XCI,
        XESTT,
        XRV,
        XLMTT,
        X0DEPG,
        X1DEPG,
        XEX0DEPG,
        XEX1DEPG,
    )

    with computation(PARALLEL), interval(...):
        if prgt > G_RTMIN and ldcompute:
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
                    * (X0DEPG * plbdag**XEX0DEPG + X1DEPG * pcj * plbdag**XEX1DEPG)
                    / (prhodref * (XLMTT - XCL * (XTT - pt)))
                )

                # Compute second freezing term
                freez2_tend = (prhodref * (XLMTT + (XCI - XCL) * (XTT - pt))) / (
                    prhodref * (XLMTT - XCL * (XTT - pt))
                )

            # Initial dry growth rate
            zrdryg_init = rcdryg_tend + ridryg_tend + rsdryg_tend + rrdryg_tend

            # Initial wet growth rate
            zrwetg_init = max(
                riwetg_tend + rswetg_tend,
                max(0.0, freez1_tend + freez2_tend * (riwetg_tend + rswetg_tend)),
            )

            # Determine if wet growth
            ldwetg = max(0.0, zrwetg_init - riwetg_tend - rswetg_tend) <= max(
                0.0, zrdryg_init - ridryg_tend - rsdryg_tend
            )

            if lnullwetg:
                ldwetg = ldwetg and (zrdryg_init > 0.0)
            else:
                ldwetg = ldwetg and (zrwetg_init > 0.0)

            if not lwetgpost:
                ldwetg = ldwetg and (pt < XTT)

            # Determine if limited dry growth
            lldryg = (
                (pt < XTT)
                and (zrdryg_init > 1.0e-20)
                and (
                    max(0.0, zrwetg_init - riwetg_tend - rswetg_tend)
                    > max(0.0, zrdryg_init - ridryg_tend - rsdryg_tend)
                )
            )
        else:
            freez1_tend = 0.0
            freez2_tend = 0.0
            zrdryg_init = 0.0
            zrwetg_init = 0.0
            ldwetg = False
            lldryg = False

    # Conversion to hail removed (if KRR == 7)

    # Compute final wet and dry growth tendencies
    with computation(PARALLEL), interval(...):
        # Wet growth (aggregated minus collected)
        if ldwetg:
            prrwetg = -(riwetg_tend + rswetg_tend + rcdryg_tend - zrwetg_init)
            prcwetg = rcdryg_tend
            priwetg = riwetg_tend
            prswetg = rswetg_tend
        else:
            prrwetg = 0.0
            prcwetg = 0.0
            priwetg = 0.0
            prswetg = 0.0

        # Dry growth (limited)
        if lldryg:
            prcdryg = rcdryg_tend
            prrdryg = rrdryg_tend
            pridryg = ridryg_tend
            prsdryg = rsdryg_tend
        else:
            prcdryg = 0.0
            prrdryg = 0.0
            pridryg = 0.0
            prsdryg = 0.0


# GT4Py stencil
def graupel_melting(
    prhodref: Field["float"],
    ppres: Field["float"],
    pdv: Field["float"],
    pka: Field["float"],
    pcj: Field["float"],
    plbdag: Field["float"],
    pt: Field["float"],
    prvt: Field["float"],
    prgt: Field["float"],
    ldcompute: Field["bool"],
    ldsoft: "bool",
    levlimit: "bool",
    prgmltr: Field["float"],
    rcdryg_tend: Field["float"],
    rrdryg_tend: Field["float"],
):
    """Compute melting of graupel"""

    from __externals__ import (
        G_RTMIN,
        XTT,
        XEPSILO,
        XALPW,
        XBETAW,
        XGAMW,
        XLVTT,
        XCPV,
        XCL,
        XESTT,
        XRV,
        XLMTT,
        X0DEPG,
        X1DEPG,
        XEX0DEPG,
        XEX1DEPG,
    )

    with computation(PARALLEL), interval(...):
        if prgt > G_RTMIN and pt > XTT and ldcompute:
            if not ldsoft:
                # Compute vapor pressure
                prs_ev = prvt * ppres / (XEPSILO + prvt)

                # Apply saturation limit if requested
                if levlimit:
                    prs_ev = min(prs_ev, exp(XALPW - XBETAW / pt - XGAMW * log(pt)))

                # Compute melting rate
                prgmltr = pka * (XTT - pt) + pdv * (
                    XLVTT + (XCPV - XCL) * (pt - XTT)
                ) * (XESTT - prs_ev) / (XRV * pt)

                prgmltr = max(
                    0.0,
                    (
                        -prgmltr
                        * (X0DEPG * plbdag**XEX0DEPG + X1DEPG * pcj * plbdag**XEX1DEPG)
                        - (rcdryg_tend + rrdryg_tend) * (prhodref * XCL * (XTT - pt))
                    )
                    / (prhodref * XLMTT),
                )
        else:
            prgmltr = 0.0
