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
"""
import numpy as np
from gt4py.cartesian.gtscript import Field
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
        ICFRR,
        EXICFRR,
        CEXVT,
        RCFRI,
        EXRCFRI,
        TT,
        CI,
        CL,
        LVTT,
    )

    with computation(PARALLEL), interval(...):
        if prit > I_RTMIN and prrt > R_RTMIN and ldcompute:
            if not ldsoft:
                # RICFRRG - pristine ice collection by rain leading to graupel
                pricfrrg = ICFRR * prit * plbdar**EXICFRR * prhodref ** (-CEXVT)

                # RRCFRIG - rain freezing by contact with pristine ice
                prrcfrig = (
                    RCFRI * pcit * plbdar**EXRCFRI * prhodref ** (-CEXVT - 1.0)
                )

                if lcrflimit:
                    # Limit based on heat balance
                    # Proportion of process that can take place
                    zzw_denominator = max(1.0e-20, LVTT * prrcfrig)
                    zzw_ratio = (pricfrrg * CI + prrcfrig * CL) * (TT - pt) / zzw_denominator
                    zzw_prop = max(0.0, min(1.0, zzw_ratio))

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
        TT,
        FCDRYG,
        FIDRYG,
        COLIG,
        COLEXIG,
        CXG,
        DG,
        CEXVT,
    )

    with computation(PARALLEL), interval(...):
        # Cloud droplet collection
        if prgt > G_RTMIN and prct > C_RTMIN and ldcompute:
            if not ldsoft:
                rcdryg_tend = plbdag ** (CXG - DG - 2.0) * prhodref ** (-CEXVT)
                rcdryg_tend = FCDRYG * prct * rcdryg_tend
            else:
                rcdryg_tend = 0.0
        else:
            rcdryg_tend = 0.0

        # Pristine ice collection
        if prgt > G_RTMIN and prit > I_RTMIN and ldcompute:
            if not ldsoft:
                base_tend = plbdag ** (CXG - DG - 2.0) * prhodref ** (-CEXVT)

                ridryg_tend = FIDRYG * exp(COLEXIG * (pt - TT)) * prit * base_tend

                riwetg_tend = ridryg_tend / (COLIG * exp(COLEXIG * (pt - TT)))
            else:
                ridryg_tend = 0.0
                riwetg_tend = 0.0
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
    ker_sdryg: NDArray,
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
    ldsoft: bool,
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
        TT,
        EPSILO,
        ALPI,
        BETAI,
        GAMI,
        LVTT,
        CPV,
        CL,
        CI,
        ESTT,
        RV,
        LMTT,
        O0DEPG,
        O1DEPG,
        EX0DEPG,
        EX1DEPG,
    )

    with computation(PARALLEL), interval(...):
        if prgt > G_RTMIN and ldcompute:
            if not ldsoft:
                # Compute vapor pressure
                prs_ev = prvt * ppres / (EPSILO + prvt)

                # Apply saturation limit if requested
                if levlimit:
                    prs_ev = min(prs_ev, exp(ALPI - BETAI / pt - GAMI * log(pt)))

                # Compute first freezing term
                freez1_tend = pka * (TT - pt) + (
                    pdv
                    * (LVTT + (CPV - CL) * (pt - TT))
                    * (ESTT - prs_ev)
                    / (RV * pt)
                )

                freez1_tend = (
                    freez1_tend
                    * (O0DEPG * plbdag**EX0DEPG + O1DEPG * pcj * plbdag**EX1DEPG)
                    / (prhodref * (LMTT - CL * (TT - pt)))
                )

                # Compute second freezing term
                freez2_tend = (prhodref * (LMTT + (CI - CL) * (TT - pt))) / (
                    prhodref * (LMTT - CL * (TT - pt))
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
                ldwetg = ldwetg and (pt < TT)

            # Determine if limited dry growth
            lldryg = (
                (pt < TT)
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
        TT,
        EPSILO,
        ALPW,
        BETAW,
        GAMW,
        LVTT,
        CPV,
        CL,
        ESTT,
        RV,
        LMTT,
        O0DEPG,
        O1DEPG,
        EX0DEPG,
        EX1DEPG,
    )

    with computation(PARALLEL), interval(...):
        if prgt > G_RTMIN and pt > TT and ldcompute:
            if not ldsoft:
                # Compute vapor pressure
                prs_ev = prvt * ppres / (EPSILO + prvt)

                # Apply saturation limit if requested
                if levlimit:
                    prs_ev = min(prs_ev, exp(ALPW - BETAW / pt - GAMW * log(pt)))

                # Compute melting rate
                zzw_temp = pka * (TT - pt) + pdv * (
                    LVTT + (CPV - CL) * (pt - TT)
                ) * (ESTT - prs_ev) / (RV * pt)

                prgmltr = max(
                    0.0,
                    (
                        -zzw_temp
                        * (O0DEPG * plbdag**EX0DEPG + O1DEPG * pcj * plbdag**EX1DEPG)
                        - (rcdryg_tend + rrdryg_tend) * (prhodref * CL * (TT - pt))
                    )
                    / (prhodref * LMTT),
                )
            else:
                prgmltr = 0.0
        else:
            prgmltr = 0.0
