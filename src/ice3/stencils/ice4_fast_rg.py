# -*- coding: utf-8 -*-
"""
Fast rain-graupel microphysical processes for ICE4 scheme.

This module implements rapid microphysical interactions between rain and
graupel particles, including rain contact freezing, dry/wet growth of graupel
by collection of cloud droplets, ice crystals, snow, and rain, plus graupel
melting. These "fast" processes occur on shorter timescales than nucleation
and vapor growth.

Source: PHYEX/src/common/micro/mode_ice4_fast_rg.F90
"""
from __future__ import annotations

from gt4py.cartesian.gtscript import (
                                        PARALLEL, Field, GlobalTable,
                                      computation, exp, interval, log)
from ice3.functions.interp_micro import (index_micro2d_dry_g,
                                         index_micro2d_dry_r,
                                         index_micro2d_dry_s)


def ice4_fast_rg(
    ldsoft: "bool",
    ldcompute: Field["bool"],
    t: Field["float"],
    rhodref: Field["float"],
    pres: Field["float"],
    rvt: Field["float"],
    rrt: Field["float"],
    rit: Field["float"],
    rgt: Field["float"],
    rct: Field["float"],
    rst: Field["float"],
    cit: Field["float"],
    ka: Field["float"],
    dv: Field["float"],
    cj: Field["float"],
    lbdar: Field["float"],
    lbdas: Field["float"],
    lbdag: Field["float"],
    ricfrrg: Field["float"],
    rrcfrig: Field["float"],
    ricfrr: Field["float"],
    rg_rcdry_tnd: Field["float"],
    rg_ridry_tnd: Field["float"],
    rg_rsdry_tnd: Field["float"],
    rg_rrdry_tnd: Field["float"],
    rg_riwet_tnd: Field["float"],
    rg_rswet_tnd: Field["float"],
    rg_freez1_tnd: Field["float"],
    rg_freez2_tnd: Field["float"],
    rgmltr: Field["float"],
    ker_sdryg: GlobalTable[float, (81, 81)],
    ker_rdryg: GlobalTable[float, (41, 41)],
    index_floor_s: Field["int"],
    index_floor_g: Field["int"],
    index_floor_r: Field["int"],
):
    """
    Compute fast graupel microphysical source terms.
    
    This function calculates tendencies for graupel growth/decay through
    various processes: rain contact freezing, dry growth collection,
    wet growth collection, and melting. It uses pre-computed lookup tables
    for collection efficiencies and determines wet vs dry growth mode based
    on environmental conditions.
    
    Parameters
    ----------
    ldsoft : bool
        If True, use previously computed tendencies without recalculation.
        If False, compute new tendencies.
    ldcompute : Field[bool]
        Mask indicating which grid points require computation.
    t : Field[float]
        Temperature (K).
    rhodref : Field[float]
        Reference air density (kg/m³).
    pres : Field[float]
        Pressure (Pa).
    rvt : Field[float]
        Water vapor mixing ratio at time t (kg/kg).
    rrt, rit, rgt, rct, rst : Field[float]
        Rain, ice, graupel, cloud, snow mixing ratios at time t (kg/kg).
    cit : Field[float]
        Ice crystal number concentration (m⁻³).
    ka : Field[float]
        Thermal conductivity of air (J/(m·s·K)).
    dv : Field[float]
        Diffusivity of water vapor in air (m²/s).
    cj : Field[float]
        Ventilation coefficient function.
    lbdar, lbdas, lbdag : Field[float]
        Slope parameters for rain, snow, graupel size distributions (m⁻¹).
    ricfrrg : Field[float]
        Output: Ice production from rain contact freezing (kg/kg/s).
    rrcfrig : Field[float]
        Output: Rain consumed by contact freezing producing graupel (kg/kg/s).
    ricfrr : Field[float]
        Output: Rain remaining after partial contact freezing (kg/kg/s).
    rg_rcdry_tnd : Field[float]
        Output: Cloud collection by graupel, dry growth (kg/kg/s).
    rg_ridry_tnd : Field[float]
        Output: Ice collection by graupel, dry growth (kg/kg/s).
    rg_rsdry_tnd : Field[float]
        Output: Snow collection by graupel, dry growth (kg/kg/s).
    rg_rrdry_tnd : Field[float]
        Output: Rain collection by graupel, dry growth (kg/kg/s).
    rg_riwet_tnd : Field[float]
        Output: Ice collection by graupel, wet growth (kg/kg/s).
    rg_rswet_tnd : Field[float]
        Output: Snow collection by graupel, wet growth (kg/kg/s).
    rg_freez1_tnd, rg_freez2_tnd : Field[float]
        Output: Freezing rate components for wet/dry growth determination.
    rgmltr : Field[float]
        Output: Graupel melting rate (kg/kg/s).
    ker_sdryg : GlobalTable[float, (40, 40)]
        Lookup table for snow-graupel dry growth collection kernel.
    ker_rdryg : GlobalTable[float, (40, 40)]
        Lookup table for rain-graupel dry growth collection kernel.
    index_floor_s, index_floor_g, index_floor_r : Field[int]
        Floor indices for table interpolation (used internally).
        
    Notes
    -----
    Process Overview:
    
    1. Rain Contact Freezing (T < 273.15 K):
       - Rain droplets freeze upon contact with ice crystals
       - Produces graupel (rrcfrig) and additional ice (ricfrrg)
       - LCRFLIMIT option limits freezing rate based on available latent heat
    
    2. Dry Growth Collection:
       - Occurs when collected water freezes immediately upon contact
       - Graupel collects: cloud droplets, ice, snow, rain
       - Uses pre-computed collection kernels from lookup tables
       - Bilinear interpolation on slope parameter space
    
    3. Wet vs Dry Growth Mode Determination:
       - Wet growth: Liquid water layer forms on graupel surface
       - Dry growth: All collected water freezes immediately
       - Mode depends on balance between collection and freezing rates
       - Controlled by LNULLWETG and LWETGPOST flags
    
    4. Graupel Melting (T > 273.15 K):
       - Heat transfer from warm environment melts graupel
       - Accounts for ventilation effects via cj parameter
       - Includes latent heat from concurrent collection processes
    
    External Parameters:
    - Microphysical constants: ICFRR, RCFRI, FCDRYG, FIDRYG, etc.
    - Size distribution parameters: CXG, DG, BS, CXS, etc.
    - Thermodynamic constants: TT, LVTT, LMTT, CI, CL, CPV, etc.
    - Threshold mixing ratios: G_RTMIN, R_RTMIN, I_RTMIN, S_RTMIN
    - Lookup table parameters: LBSDRYG1, LBSDRYG2, LBSDRYG3
    - Control flags: LCRFLIMIT, LEVLIMIT, LNULLWETG, LWETGPOST
    
    The function handles both recalculation mode (ldsoft=False) and
    reuse mode (ldsoft=True) for computational efficiency.
    """
    from __externals__ import (ALPI, ALPW, BETAI, BETAW, BS, CEXVT, CI, CL,
                               COLEXIG, COLIG, COLSG, CPV, CXG, CXS, DG,
                               EPSILO, ESTT, EX0DEPG, EX1DEPG, EXICFRR,
                               EXRCFRI, FCDRYG, FIDRYG, FRDRYG, FSDRYG,
                               G_RTMIN, GAMI, GAMW, I_RTMIN, ICFRR, LBSDRYG1,
                               LBSDRYG2, LBSDRYG3, LCRFLIMIT, LEVLIMIT, LMTT,
                               LNULLWETG, LVTT, LWETGPOST, O0DEPG, O1DEPG,
                               R_RTMIN, RCFRI, RV, S_RTMIN, TT)

    # 6.1 rain contact freezing
    with computation(PARALLEL), interval(...):
        if rit > I_RTMIN and rrt > R_RTMIN and ldcompute:
            # not LDSOFT : compute the tendencies
            if not ldsoft:
                ricfrrg = ICFRR * rit * lbdar**EXICFRR * rhodref ** (-CEXVT)
                rrcfrig = RCFRI * cit * lbdar**EXRCFRI * rhodref ** (-CEXVT)

                if LCRFLIMIT:
                    zw0d = max(
                        0,
                        min(
                            1,
                            (ricfrrg * CI + rrcfrig * CL)
                            * (TT - t)
                            / max(1e-20, LVTT * rrcfrig),
                        ),
                    )
                    rrcfrig = zw0d * rrcfrig
                    ricfrr = (1 - zw0d) * rrcfrig
                    ricfrrg = zw0d * ricfrrg

                else:
                    ricfrr = 0

        else:
            ricfrrg = 0
            rrcfrig = 0
            ricfrr = 0

    # 6.3 compute graupel growth
    with computation(PARALLEL), interval(...):
        if rgt > G_RTMIN and rct > R_RTMIN and ldcompute:
            if not ldsoft:
                rg_rcdry_tnd = lbdag ** (CXG - DG - 2.0) * rhodref ** (-CEXVT)
                rg_rcdry_tnd = rg_rcdry_tnd * FCDRYG * rct

        else:
            rg_rcdry_tnd = 0

        if rgt > G_RTMIN and rit > I_RTMIN and ldcompute:
            if not ldsoft:
                rg_ridry_tnd = lbdag ** (CXG - DG - 2.0) * rhodref ** (-CEXVT)
                rg_ridry_tnd = FIDRYG * exp(COLEXIG * (t - TT)) * rit * rg_ridry_tnd
                rg_riwet_tnd = rg_ridry_tnd / (COLIG * exp(COLEXIG * (t - TT)))

        else:
            rg_ridry_tnd = 0
            rg_riwet_tnd = 0

    # todo : move to dace
    # 6.2.1 wet and dry collection of rs on graupel
    # Translation note : l171 in mode_ice4_fast_rg.F90
    with computation(PARALLEL), interval(...):
        if rst > S_RTMIN and rgt > G_RTMIN and ldcompute:
            gdry = True  # GDRY is a boolean field in f90

        else:
            gdry = False
            rg_rsdry_tnd = 0
            rg_rswet_tnd = 0

    with computation(PARALLEL), interval(...):
        if (not ldsoft) and gdry:
            index_floor_s, index_float_s = index_micro2d_dry_s(lbdas)
            index_floor_g, index_float_g = index_micro2d_dry_g(lbdag)
            zw_tmp = index_float_g * (
                index_float_s * ker_sdryg.A[index_floor_g + 1, index_floor_s + 1]
                + (1 - index_float_s) * ker_sdryg.A[index_floor_g + 1, index_floor_s]
            ) + (1 - index_float_g) * (
                index_float_s * ker_sdryg.A[index_floor_g, index_floor_s + 1]
                + (1 - index_float_s) * ker_sdryg.A[index_floor_g, index_floor_s]
            )

    with computation(PARALLEL), interval(...):
        # Translation note : #ifdef REPRO48 l192 to l198 kept
        #                                   l200 to l206 removed
        if gdry:
            rg_rswet_tnd = (
                FSDRYG
                * zw_tmp
                / COLSG
                * (lbdas * (CXS - BS))
                * (lbdag**CXG)
                * (rhodref ** (-CEXVT))
                * (
                    LBSDRYG1 / (lbdag**2)
                    + LBSDRYG2 / (lbdag * lbdas)
                    + LBSDRYG3 / (lbdas**2)
                )
            )

            rg_rsdry_tnd = rg_rswet_tnd * COLSG * exp(t - TT)

    # todo : move to dace
    # 6.2.6 accretion of raindrops on the graupeln
    with computation(PARALLEL), interval(...):
        if rrt > R_RTMIN and rgt > G_RTMIN and ldcompute:
            gdry = True
        else:
            gdry = False
            rg_rrdry_tnd = 0

    with computation(PARALLEL), interval(...):
        if not ldsoft:
            index_floor_g, index_float_g = index_micro2d_dry_g(lbdag)
            index_floor_r, index_float_r = index_micro2d_dry_r(lbdar)
            zw_tmp = index_float_r * (
                index_float_g * ker_rdryg.A[index_floor_r + 1, index_floor_g + 1]
                + (1 - index_float_g) * ker_rdryg.A[index_floor_r + 1, index_floor_g]
            ) + (1 - index_float_r) * (
                index_float_g * ker_rdryg.A[index_floor_r, index_floor_g + 1]
                + (1 - index_float_g) * ker_rdryg.A[index_floor_r, index_floor_g]
            )

    # # l233
    with computation(PARALLEL), interval(...):
        if (not ldsoft) and gdry:
            rg_rrdry_tnd = (
                FRDRYG
                * zw_tmp
                * (lbdar ** (-4))
                * (lbdag**CXG)
                * (rhodref ** (-CEXVT - 1))
                * (
                    LBSDRYG1 / (lbdag**2)
                    + LBSDRYG2 / (lbdag * lbdar)
                    + LBSDRYG3 / (lbdar**2)
                )
            )

    # l245
    with computation(PARALLEL), interval(...):
        rdryg_init_tmp = rg_rcdry_tnd + rg_ridry_tnd + rg_rsdry_tnd + rg_rrdry_tnd

    # Translation note l300 to l316 removed (no hail)

    # Freezing rate and growth mode
    # Translation note : l251 in mode_ice4_fast_rg.F90
    with computation(PARALLEL), interval(...):
        if rgt > G_RTMIN and ldcompute:
            # Duplicated code with ice4_fast_rs
            if not ldsoft:
                rg_freez1_tnd = rvt * pres / (EPSILO + rvt)
                if LEVLIMIT:
                    rg_freez1_tnd = min(
                        rg_freez1_tnd, exp(ALPI - BETAI / t - GAMI * log(t))
                    )

                rg_freez1_tnd = ka * (TT - t) + dv * (LVTT + (CPV - CL) * (t - TT)) * (
                    ESTT - rg_freez1_tnd
                ) / (RV * t)
                rg_freez1_tnd *= (
                    O0DEPG * lbdag**EX0DEPG + O1DEPG * cj * lbdag**EX1DEPG
                ) / (rhodref * (LMTT - CL * (TT - t)))
                rg_freez2_tnd = (rhodref * (LMTT + (CI - CL) * (TT - t))) / (
                    rhodref * (LMTT - CL * (TT - t))
                )

            rwetg_init_tmp = max(
                rg_riwet_tnd + rg_rswet_tnd,
                max(0, rg_freez1_tnd + rg_freez2_tnd * (rg_riwet_tnd + rg_rswet_tnd)),
            )

            # Growth mode
            # bool calculation :
            ldwetg = (
                1
                if (
                    max(0, rwetg_init_tmp - rg_riwet_tnd - rg_rswet_tnd)
                    <= max(0, rdryg_init_tmp - rg_ridry_tnd - rg_rsdry_tnd)
                )
                else 0
            )

            if not LNULLWETG:
                ldwetg = 1 if (ldwetg == 1 and rdryg_init_tmp > 0) else 0

            else:
                ldwetg = 1 if (ldwetg == 1 and rwetg_init_tmp > 0) else 0

            if not LWETGPOST:
                ldwetg = 1 if (ldwetg == 1 and t < TT) else 0

            lldryg = (
                1
                if (
                    t < TT
                    and rdryg_init_tmp > 1e-20
                    and max(0, rwetg_init_tmp - rg_riwet_tnd - rg_rswet_tnd)
                    > max(0, rg_rsdry_tnd - rg_ridry_tnd - rg_rsdry_tnd)
                )
                else 0
            )

        else:
            rg_freez1_tnd = 0
            rg_freez2_tnd = 0
            rwetg_init_tmp = 0
            ldwetg = 0
            lldryg = 0

    # l317
    with computation(PARALLEL), interval(...):
        if ldwetg == 1:
            rr_wetg = -(rg_riwet_tnd + rg_rswet_tnd + rg_rcdry_tnd - rwetg_init_tmp)
            rc_wetg = rg_rcdry_tnd
            ri_wetg = rg_riwet_tnd
            rs_wetg = rg_rswet_tnd

        else:
            rr_wetg = 0
            rc_wetg = 0
            ri_wetg = 0
            rs_wetg = 0

        if lldryg == 1:
            rc_dry = rg_rcdry_tnd
            rr_dry = rg_rrdry_tnd
            ri_dry = rg_ridry_tnd
            rs_dry = rg_rsdry_tnd

        else:
            rc_dry = 0
            rr_dry = 0
            ri_dry = 0
            rs_dry = 0

    # 6.5 Melting of the graupel
    with computation(PARALLEL), interval(...):
        if rgt > G_RTMIN and t > TT and ldcompute:
            if not ldsoft:
                rgmltr = rvt * pres / (EPSILO + rvt)
                if LEVLIMIT:
                    rgmltr = min(rgmltr, exp(ALPW - BETAW / t - GAMW * log(t)))

                rgmltr = ka * (TT - t) + dv * (LVTT + (CPV - CL) * (t - TT)) * (
                    ESTT - rgmltr
                ) / (RV * t)
                rgmltr = max(
                    0,
                    (
                        -rgmltr
                        * (O0DEPG * lbdag**EX0DEPG + O1DEPG * cj * lbdag**EX1DEPG)
                        - (rg_rcdry_tnd + rg_rrdry_tnd) * (rhodref * CL * (TT - t))
                    )
                    / (rhodref * LMTT),
                )

        else:
            rgmltr = 0
