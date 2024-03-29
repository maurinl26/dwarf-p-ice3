# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    Field,
    GlobalTable,
    exp,
    log,
    computation,
    interval,
    PARALLEL,
)
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method

from ice3_gt4py.functions.interp_micro import (
    index_interp_micro_1d,
    index_interp_micro_2d_rs,
    interp_micro_1d,
    interp_micro_2d,
)
from ice3_gt4py.functions.sign import sign


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_fast_rs.F90")
@stencil_collection("ice4_fast_rs")
def ice4_fast_rs(
    ldcompute: Field["bool"],
    rhodref: Field["float"],
    lv_fact: Field["float"],
    ls_fact: Field["float"],
    pres: Field["float"],  # absolute pressure at t
    dv: Field["float"],  # diffusivity of water vapor in the air
    ka: Field["float"],  # thermal conductivity of the air
    cj: Field["float"],  # function to compute the ventilation coefficient
    lbda_r: Field["float"],  # Slope parameter for rain distribution
    lbda_s: Field["float"],  # Slope parameter for snow distribution
    t: Field["float"],  # Temperature
    rv_t: Field["float"],  # vapour m.r. at t
    rc_t: Field["float"],
    rr_t: Field["float"],
    rs_t: Field["float"],
    ri_aggs: Field["float"],  # ice aggregation on snow
    rc_rimss_out: Field["float"],  # cloud droplet riming of the aggregates
    rc_rimsg_out: Field["float"],
    rs_rimcg_out: Field["float"],
    rr_accss_out: Field["float"],  # rain accretion onto the aggregates
    rr_accsg_out: Field["float"],
    rs_accrg_out: Field["float"],
    rs_mltg_tnd: Field["float"],  # conversion-melting of the aggregates
    rc_mltsr_tnd: Field["float"],  # cloud droplet collection onto aggregates
    rs_rcrims_tnd: Field["float"],  # extra dimension 8 in Fortran PRS_TEND
    rs_rcrimss_tnd: Field["float"],
    rs_rsrimcg_tnd: Field["float"],
    rs_rraccs_tnd: Field["float"],
    rs_rraccss_tnd: Field["float"],
    rs_rsaccrg_tnd: Field["float"],
    rs_freez1_tnd: Field["float"],
    rs_freez2_tnd: Field["float"],
    grim_tmp: Field["bool"],
    gacc_tmp: Field["bool"],
    zw1_tmp: Field["float"],
    zw2_tmp: Field["float"],
    zw3_tmp: Field["float"],
    ldsoft: "bool",
    gaminc_rim1: GlobalTable[float, (80)],
    gaminc_rim2: GlobalTable[float, (80)],
    gaminc_rim4: GlobalTable[float, (80)],
    ker_raccs: GlobalTable[float, (40, 40)],
    ker_raccss: GlobalTable[float, (40, 40)],
    ker_saccrg: GlobalTable[float, (40, 40)],
):
    from __externals__ import (
        ALPI,
        ALPW,
        BETAI,
        BETAW,
        BS,
        C_RTMIN,
        CEXVT,
        CI,
        CL,
        CPV,
        CRIMSG,
        CRIMSS,
        CXS,
        EPSILO,
        ESTT,
        EX0DEPS,
        EX1DEPS,
        EXCRIMSG,
        EXCRIMSS,
        EXSRIMCG,
        EXSRIMCG2,
        FRACCSS,
        FSACCRG,
        FSCVMG,
        GAMI,
        GAMW,
        LBRACCS1,
        LBRACCS2,
        LBRACCS3,
        LBSACCR1,
        LBSACCR2,
        LBSACCR3,
        LEVLIMIT,
        LMTT,
        LVTT,
        O0DEPS,
        O1DEPS,
        R_RTMIN,
        RV,
        S_RTMIN,
        SNOW_RIMING,
        SRIMCG,
        SRIMCG2,
        SRIMCG3,
        TT,
    )

    # 5.0 maximum freezing rate
    with computation(PARALLEL), interval(...):
        # Translation note l106 removed not LDSOFT
        if rs_t < S_RTMIN and ldcompute:
            rs_freez1_tnd = rv_t * pres / (EPSILO + rv_t)
            if LEVLIMIT:
                rs_freez1_tnd = min(
                    rs_freez1_tnd, exp(ALPI - BETAI / t - GAMI * log(t))
                )

            rs_freez1_tnd = ka * (TT - t) + dv * (LVTT + (CPV - CL) * (t - TT)) * (
                ESTT - rs_freez1_tnd
            ) / (RV * t)

            # Translation note l115 to l177 kept #ifdef REPRO48
            # Translation note l119 to l122 removed #else REPRO48

            rs_freez1_tnd *= (
                O0DEPS * lbda_s**EX0DEPS + O1DEPS * cj * lbda_s**EX1DEPS
            ) / (rhodref * (LMTT - CL * (TT - t)))
            rs_freez2_tnd = (rhodref * (LMTT + (CI - CL) * (TT - t))) / (
                rhodref * (LMTT - CL * (TT - t))
            )

            # Translation note l129 removed
            freez_rate_tmp = 0

        else:
            rs_freez1_tnd = 0
            rs_freez2_tnd = 0
            freez_rate_tmp = 0

    # 5.1 cloud droplet riming of the aggregates
    with computation(PARALLEL), interval(...):
        if rc_t > C_RTMIN and rs_t > S_RTMIN and ldcompute:
            zw_tmp = lbda_s

            # Translation note : l144 kept
            #                    l146 removed

            grim_tmp = True

        else:
            grim_tmp = False
            rs_rcrims_tnd = 0
            rs_rcrimss_tnd = 0
            rs_rsrimcg_tnd = 0

    # Interpolation + Lookup Table
    with computation(PARALLEL), interval(...):
        if (not ldsoft) and grim_tmp:
            # Translation note : LDPACK is False l46 to l88 removed in interp_micro.func.h
            #                                    l90 to l123 kept
            index = index_interp_micro_1d(zw_tmp)
            zw1_tmp = interp_micro_1d(index, gaminc_rim1)
            zw2_tmp = interp_micro_1d(index, gaminc_rim2)
            zw3_tmp = interp_micro_1d(index, gaminc_rim4)

    # 5.1.4 riming of the small sized aggregates
    with computation(PARALLEL), interval(...):
        if grim_tmp:
            # Translation note : #ifdef REPRO48 l170 to l172 kept
            #                                   l174 to l178 removed
            rs_rcrimss_tnd = (
                CRIMSS * zw1_tmp * rc_t * lbda_s**EXCRIMSS * rhodref ** (-CEXVT)
            )

    # 5.1.6 riming convesion of the large size aggregates
    with computation(PARALLEL), interval(...):
        if grim_tmp:
            # Translation note : #ifdef REPRO48 l189 to l191 kept
            #                                   l193 to l197 removed
            rs_rcrims_tnd = CRIMSG * rc_t * lbda_s**EXCRIMSG * rhodref ** (-CEXVT)

    # if parami  csnowriming == M90
    with computation(PARALLEL), interval(...):
        # PARAMI%CSNOWRIMING == M90
        # TODO : refactor if statement out of stencil for performance
        if SNOW_RIMING == 0:
            if grim_tmp:
                zw_tmp = rs_rsrimcg_tnd - rs_rcrimss_tnd
                # Translation note : #ifdef REPRO48 l208 kept
                #                                   l210 and l211 removed
                rs_rsrimcg_tnd = SRIMCG * lbda_s**EXSRIMCG * (1 - zw2_tmp)

                # Translation note : #ifdef REPRO48 l214 to l217 kept
                #                                   l219 to l223 removed
                rs_rsrimcg_tnd = (
                    zw_tmp
                    * rs_rsrimcg_tnd
                    / max(
                        1e-20,
                        SRIMCG3 * SRIMCG2 * lbda_s**EXSRIMCG2 * (1 - zw3_tmp)
                        - SRIMCG3 * rs_rsrimcg_tnd,
                    )
                )

        else:
            rs_rsrimcg_tnd = 0

    #
    with computation(PARALLEL), interval(...):
        if grim_tmp and t < TT:
            rc_rimss_out = min(freez_rate_tmp, rs_rcrimss_tnd)
            freez_rate_tmp = max(0, freez_rate_tmp - rc_rimss_out)

            # proportion we are able to freeze
            zw0_tmp = min(1, freez_rate_tmp / max(1e-20, rs_rcrims_tnd - rc_rimss_out))
            rc_rimsg_out = zw0_tmp * max(0, rs_rcrims_tnd - rc_rimss_out)  # rc_rimsg
            freez_rate_tmp = max(0, freez_rate_tmp - rc_rimsg_out)
            rs_rimcg_out = zw0_tmp * rs_rsrimcg_tnd

            rs_rimcg_out *= max(0, -sign(1, -rc_rimsg_out))
            rc_rimsg_out = max(0, rc_rimsg_out)

        else:
            rc_rimss_out = 0
            rc_rimsg_out = 0
            rs_rimcg_out = 0

    # 5.2. rain accretion onto the aggregates
    with computation(PARALLEL), interval(...):
        if rr_t > R_RTMIN and rs_t > S_RTMIN and ldcompute:
            gacc_tmp = True
        else:
            gacc_tmp = False
            rs_rraccs_tnd = 0
            rs_rraccss_tnd = 0
            rs_rsaccrg_tnd = 0

    with computation(PARALLEL), interval(...):
        # Translation note : LDPACK is False l159 to l223 removed in interp_micro.func.h
        #                                    l226 to l266 kept

        if (not ldsoft) and gacc_tmp:
            rs_rraccs_tnd = 0
            rs_rraccss_tnd = 0
            rs_rsaccrg_tnd = 0

            index_r, index_s = index_interp_micro_2d_rs(lbda_r, lbda_s)

            zw1_tmp = interp_micro_2d(index_r, index_s, ker_raccss)
            zw2_tmp = interp_micro_2d(index_r, index_s, ker_raccs)
            zw3_tmp = interp_micro_2d(index_r, index_s, ker_saccrg)

            # CALL INTERP_MICRO_2D

    # 5.2.4. raindrop accreation on the small sized aggregates
    with computation(PARALLEL), interval(...):
        if gacc_tmp:
            # Translation note : REPRO48 l279 to l283 kept
            #                            l285 to l289 removed

            zw_tmp = (
                FRACCSS
                * (lbda_s**CXS)
                * (rhodref ** (-CEXVT))
                * (
                    LBRACCS1 / (lbda_s**2)
                    + LBRACCS2 / (lbda_s * lbda_r)
                    + LBRACCS3 / (lbda_r**2)
                )
                / lbda_r**4
            )

    # 5.2.6 raindrop accretion-conversion of the large sized aggregates
    with computation(PARALLEL), interval(...):
        if gacc_tmp:
            rs_rsaccrg_tnd = (
                FSACCRG
                * zw3_tmp
                * (lbda_s ** (CXS - BS))
                * (rhodref ** (-CEXVT - 1))
                * (
                    LBSACCR1 / (lbda_s**2)
                    + LBSACCR2 / (lbda_r * lbda_s)
                    + LBSACCR3 / (lbda_s**2)
                )
                / lbda_r
            )

    # l324
    # More restrictive ACC mask to be used for accretion by negative temperature only
    with computation(PARALLEL), interval(...):
        if gacc_tmp and t < TT:
            rr_accss_out = min(freez_rate_tmp, rs_rraccss_tnd)
            freez_rate_tmp = max(0, freez_rate_tmp - rr_accss_out)

            # proportion we are able to freeze
            zw_tmp = min(1, freez_rate_tmp / max(1e-20, rs_rraccss_tnd - rr_accss_out))
            rr_accsg_out = zw_tmp * max(0, rs_rraccs_tnd - rr_accss_out)
            freez_rate_tmp = max(0, freez_rate_tmp - rr_accsg_out)
            rs_accrg_out = zw_tmp * rs_rsaccrg_tnd

            rs_accrg_out *= max(0, -sign(1, -rr_accsg_out))
            rr_accsg_out = max(0, rr_accsg_out)

        else:
            rr_accss_out = 0
            rr_accsg_out = 0
            rs_accrg_out = 0

    # 5.3 Conversion-Melting of the aggregates
    with computation(PARALLEL), interval(...):
        if rs_t < S_RTMIN and t > TT and ldcompute:
            if not ldsoft:
                rs_mltg_tnd = rv_t * pres / (EPSILO + rv_t)
                if LEVLIMIT:
                    rs_mltg_tnd = min(
                        rs_mltg_tnd, exp(ALPW - BETAW / t - GAMW * log(t))
                    )
                rs_mltg_tnd = ka * (TT - t) + (
                    dv
                    * (LVTT + (CPV - CL) * (t - TT))
                    * (ESTT - rs_mltg_tnd)
                    / (RV * t)
                )

                # Tranlsation note : #ifdef REPRO48 l360 to l365 kept
                #                                   l367 to l374 removed
                rs_mltg_tnd = FSCVMG * max(
                    0,
                    (
                        -rs_mltg_tnd
                        * (O0DEPS * lbda_s**EX0DEPS + O1DEPS * cj * lbda_s * EX1DEPS)
                        - (rs_rcrims_tnd + rs_rraccs_tnd) * (rhodref * CL * (TT - t))
                    )
                    / (rhodref * LMTT),
                )

                # note that RSCVMG = RSMLT*XFSCVMG but no heat is exchanged (at the rate RSMLT)
                # because the graupeln produced by this process are still icy###
                #
                # When T < XTT, rc is collected by snow (riming) to produce snow and graupel
                # When T > XTT, if riming was still enabled, rc would produce snow and graupel with snow becomming graupel (conversion/melting) and graupel becomming rain (melting)
                # To insure consistency when crossing T=XTT, rc collected with T>XTT must be transformed in rain.
                # rc cannot produce iced species with a positive temperature but is still collected with a good efficiency by snow

                rc_mltsr_tnd = rs_rcrims_tnd

        else:
            rs_mltg_tnd = 0
            rc_mltsr_tnd = 0
