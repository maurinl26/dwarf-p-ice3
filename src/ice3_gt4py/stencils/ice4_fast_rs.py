# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, log

from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method
from ice3_gt4py.functions import sign


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
    grim_tmp: Field["int"],  # change to field bool
    gacc_tmp: Field["int"],
    zw1_tmp: Field["float"],  # used by interp_micro
    zw2_tmp: Field["float"],  # used by interp_micro
    zw3_tmp: Field["float"],
    ldsoft: bool,
):

    from __externals__ import (
        s_rtmin,
        c_rtmin,
        epsilo,
        levlimit,
        alpi,
        betai,
        gami,
        tt,
        cpv,
        lvtt,
        estt,
        Rv,
        o0deps,
        o1deps,
        ex0deps,
        ex1deps,
        lmtt,
        Cl,
        Ci,
        tt,
        crimss,
        excrimss,
        cexvt,
        crimsg,
        excrimsg,
        srimcg,
        exsrimcg,
        srimcg3,
        srimcg2,
        exsrimcg2,
        fraccss,
        cxs,
        lbraccs1,
        lbraccs2,
        lbraccs3,
        lbsaccr1,
        lbsaccr2,
        lbsaccr3,
        bs,
        fsaccrg,
    )

    # 5.0 maximum freezing rate
    with computation(PARALLEL), interval(...):

        # Translation note l106 removed not LDSOFT
        if rs_t < s_rtmin and ldcompute:

            rs_freez1_tnd = rv_t * pres / (epsilo + rv_t)
            if levlimit == 1:
                rs_freez1_tnd = min(
                    rs_freez1_tnd, exp(alpi - betai / t - gami * log(t))
                )

            rs_freez1_tnd = ka * (tt - t) + dv * (lvtt + (cpv - Cl) * (t - tt)) * (
                estt - rs_freez1_tnd
            ) / (Rv * t)

            # Translation note l115 to l177 kept #ifdef REPRO48
            # Translation note l119 to l122 removed #else REPRO48

            rs_freez1_tnd *= (
                o0deps * lbda_s**ex0deps + o1deps * cj * lbda_s**ex1deps
            ) / (rhodref * (lmtt - Cl * (tt - t)))
            rs_freez2_tnd = (rhodref * (lmtt + (Ci - Cl) * (tt - t))) / (
                rhodref * (lmtt - Cl * (tt - t))
            )

            # Translation note l129 removed

        else:

            rs_freez1_tnd = 0
            rs_freez2_tnd = 0
            freez_rate_tmp = 0

    # 5.1 cloud droplet riming of the aggregates
    with computation(PARALLEL), interval(...):

        if rc_t > c_rtmin and rs_t > s_rtmin and ldcompute:
            zw_tmp = lbda_s

            # Translation note : l144 kept
            #                    l146 removed

            grim_tmp = 1

        else:
            grim_tmp = 0
            rs_rcrims_tnd = 0
            rs_rcrimss_tnd = 0
            rs_rsrimcg_tnd = 0

    # TODO : interp_micro_1d l157 to l162
    # 5.1.4 riming of the small sized aggregates
    with computation(PARALLEL), interval(...):

        if grim_tmp:

            # Translation note : #ifdef REPRO48 l170 to l172 kept
            #                                   l174 to l178 removed
            rs_rcrimss_tnd = (
                crimss * zw1_tmp * rc_t * lbda_s**excrimss * rhodref ** (-cexvt)
            )

    # 5.1.6 riming convesion of the large size aggregates
    with computation(PARALLEL), interval(...):

        if grim_tmp:
            # Translation note : #ifdef REPRO48 l189 to l191 kept
            #                                   l193 to l197 removed
            rs_rcrims_tnd = crimsg * rc_t * lbda_s**excrimsg * rhodref ** (-cexvt)

    # if parami  csnowriming == M90
    with computation(PARALLEL), interval(...):

        # PARAMI%CSNOWRIMING == M90
        # TODO : refactor if statement out of stencil for performance
        if csnowriming == 0:

            if grim_tmp:
                zw_tmp = rs_rsrimcg_tnd - rs_rcrimss_tnd
                # Translation note : #ifdef REPRO48 l208 kept
                #                                   l210 and l211 removed
                rs_rsrimcg_tnd = srimcg * lbda_s**exsrimcg * (1 - zw2_tmp)

                # Translation note : #ifdef REPRO48 l214 to l217 kept
                #                                   l219 to l223 removed
                rs_rsrimcg_tnd = (
                    zw_tmp
                    * rs_rsrimcg_tnd
                    / max(
                        1e-20,
                        srimcg3 * srimcg2 * lbda_s**exsrimcg2 * (1 - zw3_tmp)
                        - srimcg3 * rs_rsrimcg_tnd,
                    )
                )

        else:
            rs_rsrimcg_tnd = 0

    #
    with computation(PARALLEL), interval(...):
        if grim_tmp and t < tt:
            rc_rimss_out = min(freez_rate_tmp, rs_rcrimss_tnd)
            freez_rate_tmp = max(0, freez_rate_tmp - rc_rimss_out)

            # proportion we are able to freeze
            zw0_tmp = min(1, freez_rate_tmp / max(1e-20, rs_rcrims_tnd - rc_rimss_out))
            rc_rimsg_out = zw0_tmp * max(0, rs_rcrims_tnd - rc_rimss_out)  # rc_rimsg
            freez_rate_tmp = max(0, freez_rate_tmp - rc_rimsg_out)
            rs_rimcg_out = zw0d * rs_rsrimcg_tnd

            rs_rimcg_out *= max(0, -sign(1, -rc_rimsg_out))
            rc_rimsg_out = max(0, rc_rimsg_out)

        else:
            rc_rimss_out = 0
            rc_rimsg_out = 0
            rs_rimcg_out = 0

    # 5.2. rain accretion onto the aggregates
    with computation(PARALLEL), interval(...):
        if rr_t > r_rtmin and rs_t > s_rtmin and ldcompute:
            gacc = True
        else:
            gacc = False
            rs_rraccs_tnd = 0
            rs_rraccss_tnd = 0
            rs_rsaccrg_tnd = 0

    # TODO: l264 to l272 interp_micro_2d
    with computation(PARALLEL), interval(...):

        # TODO: switch ldsoft statement evaluation with stencil declaration
        if ldsoft:
            rs_rraccs_tnd = 0
            rs_rraccss_tnd = 0
            rs_rsaccrg_tnd = 0

            # CALL INTERP_MICRO_2D

    # 5.2.4. raindrop accreation on the small sized aggregates
    with computation(PARALLEL), interval(...):

        if gacc_tmp:
            # Translation note : REPRO48 l279 to l283 kept
            #                            l285 to l289 removed

            zw_tmp = (
                fraccss
                * (lbda_s**cxs)
                * (rhodref ** (-cexvt))
                * (
                    lbraccs1 / (lbda_s**2)
                    + lbraccs2 / (lbda_s * lbda_r)
                    + lbraccs3 / (lbda_r**2)
                )
                / lbda_r**4
            )

    # 5.2.6 raindrop accretion-conversion of the large sized aggregates
    with computation(PARALLEL), interval(...):
        if gacc_tmp:
            rs_rsaccrg_tnd = (
                fsaccrg
                * zw3_tmp
                * (lbda_s ** (cxs - bs))
                * (rhodref ** (-cexvt - 1))
                * (
                    lbsaccr1 / (lbda_s**2)
                    + lbsaccr2 / (lbda_r * lbda_s)
                    + lbsaccr3 / (lbda_s**2)
                )
                / lbda_r
            )

    # l324
    # More restrictive ACC mask to be used for accretion by negative temperature only
    with computation(PARALLEL), interval(...):
        if gacc_tmp and t < tt:
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
        if rs_t < s_rtmin and t > tt and ldcompute:
            if not ldsoft:
                rs_mltg_tnd = rv_t * pres / (epsilo + rv_t)
                if levlimit:
                    rs_mltg_tnd = min(
                        rs_mltg_tnd, exp(alpw - betaw / t - gamw * log(t))
                    )
                rs_mltg_tnd = ka * (tt - t) + (
                    dv
                    * (lvtt + (cpv - Cl) * (t - tt))
                    * (estt - rs_mltg_tnd)
                    / (Rv * t)
                )

                # Tranlsation note : #ifdef REPRO48 l360 to l365 kept
                #                                   l367 to l374 removed
                rs_mltg_tnd = fscvmg * max(
                    0,
                    (
                        -rs_mltg_tnd
                        * (o0deps * lbda_s**ex0deps + o1deps * cj * lbda_s * ex1deps)
                        - (rs_rcrims_tnd + rs_rraccs_tnd) * (rhodref * Cl * (tt - t))
                    )
                    / (rhodref * lmtt),
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
