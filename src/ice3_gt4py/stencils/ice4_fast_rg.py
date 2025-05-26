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
    index_micro2d_dry_g,
    index_micro2d_dry_r,
    index_micro2d_dry_s,
)


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_fast_rg.F90")
@stencil_collection("ice4_fast_rg")
def ice4_fast_rg(
    ldsoft: "bool",  # bool to update tendencies
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
    ker_sdryg: GlobalTable[float, (40, 40)],
    ker_rdryg: GlobalTable[float, (40, 40)],
    index_floor_s: Field["int"],
    index_floor_g: Field["int"],
    index_floor_r: Field["int"],
):
    """Compute fast graupel sources

    Args:
        ldcompute (Field[int]): switch to compute microphysical processes on column
        t (Field[float]): temperature
        rhodref (Field[float]): reference density
        rit (Field[float]): ice mixing ratio at t

        rgt (Field[float]): graupel m.r. at t
        rct (Field[float]): cloud droplets m.r. at t
        rst (Field[float]): snow m.r. at t
        cit (Field[float]): _description_
        dv (Field[float]): diffusivity of water vapor
        ka (Field[float]): thermal conductivity of the air
        cj (Field[float]): function to compute the ventilation coefficient
        lbdar (Field[float]): slope parameter for rain
        lbdas (Field[float]): slope parameter for snow
        lbdag (Field[float]): slope parameter for graupel
        ricfrrg (Field[float]): rain contact freezing
        rrcfrig (Field[float]): rain contact freezing
        ricfrr (Field[float]): rain contact freezing
        rg_rcdry_tnd (Field[float]): Graupel wet growth
        rg_ridry_tnd (Field[float]): Graupel wet growth
        rg_riwet_tnd (Field[float]): Graupel wet growth
        rg_rsdry_tnd (Field[float]): Graupel wet growth
        rg_rswet_tnd (Field[float]): Graupel wet growth
        gdry (Field[int]): boolean field
    """

    from __externals__ import (
        LCRFLIMIT,
        ALPI,
        ALPW,
        BETAI,
        BETAW,
        BS,
        CEXVT,
        CI,
        CL,
        COLEXIG,
        COLIG,
        COLSG,
        CPV,
        CXG,
        CXS,
        DG,
        EPSILO,
        ESTT,
        EX0DEPG,
        EX1DEPG,
        EXICFRR,
        EXRCFRI,
        FCDRYG,
        FIDRYG,
        FRDRYG,
        FSDRYG,
        G_RTMIN,
        GAMI,
        GAMW,
        I_RTMIN,
        ICFRR,
        LBSDRYG1,
        LBSDRYG2,
        LBSDRYG3,
        LEVLIMIT,
        LMTT,
        LNULLWETG,
        LVTT,
        LWETGPOST,
        O0DEPG,
        O1DEPG,
        R_RTMIN,
        RCFRI,
        RV,
        S_RTMIN,
        TT,
    )

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
        if rrt < R_RTMIN and rgt < G_RTMIN and ldcompute:
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

