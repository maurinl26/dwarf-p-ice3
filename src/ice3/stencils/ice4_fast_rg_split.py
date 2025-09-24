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


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_fast_rg.F90",
               from_line=119,
               to_line=175)
@stencil_collection("rain_contact_freezing")
def rain_contact_freezing(
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
    cit: Field["float"],
    lbdar: Field["float"],
    lbdag: Field["float"],
    # outputs
    ricfrr: Field["float"],
    rrcfrig: Field["float"],
    ricfrrg: Field["float"],
    rg_ridry_tnd: Field["float"],
    rg_riwet_tnd: Field["float"],
):
    from __externals__ import (
        LCRFLIMIT,
        CEXVT,
        CI,
        CL,
        COLEXIG,
        COLIG,
        CXG,
        DG,
        EXICFRR,
        EXRCFRI,
        FCDRYG,
        FIDRYG,
        G_RTMIN,
        I_RTMIN,
        ICFRR,
        LVTT,
        R_RTMIN,
        RCFRI,
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


@stencil_collection("graupel_growth")
def graupel_growth(
    ldsoft: "bool",  # bool to update tendencies
    ldcompute: Field["bool"],
    t: Field["float"],
    rhodref: Field["float"],
    pres: Field["float"],
    rvt: Field["float"],
    rgt: Field["float"],
    ka: Field["float"],
    dv: Field["float"],
    cj: Field["float"],
    lbdar: Field["float"],
    lbdag: Field["float"],
    rg_rcdry_tnd: Field["float"],
    rg_ridry_tnd: Field["float"],
    rg_rsdry_tnd: Field["float"],
    rg_rrdry_tnd: Field["float"],
    rg_riwet_tnd: Field["float"],
    rg_rswet_tnd: Field["float"],
    rg_freez1_tnd: Field["float"],
    rg_freez2_tnd: Field["float"],
    rgmltr: Field["float"],
    zw_tmp: Field["float"],
    gdry: Field["bool"]
):
    from __externals__ import (
        ALPI,
        ALPW,
        BETAI,
        BETAW,
        BS,
        CI,
        CL,
        CPV,
        EPSILO,
        ESTT,
        EX0DEPG,
        EX1DEPG,
        FRDRYG,
        G_RTMIN,
        GAMI,
        GAMW,
        LEVLIMIT,
        LMTT,
        LNULLWETG,
        LVTT,
        LWETGPOST,
        O0DEPG,
        O1DEPG,
        RV,
        TT,
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
