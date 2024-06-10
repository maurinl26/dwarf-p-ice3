# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import (
    Field,
    exp,
    log,
    sqrt,
    computation,
    PARALLEL,
    interval,
    __INLINED,
)
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_tendencies.F90",
    from_line=454,
    to_line=559,
)
@stencil_collection("ice4_tendencies_update")
def ice4_tendencies_update(
    ls_fact: Field["float"],
    lv_fact: Field["float"],
    theta_tnd: Field["float"],
    rv_tnd: Field["float"],
    rc_tnd: Field["float"],
    rr_tnd: Field["float"],
    ri_tnd: Field["float"],
    rs_tnd: Field["float"],
    rg_tnd: Field["float"],
    rchoni: Field["float"],
    rvdeps: Field["float"],
    riaggs: Field["float"],
    riauts: Field["float"],
    rvdepg: Field["float"],
    rcautr: Field["float"],
    rcaccr: Field["float"],
    rrevav: Field["float"],
    rcberi: Field["float"],
    rsmltg: Field["float"],
    rcmltsr: Field["float"],
    rraccss: Field["float"],
    rraccsg: Field["float"],
    rsaccrg: Field["float"],
    rcrimss: Field["float"],
    rcrimsg: Field["float"],
    rsrimcg: Field["float"],
    ricfrrg: Field["float"],
    rrcfrig: Field["float"],
    ricfrr: Field["float"],
    rcwetg: Field["float"],
    riwetg: Field["float"],
    rrwetg: Field["float"],
    rswetg: Field["float"],
    rcdryg: Field["float"],
    ridryg: Field["float"],
    rrdryg: Field["float"],
    rsdryg: Field["float"],
    rgmltr: Field["float"],
    rvheni_mr: Field["float"],
    rrhong_mr: Field["float"],
    rimltc_mr: Field["float"],
    rsrimcg_mr: Field["float"],
):
    """_summary_

    Args:
        ls_fact (Field[float]): _description_
        lv_fact (Field[float]): _description_
        theta_tnd (Field[float]): _description_
        rv_tnd (Field[float]): _description_
        rc_tnd (Field[float]): _description_
        rr_tnd (Field[float]): _description_
        ri_tnd (Field[float]): _description_
        rs_tnd (Field[float]): _description_
        rg_tnd (Field[float]): _description_
        rchoni (Field[float]): _description_
        rvdeps (Field[float]): _description_
        riaggs (Field[float]): _description_
        riauts (Field[float]): _description_
        rvdepg (Field[float]): _description_
        rcautr (Field[float]): _description_
        rcaccr (Field[float]): _description_
        rrevav (Field[float]): _description_
        rcberi (Field[float]): _description_
        rsmltg (Field[float]): _description_
        rcmltsr (Field[float]): _description_
        rraccss (Field[float]): _description_
        rraccsg (Field[float]): _description_
        rsaccrg (Field[float]): _description_
        rcrimss (Field[float]): _description_
        rcrimsg (Field[float]): _description_
        rsrimcg (Field[float]): _description_
        ricfrrg (Field[float]): _description_
        rrcfrig (Field[float]): _description_
        ricfrr (Field[float]): _description_
        rcwetg (Field[float]): _description_
        riwetg (Field[float]): _description_
        rrwetg (Field[float]): _description_
        rswetg (Field[float]): _description_
        rcdryg (Field[float]): _description_
        ridryg (Field[float]): _description_
        rrdryg (Field[float]): _description_
        rsdryg (Field[float]): _description_
        rgmltr (Field[float]): _description_
        rvheni_mr (Field[float]): _description_
        rrhong_mr (Field[float]): _description_
        rimltc_mr (Field[float]): _description_
        rsrimcg_mr (Field[float]): _description_
    """
    with computation(PARALLEL), interval(...):

        theta_tnd += (
            rvdepg * ls_fact
            + rchoni * (ls_fact - lv_fact)
            + rvdeps * ls_fact
            - rrevav * lv_fact
            + rcrimss * (ls_fact - lv_fact)
            + rcrimsg * (ls_fact - lv_fact)
            + rraccss * (ls_fact - lv_fact)
            + rraccsg * (ls_fact - lv_fact)
            + (rrcfrig - ricfrr) * (ls_fact - lv_fact)
            + (rcwetg + rrwetg) * (ls_fact - lv_fact)
            + (rcdryg + rrdryg) * (ls_fact - lv_fact)
            - rgmltr * (ls_fact - lv_fact)
            + rcberi * (ls_fact - lv_fact)
        )

        # (v)
        rv_tnd += -rvdepg - rvdeps + rrevav

        # (c)
        rc_tnd += (
            -rchoni
            - rcautr
            - rcaccr
            - rcrimss
            - rcrimsg
            - rcmltsr
            - rcwetg
            - rcdryg
            - rcberi
        )

        # (r)
        rr_tnd += (
            rcautr
            + rcaccr
            - rrevav
            - rraccss
            - rraccsg
            + rcmltsr
            - rrcfrig
            + ricfrr
            - rrwetg
            - rrdryg
            + rgmltr
        )

        # (i)
        ri_tnd += rchoni - riaggs - riauts - ricfrrg - ricfrr - riwetg - ridryg + rcberi

        # (s)
        rs_tnd += (
            rvdeps
            + riaggs
            + riauts
            + rcrimss
            - rcrimsg
            + rraccss
            - rsaccrg
            - rsmltg
            - rswetg
            - rsdryg
        )

        # (g)
        rg_tnd += (
            rvdepg
            + rcrimsg
            + rsrimcg
            + rraccsg
            + rsaccrg
            + rsmltg
            + ricfrrg
            + rrcfrig
            + rcwetg
            + riwetg
            + rswetg
            + rrwetg
            + rcdryg
            + ridryg
            + rsdryg
            + rrdryg
            - rgmltr
        )


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_tendencies.F90",
    from_line=220,
    to_line=238,
)
@stencil_collection("ice4_increment_update")
def ice4_increment_update(
    ls_fact: Field["float"],
    lv_fact: Field["float"],
    theta_increment: Field["float"],
    rv_increment: Field["float"],
    rc_increment: Field["float"],
    rr_increment: Field["float"],
    ri_increment: Field["float"],
    rs_increment: Field["float"],
    rg_increment: Field["float"],
    rvheni_mr: Field["float"],
    rimltc_mr: Field["float"],
    rrhong_mr: Field["float"],
    rsrimcg_mr: Field["float"],
):
    """Update tendencies with fixed increment.

    Args:
        ls_fact (Field[float]): _description_
        lv_fact (Field[float]): _description_
        theta_increment (Field[float]): _description_
        rv_increment (Field[float]): _description_
        rc_increment (Field[float]): _description_
        rr_increment (Field[float]): _description_
        ri_increment (Field[float]): _description_
        rs_increment (Field[float]): _description_
        rg_increment (Field[float]): _description_
        rvheni_mr (Field[float]): _description_
        rimltc_mr (Field[float]): _description_
        rrhong_mr (Field[float]): _description_
        rsrimcg_mr (Field[float]): _description_
    """

    # 5.1.6 riming-conversion of the large sized aggregates into graupel
    # Translation note : l189 to l215 omitted (since CSNOWRIMING = M90 in AROME)
    with computation(PARALLEL), interval(...):
        theta_increment += (
            rvheni_mr * ls_fact
            + rrhong_mr * (ls_fact - lv_fact)
            + rimltc_mr * (ls_fact - lv_fact)
        )

        rv_increment -= rvheni_mr
        rc_increment += rimltc_mr
        rr_increment -= rrhong_mr
        ri_increment += rvheni_mr - rimltc_mr
        rs_increment -= rsrimcg_mr
        rg_increment += rrhong_mr + rsrimcg_mr


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_tendencies.F90",
    from_line=220,
    to_line=238,
)
@stencil_collection("ice4_derived_fields")
def ice4_derived_fields(
    t: Field["float"],
    rhodref: Field["float"],
    pres: Field["float"],
    ssi: Field["float"],
    ka: Field["float"],
    dv: Field["float"],
    ai: Field["float"],
    cj: Field["float"],
    rv_t: Field["float"],
):

    from __externals__ import (
        ALPI,
        BETAI,
        GAMI,
        EPSILO,
        TT,
        CI,
        CPV,
        RV,
        P00,
        LSTT,
        SCFAC,
    )

    with computation(PARALLEL), interval(...):

        zw = exp(ALPI - BETAI / t - GAMI * log(t))
        ssi = rv_t * (pres - zw) / (EPSILO * zw)  # Supersaturation over ice
        ka = 2.38e-2 + 7.1e-5 * (t - TT)
        dv = 2.11e-5 * (t / TT) ** 1.94 * (P00 / pres)
        ai = (LSTT + (CPV - CI) * (t - TT)) ** 2 / (ka**RV * t**2) + (
            RV * t / (dv * zw)
        )
        cj = SCFAC * rhodref**0.3 / sqrt(1.718e-5 + 4.9 - 8 * (t - TT))


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_tendencies.F90",
    from_line=285,
    to_line=329,
)
@stencil_collection("ice4_slope_parameters")
def ice4_slope_parameters(
    rhodref: Field["float"],
    t: Field["float"],
    rr_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    lbdar: Field["float"],
    lbdar_rf: Field["float"],
    lbdas: Field["float"],
    lbdag: Field["float"],
):
    """Compute lambda parameters for distributions of falling species (r, s, g)

    Args:
        rhodref (Field[float]): reference dry density
        t (Field[float]): temperature
        rr_t (Field[float]): rain m.r. at t
        rs_t (Field[float]): snow m.r. at t
        rg_t (Field[float]): graupel m.r. at t
        lbdar (Field[float]): lambda parameter for rain distribution
        lbdar_rf (Field[float]): _description_
        lbdas (Field[float]): lambda parameter for snow distribution
        lbdag (Field[float]): lambda parameter for graupel distribution
    """

    from __externals__ import (
        TRANS_MP_GAMMAS,
        LBR,
        LBEXR,
        R_RTMIN,
        LSNOW_T,
        LBDAG_MAX,
        LBDAS_MIN,
        LBDAS_MAX,
        LBDAS_MIN,
        LBS,
        LBEXS,
        G_RTMIN,
        R_RTMIN,
        S_RTMIN,
    )

    with computation(PARALLEL), interval(...):

        lbdar = LBR * (rhodref * max(rr_t, R_RTMIN)) ** LBEXR if rr_t > 0 else 0
        # Translation note : l293 to l298 omitted LLRFR = True (not used in AROME)
        # Translation note : l299 to l301 kept (used in AROME)
        lbdar_rf = lbdar

        if __INLINED(LSNOW_T):
            if rs_t > 0 and t > 263.15:
                lbdas = (
                    max(min(LBDAS_MAX, 10 ** (14.554 - 0.0423 * t)), LBDAS_MIN)
                    * TRANS_MP_GAMMAS
                )
            elif rs_t > 0 and t <= 263.15:
                lbdas = (
                    max(min(LBDAS_MAX, 10 ** (6.226 - 0.0106 * t)), LBDAS_MIN)
                    * TRANS_MP_GAMMAS
                )
            else:
                lbdas = 0
        else:
            lbdas = (
                min(LBDAS_MAX, LBS * (rhodref * max(rs_t, S_RTMIN)) ** LBEXS)
                if rs_t > 0
                else 0
            )

        lbdag = (
            min(LBDAG_MAX, LBS * (rhodref * max(rg_t, G_RTMIN)) ** LBEXS)
            if rg_t > 0
            else 0
        )
