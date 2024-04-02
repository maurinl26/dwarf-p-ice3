# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, log, sqrt, computation, PARALLEL, interval
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
    rchoni: Field["float"],  # 1
    rvdeps: Field["float"],  # 2
    riaggs: Field["float"],  # 3  # Aggregation on r_s
    riauts: Field["float"],  # 4  # Autoconversion of r_i for r_s production
    rvdepg: Field["float"],  # 5  # Deposition on r_g
    rcautr: Field["float"],  # 6  # Autoconversion of r_c for r_r production
    rcaccr: Field["float"],  # 7  # Accretion of r_c for r_r production
    rrevav: Field["float"],  # 8  # Evaporation of r_r
    rcberi: Field["float"],  # 9  # Bergeron-Findeisen effect
    rsmltg: Field["float"],  # 11  # Conversion-Melting of the aggregates
    rcmltsr: Field[
        "float"
    ],  # 12  # Cloud droplet collection onto aggregates by positive temperature
    rraccss: Field["float"],  # 13
    rraccsg: Field["float"],  # 14
    rsaccrg: Field["float"],  # 15  # Rain accretion onto the aggregates
    rcrimss: Field["float"],  # 16
    rcrimsg: Field["float"],  # 17
    rsrimcg: Field["float"],  # 18  # Cloud droplet riming of the aggregates
    ricfrrg: Field["float"],  # 19
    rrcfrig: Field["float"],  # 20
    ricfrr: Field["float"],  # 21  # Rain contact freezing
    rcwetg: Field["float"],  # 22
    riwetg: Field["float"],  # 23
    rrwetg: Field["float"],  # 24
    rswetg: Field["float"],  # 25  # Graupel wet growth
    rcdryg: Field["float"],  # 26
    ridryg: Field["float"],  # 27
    rrdryg: Field["float"],  # 28
    rsdryg: Field["float"],  # 29  # Graupel dry growth
    rgmltr: Field["float"],  # 31  # Melting of the graupel
    rvheni_mr: Field["float"],  # 43  # heterogeneous nucleation mixing ratio change
    rrhong_mr: Field["float"],  # 44  # Spontaneous freezing mixing ratio change
    rimltc_mr: Field["float"],  # 45  # Cloud ce melting mixing ratio change
    rsrimcg_mr: Field["float"],  # 46  # Cloud droplet riming of the aggregates
):

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
    zw: Field["float"],
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

        if LSNOW_T:
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
            lbdas = min(LBDAS_MAX, LBS * (rhodref * max(rs_t, S_RTMIN)) ** LBEXS) if rs_t > 0 else 0
                
        lbdag = min(LBDAG_MAX, LBS * (rhodref * max(rg_t, G_RTMIN)) ** LBEXS) if rg_t > 0 else 0
        
