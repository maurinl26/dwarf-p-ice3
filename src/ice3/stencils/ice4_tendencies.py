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
    from_line=129,
    to_line=134
)
@stencil_collection("init_tendencies")
def init_tendencies(
    rv_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    rv_inst: Field["float"],
    rc_inst: Field["float"],
    rr_inst: Field["float"],
    ri_inst: Field["float"],
    rs_inst: Field["float"],
    rg_inst: Field["float"],
    rv_tnd: Field["float"],
    rc_tnd: Field["float"],
    rr_tnd: Field["float"],
    ri_tnd: Field["float"],
    rs_tnd: Field["float"],
    rg_tnd: Field["float"]    
):
    
    with computation(PARALLEL), interval(...):
        
        rv_t = 0.0
        rc_t = 0.0
        rr_t = 0.0
        ri_t = 0.0
        rs_t = 0.0
        rg_t = 0.0
        rv_inst = 0.0
        rc_inst = 0.0
        rr_inst = 0.0
        ri_inst = 0.0
        rs_inst = 0.0
        rg_inst = 0.0
        rv_tnd = 0.0
        rc_tnd = 0.0
        rr_tnd = 0.0
        ri_tnd = 0.0
        rs_tnd = 0.0
        rg_tnd = 0.0
        
@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_tendencies.F90",
    from_line=136,
    to_line=140,
)
@stencil_collection("mixing_ratio_init")
def mixing_ratio_init(
    rvheni_mr: Field["float"],
    rrhong_mr: Field["float"],
    rimltc_mr: Field["float"],
    rsrimcg_mr: Field["float"],
    ldsoft: "bool"
):
    
    with computation(PARALLEL), interval(...):
        if ldsoft:
            rvheni_mr = 0.0
            rrhong_mr = 0.0
            rimltc_mr = 0.0
            rsrimcg_mr = 0.0


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_tendencies.F90",
    from_line=152,
    to_line=157,
)
@stencil_collection("ice4_nucleation_post_processing")
def ice4_nucleation_post_processing(
    t: Field["float"],
    exn: Field["float"],
    lsfact: Field["float"],
    tht: Field["float"],
    rvt: Field["float"],
    rit: Field["float"],
    rvheni_mr: Field["float"],
):
    """adjust mixing ratio with nucleation increments

    Args:
        t (Field[float]): temperature
        exn (Field[float]): exner pressure
        ls_fact (Field[float]): sublimation latent heat over heat capacity
        th_t (Field[float]): potential temperature
        rv_t (Field[float]): vapour m.r.
        ri_t (Field[float]): ice m.r.
        rvheni_mr (Field[float]): vapour m.r. increment due to HENI (heteroegenous nucleation over ice)
    """

    with computation(PARALLEL), interval(...):
        tht += rvheni_mr * lsfact
        t = tht * exn
        rvt -= rvheni_mr
        rit += rvheni_mr
        

@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_tendencies.F90",
    from_line=166,
    to_line=171,
)
@stencil_collection("ice4_rrhong_post_processing")
def ice4_rrhong_post_processing(
    t: Field["float"],
    exn: Field["float"],
    lsfact: Field["float"],
    lvfact: Field["float"],
    tht: Field["float"],
    rrt: Field["float"],
    rgt: Field["float"],
    rrhong_mr: Field["float"],
):
    """adjust mixing ratio with nucleation increments

    Args:
        t (Field[float]): temperature
        exn (Field[float]): exner pressure
        lsfact (Field[float]): sublimation latent heat over heat capacity
        tht (Field[float]): potential temperature
        rrt (Field[float]): rain m.r.
        rg_t (Field[float]): graupel m.r.
        rrhong (Field[float]): rain m.r. increment due to homogeneous nucleation
    """

    with computation(PARALLEL), interval(...):
        tht += rrhong_mr * (lsfact - lvfact)
        t = tht * exn
        rrt -= rrhong_mr
        rgt += rrhong_mr
        

@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_tendencies.F90",
    from_line=180,
    to_line=185,
)
@stencil_collection("ice4_rimltc_post_processing")
def ice4_rimltc_post_processing(
    t: Field["float"],
    exn: Field["float"],
    lsfact: Field["float"],
    lvfact: Field["float"],
    rimltc_mr: Field["float"],
    tht: Field["float"],
    rct: Field["float"],
    rit: Field["float"],
):
    """adjust mixing ratio with riming increments

    Args:
        t (Field[float]): temperature
        exn (Field[float]): exner pressure
        ls_fact (Field[float]): sublimation latent heat over heat capacity
        tht (Field[float]): potential temperature
        rr_t (Field[float]): rain m.r.
        rg_t (Field[float]): graupel m.r.
        rrhong (Field[float]): rain m.r. increment due to homogeneous nucleation
    """

    with computation(PARALLEL), interval(...):
        tht -= rimltc_mr * (lsfact - lvfact)
        t = tht * exn
        rct += rimltc_mr
        rit -= rimltc_mr


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_tendencies.F90",
    from_line=386,
    to_line=390,
)
@stencil_collection("ice4_fast_rg_pre_processing")
def ice4_fast_rg_pre_post_processing(
    rgsi: Field["float"],
    rgsi_mr: Field["float"],
    rvdepg: Field["float"],
    rsmltg: Field["float"],
    rraccsg: Field["float"],
    rsaccrg: Field["float"],
    rcrimsg: Field["float"],
    rsrimcg: Field["float"],
    rrhong_mr: Field["float"],
    rsrimcg_mr: Field["float"],
):

    with computation(PARALLEL), interval(...):
        rgsi = rvdepg + rsmltg + rraccsg + rsaccrg + rcrimsg + rsrimcg
        rgsi_mr = rrhong_mr + rsrimcg_mr


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_tendencies.F90",
    from_line=220,
    to_line=238,
)
@stencil_collection("ice4_increment_update")
def ice4_increment_update(
    lsfact: Field["float"],
    lvfact: Field["float"],
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
            rvheni_mr * lsfact
            + rrhong_mr * (lsfact - lvfact)
            - rimltc_mr * (lsfact - lvfact)
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
    rvt: Field["float"],
    zw: Field["float"]
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
        ssi = rvt * (pres - zw) / (EPSILO * zw) - 1.0 # Supersaturation over ice
        ka = 2.38e-2 + 7.1e-5 * (t - TT)
        dv = 2.11e-5 * (t / TT) ** 1.94 * (P00 / pres)
        ai = (LSTT + (CPV - CI) * (t - TT)) ** 2 / (ka**RV * t**2) + (
            (RV * t) / (dv * zw)
        )
        cj = SCFAC * rhodref**0.3 / sqrt(1.718e-5 + 4.9e-8 * (t - TT))


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_tendencies.F90",
    from_line=285,
    to_line=329,
)
@stencil_collection("ice4_slope_parameters")
def ice4_slope_parameters(
    rhodref: Field["float"],
    t: Field["float"],
    rrt: Field["float"],
    rst: Field["float"],
    rgt: Field["float"],
    lbdar: Field["float"],
    lbdar_rf: Field["float"],
    lbdas: Field["float"],
    lbdag: Field["float"],
):
    """Compute lambda parameters for distributions of falling species (r, s, g)

    Args:
        rhodref (Field[float]): reference dry density
        t (Field[float]): temperature
        rrt (Field[float]): rain m.r. at t
        rst (Field[float]): snow m.r. at t
        rgt (Field[float]): graupel m.r. at t
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
        LBG,
        LBEXS,
        LBEXG,
        G_RTMIN,
        R_RTMIN,
        S_RTMIN,
    )

    with computation(PARALLEL), interval(...):

        lbdar = LBR * (rhodref * max(rrt, R_RTMIN)) ** LBEXR if rrt > 0 else 0
        # Translation note : l293 to l298 omitted LLRFR = True (not used in AROME)
        # Translation note : l299 to l301 kept (used in AROME)
        lbdar_rf = lbdar

        if __INLINED(LSNOW_T):
            if rst > 0 and t > 263.15:
                lbdas = (
                    max(min(LBDAS_MAX, 10 ** (14.554 - 0.0423 * t)), LBDAS_MIN)
                    * TRANS_MP_GAMMAS
                )
            elif rst > 0 and t <= 263.15:
                lbdas = (
                    max(min(LBDAS_MAX, 10 ** (6.226 - 0.0106 * t)), LBDAS_MIN)
                    * TRANS_MP_GAMMAS
                )
            else:
                lbdas = 0
        else:
            lbdas = (
                min(LBDAS_MAX, LBS * (rhodref * max(rst, S_RTMIN)) ** LBEXS)
                if rst > 0
                else 0
            )

        lbdag = (
            LBG * (rhodref * max(rgt, G_RTMIN)) ** LBEXG
            if rgt > 0.0
            else 0
        )


@ported_method(
    from_file="PHYEX/src/common/micro/mode_ice4_tendencies.F90",
    from_line=454,
    to_line=559,
)
@stencil_collection("ice4_total_tendencies_update")
def ice4_total_tendencies_update(
    lsfact: Field["float"],
    lvfact: Field["float"],
    th_tnd: Field["float"],
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
    rwetgh: Field["float"]
):
    """Add contributions of processes to tendencies
    of microphysical processes.

    Args:
        lsfact (Field[float]): _description_
        lvfact (Field[float]): _description_
        th_tnd (Field[float]): _description_
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
    """
    with computation(PARALLEL), interval(...):

        th_tnd += (
            rvdepg * lsfact
            + rchoni * (lsfact - lvfact)
            + rvdeps * lsfact
            - rrevav * lvfact
            + rcrimss * (lsfact - lvfact)
            + rcrimsg * (lsfact - lvfact)
            + rraccss * (lsfact - lvfact)
            + rraccsg * (lsfact - lvfact)
            + (rrcfrig - ricfrr) * (lsfact - lvfact)
            + (rcwetg + rrwetg) * (lsfact - lvfact)
            + (rcdryg + rrdryg) * (lsfact - lvfact)
            - rgmltr * (lsfact - lvfact)
            + rcberi * (lsfact - lvfact)
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
            - rsrimcg
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
            - rwetgh
        )


