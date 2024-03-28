# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field
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
