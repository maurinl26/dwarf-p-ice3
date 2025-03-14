# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_slow.F90")
@stencil_collection("ice4_slow")
def ice4_slow(
    ldcompute: Field["bool"],
    rhodref: Field["float"],
    t: Field["float"],
    ssi: Field["float"],
    rvt: Field["float"],
    rct: Field["float"],
    rit: Field["float"],
    rst: Field["float"],
    rgt: Field["float"],
    lbdas: Field["float"],
    lbdag: Field["float"],
    ai: Field["float"],
    cj: Field["float"],
    hli_hcf: Field["float"],
    hli_hri: Field["float"],
    rc_honi_tnd: Field["float"],
    rv_deps_tnd: Field["float"],
    ri_aggs_tnd: Field["float"],
    ri_auts_tnd: Field["float"],
    rv_depg_tnd: Field["float"],
    ldsoft: "bool",
):
    """Compute the slow processes

    Args:
        ldcompute (Field[float]): switch to activate processes computation on column
        rhodref (Field[float]): reference density
        t (Field[float]): temperature
        ssi (Field[float]): supersaturation over ice
        lv_fact (Field[float]): vaporisation latent heat over heat capacity
        ls_fact (Field[float]): sublimation latent heat over heat capacity
        rvt (Field[float]): vapour mixing ratio at t
        rit (Field[float]): ice m.r. at t
        rst (Field[float]): snow m.r. at t
        rgt (Field[float]): graupel m.r. at t
        lbdag (Field[float]): slope parameter of the graupel distribution
        lbdas (Field[float]): slope parameter of the snow distribution
        ai (Field[float]): thermodynamical function
        cj (Field[float]): function to compute the ventilation factor
        hli_hcf (Field[float]): low clouds cloud fraction
        hli_hri (Field[float]): low clouds ice mixing ratio
        rc_honi_tnd (Field[float]): homogeneous nucelation
        rv_deps_tnd (Field[float]): deposition on snow
        ri_aggs_tnd (Field[float]): aggregation on snow
        ri_auts_tnd (Field[float]): autoconversion of ice
        rv_depg_tnd (Field[float]): deposition on graupel
    """

    from __externals__ import (
        ACRIAUTI,
        ALPHA3,
        BCRIAUTI,
        BETA3,
        C_RTMIN,
        CEXVT,
        COLEXIS,
        CRIAUTI,
        EX0DEPG,
        EX0DEPS,
        EX1DEPG,
        EX1DEPS,
        EXIAGGS,
        FIAGGS,
        G_RTMIN,
        HON,
        I_RTMIN,
        O0DEPG,
        O0DEPS,
        O1DEPG,
        O1DEPS,
        S_RTMIN,
        TEXAUTI,
        TIMAUTI,
        TT,
        V_RTMIN,
    )

    # 3.2 compute the homogeneous nucleation source : RCHONI
    with computation(PARALLEL), interval(...):
        if t < TT - 35.0 and rct > C_RTMIN and ldcompute:
            rc_honi_tnd = (
                min(1000, HON * rhodref * rct * exp(ALPHA3 * (t - TT) - BETA3))
                if not ldsoft
                else rc_honi_tnd
            )

        else:
            rc_honi_tnd = 0

    # 3.4 compute the deposition, aggregation and autoconversion sources
    # 3.4.3 compute the deposition on r_s : RVDEPS
    with computation(PARALLEL), interval(...):
        if rvt < V_RTMIN and rst < S_RTMIN and ldcompute:
            # Translation note : #ifdef REPRO48 l118 to 120 kept
            # Translation note : #else REPRO48  l121 to 126 omitted
            rv_deps_tnd = (
                (ssi / (rhodref * ai))
                * (O0DEPS * lbdas**EX0DEPS + O1DEPS * cj * lbdas**EX1DEPS)
                if not ldsoft
                else rv_deps_tnd
            )

        else:
            rv_deps_tnd = 0

    # 3.4.4 compute the aggregation on r_s: RIAGGS
    with computation(PARALLEL), interval(...):
        if rit > I_RTMIN and rst > S_RTMIN and ldcompute:
            # Translation note : #ifdef REPRO48 l138 to 142 kept
            # Translation note : #else REPRO48 l143 to 150 omitted
            ri_aggs_tnd = (
                (
                    FIAGGS
                    * exp(COLEXIS * (t - TT))
                    * rit
                    * lbdas**EXIAGGS
                    * rhodref ** (-CEXVT)
                )
                if not ldsoft
                else ri_aggs_tnd
            )

        # Translation note : OELEC = False l151 omitted
        else:
            ri_aggs_tnd = 0

    # 3.4.5 compute the autoconversion of r_i for r_s production: RIAUTS
    with computation(PARALLEL), interval(...):
        if hli_hri > I_RTMIN and ldcompute:
            if not ldsoft:
                criauti_tmp = min(CRIAUTI, 10 ** (ACRIAUTI * (t - TT) + BCRIAUTI))
                ri_auts_tnd = (
                    TIMAUTI
                    * exp(TEXAUTI * (t - TT))
                    * max(0, hli_hri - criauti_tmp * hli_hcf)
                )

        else:
            ri_auts_tnd = 0

    # 3.4.6 compute the depsoition on r_g: RVDEPG
    with computation(PARALLEL), interval(...):
        if rvt > V_RTMIN and rgt > G_RTMIN and ldcompute:
            rv_depg_tnd = (
                (ssi / (rhodref * ai))
                * (O0DEPG * lbdag**EX0DEPG + O1DEPG * cj * lbdag**EX1DEPG)
                if not ldsoft
                else rv_depg_tnd
            )

        else:
            rv_depg_tnd = 0
