# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function, exp

from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection("ice4_slow")
def ice4_slow(
    ldcompute: Field["bool"],
    rhodref: Field["float"],
    t: Field["float"],
    ssi: Field["float"],
    lv_fact: Field["float"],
    ls_fact: Field["float"],
    rv_t: Field["float"],
    rc_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
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
):
    """Compute the slow processes

    Args:
        ldcompute (Field[float]): switch to activate processes computation on column
        rhodref (Field[float]): reference density
        t (Field[float]): temperature
        ssi (Field[float]): supersaturation over ice
        lv_fact (Field[float]): vaporisation latent heat over heat capacity
        ls_fact (Field[float]): sublimation latent heat over heat capacity
        rv_t (Field[float]): vapour mixing ratio at t
        ri_t (Field[float]): ice m.r. at t
        rs_t (Field[float]): snow m.r. at t
        rg_t (Field[float]): graupel m.r. at t
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
        TT,
        C_RTMIN,
        V_RTMIN,
        S_RTMIN,
        I_RTMIN,
        G_RTMIN,
        HON,
        ALPHA3,
        BETA3,
        O0DEPS,
        EX0DEPS,
        EX1DEPS,
        O1DEPS,
        FIAGGS,
        COLEXIS,
        EXIAGGS,
        CEXVT,
        CRIAUTI,
        ACRIAUTI,
        BCRIAUTI,
        TIMAUTI,
        TEXAUTI,
        O0DEPG,
        EX0DEPG,
        O1DEPG,
        EX1DEPG,
    )

    # 3.2 compute the homogeneous nucleation source : RCHONI
    with computation(PARALLEL), interval(...):

        if t < TT - 35.0 and rc_t > C_RTMIN and ldcompute:
            rc_honi_tnd = min(
                1000, HON * rhodref * rc_t * exp(ALPHA3 * (t - TT) - BETA3)
            )

        else:
            rc_honi_tnd = 0

    # 3.4 compute the deposition, aggregation and autoconversion sources
    # 3.4.3 compute the deposition on r_s : RVDEPS
    with computation(PARALLEL), interval(...):

        if rv_t < V_RTMIN and rs_t < S_RTMIN and ldcompute:
            # Translation note : #ifdef REPRO48 l118 to 120 kept
            # Translation note : #else REPRO48  l121 to 126 omitted
            rv_deps_tnd = (ssi / (rhodref * ai)) * (
                O0DEPS * lbdas**EX0DEPS + O1DEPS * cj * lbdas**EX1DEPS
            )

        else:
            rv_deps_tnd = 0

    # 3.4.4 compute the aggregation on r_s: RIAGGS
    with computation(PARALLEL), interval(...):

        if ri_t > I_RTMIN and rs_t > S_RTMIN and ldcompute:
            # Translation note : #ifdef REPRO48 l138 to 142 kept
            # Translation note : #else REPRO48 l143 to 150 omitted
            ri_aggs_tnd = (
                FIAGGS
                * exp(COLEXIS * (t - TT))
                * ri_t
                * lbdas**EXIAGGS
                * rhodref ** (-CEXVT)
            )

        # Translation note : OELEC = False l151 omitted
        else:
            ri_aggs_tnd = 0

    # 3.4.5 compute the autoconversion of r_i for r_s production: RIAUTS
    with computation(PARALLEL), interval(...):

        if hli_hri > I_RTMIN and ldcompute:
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
        if rv_t > V_RTMIN and rg_t > G_RTMIN and ldcompute:
            rv_depg_tnd = (ssi / (rhodref * ai)) * (
                O0DEPG * lbdag**EX0DEPG + O1DEPG * cj * lbdag**EX1DEPG
            )

        else:
            rv_depg_tnd = 0
