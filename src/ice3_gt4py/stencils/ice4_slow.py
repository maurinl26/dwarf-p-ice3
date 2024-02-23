# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function, exp

from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection("ice4_rimltc")
def ice4_slow(
    ldcompute: Field["float"],  # boolean field to locate processes computation
    rhodref: Field["float"],
    t: Field["float"],
    ssi: Field["float"],  # supersaturation over ice
    lv_fact: Field["float"],
    ls_fact: Field["float"],
    rv_t: Field["float"],  # vapour mr at time t
    rc_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    lbdas: Field["float"],  # Slope parameter of the aggregate distribution
    lbdag: Field["float"],  # Slope parameter of the graupel distribution
    ai: Field["float"],  # Thermodynamical function PAI
    cj: Field["float"],  # to compute the ventilation coeff
    hli_hcf: Field["float"],
    hli_hri: Field["float"],
    rc_honi_tnd: Field["float"],
    rv_deps_tnd: Field["float"],
    ri_aggs_tnd: Field["float"],
    ri_auts_tnd: Field["float"],
    rv_depg_tnd: Field["float"],
):

    from __externals__ import (
        tt,
        c_rtmin,
        v_rtmin,
        s_rtmin,
        i_rtmin,
        g_rtmin,
        hon,
        alpha3,
        beta3,
        o0deps,
        ex0deps,
        ex1deps,
        o1deps,
        fiaggs,
        colexis,
        exiaggs,
        cexvt,
        xcriauti,
        acriauti,
        bcriauti,
        timauti,
        texauti,
        o0depg,
        ex0depg,
        o1depg,
        ex1depg,
    )

    # 3.2 compute the homogeneous nucleation source : RCHONI
    with computation(PARALLEL), interval(...):

        if t < tt - 35.0 and rc_t > c_rtmin and ldcompute == 1:
            rc_honi_tnd = min(
                1000, hon * rhodref * rc_t * exp(alpha3 * (t - tt) - beta3)
            )

        else:
            rc_honi_tnd = 0

    # 3.4 compute the deposition, aggregation and autoconversion sources
    # 3.4.3 compute the deposition on r_s : RVDEPS
    with computation(PARALLEL), interval(...):

        if rv_t < v_rtmin and rs_t < s_rtmin and ldcompute == 1:
            # Translation note : #ifdef REPRO48 l118 to 120 kept
            # Translation note : #else REPRO48  l121 to 126 omitted
            rv_deps_tnd = rv_deps_tnd = (ssi / (rhodref * ai)) * (
                o0deps * lbdas**ex0deps + o1deps * cj * lbdas**ex1deps
            )

        else:
            rv_deps_tnd = 0

    # 3.4.4 compute the aggregation on r_s: RIAGGS
    with computation(PARALLEL), interval(...):

        if ri_t > i_rtmin and rs_t > s_rtmin and ldcompute == 1:
            # Translation note : #ifdef REPRO48 l138 to 142 kept
            # Translation note : #else REPRO48 l143 to 150 omitted
            ri_aggs_tnd = (
                fiaggs
                * exp(colexis * (t - tt))
                * ri_t
                * lbdas**exiaggs
                * rhodref ** (-cexvt)
            )

        # Translation note : OELEC = False l151 omitted
        else:
            ri_aggs_tnd = 0

    # 3.4.5 compute the autoconversion of r_i for r_s production: RIAUTS
    with computation(PARALLEL), interval(...):

        if hli_hri > i_rtmin and ldcompute == 1:
            criauti = min(xcriauti, 10 ** (acriauti * (t - tt) + bcriauti))
            ri_auts_tnd = (
                timauti * exp(texauti * (t - tt)) * max(hli_hri - criauti * hli_hcf)
            )

        else:
            ri_auts_tnd = 0

    # 3.4.6 compute the depsoition on r_g: RVDEPG
    with computation(PARALLEL), interval(...):
        if rv_t > v_rtmin and rg_t > g_rtmin and ldcompute == 1:
            rv_depg_tnd = (ssi / (rhodref * ai)) * (
                o0depg * lbdag**ex0depg + o1depg * cj * lbdag**ex1depg
            )

        else:
            rv_depg_tnd = 0
