# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import PARALLEL, computation, interval, Field, __INLINED
from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection("cloud_fraction")
def cloud_fraction(
    lv: Field["float"],
    ls: Field["float"],
    t: Field["float"],
    cph: Field["float"],
    rhodref: Field["float"],
    exnref: Field["float"],
    rc: Field["float"],
    ri: Field["float"],
    ths: Field["float"],
    rvs: Field["float"],
    rcs: Field["float"],
    ris: Field["float"],
    rc_mf: Field["float"],
    ri_mf: Field["float"],
    cf_mf: Field["float"],
    cldfr: Field["float"],
    rc_tmp: Field["float"],
    ri_tmp: Field["float"],
    hlc_hrc: Field["float"],
    hlc_hcf: Field["float"],
    hli_hri: Field["float"],
    hli_hcf: Field["float"],
    dt: "float",
):
    """Cloud fraction computation (after condensation loop)"""

    from __externals__ import (
        LSUBG_COND,
        CRIAUTC,
        SUBG_MF_PDF,
        CRIAUTI,
        ACRIAUTI,
        BCRIAUTI,
        TT,
    )

    # l274 in ice_adjust.F90
    # 5. COMPUTE THE SOURCES AND STORES THE CLOUD FRACTION
    with computation(PARALLEL), interval(...):

        # 5.2  compute the cloud fraction cldfr
        if __INLINED(not LSUBG_COND):
            cldfr = 1 if (rcs + ris > 1e-12 / dt) else 0
        # Translation note : OCOMPUTE_SRC is taken False
        # Translation note : l320 to l322 removed

        # Translation note : LSUBG_COND = TRUE for Arome
        else:
            w1 = rc_mf / dt
            w2 = ri_mf / dt

            if w1 + w2 > rvs:
                w1 *= rvs / (w1 + w2)
                w2 = rvs - w1

            cldfr = min(1, cldfr + cf_mf)
            rcs += w1
            ris += w2
            rvs -= w1 + w2
            ths += (w1 * lv + w2 * ls) / cph / exnref

            # Droplets subgrid autoconversion
            # with computation(PARALLEL), interval(...):
            # LLHLC_H is True (AROME like)
            #
            criaut = CRIAUTC / rhodref

            # ice_adjust.F90 IF LLNONE; IF CSUBG_MF_PDF is None
            if __INLINED(SUBG_MF_PDF == 0):
                if w1 * dt > cf_mf * criaut:
                    hlc_hrc += w1 * dt
                    hlc_hcf = min(1, hlc_hcf + cf_mf)

            # Translation note : if LLTRIANGLE in .F90
            if __INLINED(SUBG_MF_PDF == 1):
                if w1 * dt > cf_mf * criaut:
                    hcf = 1 - 0.5 * (criaut * cf_mf / max(1e-20, w1 * dt)) ** 2
                    hr = w1 * dt - (criaut * cf_mf) ** 3 / (
                        3 * max(1e-20, w1 * dt) ** 2
                    )

                elif 2 * w1 * dt <= cf_mf * criaut:
                    hcf = 0
                    hr = 0

                else:
                    hcf = (2 * w1 * dt - criaut * cf_mf) ** 2 / (
                        2.0 * max(1.0e-20, w1 * dt) ** 2
                    )
                    hr = (
                        4.0 * (w1 * dt) ** 3
                        - 3.0 * w1 * dt * (criaut * cf_mf) ** 2
                        + (criaut * cf_mf) ** 3
                    ) / (3 * max(1.0e-20, w1 * dt) ** 2)

                hcf *= cf_mf
                hlc_hcf = min(1, hlc_hcf + hcf)
                hlc_hrc += hr

            # Ice subgrid autoconversion
            criaut = min(
                CRIAUTI,
                10 ** (ACRIAUTI * (t - TT) + BCRIAUTI),
            )

            if __INLINED(SUBG_MF_PDF == 0):
                if w2 * dt > cf_mf * criaut:
                    hli_hri += w2 * dt
                    hli_hcf = min(1, hli_hcf + cf_mf)

            if __INLINED(SUBG_MF_PDF == 1):
                if w2 * dt > cf_mf * criaut:
                    hli_hcf = 1 - 0.5 * ((criaut * cf_mf) / (w2 * dt)) ** 2
                    hli_hri = w2 * dt - (criaut * cf_mf) ** 3 / (3 * (w2 * dt) ** 2)

                elif 2 * w2 * dt <= cf_mf * criaut:
                    hli_hcf = 0
                    hli_hri = 0

                else:
                    hli_hcf = (2 * w2 * dt - criaut * cf_mf) ** 2 / (
                        2.0 * (w2 * dt) ** 2
                    )
                    hli_hri = (
                        4.0 * (w2 * dt) ** 3
                        - 3.0 * w2 * dt * (criaut * cf_mf) ** 2
                        + (criaut * cf_mf) ** 3
                    ) / (3 * (w2 * dt) ** 2)

                hli_hcf *= cf_mf
                hli_hcf = min(1, hli_hcf + hli_hcf)
                hli_hri += hli_hri

    # Translation note : 402 -> 427 (removed pout_x not present )
