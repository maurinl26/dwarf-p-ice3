# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, computation, interval, PARALLEL
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_rimltc.F90")
@stencil_collection("ice4_rimltc")
def ice4_rimltc(
    ldcompute: Field["bool"],
    t: Field["float"],
    exn: Field["float"],
    lvfact: Field["float"],
    lsfact: Field["float"],
    tht: Field["float"],  # theta at time t
    rit: Field["float"],  # rain water mixing ratio at t
    rimltc_mr: Field["float"],
):
    """Compute cloud ice melting process RIMLTC

    Args:
        ldcompute (Field[bool]): switch to activate microphysical sources computation on column
        t (Field[float]): temperature
        exn (Field[float]): exner pressure
        lvfact (Field[float]): vaporisation latent heat
        lsfact (Field[float]): sublimation latent heat
        tht (Field[float]): potential temperature at t
        ri_t (Field[float]): cloud ice mixing ratio at t
        rimltc_mr (Field[float]): mixing ratio change due to cloud ice melting
    """

    from __externals__ import LFEEDBACKT, TT

    with computation(PARALLEL), interval(...):
        # 7.1 cloud ice melting
        if rit > 0 and t > TT and ldcompute:
            rimltc_mr = rit

            # limitation due to zero crossing of temperature
            if LFEEDBACKT:
                rimltc_mr = min(
                    rimltc_mr, max(0, (tht - TT / exn) / (lsfact - lvfact))
                )

        else:
            rimltc_mr = 0

