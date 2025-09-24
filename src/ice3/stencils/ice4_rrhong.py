# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_rrhong.F90")
@stencil_collection("ice4_rrhong")
def ice4_rrhong(
    ldcompute: Field["bool"],
    t: Field["float"],
    exn: Field["float"],
    lvfact: Field["float"],
    lsfact: Field["float"],
    tht: Field["float"],  # theta at time t
    rrhong_mr: Field["float"],
    rrt: Field["float"],  # rain water mixing ratio at t
):
    """Compute the spontaneous frezzing source RRHONG

    Args:
        ldcompute (Field[bool]): switch to activate microphysical processes on column
        t (Field[float]): temperature at t
        exn (Field[float]): exner pressure
        lvfact (Field[float]): vaporisation latent heat
        lsfact (Field[float]): sublimation latent heat
        tht (Field[float]): potential temperature
        rrt (Field[float]): rain mixing ratio at t
        rrhong_mr (Field[float]): mixing ratio for spontaneous freezing source
    """

    from __externals__ import LFEEDBACKT, R_RTMIN, TT

    # 3.3 compute the spontaneous frezzing source: RRHONG
    with computation(PARALLEL), interval(...):
        if (
            t < TT - 35.0 
            and rrt > R_RTMIN 
            and ldcompute
        ):
            # limitation for -35 degrees crossing
            rrhong_mr = rrt
            if LFEEDBACKT:
                rrhong_mr = min(rrhong_mr, max(0., ((TT - 35.) / exn - tht) / (lsfact - lvfact)))

        else:
            rrhong_mr = 0.0

